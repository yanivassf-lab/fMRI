import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn import datasets
import argparse
from scipy import stats

from settings import BLOCKS, roi_vis_prob
from heatmap_design import color_labels, add_correlation_to_heatmap, create_bold_annot_matrix

# ==========================================
# CONFIGURATION
# ==========================================
TR = 0.75
HRF_DURATION = 32.0


def plot_temporal(temporal_sig, pc_name, peak_delay, output_path):
    plt.figure(figsize=(15, 5))
    times = np.arange(0, len(temporal_sig)) * TR
    plt.plot(times, temporal_sig, color='black', linewidth=1.5, label='Temporal')
    max_time = max(times)
    ticks = np.arange(0, max_time + 20, 20)
    plt.xticks(ticks, rotation=45)

    lag = int(peak_delay)

    for s, e, cond in BLOCKS:
        # Convert sec to TR indices
        s_idx = int(s) + lag
        e_idx = int(e) + lag

        if cond == 'v':
            plt.axvspan(s_idx, e_idx, color='red', alpha=0.3,
                        label='Visual (v)' if 'Visual (v)' not in plt.gca().get_legend_handles_labels()[1] else "")
        elif cond == 'a':
            plt.axvspan(s_idx, e_idx, color='green', alpha=0.3,
                        label='Auditory (a)' if 'Auditory (a)' not in plt.gca().get_legend_handles_labels()[1] else "")
        elif cond == 'va':
            plt.axvspan(s_idx, e_idx, color='orange', alpha=0.3,
                        label='Both (va)' if 'Both (va)' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title(f"(Temporal Trace for {pc_name})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Temporal Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    output_file = os.path.join(output_path, f'temporal_{pc_name}.png')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {output_file}")


def spm_hrf(tr, peak_delay=6, duration=30):
    dt = tr
    t = np.arange(0, duration, dt)
    hrf = stats.gamma.pdf(t, peak_delay+1)
    return hrf / np.sum(hrf)


def calculate_regressor(n_points, blocks, tr, peak_delay, stimulus_duration_sec=None):
    """
    Creates the 'Ideal' Regressors.
    1. Visual Model: Active during 'v' .
    2. Auditory Model: Active during 'a' .
    3. Interaction Model: Active during 'va'.

    Includes Convolution with HRF and Demeaning.
    Parameters:
    stimulus_duration_sec:
        If None -> Uses the full block length (Block Design).
        If value (e.g., 1.0) -> Limits the stimulus to first X seconds (Onset/Short Epoch).
    """
    timeline_len = n_points
    box_v = np.zeros(timeline_len)
    box_a = np.zeros(timeline_len)
    box_va = np.zeros(timeline_len)  # Interaction / Synergy

    for s, e, cond in blocks:
        idx_s = int(s / tr)

        # calculate effective end index
        # If defined fixed duration - use it. Else, take original block end.
        if stimulus_duration_sec is not None:
            # Conversion from seconds to indices (e.g., 1 second / 0.75 = 1.33 -> 1 index)
            duration_idx = int(stimulus_duration_sec / tr)
            # Verify at least one index will be colored (not 0)
            duration_idx = max(1, duration_idx)
            idx_e_eff = idx_s + duration_idx
        else:
            idx_e_eff = int(e / tr)

        # Prevent exceeding array length
        idx_e_eff = min(idx_e_eff, timeline_len)

        if idx_s >= timeline_len:
            continue

        if cond == 'v':
            box_v[idx_s:idx_e_eff] = 1
        if cond == 'a':  # Captures 'a'
            box_a[idx_s:idx_e_eff] = 1
        if cond == 'va':
            box_va[idx_s:idx_e_eff] = 1

    # Convolve with HRF (Simulates the delay and dispersion of blood flow)
    hrf = spm_hrf(tr, peak_delay)
    reg_v = np.convolve(box_v, hrf)[:timeline_len]
    reg_a = np.convolve(box_a, hrf)[:timeline_len]
    reg_va = np.convolve(box_va, hrf)[:timeline_len]

    # === Demeaning (Crucial) ===
    # Subtracting the mean makes the regressors centered around zero,
    # just like the PCA components.
    # Positive values = Activation, Negative values = Rest/Suppression.
    return pd.DataFrame({
        'Model_Visual': reg_v,
        'box_v': box_v,
        'Model_Auditory': reg_a,
        'box_a': box_a,
        'Model_Interaction': reg_va,
        'box_va': box_va
    })


def calculate_corr(recon_sig, design_df):
    """
    Calculates correlation ONLY during the active blocks of each condition.
    Does not zero-out data (which inflates stats), but removes non-relevant timepoints.
    """

    # === 1. Visual Correlation ===
    # בחר רק את האינדקסים שבהם הייתה קופסת וידאו פעילה
    # (אפשר להוסיף שיהוי קטן אם רוצים לתפוס את הזנב, אבל זה המימוש הטהור לבקשתך)
    mask_v = design_df['box_v'] > 0

    if np.sum(mask_v) > 2:  # מוודא שיש מספיק נקודות
        # חותכים את המערכים - לוקחים רק את הזמנים הרלוונטיים
        sig_v_sliced = recon_sig[mask_v]
        reg_v_sliced = design_df.loc[mask_v, 'Model_Visual']
        r_v, p_v = stats.pearsonr(sig_v_sliced, reg_v_sliced)
    else:
        r_v, p_v = 0, 1.0

    # === 2. Auditory Correlation ===
    mask_a = design_df['box_a'] > 0

    if np.sum(mask_a) > 2:
        sig_a_sliced = recon_sig[mask_a]
        reg_a_sliced = design_df.loc[mask_a, 'Model_Auditory']
        r_a, p_a = stats.pearsonr(sig_a_sliced, reg_a_sliced)
    else:
        r_a, p_a = 0, 1.0

    # === 3. Interaction Correlation ===
    mask_va = design_df['box_va'] > 0

    if np.sum(mask_va) > 2:
        sig_va_sliced = recon_sig[mask_va]
        reg_va_sliced = design_df.loc[mask_va, 'Model_Interaction']
        r_va, p_va = stats.pearsonr(sig_va_sliced, reg_va_sliced)
    else:
        r_va, p_va = 0, 1.0

    return r_v, p_v, r_a, p_a, r_va, p_va


def run_contrast_analysis(output_path, ts_files, nii_files, regions_class_origin,
                          regions_multimodal_class_origin, stimulus_duration_sec, peak_delay):
    """
    THE KEY CHANGE: Compute Visual - Auditory CONTRAST
    This shows DIFFERENCES, not absolute activation!
    """
    print("\n" + "=" * 60)
    print("CONTRAST ANALYSIS: Visual - Auditory")
    print("=" * 60)

    # Load atlas
    dataset = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    atlas_img = nib.load(dataset.maps)
    atlas_labels = dataset.labels
    region_vis_prob = roi_vis_prob(atlas_labels, regions_class_origin, regions_multimodal_class_origin)
    region_vis_prob_va = roi_vis_prob(atlas_labels, regions_class_origin, regions_multimodal_class_origin, use_va=True)

    # Get n_timepoints
    tmp = np.loadtxt(list(ts_files.values())[0])
    n_timepoints = tmp.shape[1] if tmp.ndim == 2 else len(tmp)

    ts_arr = np.loadtxt(ts_files['PC_0'])
    temporal_sig = ts_arr[1, :] if ts_arr.ndim == 2 else ts_arr
    n_points = len(temporal_sig)

    design_df = calculate_regressor(n_points, BLOCKS, TR, peak_delay, stimulus_duration_sec)

    results = []
    for pc_name, ts_path in ts_files.items():
        if pc_name not in nii_files: continue
        print(f">> Processing {pc_name}...")

        # Load temporal profile
        ts_arr = np.loadtxt(ts_path)
        temporal_sig = ts_arr[1, :] if ts_arr.ndim == 2 else ts_arr
        plot_temporal(temporal_sig, pc_name, peak_delay, output_path)

        n_points = len(temporal_sig)

        # Load spatial map
        pc_img = nib.load(nii_files[pc_name])
        resampled_atlas = resample_to_img(atlas_img, pc_img, interpolation='nearest',
                                          force_resample=True, copy_header=True)
        pc_data = pc_img.get_fdata()
        atlas_data = resampled_atlas.get_fdata()

        for i, label_bytes in enumerate(atlas_labels):
            roi_id = i + 1
            roi_name = label_bytes.decode('utf-8')
            roi_mask = (atlas_data == roi_id)

            if np.sum(roi_mask) == 0:
                print('\n🔴zero mask for region :', roi_name)
                continue

            spatial_weight = np.mean(pc_data[roi_mask])
            if abs(spatial_weight) < 0.0001:
                print('\n🔴spetial weight is 0 for :', roi_name)
                continue

            reconstructed_sig = temporal_sig * spatial_weight
            r_v, p_v, r_a, p_a, r_va, p_va = calculate_corr(reconstructed_sig, design_df)

            results.append({
                'PC': pc_name,
                'Region': roi_name,
                'Corr_Visual': r_v,
                'Pval_Visual': p_v,
                'Corr_Auditory': r_a,
                'Pval_Auditory': p_a,
                'Corr_Interaction': r_va,
                'Pval_Interaction': p_va,
                'region_vis_prob': region_vis_prob[roi_name],
                'region_vis_prob_va': region_vis_prob_va[roi_name],
            })

    df = pd.DataFrame(results)

    # ==========================================
    # VISUALIZATION: CONTRAST HEATMAP
    # ==========================================

    n_regions = len(df['Region'].unique())
    fig_height = max(20, n_regions * 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(35, fig_height))

    # --- 1. Visual Corrlation ---
    pivot_r_v = df.pivot(index='Region', columns='PC', values=['Corr_Visual', 'Pval_Visual'])
    annot_matrix_r_v = create_bold_annot_matrix(pivot_r_v, 'Corr_Visual', 'Pval_Visual', threshold=0.05)

    vis_max = np.percentile(df['Corr_Visual'].values, 97)
    vis_min = np.percentile(df['Corr_Visual'].values, 5)
    sns.heatmap(pivot_r_v['Corr_Visual'], ax=axes[0], cmap='coolwarm', vmin=vis_min, vmax=vis_max,
                annot=annot_matrix_r_v, fmt="", linewidths=0.5, annot_kws={'size': 8})
    color_labels(axes[0], region_vis_prob)
    corr_corr_v = add_correlation_to_heatmap(axes[0], df, pivot_r_v['Corr_Visual'],
                                             'Visual Correlations\n(Bold = p < 0.05)\nGreen - visual, Brown - audio')

    # --- 2. Auditory Corrlation ---
    pivot_r_a = df.pivot(index='Region', columns='PC', values=['Corr_Auditory', 'Pval_Auditory'])
    annot_matrix_r_a = create_bold_annot_matrix(pivot_r_a, 'Corr_Auditory', 'Pval_Auditory', threshold=0.05)
    sns.heatmap(pivot_r_a['Corr_Auditory'], ax=axes[1], cmap='coolwarm', vmin=vis_min, vmax=vis_max,
                annot=annot_matrix_r_a, fmt="", linewidths=0.5, annot_kws={'size': 8})
    color_labels(axes[1], region_vis_prob)
    corr_corr_a = add_correlation_to_heatmap(axes[1], df, pivot_r_a['Corr_Auditory'],
                                             'Auditory Correlations\n(Bold = p < 0.05)\nGreen - visual, Brown - audio')

    # --- 3. Interaction Corrlation: VA ---
    pivot_r_va = df.pivot(index='Region', columns='PC', values=['Corr_Interaction', 'Pval_Interaction'])
    annot_matrix_r_va = create_bold_annot_matrix(pivot_r_va, 'Corr_Interaction', 'Pval_Interaction', threshold=0.05)
    sns.heatmap(pivot_r_va['Corr_Interaction'], ax=axes[2], cmap='coolwarm', vmin=vis_min, vmax=vis_max,
                annot=annot_matrix_r_va, fmt="", linewidths=0.5, annot_kws={'size': 8})
    color_labels(axes[2], region_vis_prob_va, use_va=True)
    corr_corr_va = add_correlation_to_heatmap(axes[2], df, pivot_r_va['Corr_Interaction'],
                                              'Interaction Correlations\n(Bold = p < 0.05)\nGreen - visual+audio, Brown - no-common',
                                              use_va=True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'CONTRAST_Analysis.png'), dpi=100, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Saved: {os.path.join(output_path, 'CONTRAST_Analysis.png')}")
    # Save the tables
    save_correlation_tables(corr_corr_v, corr_corr_a, corr_corr_va, output_path)
    # Analyze regions
    analyze_and_export_results(df, corr_corr_v, output_path, correlation_threshold=0.2, top_n=10)

def save_correlation_tables(corr_v, corr_a, corr_va, output_path):
    """
    Saves and prints a summary table comparing spatial correlations across conditions.
    Inputs are lists of dictionaries returned by 'add_correlation_to_heatmap'.
    """
    lines = []
    headers = f"{'PC':<10} | {'Vis_r (p-val)':<18} | {'Aud_r (p-val)':<18} | {'Inter_r (p-val)':<18}"

    # --- Pearson Section ---
    lines.append("=" * 80)
    lines.append("SPATIAL CORRELATION SUMMARY (Pearson)")
    lines.append("Correlation between PC spatial map and Probability Maps (Visual/VA)")
    lines.append("-" * 80)
    lines.append(headers)
    lines.append("-" * 80)

    # Assuming all lists have the same length and order of PCs
    for i in range(len(corr_v)):
        pc = corr_v[i]['pc']

        # Format: "0.55 (***)" or "0.12 (n.s.)"
        # Using the data structure from add_correlation_to_heatmap
        v_str = f"{corr_v[i]['pearson_r']:.2f} {corr_v[i]['sig_p']}"
        a_str = f"{corr_a[i]['pearson_r']:.2f} {corr_a[i]['sig_p']}"
        va_str = f"{corr_va[i]['pearson_r']:.2f} {corr_va[i]['sig_p']}"

        lines.append(f"{pc:<10} | {v_str:<18} | {a_str:<18} | {va_str:<18}")

    lines.append("")

    # --- Spearman Section ---
    lines.append("=" * 80)
    lines.append("SPATIAL CORRELATION SUMMARY (Spearman)")
    lines.append("-" * 80)
    lines.append(headers)
    lines.append("-" * 80)

    for i in range(len(corr_v)):
        pc = corr_v[i]['pc']

        v_str = f"{corr_v[i]['spearman_r']:.2f} {corr_v[i]['sig_s']}"
        a_str = f"{corr_a[i]['spearman_r']:.2f} {corr_a[i]['sig_s']}"
        va_str = f"{corr_va[i]['spearman_r']:.2f} {corr_va[i]['sig_s']}"

        lines.append(f"{pc:<10} | {v_str:<18} | {a_str:<18} | {va_str:<18}")

    # Print to screen
    print("\n")
    for line in lines:
        print(line)

    # Save to file
    output_file = os.path.join(output_path, 'spatial_correlation_summary.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n✅ Saved Summary Table: {output_file}")


def analyze_and_export_results(df, corr_corr_v, output_path, correlation_threshold=0.2, top_n=10):
    """
    Analyzes each PC to find the regions that drive it most strongly during
    Visual, Auditory, and Interaction blocks.

    Replaces old 'Contrast' logic with explicit per-condition ranking.
    """
    output_lines = []

    def log(text):
        print(text)
        output_lines.append(text)

    log("\n" + "=" * 80)
    log("DETAILED REGION ANALYSIS PER PC")
    log("Top regions driving the signal during Visual, Auditory, and Interaction blocks")
    log("=" * 80)

    # Get list of PCs from the dataframe
    pcs = df['PC'].unique()

    for pc_name in pcs:
        # Filter data for this PC
        df_pc = df[df['PC'] == pc_name].copy()

        # Find the spatial correlation info for this PC (to display context)
        # Using corr_corr_v just for the general label (Visual-likeness)
        pc_spatial_info = next((item for item in corr_corr_v if item["pc"] == pc_name), None)
        spatial_context = ""
        if pc_spatial_info:
            r_val = pc_spatial_info['pearson_r']
            sig = pc_spatial_info['sig_p']
            if r_val > correlation_threshold:
                spatial_context = f"-> CLASSIFIED AS VISUAL SCENE (r={r_val:.2f}{sig})"
            elif r_val < -correlation_threshold:
                spatial_context = f"-> CLASSIFIED AS AUDITORY/NON-VISUAL (r={r_val:.2f}{sig})"
            else:
                spatial_context = f"-> MIXED/UNCLEAR (r={r_val:.2f}{sig})"

        log(f"\n\n{'#' * 30} {pc_name} {'#' * 30}")
        log(f"Global Context: {spatial_context}")

        # --- 1. Top Visual Drivers ---
        log(f"\n  --- Top {top_n} Regions during VISUAL Blocks (Sorted by Corr_Visual) ---")
        log(f"  {'Region':<45} {'Corr_Vis':>10} {'Vis_Prob':>10}")
        log(f"  {'-' * 67}")

        # Sort by Correlation Visual descending
        top_vis = df_pc.nlargest(top_n, 'Corr_Visual')
        for _, row in top_vis.iterrows():
            log(f"  {row['Region']:<45} {row['Corr_Visual']:+10.2f} {row['region_vis_prob']:10.2f}")

        # --- 2. Top Auditory Drivers ---
        log(f"\n  --- Top {top_n} Regions during AUDITORY Blocks (Sorted by Corr_Auditory) ---")
        log(f"  {'Region':<45} {'Corr_Aud':>10} {'Vis_Prob':>10}")
        log(f"  {'-' * 67}")

        # Sort by Correlation Auditory descending
        top_aud = df_pc.nlargest(top_n, 'Corr_Auditory')
        for _, row in top_aud.iterrows():
            log(f"  {row['Region']:<45} {row['Corr_Auditory']:+10.2f} {row['region_vis_prob']:10.2f}")

        # --- 3. Top Interaction Drivers ---
        log(f"\n  --- Top {top_n} Regions during INTERACTION Blocks (Sorted by Corr_Interaction) ---")
        log(f"  {'Region':<45} {'Corr_VA':>10} {'Vis_Prob':>10}")
        log(f"  {'-' * 67}")

        # Sort by Correlation Interaction descending
        top_va = df_pc.nlargest(top_n, 'Corr_Interaction')
        for _, row in top_va.iterrows():
            log(f"  {row['Region']:<45} {row['Corr_Interaction']:+10.2f} {row['region_vis_prob']:10.2f}")

    # Export Full Dataframe to CSV for manual check
    csv_path = os.path.join(output_path, 'full_correlation_results.csv')
    df.to_csv(csv_path, index=False)
    log(f"\n\n💾 Full data saved to: {csv_path}")

    # Save the text report
    output_file = os.path.join(output_path, 'detailed_region_analysis.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"✅ Saved Analysis Report: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run analysis')
    parser.add_argument('--input-dir', '-p', required=True,
                        help='Path to fPCA output directory')
    parser.add_argument('--output-dir', '-o', required=True,
                        help='Path to output directory')
    parser.add_argument('--peak-delay', '-d', type=float, default=6.0, help='Measurment of the lag size in seconds')
    parser.add_argument('--n-pcs', '-n', type=int, default=7, help='Number of PCs to load')
    parser.add_argument('--regions-class-origin', '-r', choices=['gemini', 'chatgpt', 'chatgpt52', 'claude_opus45'],
                        default='claude_opus45',
                        help="Origin of regions classification file ('gemini', 'chatgpt', 'chatgpt52', 'claude_opus45')")
    parser.add_argument('--regions-multimodal-class-origin', '-rr',
                        choices=['gemini', 'claude_sonnet45'],#, 'chatgpt', 'chatgpt52', 'claude_opus45'],
                        default='claude_sonnet45',
                        help="Origin of regions multimodal classification file ('gemini', 'chatgpt', 'chatgpt52', 'claude_opus45')")
    parser.add_argument('--stimulus-duration-sec', '-s', type=float, default=None,
                        help='Model of stimulus duration in seconds (None for full block)')
    args = parser.parse_args()

    ts_files = {f'PC_{i}': os.path.join(args.input_dir, f'temporal_profile_pc_{i}.txt') for i in range(args.n_pcs)}
    nii_files = {f'PC_{i}': os.path.join(args.input_dir, f'eigenfunction_{i}_importance_map.nii.gz') for i in
                 range(args.n_pcs)}
    os.makedirs(args.output_dir, exist_ok=True)
    run_contrast_analysis(args.output_dir, ts_files, nii_files, args.regions_class_origin,
                          args.regions_multimodal_class_origin, args.stimulus_duration_sec,
                          args.peak_delay)
