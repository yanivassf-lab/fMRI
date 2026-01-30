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


#
#
# def spm_hrf(tr, peak_delay=6, undershoot_delay=16, duration=HRF_DURATION):
#     """
#     Generates HRF with adjustable delays.
#     peak_delay: Time to peak response (default 6s).
#     undershoot_delay: Time to peak undershoot (default 16s).
#     """
#     dt = tr
#     t = np.arange(0, duration, dt)
#
#     # Use variable peak_delay and undershoot_delay
#     # Create HRF using gamma functions with specified delays
#     # The shape parameters are set to match the desired peak times
#     # Roughly, the peak of a gamma distribution occurs at (a-1) for shape parameter a.
#     # Thus, for a peak at 6s, we use a shape parameter of about 6.
#     # The exact SPM implementation is more complex with a second parameter, but this is a good approximation for your needs.
#
#     hrf = stats.gamma.pdf(t, peak_delay) - 1 / 6 * stats.gamma.pdf(t, undershoot_delay)
#
#     return hrf / np.sum(hrf)

def plot_temporal(temporal_sig, pc_name, peak_delay, output_path):
    plt.figure(figsize=(15, 5))
    times = np.arange(0, len(temporal_sig))*TR
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
    plt.xlabel("Time (TRs)")
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
    hrf = stats.gamma.pdf(t, peak_delay)
    return hrf / np.sum(hrf)

def create_design_matrix(n_points, blocks, tr, peak_delay, undershoot_delay, stimulus_duration_sec=None):
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
    # hrf = spm_hrf(tr, peak_delay, undershoot_delay)
    hrf = spm_hrf(tr, peak_delay)
    reg_v = np.convolve(box_v, hrf)[:timeline_len]
    reg_v[box_v == 0] = 0
    reg_a = np.convolve(box_a, hrf)[:timeline_len]
    reg_a[box_a == 0] = 0
    reg_va = np.convolve(box_va, hrf)[:timeline_len]
    reg_va[box_va == 0] = 0

    # === Demeaning (Crucial) ===
    # Subtracting the mean makes the regressors centered around zero,
    # just like the PCA components.
    # Positive values = Activation, Negative values = Rest/Suppression.
    return pd.DataFrame({
        'Model_Visual': reg_v,
        'box_v': box_v, # - np.mean(reg_v),
        'Model_Auditory': reg_a, # - np.mean(reg_a),
        'box_a': box_a,
        'Model_Interaction': reg_va, # - np.mean(reg_va)
        'box_va': box_va
    })


def calculate_amplitude(signal, blocks, tr, target_cond, peak_delay):
    """Calculate amplitude for PURE conditions only"""
    hrf_lag_sec = peak_delay
    lag_tr = int(hrf_lag_sec / tr)

    active_vals = []
    rest_vals = []

    for s, e, cond in blocks:
        s_idx = int(s / tr) + lag_tr
        e_idx = int(e / tr) + lag_tr

        if s_idx >= len(signal):
            print('Skipping block starting at', s, 's due to lag exceeding signal length.')
            continue
        e_idx = min(e_idx, len(signal))

        segment = signal[s_idx:e_idx]

        # PURE conditions only
        if target_cond == 'v' and cond == 'v':
            active_vals.extend(segment)
        elif target_cond == 'a' and cond == 'a':
            active_vals.extend(segment)
        elif target_cond == 'va' and cond == 'va':
            active_vals.extend(segment)
        elif cond == 'x':
            rest_vals.extend(segment)

    if not active_vals or not rest_vals:
        return 0.0

    return np.mean(active_vals) - np.mean(rest_vals)

def calculate_corr(recon_sig, design_df):
    recon_sig_v = recon_sig.copy()
    recon_sig_v[design_df['box_v']==0] = 0
    r_v, p_v = stats.pearsonr(recon_sig_v, design_df['Model_Visual'])
    recon_sig_a = recon_sig.copy()
    recon_sig_a[design_df['box_a'] == 0] = 0
    r_a, p_a = stats.pearsonr(recon_sig_a, design_df['Model_Auditory'])
    recon_sig_va = recon_sig.copy()
    recon_sig_va[design_df['box_va'] == 0] = 0
    r_va, p_va = stats.pearsonr(recon_sig_va, design_df['Model_Interaction'])
    return r_v, p_v, r_a, p_a, r_va, p_va

def run_contrast_analysis(output_path, ts_files, nii_files, regions_class_origin,
                          regions_multimodal_class_origin, stimulus_duration_sec, peak_delay, undershoot_delay):
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

    design_df = create_design_matrix(n_points, BLOCKS, TR, peak_delay, undershoot_delay, stimulus_duration_sec)

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
            # Calculate amplitudes
            amp_v = calculate_amplitude(reconstructed_sig, BLOCKS, TR, 'v', peak_delay)
            amp_a = calculate_amplitude(reconstructed_sig, BLOCKS, TR, 'a', peak_delay)
            amp_va = calculate_amplitude(reconstructed_sig, BLOCKS, TR, 'va', peak_delay)
            contrast = amp_v - amp_a
            # contrast_va = amp_va - amp_v - amp_a
            # Selectivity index (normalized)
            # selectivity = contrast / (abs(amp_v) + abs(amp_a) + 0.001)

            r_v, p_v, r_a, p_a, r_va, p_va = calculate_corr(reconstructed_sig, design_df)

            results.append({
                'PC': pc_name,
                'Region': roi_name,
                'Amp_Visual': amp_v,
                'Amp_Auditory': amp_a,
                'Amp_Interaction': amp_va,
                'Corr_Visual': r_v,
                'Pval_Visual': p_v,
                'Corr_Auditory': r_a,
                'Pval_Auditory': p_a,
                'Corr_Interaction': r_va,
                'Pval_Interaction': p_va,
                'Contrast_V_minus_A': contrast,
                # 'Selectivity': selectivity,
                'region_vis_prob': region_vis_prob[roi_name],
                'region_vis_prob_va': region_vis_prob_va[roi_name],
                # 'region_vis': True if region_vis_prob[roi_name] > 0.5 else False
            })

    df = pd.DataFrame(results)

    # ==========================================
    # VISUALIZATION: CONTRAST HEATMAP
    # ==========================================

    n_regions = len(df['Region'].unique())
    fig_height = max(20, n_regions * 0.5)

    fig, axes = plt.subplots(1, 7, figsize=(70, fig_height))

    # --- 1. Visual Corrlation ---
    pivot_r_v = df.pivot(index='Region', columns='PC', values=['Corr_Visual', 'Pval_Visual'])
    annot_matrix_r_v = create_bold_annot_matrix(pivot_r_v, 'Corr_Visual', 'Pval_Visual', threshold=0.05)

    vis_max = np.percentile(df['Corr_Visual'].values, 97)
    vis_min = np.percentile(df['Corr_Visual'].values, 5)
    sns.heatmap(pivot_r_v['Corr_Visual'], ax=axes[0], cmap='coolwarm', vmin=vis_min, vmax=vis_max,
                annot=annot_matrix_r_v, fmt="", linewidths=0.5, annot_kws={'size': 8})
    color_labels(axes[0], region_vis_prob)
    corr_corr_v = add_correlation_to_heatmap(axes[0], df, pivot_r_v['Corr_Visual'], 'Visual Correlations\n(Bold = p < 0.05)\nGreen - visual, Brown - audio')

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
                                             'Interaction Correlations\n(Bold = p < 0.05)\nGreen - visual+audio, Brown - no-common', use_va=True)

    # ===================================
    # Amplitude
    # ===================================

    # --- 1. Visual Amplitude ---
    pivot_v = df.pivot(index='Region', columns='PC', values='Amp_Visual')
    vis_max = np.percentile(df['Amp_Visual'].values, 97)
    vis_min = np.percentile(df['Amp_Visual'].values, 5)
    sns.heatmap(pivot_v, ax=axes[3], cmap='coolwarm', vmin=vis_min, vmax=vis_max,
                annot=True, fmt='.1f', linewidths=0.5, annot_kws={'size': 8})
    color_labels(axes[3], region_vis_prob)
    corr_amp_v = add_correlation_to_heatmap(axes[3], df, pivot_v, 'Visual Amplitude\nGreen - visual, Brown - audio')

    # --- 2. Auditory Amplitude ---
    pivot_a = df.pivot(index='Region', columns='PC', values='Amp_Auditory')
    # aud_max = np.percentile(df['Amp_Auditory'].values, 95)
    # aud_min = np.percentile(df['Amp_Auditory'].values, 5)
    sns.heatmap(pivot_a, ax=axes[4], cmap='coolwarm', vmin=vis_min, vmax=vis_max,
                annot=True, fmt='.1f', linewidths=0.5, annot_kws={'size': 8})
    color_labels(axes[4], region_vis_prob)  # Just for coloring
    corr_amp_a = add_correlation_to_heatmap(axes[4], df, pivot_a, 'Auditory Amplitude\nGreen - visual, Brown - audio')

    # --- 3. Interaction Amplitude: VA
    pivot_va = df.pivot(index='Region', columns='PC', values='Amp_Interaction')
    # inter_max = np.percentile(df['Amp_Interaction'].values, 95)
    # inter_min = np.percentile(df['Amp_Interaction'].values, 5)
    sns.heatmap(pivot_va, ax=axes[5], cmap='coolwarm', center=0,
                vmin=vis_min, vmax=vis_max,
                annot=True, fmt='.1f', linewidths=0.5, annot_kws={'size': 8})
    color_labels(axes[5], region_vis_prob_va, use_va=True)
    corr_amp_va = add_correlation_to_heatmap(axes[5], df, pivot_va, 'Interaction: VA\nGreen - visual+audio, Brown - no-common', use_va=True)

    # --- 4. CONTRAST: V - A  ---
    pivot_contrast = df.pivot(index='Region', columns='PC', values='Contrast_V_minus_A')
    # contrast_max = np.percentile(df['Contrast_V_minus_A'].values, 95)
    # contrast_min = np.percentile(df['Contrast_V_minus_A'].values, 5)
    sns.heatmap(pivot_contrast, ax=axes[6], cmap='coolwarm', center=0,
                vmin=vis_min, vmax=vis_max,
                annot=True, fmt='.1f', linewidths=0.5, annot_kws={'size': 8})
    color_labels(axes[6], region_vis_prob)
    corr_amp_contrast = add_correlation_to_heatmap(axes[6], df, pivot_contrast, 'CONTRAST: Visual - Auditory\nGreen - visual, Brown - audio')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'CONTRAST_Analysis.png'), dpi=100, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Saved: {os.path.join(output_path, 'CONTRAST_Analysis.png')}")
    # Replace the two print loops with:
    save_correlation_tables(corr_amp_v, corr_amp_a, corr_amp_contrast, corr_amp_va, output_path)

    # Add after that:
    analyze_and_export_results(df, corr_amp_contrast, output_path,
                               correlation_threshold=0.2, top_n=10)

    return df, corr_amp_contrast


def analyze_and_export_results(df, corr_contrast, output_path,
                               correlation_threshold=0.2, top_n=10):
    """
    Analyze PCs based on their correlation with visual probability and export results.

    For PCs with high POSITIVE correlation:
        - Classified as VISUAL PC
        - Top regions by highest contrast = most visual
        - Top regions by lowest contrast = most auditory

    For PCs with high NEGATIVE correlation:
        - Classified as AUDITORY PC
        - Top regions by highest contrast = most auditory (inverted)
        - Top regions by lowest contrast = most visual (inverted)
    """

    output_lines = []

    def log(text):
        print(text)
        output_lines.append(text)

    log("=" * 80)
    log("PC CLASSIFICATION AND TOP REGIONS ANALYSIS")
    log("=" * 80)

    for corr in corr_contrast:
        pc_name = corr['pc']
        pearson_r = corr['pearson_r']
        spearman_r = corr['spearman_r']
        sig_p = corr['sig_p']
        sig_s = corr['sig_s']

        # Filter df for this PC
        df_pc = df[df['PC'] == pc_name].copy()

        log(f"\n{'=' * 70}")
        log(f"{pc_name}: Pearson r={pearson_r:+.3f}{sig_p}, Spearman ρ={spearman_r:+.3f}{sig_s}")
        log(f"{'=' * 70}")

        if pearson_r > correlation_threshold:
            # VISUAL PC
            log(f"\n🎬 {pc_name} classified as VISUAL (positive correlation with vis_prob)")

            log(f"\n  TOP {top_n} VISUAL-SELECTIVE REGIONS (highest contrast):")
            log(f"  {'Region':<45} {'Contrast':>10} {'Vis_Prob':>10}")
            log(f"  {'-' * 65}")
            top_visual = df_pc.nlargest(top_n, 'Contrast_V_minus_A')
            for _, row in top_visual.iterrows():
                log(f"  {row['Region']:<45} {row['Contrast_V_minus_A']:+10.2f} {row['region_vis_prob']:10.2f}")

            log(f"\n  TOP {top_n} AUDITORY-SELECTIVE REGIONS (lowest contrast):")
            log(f"  {'Region':<45} {'Contrast':>10} {'Vis_Prob':>10}")
            log(f"  {'-' * 65}")
            top_auditory = df_pc.nsmallest(top_n, 'Contrast_V_minus_A')
            for _, row in top_auditory.iterrows():
                log(f"  {row['Region']:<45} {row['Contrast_V_minus_A']:+10.2f} {row['region_vis_prob']:10.2f}")

        elif pearson_r < -correlation_threshold:
            # AUDITORY PC (inverted polarity)
            log(f"\n🎵 {pc_name} classified as AUDITORY (negative correlation with vis_prob)")
            log(f"   Note: This PC has inverted polarity - high contrast = auditory regions")

            log(f"\n  TOP {top_n} AUDITORY-SELECTIVE REGIONS (highest contrast = auditory in this PC):")
            log(f"  {'Region':<45} {'Contrast':>10} {'Vis_Prob':>10}")
            log(f"  {'-' * 65}")
            top_auditory = df_pc.nlargest(top_n, 'Contrast_V_minus_A')
            for _, row in top_auditory.iterrows():
                log(f"  {row['Region']:<45} {row['Contrast_V_minus_A']:+10.2f} {row['region_vis_prob']:10.2f}")

            log(f"\n  TOP {top_n} VISUAL-SELECTIVE REGIONS (lowest contrast = visual in this PC):")
            log(f"  {'Region':<45} {'Contrast':>10} {'Vis_Prob':>10}")
            log(f"  {'-' * 65}")
            top_visual = df_pc.nsmallest(top_n, 'Contrast_V_minus_A')
            for _, row in top_visual.iterrows():
                log(f"  {row['Region']:<45} {row['Contrast_V_minus_A']:+10.2f} {row['region_vis_prob']:10.2f}")
        else:
            # MIXED / UNCLEAR
            log(f"\n❓ {pc_name} classified as MIXED/UNCLEAR (|correlation| < {correlation_threshold})")
            log(f"   This PC may represent shared processes, attention, or noise.")

    # Save to file
    output_file = os.path.join(output_path, 'PC_classification_and_top_regions.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"\n✅ Saved: {output_file}")

    return output_lines


def save_correlation_tables(corr_v, corr_a, corr_contrast, corr_va, output_path):
    """
    Save correlation tables to file AND print to screen.
    Call this instead of the inline print statements.
    """

    lines = []

    lines.append("=" * 70)
    lines.append("PEARSON CORRELATION SUMMARY WITH VIDEO PROBABILITY")
    lines.append("=" * 70)
    lines.append(f"{'PC':<8} {'Visual Amp':>12} {'Auditory Amp':>14} {'Contrast V-A':>14} {'Contrast VA':>12}")
    lines.append("-" * 70)

    for i in range(len(corr_v)):
        pc = corr_v[i]['pc']
        v_r = f"{corr_v[i]['pearson_r']:+.2f}{corr_v[i]['sig_p']}"
        a_r = f"{corr_a[i]['pearson_r']:+.2f}{corr_a[i]['sig_p']}"
        c_r = f"{corr_contrast[i]['pearson_r']:+.2f}{corr_contrast[i]['sig_p']}"
        va_r = f"{corr_va[i]['pearson_r']:+.2f}{corr_va[i]['sig_p']}"
        lines.append(f"{pc:<8} {v_r:>12} {a_r:>14} {c_r:>14} {va_r:>12}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("SPEARMAN CORRELATION SUMMARY WITH VIDEO PROBABILITY")
    lines.append("=" * 70)
    lines.append(f"{'PC':<8} {'Visual Amp':>12} {'Auditory Amp':>14} {'Contrast V-A':>14} {'Contrast VA':>12}")
    lines.append("-" * 70)

    for i in range(len(corr_v)):
        pc = corr_v[i]['pc']
        v_r = f"{corr_v[i]['spearman_r']:+.2f}{corr_v[i]['sig_s']}"
        a_r = f"{corr_a[i]['spearman_r']:+.2f}{corr_a[i]['sig_s']}"
        c_r = f"{corr_contrast[i]['spearman_r']:+.2f}{corr_contrast[i]['sig_s']}"
        va_r = f"{corr_va[i]['spearman_r']:+.2f}{corr_va[i]['sig_s']}"
        lines.append(f"{pc:<8} {v_r:>12} {a_r:>14} {c_r:>14} {va_r:>12}")

    # Print to screen
    for line in lines:
        print(line)

    # Save to file
    output_file = os.path.join(output_path, 'correlation_summary.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n✅ Saved: {output_file}")

    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run contrast analysis')
    parser.add_argument('--path', '-p',
                        default='/Users/user/Documents/pythonProject/fMRI-runs/validation-experiment/tests/fmri_output_basis100_lam-6to12_derivP1U0',
                        help='Path to fmri output directory (overrides `PATH`)')
    parser.add_argument('--peak-delay', '-d', type=float, default=6.0, help='Measurment of the lag size in seconds')
    parser.add_argument('--undershoot-delay', '-u', type=float, default=16.0, help='Undershoot delay in seconds')
    parser.add_argument('--n-pcs', '-n', type=int, default=7, help='Number of PCs to load')
    parser.add_argument('--regions-class-origin', '-r', choices=['gemini', 'chatgpt', 'chatgpt52', 'claude_opus45'],
                        default='claude_opus45',
                        help="Origin of regions classification file ('gemini', 'chatgpt', 'chatgpt52', 'claude_opus45')")
    parser.add_argument('--regions-multimodal-class-origin', '-rr',
                        choices=['gemini', 'chatgpt', 'chatgpt52', 'claude_opus45'],
                        default='claude_opus45',
                        help="Origin of regions multimodal classification file ('gemini', 'chatgpt', 'chatgpt52', 'claude_opus45')")
    parser.add_argument('--stimulus-duration-sec', '-s', type=float, default=None,
                        help='Model of stimulus duration in seconds (None for full block)')
    args = parser.parse_args()

    ts_files = {f'PC_{i}': os.path.join(args.path, f'temporal_profile_pc_{i}.txt') for i in range(args.n_pcs)}
    nii_files = {f'PC_{i}': os.path.join(args.path, f'eigenfunction_{i}_importance_map.nii.gz') for i in
                 range(args.n_pcs)}

    df, corr_contrast = run_contrast_analysis(args.path, ts_files, nii_files, args.regions_class_origin,
                                              args.regions_multimodal_class_origin, args.stimulus_duration_sec,
                                              args.peak_delay, args.undershoot_delay)
