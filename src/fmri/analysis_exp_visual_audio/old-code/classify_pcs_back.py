import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn import datasets
import sys
# ==========================================
# CONFIGURATION
# ==========================================
TR = 0.75

BLOCKS = [
    (0, 20, 'va'), (20, 40, 'a'), (40, 60, 'v'),
    (60, 80, 'a'), (80, 100, 'va'), (100, 120, 'v'),
    (120, 140, 'va'), (140, 160, 'x'), (160, 180, 'v'),
    (180, 200, 'a'), (200, 220, 'x'), (220, 240, 'a'),
    (240, 260, 'va'), (260, 280, 'v'), (280, 300, 'va'),
    (300, 320, 'v'), (320, 340, 'a'), (340, 360, 'v'),
    (360, 380, 'a'), (380, 400, 'va'), (400, 420, 'v'),
    (420, 440, 'x'), (440, 450, 'v'), (450, 460, 'va')
]


def spm_hrf(tr, duration=32):
    t = np.arange(0, duration, tr)
    hrf = stats.gamma.pdf(t, 6) - 1 / 6 * stats.gamma.pdf(t, 16)
    return hrf / np.sum(hrf)


def calculate_amplitude(signal, blocks, tr, target_cond):
    """Calculate amplitude for PURE conditions only"""
    hrf_lag_sec = 6.0
    lag_tr = int(hrf_lag_sec / tr)

    active_vals = []
    rest_vals = []

    for s, e, cond in blocks:
        s_idx = int(s / tr) + lag_tr
        e_idx = int(e / tr) + lag_tr

        if s_idx >= len(signal): continue
        e_idx = min(e_idx, len(signal))

        segment = signal[s_idx:e_idx]

        # PURE conditions only
        if target_cond == 'v' and cond == 'v':
            active_vals.extend(segment)
        elif target_cond == 'a' and cond == 'a':
            active_vals.extend(segment)
        elif cond == 'x':
            rest_vals.extend(segment)

    if not active_vals or not rest_vals:
        return 0.0

    return np.mean(active_vals) - np.mean(rest_vals)


def run_contrast_analysis(TS_FILES, NII_FILES):
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

    # Get n_points
    tmp = np.loadtxt(list(TS_FILES.values())[0])
    n_points = tmp.shape[1] if tmp.ndim == 2 else len(tmp)

    results = []

    for pc_name, ts_path in TS_FILES.items():
        if pc_name not in NII_FILES: continue
        print(f">> Processing {pc_name}...")

        # Load temporal profile
        ts_arr = np.loadtxt(ts_path)
        temporal_sig = ts_arr[1, :] if ts_arr.ndim == 2 else ts_arr

        # Load spatial map
        pc_img = nib.load(NII_FILES[pc_name])
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

            reconstructed = temporal_sig * spatial_weight

            # Calculate amplitudes
            amp_v = calculate_amplitude(reconstructed, BLOCKS, TR, 'v')
            amp_a = calculate_amplitude(reconstructed, BLOCKS, TR, 'a')

            # === THE KEY METRIC: CONTRAST ===
            contrast = amp_v - amp_a

            # Selectivity index (normalized)
            selectivity = contrast / (abs(amp_v) + abs(amp_a) + 0.001)

            results.append({
                'PC': pc_name,
                'Region': roi_name,
                'Amp_Visual': amp_v,
                'Amp_Auditory': amp_a,
                'Contrast_V_minus_A': contrast,
                'Selectivity': selectivity
            })

    df = pd.DataFrame(results)

    # ==========================================
    # VISUALIZATION: CONTRAST HEATMAP
    # ==========================================

    n_regions = len(df['Region'].unique())
    fig_height = max(20, n_regions * 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(30, fig_height))

    # Color labels function
    def color_labels(ax):
        for label in ax.get_yticklabels():
            txt = label.get_text()
            label.set_fontsize(9)
            if 'Vis' in txt:
                label.set_color('green')
                label.set_fontweight('bold')
            elif 'Aud' in txt or 'SomMot' in txt:
                label.set_color('brown')
                label.set_fontweight('bold')

    # --- 1. Visual Amplitude ---
    pivot_v = df.pivot(index='Region', columns='PC', values='Amp_Visual')
    vmax = np.percentile(np.abs(df['Amp_Visual'].values), 95)
    sns.heatmap(pivot_v, ax=axes[0], cmap='Reds', vmin=0, vmax=vmax,
                annot=True, fmt='.1f', linewidths=0.5, annot_kws={'size': 8})
    axes[0].set_title('Visual Amplitude\n(Both V and A regions are red = BAD)', fontsize=14)
    color_labels(axes[0])

    # --- 2. Auditory Amplitude ---
    pivot_a = df.pivot(index='Region', columns='PC', values='Amp_Auditory')
    sns.heatmap(pivot_a, ax=axes[1], cmap='Blues', vmin=0, vmax=vmax,
                annot=True, fmt='.1f', linewidths=0.5, annot_kws={'size': 8})
    axes[1].set_title('Auditory Amplitude\n(Both V and A regions are blue = BAD)', fontsize=14)
    color_labels(axes[1])

    # --- 3. CONTRAST: V - A (THE SOLUTION!) ---
    pivot_contrast = df.pivot(index='Region', columns='PC', values='Contrast_V_minus_A')
    contrast_max = np.percentile(np.abs(df['Contrast_V_minus_A'].values), 95)
    sns.heatmap(pivot_contrast, ax=axes[2], cmap='coolwarm', center=0,
                vmin=-contrast_max, vmax=contrast_max,
                annot=True, fmt='.1f', linewidths=0.5, annot_kws={'size': 8})
    axes[2].set_title('CONTRAST: Visual - Auditory\n(Red=Visual selective, Blue=Auditory selective)',
                      fontsize=14, fontweight='bold')
    color_labels(axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(PATH, 'CONTRAST_Analysis_V_minus_A.png'), dpi=100, bbox_inches='tight')
    plt.close()

    print("\n✅ Saved: CONTRAST_Analysis_V_minus_A.png")

    # ==========================================
    # SUMMARY: Top Selective Regions
    # ==========================================

    print("\n" + "=" * 60)
    print("TOP VISUAL-SELECTIVE REGIONS (Contrast > 0)")
    print("=" * 60)

    # Aggregate across PCs
    df_agg = df.groupby('Region').agg({
        'Contrast_V_minus_A': 'mean',
        'Selectivity': 'mean',
        'Amp_Visual': 'mean',
        'Amp_Auditory': 'mean'
    }).reset_index()

    # Top Visual
    top_visual = df_agg.nlargest(10, 'Contrast_V_minus_A')
    print("\n🔴 Most VISUAL-selective regions:")
    for _, row in top_visual.iterrows():
        marker = "✓ Vis" if 'Vis' in row['Region'] else ""
        print(f"  {row['Region']:<45} Contrast={row['Contrast_V_minus_A']:+.2f}  {marker}")

    # Top Auditory
    print("\n" + "=" * 60)
    print("TOP AUDITORY-SELECTIVE REGIONS (Contrast < 0)")
    print("=" * 60)

    top_auditory = df_agg.nsmallest(10, 'Contrast_V_minus_A')
    print("\n🔵 Most AUDITORY-selective regions:")
    for _, row in top_auditory.iterrows():
        marker = "✓ Aud" if ('Aud' in row['Region'] or 'SomMot' in row['Region']) else ""
        print(f"  {row['Region']:<45} Contrast={row['Contrast_V_minus_A']:+.2f}  {marker}")

    # ==========================================
    # VALIDATION CHECK
    # ==========================================

    print("\n" + "=" * 60)
    print("VALIDATION: Do Visual regions have positive contrast?")
    print("=" * 60)

    vis_regions = df_agg[df_agg['Region'].str.contains('Vis')]
    aud_regions = df_agg[df_agg['Region'].str.contains('SomMot|Aud')]

    print(f"\nVisual regions (Vis_*): Mean contrast = {vis_regions['Contrast_V_minus_A'].mean():+.2f}")
    print(f"Auditory regions (SomMot_*): Mean contrast = {aud_regions['Contrast_V_minus_A'].mean():+.2f}")

    if vis_regions['Contrast_V_minus_A'].mean() > aud_regions['Contrast_V_minus_A'].mean():
        print("\n✅ SUCCESS! Visual regions have MORE positive contrast than auditory regions!")
        print("   The data IS separating visual from auditory - you just need to look at CONTRAST!")
    else:
        print("\n⚠️ WARNING: Something may be wrong with the data or experimental design.")

    return df

if __name__ == '__main__':
    import argparse

    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Run contrast analysis')
        parser.add_argument('--path', '-p',
                            default='/Users/user/Documents/pythonProject/fMRI-runs/validation-experiment/tests/fmri_output_basis100_lam-6to12_derivP1U0',
                            help='Path to fmri output directory (overrides `PATH`)')
        parser.add_argument('--n-pcs', '-n', type=int, default=7, help='Number of PCs to load')
        args = parser.parse_args()

        PATH = args.path
        TS_FILES = {f'PC_{i}': os.path.join(PATH, f'temporal_profile_pc_{i}.txt') for i in range(args.n_pcs)}
        NII_FILES = {f'PC_{i}': os.path.join(PATH, f'eigenfunction_{i}_importance_map.nii.gz') for i in range(args.n_pcs)}

        df = run_contrast_analysis(TS_FILES, NII_FILES)
