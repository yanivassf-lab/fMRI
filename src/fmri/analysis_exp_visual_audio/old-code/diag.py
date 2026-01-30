
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import nibabel as nib
from nilearn import datasets

from classify_pcs_v4 import spm_hrf, create_design_matrix

# ==========================================
# 1. Configuration & Constants
# ==========================================
TR = 0.75  # Repetition Time in seconds. Critical for timing accuracy.
PATH = '/Users/user/Documents/pythonProject/fMRI-runs/validation-experiment/fmri_output_no_flip_fast'
# Dictionary of Temporal Profiles (The "When")
# These are the time-series extracted by PCA
TS_FILES = {
    'PC_0': os.path.join(PATH, 'temporal_profile_pc_0.txt'),
    'PC_1': os.path.join(PATH, 'temporal_profile_pc_1.txt'),
    'PC_2': os.path.join(PATH, 'temporal_profile_pc_2.txt'),
    'PC_3': os.path.join(PATH, 'temporal_profile_pc_3.txt'),
    'PC_4': os.path.join(PATH, 'temporal_profile_pc_4.txt'),
    'PC_5': os.path.join(PATH, 'temporal_profile_pc_5.txt'),
    'PC_6': os.path.join(PATH, 'temporal_profile_pc_6.txt')
}

# Dictionary of Spatial Maps (The "Where")
# These are the NIfTI files corresponding to the PCs
NII_FILES = {
    'PC_0': os.path.join(PATH, 'eigenfunction_0_importance_map.nii.gz'),
    'PC_1': os.path.join(PATH, 'eigenfunction_1_importance_map.nii.gz'),
    'PC_2': os.path.join(PATH, 'eigenfunction_2_importance_map.nii.gz'),
    'PC_3': os.path.join(PATH, 'eigenfunction_3_importance_map.nii.gz'),
    'PC_4': os.path.join(PATH, 'eigenfunction_4_importance_map.nii.gz'),
    'PC_5': os.path.join(PATH, 'eigenfunction_5_importance_map.nii.gz'),
    'PC_6': os.path.join(PATH, 'eigenfunction_6_importance_map.nii.gz'),
}

# The Global Signal file (Average of the whole brain before PCA)
GLOBAL_SIGNAL_FILE = os.path.join(PATH, 'original_averaged_signal_intensity.txt')

# Experimental Block Design
# Format: (Start_Time, End_Time, Condition)
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


def diagnose_temporal_profiles():
    """
    בודק האם הפרופילים הזמניים של ה-PCs מבדילים בין תנאים
    """
    print("\n" + "=" * 50)
    print("DIAGNOSTIC: Are PCs condition-specific?")
    print("=" * 50)

    # טען את ה-design matrix
    tmp = np.loadtxt(list(TS_FILES.values())[0])
    n_points = tmp.shape[1] if tmp.ndim == 2 else len(tmp)
    design_df = create_design_matrix(n_points, BLOCKS, TR)

    # צור רגרסורים טהורים להשוואה
    pure_v = np.zeros(n_points)
    pure_a = np.zeros(n_points)
    for s, e, cond in BLOCKS:
        idx_s, idx_e = int(s / TR), min(int(e / TR), n_points)
        if cond == 'v': pure_v[idx_s:idx_e] = 1
        if cond == 'a': pure_a[idx_s:idx_e] = 1

    hrf = spm_hrf(TR)
    pure_v = np.convolve(pure_v, hrf)[:n_points]
    pure_a = np.convolve(pure_a, hrf)[:n_points]
    pure_v = pure_v - np.mean(pure_v)
    pure_a = pure_a - np.mean(pure_a)

    print("\nCorrelation of each PC with PURE regressors:")
    print("-" * 45)

    results = []
    for pc_name, ts_path in TS_FILES.items():
        ts_arr = np.loadtxt(ts_path)
        temporal_sig = ts_arr[1, :] if ts_arr.ndim == 2 else ts_arr

        r_v, _ = stats.pearsonr(temporal_sig, pure_v)
        r_a, _ = stats.pearsonr(temporal_sig, pure_a)

        # מדד הספציפיות
        specificity = abs(r_v) - abs(r_a)

        results.append({
            'PC': pc_name,
            'r_Visual_Pure': r_v,
            'r_Auditory_Pure': r_a,
            'Specificity (V-A)': specificity
        })

        # קביעת אופי ה-PC
        if abs(r_v) > 0.3 and abs(r_a) < 0.15:
            char = "📺 VISUAL SPECIFIC"
        elif abs(r_a) > 0.3 and abs(r_v) < 0.15:
            char = "🔊 AUDITORY SPECIFIC"
        elif abs(r_v) > 0.2 and abs(r_a) > 0.2:
            char = "⚠️ MIXED (both conditions)"
        else:
            char = "❓ Unclear"

        print(f"{pc_name}: r_vis={r_v:+.3f}, r_aud={r_a:+.3f} → {char}")

    # גרף
    df_diag = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # גרף 1: קורלציות
    x = np.arange(len(df_diag))
    width = 0.35
    axes[0].bar(x - width / 2, df_diag['r_Visual_Pure'], width, label='Visual', color='red', alpha=0.7)
    axes[0].bar(x + width / 2, df_diag['r_Auditory_Pure'], width, label='Auditory', color='blue', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_diag['PC'])
    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].set_ylabel('Correlation with Pure Regressor')
    axes[0].set_title('Is each PC condition-specific?')
    axes[0].legend()

    # גרף 2: ספציפיות
    colors = ['green' if s > 0.1 else 'orange' if s > -0.1 else 'purple'
              for s in df_diag['Specificity (V-A)']]
    axes[1].bar(df_diag['PC'], df_diag['Specificity (V-A)'], color=colors)
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].axhline(0.1, color='green', linestyle='--', alpha=0.5, label='Visual-leaning')
    axes[1].axhline(-0.1, color='purple', linestyle='--', alpha=0.5, label='Auditory-leaning')
    axes[1].set_ylabel('Specificity: |r_vis| - |r_aud|')
    axes[1].set_title('Positive = More Visual, Negative = More Auditory')

    plt.tight_layout()
    plt.show()

    return df_diag


def debug_pc_signal(pc_name='PC_0', target_region_str='Vis'):
    """
    Plots the actual reconstructed signal for a specific region to debug
    why Auditory blocks might look active.
    """
    print(f"\n--- DEBUGGING {pc_name} SIGNAL ---")

    # 1. Load Data
    if pc_name not in TS_FILES or pc_name not in NII_FILES:
        print("PC files not found.")
        return

    # Load Time Series
    ts_arr = np.loadtxt(TS_FILES[pc_name])
    temporal = ts_arr[1, :] if ts_arr.ndim == 2 else ts_arr

    # Load Atlas & Map
    dataset = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    atlas_img = nib.load(dataset.maps)
    pc_img = nib.load(NII_FILES[pc_name])

    # Resample
    from nilearn.image import resample_to_img
    resampled_atlas = resample_to_img(atlas_img, pc_img, interpolation='nearest', force_resample=True)
    atlas_data = resampled_atlas.get_fdata()
    pc_data = pc_img.get_fdata()

    # 2. Find a Visual Region
    # We look for the first region that matches the string (e.g., 'Vis')
    target_id = None
    target_name = ""

    for i, label in enumerate(dataset.labels):
        name = label.decode('utf-8')
        if target_region_str in name:
            target_id = i + 1
            target_name = name
            break

    if target_id is None:
        print(f"No region found containing '{target_region_str}'")
        return

    print(f"Analyzing Region: {target_name}")

    # 3. Reconstruct Signal
    roi_mask = (atlas_data == target_id)
    spatial_weight = np.mean(pc_data[roi_mask])
    reconstructed_signal = temporal * spatial_weight

    # 4. Plot with Background Colors
    plt.figure(figsize=(15, 5))
    plt.plot(reconstructed_signal, color='black', linewidth=1.5, label='Reconstructed Signal')

    # Add colored spans for blocks
    # Shift by 6 seconds (8 TRs) for visualization match
    lag = int(6.0 / TR)

    for s, e, cond in BLOCKS:
        # Convert sec to TR indices
        s_idx = int(s / TR) + lag
        e_idx = int(e / TR) + lag

        if cond == 'v':
            plt.axvspan(s_idx, e_idx, color='red', alpha=0.3,
                        label='Visual (v)' if 'Visual (v)' not in plt.gca().get_legend_handles_labels()[1] else "")
        elif cond == 'a':
            plt.axvspan(s_idx, e_idx, color='green', alpha=0.3,
                        label='Auditory (a)' if 'Auditory (a)' not in plt.gca().get_legend_handles_labels()[1] else "")
        elif cond == 'va':
            plt.axvspan(s_idx, e_idx, color='orange', alpha=0.3,
                        label='Both (va)' if 'Both (va)' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title(f"(Signal Trace for {pc_name})")
    plt.xlabel("Time (TRs)")
    plt.ylabel("Signal Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.show()


def analyze_pc_content(temporal_profiles, times, blocks, tr):
    """
    Analyze what each PC actually captures in terms of experimental conditions.
    """
    from scipy import stats
    n_timepoints = len(times)
    n_pcs = temporal_profiles.shape[1]

    # Create condition indicators
    def make_condition_timeseries(blocks, n_timepoints, tr, target_conditions):
        ts = np.zeros(n_timepoints)
        for s, e, cond in blocks:
            if cond in target_conditions:
                idx_s = int(s / tr)
                idx_e = min(int(e / tr), n_timepoints)
                ts[idx_s:idx_e] = 1
        return ts

    # Pure conditions (no overlap)
    vis_only = make_condition_timeseries(blocks, n_timepoints, tr, ['v'])
    aud_only = make_condition_timeseries(blocks, n_timepoints, tr, ['a'])
    combined = make_condition_timeseries(blocks, n_timepoints, tr, ['va'])
    any_task = make_condition_timeseries(blocks, n_timepoints, tr, ['v', 'a', 'va'])
    rest = make_condition_timeseries(blocks, n_timepoints, tr, ['x'])

    # Convolve with HRF
    def spm_hrf(tr, duration=32):
        t = np.arange(0, duration, tr)
        hrf = stats.gamma.pdf(t, 6) - 1 / 6 * stats.gamma.pdf(t, 16)
        return hrf / np.sum(hrf)

    hrf = spm_hrf(tr)
    vis_only_conv = np.convolve(vis_only, hrf)[:n_timepoints]
    aud_only_conv = np.convolve(aud_only, hrf)[:n_timepoints]
    combined_conv = np.convolve(combined, hrf)[:n_timepoints]
    any_task_conv = np.convolve(any_task, hrf)[:n_timepoints]

    print("\n" + "=" * 70)
    print("PC CONTENT ANALYSIS: What does each PC actually represent?")
    print("=" * 70)
    print(f"\n{'PC':<6} {'Task(any)':<12} {'Vis_only':<12} {'Aud_only':<12} {'VA_only':<12} {'INTERPRETATION'}")
    print("-" * 70)

    interpretations = []

    for i in range(n_pcs):
        pc = temporal_profiles[:, i]
        print(len(pc), len(any_task_conv))
        r_task, _ = stats.pearsonr(pc, any_task_conv)
        r_vis, _ = stats.pearsonr(pc, vis_only_conv)
        r_aud, _ = stats.pearsonr(pc, aud_only_conv)
        r_va, _ = stats.pearsonr(pc, combined_conv)

        # Interpretation logic
        if abs(r_task) > 0.5:
            if abs(r_vis - r_aud) < 0.15:
                interp = "⚠️ GENERAL TASK (not condition-specific!)"
            elif r_vis > r_aud + 0.15:
                interp = "📺 Visual-dominant task response"
            elif r_aud > r_vis + 0.15:
                interp = "🔊 Auditory-dominant task response"
            else:
                interp = "Mixed task response"
        elif abs(r_vis) > 0.3 and abs(r_aud) < 0.15:
            interp = "✅ VISUAL SPECIFIC"
        elif abs(r_aud) > 0.3 and abs(r_vis) < 0.15:
            interp = "✅ AUDITORY SPECIFIC"
        elif abs(r_task) < 0.2:
            interp = "❓ Non-task related (drift/noise?)"
        else:
            interp = "Complex pattern"

        interpretations.append(interp)

        print(f"PC_{i:<3} {r_task:>+.3f}       {r_vis:>+.3f}       {r_aud:>+.3f}       {r_va:>+.3f}       {interp}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    general_task_pcs = sum(1 for i in interpretations if "GENERAL TASK" in i)
    if general_task_pcs > 0:
        print(f"⚠️  {general_task_pcs} PCs capture GENERAL task activation (V≈A)")
        print("   This is why your Visual and Auditory heatmaps look similar!")
        print("   The PCA did NOT separate the conditions.")

    return interpretations


if __name__ == '__main__':
    # Run the debug
    debug_pc_signal('PC_2', 'Vis')
    # הרץ את האבחון:
    diagnose_temporal_profiles()

