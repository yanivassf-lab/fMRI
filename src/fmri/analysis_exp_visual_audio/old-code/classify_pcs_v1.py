import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn import datasets

# ==========================================
# 1. Configuration & Constants
# ==========================================
TR = 0.75  # Repetition Time in seconds. Critical for timing accuracy.
PATH = '/Users/user/Documents/pythonProject/fMRI-runs/validation-experiment/fmri_output'
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


# ==========================================
# 2. Helper Functions: Modeling & Stats
# ==========================================

def spm_hrf(tr, duration=32):
    """
    Generates a canonical Hemodynamic Response Function (HRF).
    This models how blood flow increases in response to neural activity.
    """
    dt = tr
    t = np.arange(0, duration, dt)
    # Double-Gamma function (Peak + Undershoot)
    hrf = stats.gamma.pdf(t, 6) - 1 / 6 * stats.gamma.pdf(t, 16)
    return hrf / np.sum(hrf)


def create_design_matrix(n_points, blocks, tr):
    """
    Creates the 'Ideal' Regressors.
    1. Visual Model: Active during 'v' and 'va'.
    2. Auditory Model: Active during 'a' and 'va'.
    3. Interaction Model: Active ONLY during 'va'.

    Includes Convolution with HRF and Demeaning.
    """
    timeline_len = n_points
    box_v = np.zeros(timeline_len)
    box_a = np.zeros(timeline_len)
    box_int = np.zeros(timeline_len)  # Interaction / Synergy

    for s, e, cond in blocks:
        idx_s = int(s / tr)
        idx_e = int(e / tr)
        idx_e = min(idx_e, timeline_len)
        if idx_s >= timeline_len: continue

        # --- Main Effects ---
        if 'v' in cond:  # Captures 'v' AND 'va'
            box_v[idx_s:idx_e] = 1

        if 'a' in cond:  # Captures 'a' AND 'va'
            box_a[idx_s:idx_e] = 1

        # --- Interaction Effect ---
        # Stricter condition: Only 'va' blocks
        if cond == 'va':
            box_int[idx_s:idx_e] = 1

    # Convolve with HRF (Simulates the delay and dispersion of blood flow)
    hrf = spm_hrf(tr)
    reg_v = np.convolve(box_v, hrf)[:timeline_len]
    reg_a = np.convolve(box_a, hrf)[:timeline_len]
    reg_int = np.convolve(box_int, hrf)[:timeline_len]

    # === Demeaning (Crucial) ===
    # Subtracting the mean makes the regressors centered around zero,
    # just like the PCA components.
    # Positive values = Activation, Negative values = Rest/Suppression.
    return pd.DataFrame({
        'Model_Visual': reg_v - np.mean(reg_v),
        'Model_Auditory': reg_a - np.mean(reg_a),
        'Model_Interaction': reg_int - np.mean(reg_int)
    })


def calculate_amplitude(reconstructed_signal, blocks, tr, target_cond='v'):
    """
    Calculates the actual AMPLITUDE difference between Active and Rest blocks.
    Unlike correlation (which checks shape/timing), this checks Magnitude.

    Note: We manually add a lag of 8 TRs (~6 seconds) to account for HRF
    since we are doing raw averaging here.
    """
    lag_tr = int(6.0 / tr)  # Approx 6 seconds lag
    active_vals = []
    rest_vals = []

    for s, e, cond in blocks:
        s_idx = int(s / tr) + lag_tr
        e_idx = int(e / tr) + lag_tr

        # Boundary check
        if s_idx >= len(reconstructed_signal): continue
        e_idx = min(e_idx, len(reconstructed_signal))

        current_segment = reconstructed_signal[s_idx:e_idx]

        # Check if this block matches the target condition
        # (e.g., if target is 'v', we include 'v' and 'va')
        is_active = False
        if target_cond == 'v' and 'v' in cond:
            is_active = True
        elif target_cond == 'a' and 'a' in cond:
            is_active = True
        elif target_cond == 'va' and cond == 'va':
            is_active = True

        if is_active:
            active_vals.extend(current_segment)
        elif cond == 'x':  # Rest condition
            rest_vals.extend(current_segment)

    if not active_vals or not rest_vals:
        return 0.0

    # Amplitude = Mean Activation - Mean Rest
    return np.mean(active_vals) - np.mean(rest_vals)


def classify_component_advanced(r_vis, r_aud, r_int, thresh=0.25):
    """
    Automatically classifies the component type based on correlation strength.
    """
    is_v = r_vis > thresh
    is_a = r_aud > thresh
    is_int = r_int > thresh

    # 1. Pure Synergy: Interaction is significantly higher than single senses
    if is_int and (r_int > r_vis + 0.1) and (r_int > r_aud + 0.1):
        return "Pure Synergy (VA Specific)"

    # 2. General Multisensory: Responds to both
    elif is_v and is_a:
        return "Multisensory (General)"

    # 3. Unisensory Dominance
    elif is_v and not is_a:
        return "Visual Dominant"
    elif is_a and not is_v:
        return "Auditory Dominant"

    # 4. Suppression: Negative correlation (Deactivation)
    elif r_vis < -thresh and r_aud < -thresh:
        return "Suppressed (Deactivated)"

    else:
        return "Unspecified / Noise"


# ==========================================
# 3. Global Signal Analysis
# ==========================================
def analyze_global_signal():
    """
    Analyzes the original averaged signal (before PCA) to see
    if there is a global arousal effect or global response to tasks.
    """
    print("\n" + "=" * 50)
    print("ANALYZING GLOBAL SIGNAL (ORIGINAL)")
    print("=" * 50)

    if not os.path.exists(GLOBAL_SIGNAL_FILE):
        print(f"File {GLOBAL_SIGNAL_FILE} not found. Skipping.")
        return

    # Load data
    ts_arr = np.loadtxt(GLOBAL_SIGNAL_FILE)
    # Handle dimensions (ensure 1D array)
    global_sig = ts_arr[1, :] if ts_arr.ndim == 2 else ts_arr
    n_points = len(global_sig)

    # Create models
    design_df = create_design_matrix(n_points, BLOCKS, TR)

    # Calculate Correlations
    r_v, p_v = stats.pearsonr(global_sig, design_df['Model_Visual'])
    r_a, p_a = stats.pearsonr(global_sig, design_df['Model_Auditory'])
    r_int, p_int = stats.pearsonr(global_sig, design_df['Model_Interaction'])

    # Calculate Amplitude (Magnitude of change)
    amp_v = calculate_amplitude(global_sig, BLOCKS, TR, 'v')
    amp_a = calculate_amplitude(global_sig, BLOCKS, TR, 'a')

    print(f"Global Signal Stats:")
    print(f"  > Correlation with Visual Model:    {r_v:.3f} (p={p_v:.4f})")
    print(f"  > Amplitude Change (Vis vs Rest):   {amp_v:.3f}")
    print(f"  > Correlation with Auditory Model:  {r_a:.3f} (p={p_a:.4f})")
    print(f"  > Amplitude Change (Aud vs Rest):   {amp_a:.3f}")
    print(f"  > Correlation with Interaction:     {r_int:.3f}")

    # Plot
    plt.figure(figsize=(12, 4))
    # Normalize global signal for visualization overlap
    norm_sig = (global_sig - np.mean(global_sig)) / np.std(global_sig)

    plt.plot(norm_sig, label='Global Signal (Z-scored)', color='black', alpha=0.6, linewidth=1)
    plt.plot(design_df['Model_Visual'], label='Visual Model', color='red', alpha=0.4)
    plt.plot(design_df['Model_Auditory'], label='Auditory Model', color='green', alpha=0.4)
    plt.title("Original Global Signal vs. Task Models")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. Main ROI Analysis (PCA Components)
# ==========================================

def run_advanced_roi_analysis():
    print("\n" + "=" * 50)
    print("STARTING ADVANCED ROI ANALYSIS (Schaefer 2018)")
    print("=" * 50)

    # 1. Automatic Atlas Download
    print(">> Fetching Atlas...")
    # This downloads the Schaefer atlas (100 parcels) to your nilearn_data folder
    dataset = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    atlas_filename = dataset.maps
    atlas_labels = dataset.labels  # List of label names (in bytes)
    print(f"   Atlas loaded successfully: {atlas_filename}")

    # 2. Check Input Files
    if not TS_FILES:
        print("Error: No Time Series files defined.")
        return

    # 3. Setup Design Matrix (using the length of the first file)
    tmp = np.loadtxt(list(TS_FILES.values())[0])
    n_points = tmp.shape[1] if tmp.ndim == 2 else len(tmp)
    design_df = create_design_matrix(n_points, BLOCKS, TR)

    # Plot Design Matrix for verification
    plt.figure(figsize=(10, 3))
    plt.plot(design_df['Model_Visual'], label='Vis', alpha=0.7)
    plt.plot(design_df['Model_Auditory'], label='Aud', alpha=0.7)
    plt.plot(design_df['Model_Interaction'], label='Synergy', linestyle='--', color='k', alpha=0.5)
    plt.title('Design Matrix (Regressors)')
    plt.legend()
    plt.show()

    # Load Atlas Image
    atlas_img = nib.load(atlas_filename)
    final_results = []

    # 4. Loop through each PCA Component
    for pc_name, ts_path in TS_FILES.items():
        if pc_name not in NII_FILES: continue
        print(f">> Analyzing {pc_name}...")

        # A. Load Temporal Profile (The "When")
        ts_arr = np.loadtxt(ts_path)
        temporal_sig = ts_arr[1, :] if ts_arr.ndim == 2 else ts_arr

        # B. Load Spatial Map & Resample Atlas (The "Where")
        # We must align the Atlas resolution to the PCA map resolution
        pc_img = nib.load(NII_FILES[pc_name])
        resampled_atlas = resample_to_img(atlas_img, pc_img, interpolation='nearest', force_resample=True,
                                          copy_header=True)
        pc_data = pc_img.get_fdata()
        atlas_data = resampled_atlas.get_fdata()

        # C. Loop through Atlas Regions
        for i, label_bytes in enumerate(atlas_labels):
            roi_id = i + 1  # Atlas IDs start at 1
            roi_name = label_bytes.decode('utf-8')  # Convert bytes to string

            # Create boolean mask for current ROI
            roi_mask = (atlas_data == roi_id)
            if np.sum(roi_mask) == 0: continue  # Skip empty regions

            # --- D. Signal Reconstruction ---
            # 1. Calculate Average Spatial Weight for this ROI in this PC
            spatial_weight = np.mean(pc_data[roi_mask])

            # Optimization: Skip regions that are irrelevant for this PC (weight near 0)
            if abs(spatial_weight) < 0.0001: continue

            # 2. Reconstruct Signal: Time * Weight
            # This fixes sign ambiguity and gives local signal approximation
            reconstructed = temporal_sig * spatial_weight

            # --- E. Statistics ---
            # 1. Timing Check (Correlation with Model)
            r_v, _ = stats.pearsonr(reconstructed, design_df['Model_Visual'])
            r_a, _ = stats.pearsonr(reconstructed, design_df['Model_Auditory'])
            r_int, _ = stats.pearsonr(reconstructed, design_df['Model_Interaction'])

            # 2. Magnitude Check (Amplitude Difference: Active - Rest)
            amp_v = calculate_amplitude(reconstructed, BLOCKS, TR, 'v')
            amp_a = calculate_amplitude(reconstructed, BLOCKS, TR, 'a')

            # Store Results
            final_results.append({
                'PC': pc_name,
                'Region': roi_name,
                'Spatial_Weight': spatial_weight,
                'Corr_Visual': r_v,
                'Corr_Auditory': r_a,
                'Corr_Interaction': r_int,
                'Amp_Visual': amp_v,  # NEW: Actual strength check
                'Amp_Auditory': amp_a,  # NEW: Actual strength check
                'Class': classify_component_advanced(r_v, r_a, r_int)
            })

    # ==========================================
    # 5. Visualization & Summary
    # ==========================================
    df = pd.DataFrame(final_results)

    if df.empty:
        print("No results found.")
        return

    # Filter for interesting findings (High correlation OR High Amplitude)
    # Using Amplitude filter ensures we don't just get noise that "looks like" the task
    interesting = df[
        (df['Corr_Visual'].abs() > 0.1) |
        (df['Corr_Auditory'].abs() > 0.1) |
        (df['Corr_Interaction'].abs() > 0.1)
        ].sort_values(by=['PC', 'Corr_Visual'], ascending=False)

    print("\n=== TOP SIGNIFICANT FINDINGS ===")
    print(interesting[['PC', 'Region', 'Class', 'Corr_Visual', 'Corr_Auditory', 'Corr_Interaction', 'Amp_Visual']].head(
        20))
    # --- Plotting 3 Heatmaps (FIXED) ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))  # הגדלתי קצת את הגובה

    # 1. Visual Fit
    p_v = df.pivot(index='Region', columns='PC', values='Corr_Visual')
    # annot=False : מבטל את המספרים
    # cbar=True : מוסיף מקרא צבעים כדי שנבין את הערכים
    sns.heatmap(p_v, ax=axes[0], cmap='coolwarm', center=0, vmin=-0.8, vmax=0.8, annot=False, cbar=True)
    axes[0].set_title('Visual Model Fit')

    # 2. Auditory Fit
    p_a = df.pivot(index='Region', columns='PC', values='Corr_Auditory')
    sns.heatmap(p_a, ax=axes[1], cmap='coolwarm', center=0, vmin=-0.8, vmax=0.8, annot=False, cbar=True)
    axes[1].set_title('Auditory Model Fit')

    # 3. Synergy Fit
    p_i = df.pivot(index='Region', columns='PC', values='Corr_Interaction')
    # כאן הסקאלה שונה (סגול), מתאים לערכים חיוביים
    sns.heatmap(p_i, ax=axes[2], cmap='RdPu', vmin=0, vmax=0.8, annot=False, cbar=True)
    axes[2].set_title('Synergy Fit (VA Only)')

    plt.tight_layout()
    plt.show()


# ==========================================
# 6. Execution
# ==========================================

# 1. First, check the global signal to understand overall brain state
analyze_global_signal()

# 2. Then, run the detailed component-wise analysis
run_advanced_roi_analysis()

# pip install nilearn nibabel pandas matplotlib seaborn scipy numpy
