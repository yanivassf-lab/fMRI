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
PATH = '/Users/user/Documents/pythonProject/fMRI-runs/validation-experiment/tests/fmri_output_basis500_lam-6to12_derivP2U0'
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
    Calculates the signal amplitude during specific blocks compared to rest.

    CRITICAL CHANGE: This function now operates in 'Strict Mode'.
    - If target is 'v': It ONLY considers 'v' blocks (ignores 'va').
    - If target is 'a': It ONLY considers 'a' blocks (ignores 'va').
    - If target is 'va': It ONLY considers 'va' blocks.

    This ensures that the massive visual response in 'va' blocks does not
    pollute the auditory amplitude calculation.
    """

    # HRF Lag: The BOLD signal peaks ~6 seconds after neural activity.
    # We shift the sampling window to capture the peak response.
    hrf_lag_sec = 6.0
    lag_tr = int(hrf_lag_sec / tr)

    active_vals = []
    rest_vals = []

    for s, e, cond in blocks:
        # Shift start and end indices to account for hemodynamic delay
        s_idx = int(s / tr) + lag_tr
        e_idx = int(e / tr) + lag_tr

        # Boundary checks to prevent crashing
        if s_idx >= len(reconstructed_signal): continue
        e_idx = min(e_idx, len(reconstructed_signal))

        current_segment = reconstructed_signal[s_idx:e_idx]

        # --- Strict Condition Matching ---

        # 1. Visual Only (Pure 'v', ignore mixed 'va')
        if target_cond == 'v':
            if cond == 'v':
                active_vals.extend(current_segment)

        # 2. Auditory Only (Pure 'a', ignore mixed 'va')
        elif target_cond == 'a':
            if cond == 'a':
                active_vals.extend(current_segment)

        # 3. Synergy/Interaction Only (Pure 'va')
        elif target_cond == 'va':
            if cond == 'va':
                active_vals.extend(current_segment)

        # 4. Rest Condition (Baseline)
        if cond == 'x':
            rest_vals.extend(current_segment)

    # Handle cases where a specific condition might not exist in the run
    if not active_vals or not rest_vals:
        return 0.0

    # Calculate difference: Mean Active - Mean Rest
    return np.mean(active_vals) - np.mean(rest_vals)


def classify_component_advanced(r_vis, r_aud, r_int, thresh=0.25):
    """
    סיווג חכם שמבין שגם קורלציה שלילית חזקה היא מידע חשוב.
    """
    # עבודה עם ערכים מוחלטים כדי למדוד "עוצמת קשר"
    abs_v = abs(r_vis)
    abs_a = abs(r_aud)
    abs_int = abs(r_int)

    # אם שום דבר לא עובר את הסף - זה רעש
    if max(abs_v, abs_a, abs_int) < thresh:
        return "Unspecified / Noise"

    # 1. בדיקת אינטגרציה ייחודית (Synergy)
    # כאן אנחנו עדיין רוצים חיוביות, כי סינרגיה שלילית היא מסובכת לפירוש
    if r_int > thresh and (r_int > r_vis + 0.1) and (r_int > r_aud + 0.1):
        return "Pure Synergy (VA Specific)"

    # 2. בדיקת דומיננטיות ויזואלית
    if abs_v > abs_a and abs_v > thresh:
        if r_vis > 0:
            return "Visual Dominant"
        else:
            return "Visual (Inverted/Negative)"  # זיהוי מפורש של ההיפוך

    # 3. בדיקת דומיננטיות שמיעתית
    elif abs_a > abs_v and abs_a > thresh:
        if r_aud > 0:
            return "Auditory Dominant"
        else:
            return "Auditory (Inverted/Negative)"

    # 4. רב-חושי כללי
    elif abs_v > thresh and abs_a > thresh:
        return "Multisensory (General)"

    else:
        return "Complex / Mixed"


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
    # plt.show()


# ==========================================
# 4. Main ROI Analysis (PCA Components)
# ==========================================
def create_pure_regressors(n_points, blocks, tr):
    """
    יוצר רגרסורים שמבוססים רק על בלוקים טהורים.
    'v' בלבד (לא 'va')
    'a' בלבד (לא 'va')
    """
    box_v_pure = np.zeros(n_points)
    box_a_pure = np.zeros(n_points)

    for s, e, cond in blocks:
        idx_s = int(s / tr)
        idx_e = min(int(e / tr), n_points)
        if idx_s >= n_points: continue

        # רק 'v' טהור (לא 'va'!)
        if cond == 'v':
            box_v_pure[idx_s:idx_e] = 1

        # רק 'a' טהור (לא 'va'!)
        elif cond == 'a':
            box_a_pure[idx_s:idx_e] = 1

    # קונבולוציה עם HRF
    hrf = spm_hrf(tr)
    reg_v = np.convolve(box_v_pure, hrf)[:n_points]
    reg_a = np.convolve(box_a_pure, hrf)[:n_points]

    return pd.DataFrame({
        'Model_Visual_Pure': reg_v - np.mean(reg_v),
        'Model_Auditory_Pure': reg_a - np.mean(reg_a)
    })

    # השתמש ברגרסורים הטהורים:
    # pure_design = create_pure_regressors(n_points, BLOCKS, TR)
    #
    # עכשיו הקורלציה בין הרגרסורים צריכה להיות קרובה ל-0!
    # print("Correlation between PURE regressors:")
    # print(pure_design.corr())


from scipy import stats
import numpy as np


def partial_correlation(x, y, covariate):
    """
    קורלציה חלקית: הקשר בין x ל-y אחרי הסרת ההשפעה של covariate.
    """
    # רגרסיה של x על הקו-וריאט
    slope_x, intercept_x, _, _, _ = stats.linregress(covariate, x)
    residual_x = x - (slope_x * covariate + intercept_x)

    # רגרסיה של y על הקו-וריאט
    slope_y, intercept_y, _, _, _ = stats.linregress(covariate, y)
    residual_y = y - (slope_y * covariate + intercept_y)

    # קורלציה בין השאריות
    r, p = stats.pearsonr(residual_x, residual_y)
    return r, p


# === בתוך הלולאה הראשית, החלף את חישוב הקורלציות ב: ===

# קורלציה פשוטה (לשם השוואה)
# r_v_simple, _ = stats.pearsonr(reconstructed, design_df['Model_Visual'])
# r_a_simple, _ = stats.pearsonr(reconstructed, design_df['Model_Auditory'])
#
# # קורלציה חלקית (הפתרון!)
# r_v_partial, _ = partial_correlation(
#     reconstructed,
#     design_df['Model_Visual'].values,
#     design_df['Model_Auditory'].values
# )
# r_a_partial, _ = partial_correlation(
#     reconstructed,
#     design_df['Model_Auditory'].values,
#     design_df['Model_Visual'].values
# )

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def glm_analysis(signal, design_df):
    """
    General Linear Model - הסטנדרט בניתוח fMRI.
    מחזיר Beta weights שמייצגים תרומה ייחודית של כל רגרסור.
    """
    # הכנת הנתונים
    X = design_df[['Model_Visual', 'Model_Auditory', 'Model_Interaction']].values
    y = signal

    # נרמול (Z-score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = (y - np.mean(y)) / np.std(y)

    # התאמת המודל
    model = LinearRegression(fit_intercept=True)
    model.fit(X_scaled, y_scaled)

    # Beta weights (מנורמלים)
    betas = model.coef_

    # חישוב T-statistics (פשוט)
    y_pred = model.predict(X_scaled)
    residuals = y_scaled - y_pred
    mse = np.mean(residuals ** 2)
    se = np.sqrt(mse / len(y))
    t_stats = betas / se

    return {
        'Beta_Visual': betas[0],
        'Beta_Auditory': betas[1],
        'Beta_Interaction': betas[2],
        'T_Visual': t_stats[0],
        'T_Auditory': t_stats[1],
        'T_Interaction': t_stats[2],
        'R_squared': model.score(X_scaled, y_scaled)
    }


# === בתוך הלולאה הראשית: ===
# glm_results = glm_analysis(reconstructed, design_df)
#
# final_results.append({
#     'PC': pc_name,
#     'Region': roi_name,
#     'Spatial_Weight': spatial_weight,
#     # GLM Results (הפתרון הנכון!)
#     'Beta_Visual': glm_results['Beta_Visual'],
#     'Beta_Auditory': glm_results['Beta_Auditory'],
#     'Beta_Interaction': glm_results['Beta_Interaction'],
#     'R_squared': glm_results['R_squared'],
#     # השאר גם את הישנים להשוואה
#     'Corr_Visual_Simple': r_v_simple,
#     'Corr_Auditory_Simple': r_a_simple,
# })
# עכשיו r_v_partial מייצג את הקשר הייחודי לראייה (אחרי הסרת שמיעה)
# ו-r_a_partial מייצג את הקשר הייחודי לשמיעה (אחרי הסרת ראייה)
def run_advanced_roi_analysis_v2():
    """
    Improved version with GLM and visualization
    """
    print("\n" + "=" * 50)
    print("IMPROVED ROI ANALYSIS (with GLM)")
    print("=" * 50)

    # Setup
    dataset = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    atlas_filename = dataset.maps
    atlas_labels = dataset.labels

    tmp = np.loadtxt(list(TS_FILES.values())[0])
    n_points = tmp.shape[1] if tmp.ndim == 2 else len(tmp)

    # Two types of Design Matrix
    design_df = create_design_matrix(n_points, BLOCKS, TR)
    pure_design = create_pure_regressors(n_points, BLOCKS, TR)

    # Diagnose regressor correlations
    print("\n--- Standard Regressors ---")
    print(design_df.corr().round(3))
    print("\n--- Pure Regressors ---")
    print(pure_design.corr().round(3))

    # Combine into one design matrix
    full_design = pd.concat([design_df, pure_design], axis=1)

    atlas_img = nib.load(atlas_filename)
    final_results = []

    for pc_name, ts_path in TS_FILES.items():
        if pc_name not in NII_FILES: continue
        print(f">> Analyzing {pc_name}...")

        ts_arr = np.loadtxt(ts_path)
        temporal_sig = ts_arr[1, :] if ts_arr.ndim == 2 else ts_arr

        pc_img = nib.load(NII_FILES[pc_name])
        resampled_atlas = resample_to_img(atlas_img, pc_img, interpolation='nearest',
                                          force_resample=True, copy_header=True)
        pc_data = pc_img.get_fdata()
        atlas_data = resampled_atlas.get_fdata()

        for i, label_bytes in enumerate(atlas_labels):
            roi_id = i + 1
            roi_name = label_bytes.decode('utf-8')
            roi_mask = (atlas_data == roi_id)
            if np.sum(roi_mask) == 0: continue

            spatial_weight = np.mean(pc_data[roi_mask])
            if abs(spatial_weight) < 0.0001: continue

            reconstructed = temporal_sig * spatial_weight

            # 1. Simple correlations
            r_v_simple, _ = stats.pearsonr(reconstructed, design_df['Model_Visual'])
            r_a_simple, _ = stats.pearsonr(reconstructed, design_df['Model_Auditory'])

            # 2. Pure regressor correlations
            r_v_pure, _ = stats.pearsonr(reconstructed, pure_design['Model_Visual_Pure'])
            r_a_pure, _ = stats.pearsonr(reconstructed, pure_design['Model_Auditory_Pure'])

            # 3. Partial correlations
            r_v_partial, _ = partial_correlation(
                reconstructed,
                design_df['Model_Visual'].values,
                design_df['Model_Auditory'].values
            )
            r_a_partial, _ = partial_correlation(
                reconstructed,
                design_df['Model_Auditory'].values,
                design_df['Model_Visual'].values
            )

            # 4. GLM
            glm_results = glm_analysis(reconstructed, design_df)

            # 5. Amplitudes
            amp_v = calculate_amplitude(reconstructed, BLOCKS, TR, 'v')
            amp_a = calculate_amplitude(reconstructed, BLOCKS, TR, 'a')

            final_results.append({
                'PC': pc_name,
                'Region': roi_name,
                'Spatial_Weight': spatial_weight,
                'Corr_Visual_Simple': r_v_simple,
                'Corr_Auditory_Simple': r_a_simple,
                'Corr_Visual_Pure': r_v_pure,
                'Corr_Auditory_Pure': r_a_pure,
                'Corr_Visual_Partial': r_v_partial,
                'Corr_Auditory_Partial': r_a_partial,
                'Beta_Visual': glm_results['Beta_Visual'],
                'Beta_Auditory': glm_results['Beta_Auditory'],
                'Beta_Interaction': glm_results['Beta_Interaction'],
                'Amp_Visual': amp_v,
                'Amp_Auditory': amp_a,
            })

    df = pd.DataFrame(final_results)

    # Print comparison
    print("\n" + "=" * 50)
    print("COMPARISON: Different Methods")
    print("=" * 50)

    vis_region = df[df['Region'].str.contains('Vis')].iloc[0] if any(df['Region'].str.contains('Vis')) else df.iloc[0]

    print(f"\nExample Region: {vis_region['Region']}")
    print(f"  Simple Corr Visual:    {vis_region['Corr_Visual_Simple']:.3f}")
    print(f"  Simple Corr Auditory:  {vis_region['Corr_Auditory_Simple']:.3f}")
    print(f"  ---")
    print(f"  Pure Corr Visual:      {vis_region['Corr_Visual_Pure']:.3f}")
    print(f"  Pure Corr Auditory:    {vis_region['Corr_Auditory_Pure']:.3f}")
    print(f"  ---")
    print(f"  Partial Corr Visual:   {vis_region['Corr_Visual_Partial']:.3f}")
    print(f"  Partial Corr Auditory: {vis_region['Corr_Auditory_Partial']:.3f}")
    print(f"  ---")
    print(f"  GLM Beta Visual:       {vis_region['Beta_Visual']:.3f}")
    print(f"  GLM Beta Auditory:     {vis_region['Beta_Auditory']:.3f}")

    # ==========================================
    # VISUALIZATION - THE MISSING PART!
    # ==========================================

    print("\n>> Generating Heatmaps...")

    # Calculate dynamic height
    n_regions = len(df['Region'].unique())
    fig_height = max(15, (n_regions * 0.6) + 4)

    # Create figure with 4 subplots (comparing methods)
    fig, axes = plt.subplots(1, 4, figsize=(36, fig_height))

    # Dynamic scale
    vmax = 0.8
    vmin = -0.8

    heatmap_kwargs = {
        'cmap': 'coolwarm',
        'center': 0,
        'vmin': vmin,
        'vmax': vmax,
        'annot': True,
        'fmt': ".2f",
        'cbar': True,
        'cbar_kws': {'shrink': 0.3, 'aspect': 20, 'location': 'top'},
        'linewidths': 0.5,
        'linecolor': 'white',
        'yticklabels': True,
        'annot_kws': {"size": 8}
    }

    def color_y_labels(ax):
        """Color visual regions green, auditory regions brown"""
        for label in ax.get_yticklabels():
            txt = label.get_text()
            label.set_fontsize(9)
            if 'Vis' in txt:
                label.set_color('green')
                label.set_fontweight('bold')
            elif 'Aud' in txt or 'SomMot' in txt:
                label.set_color('brown')
                label.set_fontweight('bold')

    # --- 1. Simple Correlation - Visual ---
    pivot_v_simple = df.pivot(index='Region', columns='PC', values='Corr_Visual_Simple')
    sns.heatmap(pivot_v_simple, ax=axes[0], **heatmap_kwargs)
    axes[0].set_title('Simple Corr: VISUAL\n(Original Method)', fontsize=14, pad=20)
    axes[0].tick_params(axis='x', rotation=90, labelsize=10)
    color_y_labels(axes[0])

    # --- 2. Simple Correlation - Auditory ---
    pivot_a_simple = df.pivot(index='Region', columns='PC', values='Corr_Auditory_Simple')
    sns.heatmap(pivot_a_simple, ax=axes[1], **heatmap_kwargs)
    axes[1].set_title('Simple Corr: AUDITORY\n(Original Method)', fontsize=14, pad=20)
    axes[1].tick_params(axis='x', rotation=90, labelsize=10)
    color_y_labels(axes[1])

    # --- 3. GLM Beta - Visual ---
    pivot_v_glm = df.pivot(index='Region', columns='PC', values='Beta_Visual')
    sns.heatmap(pivot_v_glm, ax=axes[2], **heatmap_kwargs)
    axes[2].set_title('GLM Beta: VISUAL\n(Improved Method)', fontsize=14, pad=20)
    axes[2].tick_params(axis='x', rotation=90, labelsize=10)
    color_y_labels(axes[2])

    # --- 4. GLM Beta - Auditory ---
    pivot_a_glm = df.pivot(index='Region', columns='PC', values='Beta_Auditory')
    sns.heatmap(pivot_a_glm, ax=axes[3], **heatmap_kwargs)
    axes[3].set_title('GLM Beta: AUDITORY\n(Improved Method)', fontsize=14, pad=20)
    axes[3].tick_params(axis='x', rotation=90, labelsize=10)
    color_y_labels(axes[3])

    plt.subplots_adjust(left=0.15, right=0.98, bottom=0.05, top=0.93, wspace=0.3)

    # Save
    filename = os.path.join(PATH, "GLM_vs_Simple_Correlation_Heatmaps.png")
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"\n✅ SUCCESS! Heatmap saved to: {filename}")

    # ==========================================
    # SECOND FIGURE: Amplitude comparison
    # ==========================================

    fig2, axes2 = plt.subplots(1, 2, figsize=(20, fig_height))

    # Amplitude scale
    amp_vals = np.concatenate([df['Amp_Visual'].values, df['Amp_Auditory'].values])
    amp_vmax = np.percentile(np.abs(amp_vals), 95)

    amp_kwargs = {
        'cmap': 'coolwarm',
        'center': 0,
        'vmin': -amp_vmax,
        'vmax': amp_vmax,
        'annot': True,
        'fmt': ".1f",
        'cbar': True,
        'linewidths': 0.5,
        'linecolor': 'white',
        'annot_kws': {"size": 8}
    }

    # Visual Amplitude
    pivot_amp_v = df.pivot(index='Region', columns='PC', values='Amp_Visual')
    sns.heatmap(pivot_amp_v, ax=axes2[0], **amp_kwargs)
    axes2[0].set_title('Visual Amplitude\n(Pure V blocks only)', fontsize=14, pad=20)
    color_y_labels(axes2[0])

    # Auditory Amplitude
    pivot_amp_a = df.pivot(index='Region', columns='PC', values='Amp_Auditory')
    sns.heatmap(pivot_amp_a, ax=axes2[1], **amp_kwargs)
    axes2[1].set_title('Auditory Amplitude\n(Pure A blocks only)', fontsize=14, pad=20)
    color_y_labels(axes2[1])

    plt.tight_layout()

    filename2 = os.path.join(PATH, "Amplitude_Heatmaps.png")
    plt.savefig(filename2, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"✅ SUCCESS! Amplitude heatmap saved to: {filename2}")

    return df


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
    # plt.show()

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
    # 5. Visualization (SCROLLABLE, AMPLITUDE, COLORED TEXT)
    # ==========================================
    df = pd.DataFrame(final_results)

    if df.empty:
        print("No results found.")
        return

    # מיון וסינון
    df['Max_Abs_Corr'] = df[['Corr_Visual', 'Corr_Auditory', 'Corr_Interaction']].abs().max(axis=1)
    interesting = df[df['Max_Abs_Corr'] > 0.15].sort_values(by='Max_Abs_Corr', ascending=False)
    print("\n" + "=" * 60)
    print("TOP FINDINGS")
    print("=" * 60)
    cols_to_show = ['PC', 'Region', 'Class', 'Corr_Visual', 'Corr_Auditory', 'Amp_Visual']
    print(interesting[cols_to_show].to_string(index=False))

    print("\nGenerating High-Resolution SCROLLABLE Image file...")

    # חישוב גובה דינמי למניעת דריסות (0.6 אינץ' לשורה)
    n_regions = len(df['Region'].unique())
    fig_height = max(15, (n_regions * 0.6) + 4)

    # יצירת הקנבס (בלי להציג אותו)
    fig, axes = plt.subplots(1, 3, figsize=(28, fig_height))

    # חישוב סקאלה דינמית לאמפליטודה (כדי שהצבעים לא יישרפו)
    # לוקחים את הערכים מכל העמודות כדי שהסקאלה תהיה אחידה
    all_vals = np.concatenate([df['Amp_Visual'], df['Amp_Auditory']])
    # נשתמש באחוזון 98 כדי לסנן קיצון
    vmax = np.percentile(np.abs(all_vals), 98)
    vmin = -vmax

    # הגדרות עיצוב ל-Heatmap
    heatmap_kwargs = {
        'cmap': 'coolwarm',
        'center': 0,
        'vmin': vmin,
        'vmax': vmax,
        'annot': True,
        'fmt': ".1f",
        'cbar': True,
        'cbar_kws': {'shrink': 0.5, 'aspect': 20, 'location': 'top',
                     'label': 'Signal Amplitude (Change from Rest)'},
        'linewidths': 1,
        'linecolor': 'white',
        'yticklabels': True,
        'annot_kws': {"size": 10}
    }

    # פונקציית צביעה מתקדמת
    def color_y_labels(ax, target_colors):
        ax.tick_params(axis='y', labelsize=11)
        for label in ax.get_yticklabels():
            txt = label.get_text()
            # איפוס צבע
            label.set_color('black')

            # צביעה לפי מילות מפתח
            if 'Vis' in txt and 'green' in target_colors:
                label.set_color('green')
                label.set_fontweight('bold')
            elif ('Aud' in txt or 'SomMot' in txt) and 'brown' in target_colors:
                label.set_color('brown')
                label.set_fontweight('bold')

    # --- 1. Visual Amplitude ---
    # מציגים את Amp_Visual במקום Corr_Visual
    p_v = df.pivot(index='Region', columns='PC', values='Amp_Visual')
    sns.heatmap(p_v, ax=axes[0], **heatmap_kwargs)
    axes[0].set_title('Visual Amplitude (Strength)', fontsize=22, pad=20)
    axes[0].tick_params(axis='x', rotation=90, labelsize=12)
    color_y_labels(axes[0], ['green'])  # רק ירוק בגרף הראייה

    # --- 2. Auditory Amplitude ---
    p_a = df.pivot(index='Region', columns='PC', values='Amp_Auditory')
    sns.heatmap(p_a, ax=axes[1], **heatmap_kwargs)
    axes[1].set_title('Auditory Amplitude (Strength)', fontsize=22, pad=20)
    axes[1].tick_params(axis='x', rotation=90, labelsize=12)
    color_y_labels(axes[1], ['brown'])  # רק חום בגרף השמיעה

    # --- 3. Synergy (Weighted Interaction) ---
    # כאן נשתמש בקורלציה מוכפלת במשקל, כי אין לנו "אמפליטודת אינטראקציה" ישירה מהפונקציה
    # אבל ננרמל את זה שייראה בסקאלה דומה
    df['Synergy_Index'] = df['Corr_Interaction'] * df['Spatial_Weight'] * 10

    p_i = df.pivot(index='Region', columns='PC', values='Synergy_Index')
    sns.heatmap(p_i, ax=axes[2], **heatmap_kwargs)
    axes[2].set_title('Synergy Index (Weighted)', fontsize=22, pad=20)
    axes[2].tick_params(axis='x', rotation=90, labelsize=12)
    color_y_labels(axes[2], ['green', 'brown'])  # גם וגם בגרף הסינרגיה

    # מרווחים
    plt.subplots_adjust(left=0.2, right=0.98, bottom=0.05, top=0.93, wspace=0.3)

    # שמירה
    filename = "Final_Analysis_Amplitude_Scrollable.png"
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close()

    print(f"\nSUCCESS! High-Res Image saved to: {filename}")
    print(">> Please open the file externally to verify colors and scroll. <<")

    # # ==========================================
    # # 5. Visualization & Summary (HUGE SCROLLABLE IMAGE)
    # # ==========================================
    # df = pd.DataFrame(final_results)
    #
    # if df.empty:
    #     print("No results found.")
    #     return
    #
    # # 1. מיון וסינון
    # df['Max_Abs_Corr'] = df[['Corr_Visual', 'Corr_Auditory', 'Corr_Interaction']].abs().max(axis=1)
    # interesting = df[df['Max_Abs_Corr'] > 0.20].sort_values(by='Max_Abs_Corr', ascending=False)
    #
    # print("\n" + "=" * 60)
    # print("TOP FINDINGS")
    # print("=" * 60)
    # cols_to_show = ['PC', 'Region', 'Class', 'Corr_Visual', 'Corr_Auditory', 'Amp_Visual']
    # print(interesting[cols_to_show].head(20).to_string(index=False))
    #
    # # --- חישוב גובה מפלצתי ---
    # # נותנים 0.8 אינץ' לכל שורה. אם יש 100 אזורים -> גובה 80 אינץ' (כ-2 מטר)
    # n_regions = len(df['Region'].unique())
    # # הגובה המינימלי הוא 15, אבל הוא יגדל ככל שיש יותר אזורים
    # fig_height = max(15, n_regions * 0.8)
    #
    # print(f"\nGenerating HUGE image (Height: {fig_height:.1f} inches).")
    # print("Please wait, this might take a moment...")
    #
    # # יצירת הקנבס הענק
    # fig, axes = plt.subplots(1, 3, figsize=(25, fig_height))
    #
    # # הגדרות עיצוב - פונט קטן יותר למספרים
    # heatmap_kwargs = {
    #     'cmap': 'coolwarm',
    #     'center': 0,
    #     'vmin': -0.8,
    #     'vmax': 0.8,
    #     'annot': True,
    #     'fmt': ".2f",
    #     # 'cbar': True,
    #     'linewidths': 1,  # קווים עבים יותר להפרדה
    #     'linecolor': 'black',  # צבע שחור להפרדה ברורה
    #     'yticklabels': True,
    #     'annot_kws': {"size": 10}  # הקטנת המספרים בתוך התאים
    # }
    #
    # # פונקציית עזר לצביעת תוויות
    # def color_y_labels(ax):
    #     for label in ax.get_yticklabels():
    #         txt = label.get_text()
    #         if 'Vis' in txt:
    #             label.set_color('green')
    #             label.set_fontweight('bold')
    #         elif 'Aud' in txt or 'SomMot' in txt:
    #             label.set_color('brown')  # חום
    #             label.set_fontweight('bold')
    #         label.set_fontsize(10)  # פונט קריא לאזורים
    #
    # # --- 1. Visual Fit ---
    # p_v = df.pivot(index='Region', columns='PC', values='Corr_Visual')
    # sns.heatmap(p_v, ax=axes[0], cbar=False, **heatmap_kwargs)
    # axes[0].set_title('Visual Model Fit', fontsize=20, pad=20)
    # axes[0].tick_params(axis='x', rotation=90, labelsize=12)
    # color_y_labels(axes[0])
    #
    # # --- 2. Auditory Fit ---
    # p_a = df.pivot(index='Region', columns='PC', values='Corr_Auditory')
    # sns.heatmap(p_a, ax=axes[1], cbar=False, **heatmap_kwargs)
    # axes[1].set_title('Auditory Model Fit', fontsize=20, pad=20)
    # axes[1].tick_params(axis='x', rotation=90, labelsize=12)
    # color_y_labels(axes[1])
    #
    # # --- 3. Synergy Fit ---
    # p_i = df.pivot(index='Region', columns='PC', values='Corr_Interaction')
    # sns.heatmap(p_i, ax=axes[2], cbar=True, **heatmap_kwargs)
    # axes[2].set_title('Synergy Fit (VA Only)', fontsize=20, pad=20)
    # axes[2].tick_params(axis='x', rotation=90, labelsize=12)
    # color_y_labels(axes[2])
    #
    # # ריווח בין הגרפים
    # plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.98, wspace=0.3)
    #
    # # שמירה לקובץ (הפתרון האמיתי לגלילה)
    # filename = "Full_Analysis_Heatmap.png"
    # plt.savefig(filename, dpi=100, bbox_inches='tight')  # dpi נמוך יחסית כדי שהקובץ לא ישקול טרה-בייט
    # print(f"Done! Saved to {filename}")
    # print(">> OPEN THE FILE EXTERNALLY TO SCROLL <<")
    #
    # # מנסה להציג גם בחלון (אבל זה יהיה ענק)
    # try:
    #     plt.show()
    # except:
    #     pass


# ==========================================
# 6. Execution
# ==========================================
if __name__ == '__main__':
    # 1. First, check the global signal to understand overall brain state
    analyze_global_signal()

    # 2. Then, run the detailed component-wise analysis
    run_advanced_roi_analysis_v2()

# pip install nilearn nibabel pandas matplotlib seaborn scipy numpy
