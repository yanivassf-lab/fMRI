import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import datetime
import math
import nibabel as nib  # pip install nibabel

# ==========================================
# 1. הגדרות וקונפיגורציה
# ==========================================
TR = 0.75
HRF_LAG_SEC = 6.0
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# קבצי הטקסט (Time Series)
TS_FILES = {
    'Original_Signal': 'original_averaged_signal_intensity.txt',
    'PC_0': 'temporal_profile_pc_0.txt',
    'PC_1': 'temporal_profile_pc_1.txt',
    'PC_2': 'temporal_profile_pc_2.txt',
    'PC_3': 'temporal_profile_pc_3.txt',
    'PC_4': 'temporal_profile_pc_4.txt',
    'PC_5': 'temporal_profile_pc_5.txt',
    'PC_6': 'temporal_profile_pc_6.txt'
}

# קבצי המפות (Spatial Maps) - אופציונלי
NII_FILES = {
    'PC_0': 'eigenfunction_0_importance_map.nii.gz',
    'PC_1': 'eigenfunction_1_importance_map.nii.gz',
    'PC_2': 'eigenfunction_2_importance_map.nii.gz',
    'PC_3': 'eigenfunction_3_importance_map.nii.gz',
    'PC_4': 'eigenfunction_4_importance_map.nii.gz',
    'PC_5': 'eigenfunction_5_importance_map.nii.gz',
    'PC_6': 'eigenfunction_6_importance_map.nii.gz',
    # ניתן להוסיף עוד אם יש
}

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

COLORS = {'va': '#9b59b6', 'v': '#3498db', 'a': '#2ecc71', 'x': '#95a5a6'}


# ==========================================
# 2. פונקציות עיבוד
# ==========================================

def load_time_series(file_dict):
    data = {}
    valid_files = {k: v for k, v in file_dict.items() if os.path.exists(v)}
    if not valid_files: raise FileNotFoundError("No time series files found.")

    first = np.loadtxt(list(valid_files.values())[0])
    n = first.shape[1] if first.ndim == 2 else len(first)
    times = np.arange(n) * TR

    for name, f in valid_files.items():
        arr = np.loadtxt(f)
        data[name] = arr[1, :] if arr.ndim == 2 else arr

    return pd.DataFrame(data).assign(Time=times)


def apply_labels(df, blocks, lag):
    df['Condition'] = 'unknown'
    for s, e, c in blocks:
        mask = (df['Time'] >= s + lag) & (df['Time'] < e + lag)
        df.loc[mask, 'Condition'] = c
    return df


def align_pca_sign_and_record(df, reference_condition='v', baseline_condition='x'):
    """
    מיישר כיוונים כך ש-V תמיד חיובי ביחס ל-X.
    מחזיר גם מילון של ה-Flips כדי שנוכל ליישם אותם על המפות המרחביות!
    """
    print("\n--- Fixing PCA Signs (Time Series) ---")
    pc_cols = [c for c in df.columns if c.startswith('PC_') or c == 'Original_Signal']
    means = df.groupby('Condition').mean()
    flip_record = {}

    for col in pc_cols:
        flip_record[col] = 1  # ברירת מחדל: לא הופכים

        if reference_condition in means.index and baseline_condition in means.index:
            val_ref = means.loc[reference_condition, col]
            val_base = means.loc[baseline_condition, col]

            if val_ref < val_base:
                print(f"  Flipping {col} (Time Series)")
                df[col] = df[col] * -1
                flip_record[col] = -1  # רושמים שהפכנו

    return df, flip_record


def analyze_spatial_map(nii_path, flip_factor):
    """טוען NIfTI, מחיל Flip במידת הצורך, ומחזיר קואורדינטות שיא"""
    if not os.path.exists(nii_path):
        return "File not found"

    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        affine = img.affine

        # === החלק החכם: יישום ה-Flip גם על המרחב ===
        # אם הפכנו את הזמן, חייבים להפוך את המרחב כדי שהמשמעות תישמר
        data = data * flip_factor

        # מציאת המקסימום (האזור שהכי מזוהה עם הרכיב)
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        max_val = data[max_idx]
        max_mni = nib.affines.apply_affine(affine, max_idx)

        # מציאת המינימום (האזור שעובד הפוך לרכיב - Anti-correlated)
        min_idx = np.unravel_index(np.argmin(data), data.shape)
        min_val = data[min_idx]
        min_mni = nib.affines.apply_affine(affine, min_idx)

        return {
            'positive_peak': {
                'mni': max_mni,
                'val': max_val,
                'voxel': max_idx
            },
            'negative_peak': {
                'mni': min_mni,
                'val': min_val,
                'voxel': min_idx
            }
        }

    except Exception as e:
        return f"Error analyzing NIfTI: {str(e)}"


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pool_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    return (np.mean(x) - np.mean(y)) / pool_std


# ==========================================
# 3. הריצה הראשית (Main Execution)
# ==========================================

print(f"--- Starting Full Stack Analysis: {timestamp} ---")

try:
    # 1. טעינה ועיבוד זמן
    df = load_time_series(TS_FILES)
    df = apply_labels(df, BLOCKS, HRF_LAG_SEC)

    # יישור כיוונים ושמירת המידע על ההיפוך
    df, flip_dict = align_pca_sign_and_record(df, 'v', 'x')

    df_clean = df[df['Condition'] != 'unknown'].copy()

    # 2. יצירת דוח משולב (זמן + מרחב)
    print("\n--- Generating Integrated Report ---")
    output_text = []
    output_text.append(f"Full Analysis Report - Run ID: {timestamp}\n")
    output_text.append("=" * 60 + "\n")
    output_text.append("Directionality Note: All components aligned so Visual(v) > Rest(x).\n")
    output_text.append("Spatial maps are flipped accordingly to match time-series.\n\n")

    comparisons = [('v', 'va'), ('v', 'a'), ('v', 'x'), ('va', 'a')]
    signals = [c for c in df.columns if c not in ['Time', 'Condition']]

    for sig in signals:
        output_text.append(f"COMPONENT: {sig}\n" + "-" * 30 + "\n")

        # --- 1. הכנת נתונים ברמת הבלוק (Block-wise) ---
        # מחלצים ממוצע לכל הופעה של בלוק כדי להבטיח אי-תלות
        block_data = {cond: [] for cond in ['v', 'a', 'va', 'x']}
        for s, e, cond in BLOCKS:
            if cond in block_data:
                # שימוש באותה לוגיקה של ה-Lag כמו ב-apply_labels
                mask = (df['Time'] >= s + HRF_LAG_SEC) & (df['Time'] < e + HRF_LAG_SEC)
                block_avg = df.loc[mask, sig].mean()
                if not np.isnan(block_avg):
                    block_data[cond].append(block_avg)

        # --- 2. סטטיסטיקה של הזמן (כל הנקודות - למען ההשוואה) ---
        output_text.append(">> Time Series Stats (Point-by-Point):\n")
        for c1, c2 in comparisons:
            d1 = df_clean.loc[df_clean['Condition'] == c1, sig]
            d2 = df_clean.loc[df_clean['Condition'] == c2, sig]
            if len(d1) > 1 and len(d2) > 1:
                t, p = stats.ttest_ind(d1, d2, equal_var=False)
                d = cohens_d(d1, d2)
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                output_text.append(f"   {c1.upper()} vs {c2.upper()}: p={p:.6f} {stars}, d={d:.3f} (N={len(d1)},{len(d2)})\n")

        # --- 3. סטטיסטיקה ברמת הבלוק (שמרנית ומדויקת יותר) ---
        output_text.append("\n>> Block-Level Stats (Independent Samples):\n")
        for c1, c2 in comparisons:
            b1 = block_data[c1]
            b2 = block_data[c2]
            if len(b1) > 1 and len(b2) > 1:
                t_b, p_b = stats.ttest_ind(b1, b2, equal_var=False)
                d_b = cohens_d(b1, b2)
                stars_b = "***" if p_b < 0.001 else "**" if p_b < 0.01 else "*" if p_b < 0.05 else ""
                output_text.append(
                    f"   {c1.upper()} vs {c2.upper()}: p={p_b:.4f} {stars_b}, d={d_b:.3f} (N={len(b1)},{len(b2)})\n")

        # --- 4. בדיקת אדיטיביות ---
        means = df_clean.groupby('Condition')[sig].mean()
        if all(k in means.index for k in ['v', 'a', 'x', 'va']):
            diff = (means['va'] - means['x']) - ((means['v'] - means['x']) + (means['a'] - means['x']))
            output_text.append(f"\n   Additivity (Diff from Sum): {diff:.3f} (Negative = Suppression)\n")


        # --- חלק ב: ניתוח מרחבי (Spatial Analysis) ---
        # בודקים אם יש קובץ NIfTI תואם לרכיב הזה
        # התאמת שמות: PC_0 -> PC_0 ב-NII_FILES (או שם דומה)
        output_text.append("\n>> Spatial Analysis (Peak Regions):\n")

        nii_key = sig  # מניחים שהשם ב-DF זהה למפתח ב-NII_FILES
        if nii_key in NII_FILES and os.path.exists(NII_FILES[nii_key]):
            # בדיקת ה-Flip הספציפי לרכיב הזה
            flip = flip_dict.get(sig, 1)
            flip_str = "(Flipped)" if flip == -1 else "(Original)"

            spatial_res = analyze_spatial_map(NII_FILES[nii_key], flip)

            if isinstance(spatial_res, dict):
                pos = spatial_res['positive_peak']
                neg = spatial_res['negative_peak']

                output_text.append(f"   Alignment: {flip_str}\n")
                output_text.append(f"   [+] MAX PEAK (Correlated with Visual): \n")
                output_text.append(f"       MNI: {pos['mni']} | Value: {pos['val']:.2f}\n")
                output_text.append(f"   [-] MIN PEAK (Anti-correlated): \n")
                output_text.append(f"       MNI: {neg['mni']} | Value: {neg['val']:.2f}\n")
            else:
                output_text.append(f"   Error reading map: {spatial_res}\n")
        else:
            output_text.append("   No spatial map file attached for this component.\n")

        output_text.append("\n" + "=" * 60 + "\n\n")

    # שמירת הדוח
    report_fname = f"full_report_{timestamp}.txt"
    with open(report_fname, "w") as f:
        f.writelines(output_text)
    print(f"Saved Full Report to: {report_fname}")

    # 3. יצירת גרפים (נשמר מהגרסה הקודמת)
    print("--- Generating Plots ---")

    # Boxplots
    cols = 3
    rows = math.ceil(len(signals) / cols)
    plt.figure(figsize=(15, 4 * rows))
    for i, sig in enumerate(signals):
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(x='Condition', y=sig, data=df_clean, order=['x', 'a', 'v', 'va'], palette=COLORS)
        plt.title(sig)
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"boxplots_ALL_{timestamp}.png", dpi=300)
    plt.close()

    # Waveforms
    plt.figure(figsize=(15, 4 * rows))
    for i, sig in enumerate(signals):
        ax = plt.subplot(rows, cols, i + 1)
        window_pts = int(20 / TR)
        t_epoch = np.arange(window_pts) * TR
        for cond in ['v', 'va']:
            starts = [b[0] for b in BLOCKS if b[2] == cond]
            epochs = []
            for s in starts:
                s_idx = int((s + 2) / TR)
                if s_idx + window_pts < len(df):
                    seg = df[sig].iloc[s_idx: s_idx + window_pts].values
                    epochs.append(seg - seg[0])
            if epochs:
                avg = np.mean(epochs, axis=0)
                sem = np.std(epochs, axis=0) / np.sqrt(len(epochs))
                ax.plot(t_epoch, avg, label=cond, color=COLORS[cond], lw=2)
                ax.fill_between(t_epoch, avg - sem, avg + sem, color=COLORS[cond], alpha=0.2)
        ax.set_title(sig)
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()
    plt.tight_layout()
    plt.savefig(f"waveforms_ALL_{timestamp}.png", dpi=300)
    plt.close()

    print("\nAnalysis Complete! Files generated:")
    print(f"1. {report_fname} (Text Report + MNI Coords)")
    print(f"2. boxplots_ALL_{timestamp}.png")
    print(f"3. waveforms_ALL_{timestamp}.png")

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    import traceback

    traceback.print_exc()
