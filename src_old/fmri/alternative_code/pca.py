from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib
import datetime


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import detrend

# --- הגדרות בסיס ---
nifti_file = 'sub-01_task-movie_bold.nii.gz'  # עדכן אם צריך
n_components = 5  # מספר רכיבים ל-fPCA

# --- שלב 1: טעינת הקובץ ---
print("Loading NIfTI file...")
img = nib.load(nifti_file)
data = img.get_fdata()  # צורת הנתונים: (X, Y, Z, Time)
X, Y, Z, T = data.shape
print(f"Data shape: {data.shape}")

# --- שלב 2: בניית מסכת מח פשוטה ---
print("Building brain mask...")
brain_mask = np.mean(data, axis=3) > 100  # ערך סף פשוט
n_voxels = np.sum(brain_mask)
print(f"Number of brain voxels: {n_voxels}")

# --- שלב 3: פריסת המידע למטריצה ---
print("Reshaping data...")
data_2d = data[brain_mask].T  # צורת הנתונים: (Time, Voxels)

# --- שלב 4: ניקוי בסיסי (למשל Detrend) ---
print("Preprocessing signals...")
data_2d = detrend(data_2d, axis=0)

# --- שלב 5: ביצוע PCA ---
print(f"Running PCA with {n_components} components...")
pca = PCA(n_components=n_components)
components = pca.fit_transform(data_2d)  # צורת הרכיבים: (Time, n_components)

# --- שלב 6: הצגת הרכיבים ---
print("Plotting components...")
plt.figure(figsize=(12, 8))
for i in range(n_components):
    plt.subplot(n_components, 1, i+1)
    plt.plot(components[:, i])
    plt.title(f'Component {i+1}')
    plt.xlabel('Timepoints')
    plt.ylabel('Intensity')
plt.tight_layout()
plt.show()

# --- שלב 7: (אופציונלי) הצגת Variance מוסבר ---
explained = pca.explained_variance_ratio_
print(f"Explained variance per component: {explained}")

import os
from nibabel import Nifti1Image

# --- שלב 8: שחזור Spatial Maps ---
print("Reconstructing spatial maps for components...")

# קובץ לשמירת תוצאות
output_dir = "pca_components"
os.makedirs(output_dir, exist_ok=True)

# כל רכיב -> תוצאה במוח
for i in range(n_components):
    # PCA מרחבי: העומסים של הרכיב על כל ווקסל
    spatial_map_flat = pca.components_[i, :]  # צורת וקטור: (voxels,)

    # נבנה מפה בצורת מח
    spatial_map = np.zeros(brain_mask.shape)
    spatial_map[brain_mask] = spatial_map_flat

    # ניצור קובץ NIfTI חדש
    comp_img = Nifti1Image(spatial_map, affine=img.affine, header=img.header)

    # שמור לקובץ
    comp_filename = os.path.join(output_dir, f"component_{i+1:02d}.nii.gz")
    nib.save(comp_img, comp_filename)
    print(f"Saved: {comp_filename}")

print("All spatial maps saved!")


from nilearn import datasets, image, masking
import pandas as pd

# --- טען את אטלס Harvard-Oxford cortical ---
print("Loading brain atlas...")
atlas_dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_img = nib.load(atlas_dataset.maps)
atlas_labels = atlas_dataset.labels  # רשימת שמות האזורים

# --- הכנה ---
# רגרסיה של Spatial Maps לאטלס
print("Matching components to brain regions...")

for i in range(n_components):
    comp_file = os.path.join(output_dir, f"component_{i+1:02d}.nii.gz")
    comp_img = nib.load(comp_file)

    # התאמת רזולוציות אם צריך (רישמית: נעשה resample)
    comp_img_resampled = image.resample_to_img(comp_img, atlas_img)

    # מסיכה
    comp_data = masking.apply_mask(comp_img_resampled, atlas_img)
    atlas_data = masking.apply_mask(atlas_img, atlas_img)

    # לכל אזור באטלס: ממוצע העוצמות
    results = []
    for region_index in np.unique(atlas_data):
        if region_index == 0:
            continue  # רקע
        region_mask = (atlas_data == region_index)
        mean_intensity = comp_data[region_mask].mean()
        results.append((region_index, mean_intensity))

    # מיון לפי עוצמה
    results_sorted = sorted(results, key=lambda x: -x[1])

    # הדפסת האזור הכי פעיל
    best_region_index, best_intensity = results_sorted[0]
    best_region_name = atlas_labels[best_region_index]

    print(f"Component {i+1}: strongest in {best_region_name} (mean intensity: {best_intensity:.4f})")

from nilearn import plotting

# --- שלב 9: הצגת רכיב על גבי תמונת מוח ---
print("Plotting components on brain background...")

# בחר איזה רכיב להציג (למשל 1)
component_to_plot = 1

# טען את המפה
comp_file = os.path.join(output_dir, f"component_{component_to_plot:02d}.nii.gz")
comp_img = nib.load(comp_file)

# הגדרות תצוגה
threshold = 0.2  # סף תצוגה: רק ערכים מעל 0.2 יוצגו
display_mode = 'ortho'  # אפשר גם 'z', 'x', 'y', 'ortho'

# הצגה
plotting.plot_stat_map(
    comp_img,
    threshold=threshold,
    display_mode=display_mode,
    colorbar=True,
    title=f"Component {component_to_plot}",
    cmap="cold_hot",  # colormap
    cut_coords=(0, 0, 0)  # אפשר להוריד ולתת לו לבחור לבד
)
plotting.show()


# --- 1. שמירת כל רכיב כתמונה ---
print("Saving brain images of components...")

brain_plots_dir = "pca_component_plots"
os.makedirs(brain_plots_dir, exist_ok=True)

for i in range(n_components):
    comp_file = os.path.join(output_dir, f"component_{i+1:02d}.nii.gz")
    comp_img = nib.load(comp_file)

    display = plotting.plot_stat_map(
        comp_img,
        threshold=0.2,
        display_mode='ortho',
        colorbar=True,
        title=f"Component {i+1}",
        cmap="cold_hot"
    )

    plot_filename = os.path.join(brain_plots_dir, f"component_{i+1:02d}.png")
    display.savefig(plot_filename)
    display.close()
    print(f"Saved brain plot: {plot_filename}")

print("All brain images saved!")

# --- 2. תיאור אוטומטי ---
print("\nComponent Area Descriptions:")

# מילון פשוט: שם -> קטגוריה
area_categories = {
    'occipital': 'Vision-related',
    'temporal': 'Auditory-related',
    'precentral': 'Motor-related',
    'postcentral': 'Somatosensory-related',
    'cingulate': 'Attention-related',
    'frontal': 'Executive function',
    'parietal': 'Spatial processing'
}

for i in range(n_components):
    comp_file = os.path.join(output_dir, f"component_{i+1:02d}.nii.gz")
    comp_img = nib.load(comp_file)

    comp_img_resampled = image.resample_to_img(comp_img, atlas_img)
    comp_data = masking.apply_mask(comp_img_resampled, atlas_img)
    atlas_data = masking.apply_mask(atlas_img, atlas_img)

    results = []
    for region_index in np.unique(atlas_data):
        if region_index == 0:
            continue
        region_mask = (atlas_data == region_index)
        mean_intensity = comp_data[region_mask].mean()
        results.append((region_index, mean_intensity))

    results_sorted = sorted(results, key=lambda x: -x[1])
    best_region_index, _ = results_sorted[0]
    best_region_name = atlas_labels[best_region_index]

    # תיאור לפי מילה בתוך שם האזור
    description = 'Unknown function'
    for keyword, category in area_categories.items():
        if keyword.lower() in best_region_name.lower():
            description = category
            break

    print(f"Component {i+1}: {best_region_name} -> {description}")

# --- 3. הצגת Glass Brain של כמה רכיבים ביחד ---
print("\nPlotting multiple components on glass brain...")

# בחר רכיבים להציג
components_to_plot = [1, 2, 3]  # לדוגמה: רכיבים 1,2,3

# טען את התמונות
imgs = []
for comp_idx in components_to_plot:
    comp_file = os.path.join(output_dir, f"component_{comp_idx:02d}.nii.gz")
    comp_img = nib.load(comp_file)
    imgs.append(comp_img)

# הצג הכל
plotting.plot_glass_brain(
    imgs[0],
    threshold=0.2,
    colorbar=True,
    title=f"Component {components_to_plot[0]}",
    plot_abs=False
)

for img in imgs[1:]:
    plotting.plot_glass_brain(
        img,
        threshold=0.2,
        colorbar=False,
        plot_abs=False,
        black_bg=True
    )

plotting.show()


# --- הגדרות ---
pdf_filename = "fmri_full_report.pdf"
page_width, page_height = A4

brain_plots_dir = "pca_component_plots"
time_series_dir = "pca_component_timeseries"
os.makedirs(time_series_dir, exist_ok=True)

colormap = plt.get_cmap('tab10')

# --- נבנה מידע לכל רכיב ---
component_info = {}

for i in range(n_components):
    comp_file = os.path.join(output_dir, f"component_{i+1:02d}.nii.gz")
    comp_img = nib.load(comp_file)

    comp_img_resampled = image.resample_to_img(comp_img, atlas_img)
    comp_data = masking.apply_mask(comp_img_resampled, atlas_img)
    atlas_data = masking.apply_mask(atlas_img, atlas_img)

    results = []
    for region_index in np.unique(atlas_data):
        if region_index == 0:
            continue
        region_mask = (atlas_data == region_index)
        mean_intensity = comp_data[region_mask].mean()
        results.append((region_index, mean_intensity))

    results_sorted = sorted(results, key=lambda x: -x[1])
    best_region_index, _ = results_sorted[0]
    best_region_name = atlas_labels[best_region_index]

    description = 'Unknown function'
    for keyword, category in area_categories.items():
        if keyword.lower() in best_region_name.lower():
            description = category
            break

    # חישוב עוצמה מקסימלית בסיגנל זמן
    time_series = components_time_series[:, i]
    max_intensity = np.max(np.abs(time_series))

    component_info[i+1] = (best_region_name, description, max_intensity)

# --- יצירת PDF ---
c = canvas.Canvas(pdf_filename, pagesize=A4)

# --- עמוד ראשון: סיכום ואינדקס ---
c.setFont("Helvetica-Bold", 24)
c.drawString(50, page_height - 70, "fMRI Components Report")

c.setFont("Helvetica", 14)
c.drawString(50, page_height - 120, f"Number of components: {n_components}")
c.drawString(50, page_height - 150, f"Date generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

c.setFont("Helvetica", 12)
c.drawString(50, page_height - 190, "This report presents spatial maps and time series of each principal component extracted from fMRI data.")
c.drawString(50, page_height - 210, "Each component is shown with its brain activation map, time-intensity curve, and dominant brain region.")

# אינדקס
c.setFont("Helvetica-Bold", 16)
c.drawString(50, page_height - 260, "Index of Components:")

c.setFont("Helvetica", 12)
y_index = page_height - 290
for i in range(1, n_components+1):
    best_region_name, _, _ = component_info[i]
    c.drawString(60, y_index, f"Component {i}: {best_region_name}")
    y_index -= 20
    if y_index < 100:  # דף חדש אם אין מקום
        c.showPage()
        y_index = page_height - 50

c.showPage()

# --- עמודים לכל רכיב ---
for i in range(1, n_components+1):
    brain_img_path = os.path.join(brain_plots_dir, f"component_{i:02d}.png")
    best_region_name, description, max_intensity = component_info[i]

    # צבע ייחודי לכל רכיב
    color = colormap((i-1) % 10)

    # יצירת גרף זמן
    time_series = components_time_series[:, i-1]
    plt.figure(figsize=(6, 2))
    plt.plot(time_series, color=color)
    plt.title(f"Component {i} Time Series")
    plt.xlabel('Time (scans)')
    plt.ylabel('Intensity')
    plt.tight_layout()
    timeseries_img_path = os.path.join(time_series_dir, f"component_{i:02d}_timeseries.png")
    plt.savefig(timeseries_img_path)
    plt.close()

    # עמוד של רכיב
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, page_height - 50, f"Component {i}")

    c.setFont("Helvetica", 14)
    c.setFillColor(colors.darkblue)
    c.drawString(50, page_height - 90, f"Dominant Region: {best_region_name}")
    c.setFillColor(colors.black)

    c.setFont("Helvetica", 12)
    c.drawString(50, page_height - 120, f"Function: {description}")
    c.drawString(50, page_height - 140, f"Max Intensity: {max_intensity:.2f}")

    # תמונת מוח
    brain_img = ImageReader(brain_img_path)
    c.drawImage(brain_img, 50, 300, width=250, height=250, preserveAspectRatio=True)

    # גרף סיגנל
    ts_img = ImageReader(timeseries_img_path)
    c.drawImage(ts_img, 320, 300, width=250, height=250, preserveAspectRatio=True)

    c.showPage()

# --- עמוד אחרון: טבלה מסכמת ---
c.setFont("Helvetica-Bold", 20)
c.drawString(50, page_height - 50, "Summary Table")

# בניית טבלה
data = [["Component", "Dominant Region", "Function", "Max Intensity"]]
for i in range(1, n_components+1):
    best_region_name, description, max_intensity = component_info[i]
    data.append([str(i), best_region_name, description, f"{max_intensity:.2f}"])

# ציור הטבלה
table = Table(data, colWidths=[70, 150, 150, 80])

table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
]))

# מיקום הטבלה
table.wrapOn(c, width=500, height=400)
table.drawOn(c, 30, page_height - 100 - 20 * len(data))

c.save()

print(f"Full PDF report saved: {pdf_filename}")
