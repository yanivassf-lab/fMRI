import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, binary_closing

def create_brain_mask(fmri_data, threshold_ratio=0.05):
    """
    Creates a binary mask for fMRI data by thresholding the mean signal across time.

    Parameters:
    - fmri_data: 4D numpy array (X, Y, Z, T)
    - threshold_ratio: fraction of max mean signal to use as threshold

    Returns:
    - mask: 3D numpy array (X, Y, Z) of booleans
    """
    # Compute mean signal across time
    mean_img = np.mean(fmri_data, axis=-1)

    # Threshold: voxels with mean signal above fraction of max are kept
    threshold = threshold_ratio * np.max(mean_img)
    mask = mean_img > threshold

    # Optional: remove noise using morphological operations
    mask = binary_opening(mask, structure=np.ones((3, 3, 3)))
    mask = binary_closing(mask, structure=np.ones((3, 3, 3)))

    return mask.astype(np.uint8)

# Example usage:
nii = nib.load("11001_drc144images.nii")  # or .nii.gz
fmri_data = nii.get_fdata()
mask = create_brain_mask(fmri_data)

# Save the mask as a new NIfTI file (optional)
mask_img = nib.Nifti1Image(mask, affine=nii.affine)
nib.save(mask_img, "auto_generated_mask.nii.gz")

# Visualize a few slices (optional)
plt.imshow(mask[:, :, fmri_data.shape[2] // 2], cmap='gray')
plt.title("Center Slice of Auto Mask")
plt.axis('off')
plt.show()
