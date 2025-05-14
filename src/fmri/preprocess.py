import nibabel as nib
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import butter, filtfilt


class LoadData:
    def __init__(self, nii_file: str, mask_file: str):
        self.nii_file = nii_file
        self.mask_file = mask_file

    def run_preprocessing(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Preprocess fMRI data for functional Principal Component Analysis (fPCA).

        Returns:
        tuple:
            - fdata_flat (ndarray): Filtered fMRI data, Time series extracted from masked voxels (shape: n_timepoints, n_voxels)
            - fdata (ndarray): Filtered 4D fMRI data.
            - srdata (ndarray): Spatially smoothed fMRI data.
            - mask (ndarray): Binary mask array indicating brain voxels (same shape as volume).
            - nii (Nifti1Image): The loaded fMRI NIfTI image object.
            - TR (float): Repetition time (in seconds), used to determine Nyquist frequency.
            # - rawdat (ndarray): Time series extracted from masked voxels (shape: n_timepoints, n_voxels).



        Pipeline Steps:
        1. Load subject's 4D fMRI data (NIfTI format).
        2. Load brain mask indicating voxels of interest.
        3. Apply 3D box smoothing to each time frame.
        4. Extract voxel-wise time series using the mask.
        5. Apply temporal bandpass filtering (0.01–0.4 Hz).

        Notes:
        - The function assumes file naming follows pattern: `{subjno}_drc144images.nii` and `{subjno}_mask.nii`
        - Prepares data for functional PCA, which requires denoised, preprocessed time series data.
        """
        # nii_file = f"{subjno}_drc144images.nii"
        nii, rdata = self.load_nifti_data(self.nii_file)  # Load the 4D fMRI data
        sx, sy, sz, st = rdata.shape  # Extract shape for later use

        bsl = rdata[..., 0]  # First time frame often used as baseline or reference image
        TR = 1  # Assumed repetition time in seconds (can be changed if known)

        # Load binary brain mask (1 = brain voxel, 0 = background)
        # mask_file = f"{subjno}_mask.nii"
        nii, mask = self.load_nifti_data(self.mask_file)

        # Identify the coordinates of voxels within the mask
        xlocs, ylocs, zlocs = np.where(mask == 1)

        # Initialize smoothed data array
        srdata = np.zeros_like(rdata)

        # Apply 3D smoothing to each time point independently
        for i in range(st):
            vol1 = rdata[..., i]
            srdata[..., i] = self.smooth3_box(vol1)  # Spatial smoothing

        # Extract raw time series data from voxels inside the mask
        rawdat = np.zeros((st, len(xlocs)))  # [time, voxels]
        for i in range(len(xlocs)):
            rawdat[:, i] = srdata[xlocs[i], ylocs[i], zlocs[i], :]

        # Apply temporal filtering
        fdata = self.filter_fMRI(srdata, TR)

        # Extract raw time series data from voxels inside the mask
        fdata_flat = np.zeros((st, len(xlocs)))  # [time, voxels]
        for i in range(len(xlocs)):
            fdata_flat[:, i] = fdata[xlocs[i], ylocs[i], zlocs[i], :]

        # TODO: original code. check if srdata need to be after mask
        # return nii, fdata_flat.T, fdata, rawdat.T, srdata, mask
        zoom = nii.header.get_zooms()
        TR = zoom[3] if len(zoom) >= 4 else 2.0

        return fdata_flat.T, fdata, srdata, rawdat.T, mask, nii.affine, TR

    def load_nifti_data(self, filename: str):
        """
        Load a NIfTI (.nii or .nii.gz) file using nibabel.

        Parameters:
        filename (str): Path to the NIfTI file.

        Returns:
        tuple:
            - nib.Nifti1Image: The NIfTI object.
            - numpy.ndarray: The image data array extracted from the NIfTI file.
        """
        nii = nib.load(filename)
        return nii, nii.get_fdata()

    def smooth3_box(self, volume, size=(5, 5, 5)):
        """
        Apply 3D box smoothing (mean filter) to a volume.

        Parameters:
        volume (ndarray): 3D volume to smooth.
        size (tuple): Size of the smoothing kernel in 3D.

        Returns:
        ndarray: Smoothed 3D volume.
        """
        # Uniform (box) smoothing to reduce noise while preserving spatial structure.
        # This is often used before temporal filtering to reduce voxel-level noise.
        return uniform_filter(volume, size=size, mode='nearest')

    def filter_fMRI(self, srdata, TR):
        """
        Apply temporal bandpass filtering to fMRI data using a Butterworth filter.

        Parameters:
        srdata (ndarray): 4D fMRI data after spatial smoothing (shape: x, y, z, t).
        TR (float): Repetition time (in seconds), used to determine Nyquist frequency.

        Returns:
        ndarray: Temporally filtered 4D fMRI data.

        Notes:
        - Applies a 3rd-order Butterworth bandpass filter in the range 0.01–0.4 Hz.
        - This is a standard preprocessing step to remove physiological noise and low-frequency drift.
        - Equation Reference: See bandpass filtering equations, typically Eq. (3) in many fMRI preprocessing papers.
        """
        time_series = srdata.reshape(-1, srdata.shape[3])  # reshape to [n_voxels, time]
        f_high = 0.01  # lower cutoff frequency in Hz
        f_low = 0.4  # upper cutoff frequency in Hz
        nyquist = 1 / (2 * TR)
        Wn = [f_high / nyquist, f_low / nyquist]  # normalized cutoff frequencies

        order = 3  # Butterworth filter order
        b, a = butter(order, Wn, btype='band')

        # Apply the bandpass filter to each voxel's time series
        filtered = np.zeros_like(time_series)
        for i in range(time_series.shape[0]):
            filtered[i, :] = filtfilt(b, a, time_series[i, :])

        return filtered.reshape(srdata.shape)

# if __name__ == '__main__':
#     Run the pipeline on subject 11001 and print basic output information
# print(fMRI4fPCA("11001"))
