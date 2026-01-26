import logging
from typing import Any

import nibabel as nib
import numpy as np
from numpy import ndarray, dtype, float64
from scipy.ndimage import uniform_filter
from scipy.signal import butter, filtfilt

logger = logging.getLogger("fmri_logger")


class LoadData:
    def __init__(self, nii_file: str, mask_file: str, TR: float = None, smooth_size: int = 5, highpass: float = 0.01,
                 lowpass: float = 0.08, ):
        """
        Args:
            nii_file (str): Path to the 4D fMRI NIfTI file.
            mask_file (str): Path to the 3D mask NIfTI file.
            TR (float): Repetition time in seconds. If None, it will be extracted from the NIfTI header.
            smooth_size (int): Box size for smoothing kernel.
            highpass (float): High-pass filter cutoff frequency in Hz. Filters out slow drifts below this frequency.
            lowpass (float): Low-pass filter cutoff frequency in Hz. Filters out high-frequency noise above this frequency.
        """
        self.nii_file = nii_file
        self.mask_file = mask_file
        self.TR = TR
        self.smooth_size = smooth_size
        self.highpass = highpass
        self.lowpass = lowpass
        self.filtered_file = None
        self.filtered_nii_rdata = None

    def load_data(self, processed: bool = True, save_filtered_file: bool = False) -> tuple[
        ndarray[tuple[Any, int], dtype[float64]], Any, Any]:
        """
        Load and preprocess fMRI data from a NIfTI file.
        This function loads 4D fMRI data and a binary brain mask from specified NIfTI files.
        It applies spatial smoothing and temporal filtering to the fMRI data if not already processed.
        The function returns the processed time series data, the mask, the affine transformation matrix,
        and the repetition time (TR).

        Parameters:
            processed: If True, returns reorganized data without pre-processing; otherwise, applies preprocessing steps.
            save_filtered_file: If True, store the processed nii file for saving (by calling to save_filtered_data() method).
        Returns:
            tuple: A tuple containing:
                - data_flat: 2D array of time series data (shape: [voxels, times]).
                - mask: 3D binary mask array indicating brain voxels.
                - affine: Affine transformation matrix from the NIfTI header.
        """

        nii_rdata, data = self.load_nifti_data(self.nii_file)  # Load the 4D fMRI data
        sx, sy, sz, st = data.shape  # Extract shape for later use

        # Load binary brain mask (1 = brain voxel, 0 = background)
        nii_mask, mask = self.load_nifti_data(self.mask_file)
        if not self.TR:
            zoom = nii_rdata.header.get_zooms()
            self.TR = zoom[3] if len(zoom) >= 4 else 1.0

        # Identify the coordinates of voxels within the mask
        xlocs, ylocs, zlocs = np.where(mask == 1)

        if not processed:
            logger.info("Applying preprocessing steps: spatial smoothing and temporal filtering...")
            # Apply smoothing and temporal filtering
            data = self.filter_fMRI(data, st)
            if save_filtered_file:
                self.filtered_file = data
                self.filtered_nii_rdata = nii_rdata if nii_rdata else nii_mask
        else:
            logger.info("Loading data files without additional filtering.")

        # Extract raw time series data from voxels inside the mask
        data_flat = np.zeros((st, len(xlocs)))  # [time, voxels]
        for i in range(len(xlocs)):
            data_flat[:, i] = data[xlocs[i], ylocs[i], zlocs[i], :]
        return data_flat.T, mask, nii_mask.affine # [voxels, times]

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

    def filter_fMRI(self, data, st):
        """
        Apply smoothing and temporal bandpass filtering to fMRI data using a Butterworth filter.

        Parameters:
        srdata (ndarray): 4D fMRI data after spatial smoothing (shape: x, y, z, t).
        TR (float): Repetition time (in seconds), used to determine Nyquist frequency.
        smooth_size: Box size for smoothing kernel.
        st: Number of time points.

        Returns:
        ndarray: Temporally filtered 4D fMRI data.

        Notes:
        - Applies a 3rd-order Butterworth bandpass filter in the range 0.01–0.4 Hz.
        - This is a standard preprocessing step to remove physiological noise and low-frequency drift.
        - Equation Reference: See bandpass filtering equations, typically Eq. (3) in many fMRI preprocessing papers.
        """
        # Apply 3D smoothing to each time point independently
        srdata = np.zeros_like(data)
        for i in range(st):
            srdata[..., i] = uniform_filter(data[..., i], size=self.smooth_size, mode='nearest')  # Spatial smoothing

        time_series = srdata.reshape(-1, srdata.shape[3])  # reshape to [n_voxels, time]
        f_high = 0.01  # lower cutoff frequency in Hz
        f_low = 0.08  # upper cutoff frequency in Hz
        nyquist = 1 / (2 * self.TR)
        Wn = [f_high / nyquist, f_low / nyquist]  # normalized cutoff frequencies

        # Ensure Wn is within (0, 1)
        if not (0 < Wn[0] < 1 and 0 < Wn[1] < 1):
            raise ValueError(
                f"Invalid filter frequencies: Wn={Wn}. Generally it caused because of an incorrect value of "
                f"TR. Check TR value.")

        order = 3  # Butterworth filter order
        b, a = butter(order, Wn, btype='band')

        # Apply the bandpass filter to each voxel's time series
        filtered = np.zeros_like(time_series)
        for i in range(time_series.shape[0]):
            filtered[i, :] = filtfilt(b, a, time_series[i, :])

        return filtered.reshape(srdata.shape)

    def save_filtered_data(self, path):
        # Create a NIfTI image object
        nifti_img = nib.Nifti1Image(self.filtered_file, self.filtered_nii_rdata.affine)
        # Save the NIfTI image to the specified path
        nib.save(nifti_img, path)

# if __name__ == '__main__':
#     Run the pipeline on subject 11001 and print basic output information
# print(fMRI4fPCA("11001"))
