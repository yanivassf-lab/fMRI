from typing import Tuple, Any

import nibabel as nib
import numpy as np
from numpy import ndarray, dtype, float64
from scipy.ndimage import uniform_filter
from scipy.signal import butter, filtfilt


class LoadData:
    def __init__(self, nii_file: str, mask_file: str):
        self.nii_file = nii_file
        self.mask_file = mask_file


    def load_data(self, TR: float = None, processed: bool = True) -> tuple[
        ndarray[tuple[Any, int], dtype[float64]], Any, Any, float | Any]:
        """
        Load and preprocess fMRI data from a NIfTI file.
        This function loads 4D fMRI data and a binary brain mask from specified NIfTI files.
        It applies spatial smoothing and temporal filtering to the fMRI data if not already processed.
        The function returns the processed time series data, the mask, the affine transformation matrix,
        and the repetition time (TR).

        Parameters:
            filename: Path to the NIfTI file containing fMRI data.
            TR: Repetition time in seconds. If None, it will be extracted from the NIfTI header.
            processed: If True, returns preprocessed data; otherwise, applies preprocessing steps.

        Returns:
            tuple: A tuple containing:
                - data_flat: 2D array of time series data (shape: [time, voxels]).
                - mask: 3D binary mask array indicating brain voxels.
                - affine: Affine transformation matrix from the NIfTI header.
                - TR: Repetition time in seconds.
        """

        nii_rdata, data = self.load_nifti_data(self.nii_file)  # Load the 4D fMRI data
        sx, sy, sz, st = data.shape  # Extract shape for later use

        # Load binary brain mask (1 = brain voxel, 0 = background)
        nii_mask, mask = self.load_nifti_data(self.mask_file)
        if not TR:
            zoom = nii_rdata.header.get_zooms()
            TR = zoom[3] if len(zoom) >= 4 else 2.0

        # Identify the coordinates of voxels within the mask
        xlocs, ylocs, zlocs = np.where(mask == 1)

        if not processed:
            # Apply 3D smoothing to each time point independently
            srdata = np.zeros_like(data)
            for i in range(st):
                vol1 = data[..., i]
                srdata[..., i] = self.smooth3_box(vol1)  # Spatial smoothing
            # Apply temporal filtering
            data = self.filter_fMRI(srdata, TR)

        # Extract raw time series data from voxels inside the mask
        data_flat = np.zeros((st, len(xlocs)))  # [time, voxels]
        for i in range(len(xlocs)):
            data_flat[:, i] = data[xlocs[i], ylocs[i], zlocs[i], :]
        return data_flat.T, mask, nii_mask.affine, TR

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
