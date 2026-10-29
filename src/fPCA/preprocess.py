import logging
from typing import Any

import nibabel as nib
import numpy as np
from numpy import ndarray, dtype, float64
from scipy.ndimage import uniform_filter
from scipy.signal import butter, filtfilt

from nilearn.interfaces.fmriprep import load_confounds
from nilearn import maskers

logger = logging.getLogger("fmri_logger")


class LoadData:
    def __init__(self, nii_file: str, mask_file: str, TR: float = None, smooth_size: int = 5, highpass: float = 0.01,
                 lowpass: float = 0.08, use_nilearn: bool = False, n_compcor: int = 5,
                 nilearn_smoothing_fwhm: float = 6.0):
        """
        Args:
            nii_file (str): Path to the 4D fMRI NIfTI file.
            mask_file (str): Path to the 3D mask NIfTI file.
            TR (float): Repetition time in seconds. If None, it will be extracted from the NIfTI header.
            smooth_size (int): Box size for smoothing kernel.
            highpass (float): High-pass filter cutoff frequency in Hz. Filters out slow drifts below this frequency.
            lowpass (float): Low-pass filter cutoff frequency in Hz. Filters out high-frequency noise above this frequency.
            use_nilearn (bool): Whether to use nilearn's masker for data preprocessing.
            n_compcor (int): Number of components to use for compcor.
            nilearn_smoothing_fwhm (float): FWHM for smoothing in Nilearn.
        """
        self.nii_file = nii_file
        self.mask_file = mask_file
        self.TR = TR
        self.smooth_size = smooth_size
        self.highpass = highpass
        self.lowpass = lowpass
        self.use_nilearn = use_nilearn
        self.n_compcor = n_compcor
        # Explicit parameter for Nilearn smoothing in millimeters (FWHM)
        self.nilearn_smoothing_fwhm = nilearn_smoothing_fwhm
        self.filtered_file = None
        self.filtered_nii_rdata = None

    def load_data(self, processed: bool = False, save_filtered_file: bool = False) -> tuple[
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
        xlocs, ylocs, zlocs = np.where(mask > 0)

        if not processed:
            logger.info("Applying preprocessing steps: spatial smoothing and temporal filtering...")
            # Determine which filtering pipeline to use
            if self.use_nilearn:
                logger.info("Using Nilearn for confound regression and filtering.")
                data_flat = self.filter_fMRI_nilearn()
            else:
                logger.info("Using scipy spatial smoothing and temporal filtering.")
                data_flat = self.filter_fMRI(data, st, xlocs, ylocs, zlocs)
            # Apply smoothing and temporal filtering
            if save_filtered_file:
                logger.info("Reconstructing the 4D file for saving...")
                # Initialize a zeros array with the same shape as the original 4D data
                reconstructed_4d = np.zeros_like(data)

                # Place the filtered time series back into their spatial coordinates
                reconstructed_4d[xlocs, ylocs, zlocs, :] = data_flat

                # Save the reconstructed 4D array and the header info
                self.filtered_file = reconstructed_4d
                self.filtered_nii_rdata = nii_rdata if nii_rdata else nii_mask
        else:
            logger.info("Loading data files without additional filtering.")
            data_flat = data[xlocs, ylocs, zlocs, :]

        return data_flat, mask, nii_mask.affine # [voxels, times]

    def filter_fMRI_nilearn(self):
        """
        Apply the original Nilearn preprocessing and report flagged volumes.

        Preserve the original motion, cosine, and CompCor regressors.
        Use sample_mask for diagnostics only: retain every time point,
        without adding spike regressors or forcing values to zero.

        Write one CRITICAL-tagged warning per affected input. Indices refer
        to the original input volume sequence, start at zero, and use
        inclusive ranges. Reporting failures do not stop preprocessing.

        Returns:
            ndarray: Filtered masked data with shape (voxels, time).
        """
        confounds_df, sample_mask = load_confounds(
            self.nii_file,
            strategy=("motion", "high_pass", "compcor"),
            motion="basic",
            compcor="anat_combined",
            n_compcor=self.n_compcor,
        )

        # Diagnostics only: keep the original confounds and all time points.
        if sample_mask is not None:
            try:
                n_scans = len(confounds_df)
                keep = np.asarray(sample_mask)

                if keep.dtype.kind == "b":
                    keep = np.flatnonzero(keep)

                flagged = np.setdiff1d(np.arange(n_scans), keep)

                if flagged.size:
                    # Compress consecutive indices without losing locations.
                    breaks = np.flatnonzero(np.diff(flagged) > 1)
                    starts = np.r_[flagged[0], flagged[breaks + 1]]
                    ends = np.r_[flagged[breaks], flagged[-1]]

                    ranges = ",".join(
                        str(int(a)) if a == b else f"{int(a)}-{int(b)}"
                        for a, b in zip(starts, ends)
                    )

                    leading = int(ends[0] + 1) if starts[0] == 0 else 0
                    n_flagged = int(flagged.size)

                    # This strategy does not request motion scrubbing.
                    # The reported flags therefore concern non-steady states.
                    # %r escapes any newlines in paths and parameters.
                    logger.warning(
                        "CRITICAL SAMPLE_MASK_IGNORED | file=%r "
                        "| flag_type=non_steady_state "
                        "| total_volumes=%d | flagged_volumes=%d "
                        "| flagged_pct=%.3f | leading_flagged=%d "
                        "| nonleading_flagged=%d | flagged_blocks=%d "
                        "| longest_block=%d | indices_0based=%s "
                        "| tr_s=%r | n_compcor=%r | smoothing_fwhm_mm=%r "
                        "| action=KEEP_ALL_ORIGINAL_DENOISING",
                        str(self.nii_file),
                        n_scans,
                        n_flagged,
                        100.0 * n_flagged / n_scans,
                        leading,
                        n_flagged - leading,
                        len(starts),
                        int(np.max(ends - starts + 1)),
                        ranges,
                        self.TR,
                        self.n_compcor,
                        self.nilearn_smoothing_fwhm,
                    )

            except (TypeError, ValueError, IndexError) as exc:
                # A diagnostic failure must not prevent normal preprocessing.
                logger.warning(
                    "CRITICAL SAMPLE_MASK_REPORT_FAILED | file=%r | error=%r "
                    "| action=KEEP_ALL_ORIGINAL_DENOISING",
                    str(self.nii_file),
                    str(exc),
                )

        # Original preprocessing settings.
        masker = maskers.NiftiMasker(
            mask_img=self.mask_file,
            smoothing_fwhm=self.nilearn_smoothing_fwhm,
            standardize="zscore_sample",
            detrend=False,
            low_pass=None,
            high_pass=None,
            t_r=self.TR,
        )

        # Preserve the original behavior: no sample_mask is applied.
        time_series = masker.fit_transform(
            self.nii_file,
            confounds=confounds_df,
        )

        return time_series.T

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

    def filter_fMRI(self, data, st, xlocs, ylocs, zlocs):
        """
        Apply smoothing and temporal bandpass filtering to fMRI data.

        Parameters:
            data (ndarray): 4D raw fMRI data (shape: x, y, z, t).
            xlocs, ylocs, zlocs (ndarray): Coordinates of voxels within the brain mask.

        Returns:
            ndarray: Temporally filtered 2D fMRI data for masked voxels (shape: [voxels, time]).
        """
        # Apply 3D spatial smoothing to all time points in one call
        # Use a kernel size only in spatial dims and 1 in time dim
        srdata = uniform_filter(
            data,
            size=(self.smooth_size, self.smooth_size, self.smooth_size, 1),
            mode='nearest',
        )
        # 2. Extract raw time series data ONLY from voxels inside the mask
        # Shape will be [voxels, time]
        masked_time_series = srdata[xlocs, ylocs, zlocs, :]

        # Reshape to [n_voxels, time] for vectorized temporal filtering
        # time_series = srdata.reshape(-1, srdata.shape[3])
        nyquist = 1 / (2 * self.TR)
        Wn = [self.highpass / nyquist, self.lowpass / nyquist]  # normalized cutoff frequencies

        # Ensure Wn is within (0, 1)
        if not (0 < Wn[0] < 1 and 0 < Wn[1] < 1):
            raise ValueError(
                f"Invalid filter frequencies: Wn={Wn}. Generally it caused because of an incorrect value of "
                f"TR. Check TR value.")

        order = 3  # Butterworth filter order
        b, a = butter(order, Wn, btype='band')

        # Apply the bandpass filter to all voxel time series in a vectorized way
        # Shape is [n_voxels, time], so filter along axis=1
        filtered = filtfilt(b, a, masked_time_series, axis=1)

        return filtered

    def save_filtered_data(self, path):
        """
        Save filtered data with at least the same numerical precision
        as the original simulator outputs.

        We store as float32 (like the synthetic generator), letting gzip
        handle compression, with no extra quantization.
        """
        data_to_save = np.asarray(self.filtered_file, dtype=np.float32)
        nifti_img = nib.Nifti1Image(data_to_save, self.filtered_nii_rdata.affine)
        nib.save(nifti_img, path)

# if __name__ == '__main__':
#     Run the pipeline on subject 11001 and print basic output information
# print(fMRI4fPCA("11001"))
