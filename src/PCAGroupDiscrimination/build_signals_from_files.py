# Standard Library
import os
import re
import logging
import glob

import warnings
import pathlib
import platform

# Data Processing
import numpy as np

# Neuroimaging
import nibabel as nib
from nilearn import datasets
from nilearn.image import resample_to_img

import joblib # Added for efficient serialization of NumPy arrays

from fPCA.preprocess import LoadData

from .utils import setup_logger

# Suppress warnings
warnings.filterwarnings('ignore')



# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================

def load_temporal_profile(filepath):
    """
    Load temporal profile from a text file and return as 1D array.

    Args:
        filepath (str): Path to file containing temporal profile data after '#PC temporal profile:' delimiter.

    Returns:
        np.ndarray: 1D array of float values representing temporal profile.
    """
    # Read the temporal profile and extract the data array
    with open(filepath, 'r') as f:
        content = f.read()
    profile_str = content.split('#PC temporal profile:\n')[1]
    return np.array([float(x) for x in profile_str.split()])


def get_region_weights(nii_file, atlas_img, atlas_labels):
    """
    Calculate mean weight per ROI region from a spatial map (PC eigenfunction).

    Args:
        nii_file (str): Path to NIfTI file containing spatial map (eigenfunction).
        atlas_img (nibabel.Nifti1Image): Loaded atlas image object.
        atlas_labels (list): List of ROI label names from atlas.

    Returns:
        np.ndarray: 1D array of mean weights for each ROI (length = number of atlas labels).
    """
    # Load spatial map and calculate mean weight per region
    pc_img = nib.load(nii_file)
    resampled = resample_to_img(atlas_img, pc_img, interpolation='nearest', force_resample=True, copy_header=True)
    pc_data = pc_img.get_fdata()
    atlas_data = resampled.get_fdata()

    weights = np.zeros(len(atlas_labels))
    for idx, _ in enumerate(atlas_labels):
        roi_mask = (atlas_data == (idx + 1))
        if np.sum(roi_mask) > 0:
            weights[idx] = np.mean(pc_data[roi_mask])
    return weights


def scan_input_folder(input_dir, file_name_pattern, logger=None):
    """
    Scans a directory for BOLD files and their corresponding brain masks,
    categorizing them into movement 1 and movement 2 per subject.

    Args:
        input_dir (str): Path to the directory containing the raw files.
        file_name_pattern (str): Regex pattern to extract subject ID (e.g., r'(sub-[A-Z0-9]+)').
        logger (logging.Logger, optional): Logger instance.

    Returns:
        dict: A nested dictionary mapping subjects to their movements, containing
              both the NIfTI file and its corresponding mask file.
              Format: {sub_id: {1: {'nii': bold_file, 'mask': mask_file}, ...}}
    """
    if logger is None:
        logger = logging.getLogger('clustering')

    logger.info(f"Scanning directory for files: {input_dir}")

    subject_files_map = {}

    # Iterate over all .nii.gz files in the input directory
    search_pattern = os.path.join(input_dir, "*.nii.gz")
    for bold_file in glob.glob(search_pattern):
        filename = os.path.basename(bold_file)

        # Skip mask files to avoid processing them as functional data
        if "brain_mask" in filename:
            continue

        # Replicate Bash logic: extract base name by stripping '-preproc_bold*'
        base = filename.split('-preproc_bold')[0]
        mask_file = os.path.join(input_dir, f"{base}-brain_mask.nii.gz")

        # Proceed only if the corresponding mask file exists
        if os.path.isfile(mask_file):
            match = re.search(file_name_pattern, filename)
            if not match:
                continue

            sub_id = match.group(1)

            if sub_id not in subject_files_map:
                subject_files_map[sub_id] = {}

            # Route to mov1 or mov2 based on filename string
            if "task-movement1" in filename:
                subject_files_map[sub_id][1] = {'nii': bold_file, 'mask': mask_file}
            elif "task-movement2" in filename:
                subject_files_map[sub_id][2] = {'nii': bold_file, 'mask': mask_file}

    # Filter out subjects that do not have BOTH movement 1 and movement 2
    valid_subjects = {
        sub: files for sub, files in subject_files_map.items()
        if 1 in files and 2 in files
    }

    logger.info(f"Found {len(valid_subjects)} valid subjects with both movements and masks.")

    return valid_subjects


def _process_subject_raw(args):
    """
    Helper function for multiprocessing: processes a single subject's raw fMRI data.
    Loads raw NIfTI files, applies LoadData filtering, averages signals across
    dynamic ROIs to save memory, and concatenates the two movements.

    Args:
        args (tuple): Tuple containing:
            - sub (str): Subject ID.
            - sub_files (dict): Dict mapping movement IDs to their respective NIfTI and mask files.
            - atlas_img (nibabel.Nifti1Image): Loaded atlas image.
            - n_rois (int): Number of Regions of Interest.
            - load_params (dict): Dictionary with parameters for LoadData and cropping.

    Returns:
        tuple: (sub, [full_matrix]) where full_matrix is a (n_rois x Time) matrix.
    """
    sub, sub_files, atlas_img, n_rois, load_params = args
    log_dir = load_params.get('log_dir', './logs')
    logger = setup_logger(log_dir=log_dir)

    concatenated = []

    # Extract temporal cropping parameters
    n_skip_vols_start = load_params.get('n_skip_vols_start', 0)
    n_skip_vols_end = load_params.get('n_skip_vols_end', 0)

    # Iterate over the expected movements (e.g., 1 and 2)
    for mov in sorted(sub_files.keys()):
        nii_file = sub_files[mov]['nii']
        mask_file = sub_files[mov]['mask']

        if not nii_file or not mask_file:
            continue

        logger.info(f"Processing subject {sub}, movement {mov}: {nii_file} with mask {mask_file}")
        # Initialize LoadData for the specific file using its dedicated mask
        data_loader = LoadData(
            nii_file=nii_file,
            mask_file=mask_file,
            TR=load_params.get('TR', 0.75),
            smooth_size=load_params.get('smooth_size', None),
            highpass=load_params.get('highpass', None),
            lowpass=load_params.get('lowpass', None),
            use_nilearn=load_params.get('use_nilearn', False),
            n_compcor_nilearn_filter=load_params.get('n_compcor_nilearn_filter', 5),
            smoothing_fwhm_nilearn_filter=load_params.get('smoothing_fwhm_nilearn_filter', 6.0)
        )

        # Load filtered data; fmri_data_all shape is (n_voxels, n_timepoints)
        fmri_data_all, mask, _ = data_loader.load_data(processed=load_params.get('processed', False))

        # Crop timepoints
        end_idx = fmri_data_all.shape[1] - n_skip_vols_end if n_skip_vols_end > 0 else fmri_data_all.shape[1]
        fmri_data = fmri_data_all[:, n_skip_vols_start:end_idx]

        # Resample the atlas to match the functional image space
        resampled_atlas = resample_to_img(atlas_img, nii_file, interpolation='nearest')
        atlas_data = resampled_atlas.get_fdata()

        # Extract atlas labels only for voxels within the functional mask
        voxel_labels = atlas_data[mask > 0]

        # Average the signal dynamically for each ROI
        roi_signals = []
        for roi_idx in range(1, n_rois + 1):
            roi_voxels = fmri_data[voxel_labels == roi_idx, :]
            if roi_voxels.shape[0] > 0:
                roi_mean = np.mean(roi_voxels, axis=0)
            else:
                roi_mean = np.zeros(fmri_data.shape[1])
            roi_signals.append(roi_mean)

        # Stack into an (n_rois x Time) matrix
        roi_matrix = np.vstack(roi_signals)
        concatenated.append(roi_matrix)

        # Free memory explicitly before processing the next movement
        del fmri_data_all, fmri_data, mask, atlas_data, voxel_labels, roi_voxels

    # Concatenate movements along the time axis (axis 1)
    if len(concatenated) == len(sub_files):
        mov1_times = concatenated[0].shape[1]
        mov2_times = concatenated[1].shape[1]
        full_matrix = np.concatenate(concatenated, axis=1)
        return sub, [full_matrix], mov1_times, mov2_times
    else:
        return sub, [], None, None


def build_subject_raw_data(subject_files_map, load_params, jobs, cache_path=None, logger=None):
    """
    Build raw spatiotemporal matrices using multiprocessing, with caching support.

    Args:
        subject_files_map (dict): Dict mapping {subject_id: {movement_id: {'nii': path, 'mask': path}}}.
        load_params (dict): Parameters dictionary for LoadData and cropping logic.
        jobs (int): Number of parallel jobs.
        cache_path (str, optional): Path to a .pkl or .joblib file to load/save the processed data.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        tuple: (data_dict, atlas_labels)
    """
    if logger is None:
        logger = logging.getLogger('clustering')
    # Check if cached data exists to avoid re-processing
    if cache_path and os.path.exists(cache_path):
        if platform.system() == 'Windows':
            pathlib.PosixPath = pathlib.WindowsPath
        logger.info(f"Loading cached raw data from: {cache_path}")
        try:
            # 1. Load the tuple
            cached_data = joblib.load(cache_path)
            # Extract the base data which is always present
            data_dict = cached_data[0]
            atlas_labels = cached_data[1]
            mov1_times = cached_data[2]
            mov2_times = cached_data[3]
            atlas_labels = [label.decode('utf-8') if isinstance(label, bytes) else str(label) for label in atlas_labels]
            return data_dict, atlas_labels, mov1_times, mov2_times
        except Exception as e:
            logger.error(f"Failed to load cache from {cache_path}. Error: {e}")
            logger.info("Proceeding with full data processing...")

    logger.info("Initializing atlas and preparing parallel ROI extraction...")

    # Load the global Schaefer atlas to determine n_rois dynamically
    dataset = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    atlas_img = nib.load(dataset.maps)
    # atlas_labels = [label.decode('utf-8') for label in dataset.labels]
    atlas_labels = [label.decode('utf-8') if isinstance(label, bytes) else str(label) for label in dataset.labels]
    n_rois = len(atlas_labels)

    # Build the arguments list based directly on the mapped subjects
    args_list = [(sub, files, atlas_img, n_rois, load_params) for sub, files in subject_files_map.items()]

    logger.info(f"Processing {len(args_list)} subjects...")

    if jobs == 1:
        results = [_process_subject_raw(args) for args in args_list]
    else:
        from joblib import Parallel, delayed
        # Using the default 'loky' backend is usually best, but you can explicitly define it
        results = Parallel(n_jobs=jobs)(delayed(_process_subject_raw)(args) for args in args_list)

    # Reconstruct the output dictionary
    data_dict = {}
    subs, matrix_lists, mov1_times_list, mov2_times_list = zip(*results)
    valid_mov1 = next(t for t in mov1_times_list if t is not None)
    valid_mov2 = next(t for t in mov2_times_list if t is not None)
    mov1_times = valid_mov1
    mov2_times = valid_mov2
    for sub, matrix_list in zip(subs, matrix_lists):
        if matrix_list:
            data_dict[sub] = matrix_list

    logger.info(f"Raw data mapped and averaged successfully for {len(data_dict)} subjects.")

    # Save the processed data to the cache file if a path was provided
    if cache_path:
        if platform.system() == 'Windows':
            pathlib.PosixPath = pathlib.WindowsPath
        logger.info(f"Saving processed data to cache: {cache_path}")
        # Create the parent directory if it does not exist
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            joblib.dump((data_dict, atlas_labels, mov1_times, mov2_times), cache_path)
            logger.info("Cache saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path}. Error: {e}")
    return data_dict, atlas_labels, mov1_times, mov2_times











####################################################################################


# A version of the _process_subject_pc and build_subject_data functions that uses the global
# temporal profiles for each movement, rather than subject-specific temporal profiles.
# This ensures that all subjects are aligned to the same temporal basis for each PC,
# which is important for downstream clustering and analysis.
def _process_subject_pc(args):
    """
    Helper function for multiprocessing: processes a single subject-PC combination.

    Args:
        args (tuple): Tuple containing:
            - sub (str): Subject ID.
            - pc (int): Number of PCs to process.
            - raw_data (dict): Dict mapping {subject_id: {movement(1 or 2): subject_folder_path}}.
            - group_dirs (dict): Dict mapping {movement(1 or 2): group_output_directory_path}.
            - min_len (dict): Dict mapping {movement: min_temporal_length} to align time series.
            - atlas_img(nibabel.Nifti1Image): Loaded atlas image.
            - atlas_labels (list): List of ROI label names.
            - is_test (bool): Is the step of prediction on the test samples

    Returns:
        tuple: (sub, pcs_for_subject) where sub is subject ID and pcs_for_subject is list of
               (ROI x Time) matrices, one per PC.
    """
    sub, pc, raw_data, group_dirs, min_len, atlas_img, atlas_labels, is_test = args
    pcs_for_subject = []

    mov1_times = None
    mov2_times = None
    for pc_idx in range(pc):
        concatenated = []
        for mov in [1, 2]:
            # Load the global temporal profile from the main movement directory
            ts_file = os.path.join(group_dirs[mov], f'temporal_profile_pc_{pc_idx}_group.txt')

            # Load the subject-specific importance map from the subject directory
            if is_test:
                print(f"Loading test projection importance map for subject {sub}, movement {mov}, PC {pc_idx}")
                nii_file = os.path.join(raw_data[sub][mov], f'eigenfunction_{pc_idx}_importance_map_group_test_proj.nii.gz')
            else:
                print(f"Loading importance map for subject {sub}, movement {mov}, PC {pc_idx}")
                nii_file = os.path.join(raw_data[sub][mov], f'eigenfunction_{pc_idx}_importance_map_group.nii.gz')

            # Note: The global matrices file is available at os.path.join(group_dirs[mov], 'global_F_U_matrices.npz')
            # if required for downstream mathematical operations.

            if not os.path.exists(ts_file) or not os.path.exists(nii_file):
                # Skip if files are missing to prevent breaking the concatenation
                continue

            w = get_region_weights(nii_file, atlas_img, atlas_labels)
            t = load_temporal_profile(ts_file)[:min_len[mov]]

            # Create Region x Time matrix
            st_matrix = np.outer(w, t)
            concatenated.append(st_matrix)


        if len(concatenated) == 2:
            mov1_times = concatenated[0].shape[1]
            mov2_times = concatenated[1].shape[1]
            # Concatenate mov1 and mov2 along the time axis
            full_matrix = np.concatenate(concatenated, axis=1)
            pcs_for_subject.append(full_matrix)

    return sub, pcs_for_subject, mov1_times, mov2_times

# A version of the _process_subject_pc and build_subject_data functions that uses the global
# temporal profiles for each movement, rather than subject-specific temporal profiles.
# This ensures that all subjects are aligned to the same temporal basis for each PC,
# which is important for downstream clustering and analysis.
def build_subject_data(examples_dir, pattern, n_pcs, sample_pct, skip_nm, is_test, jobs, cache_path=None, logger=None):
    r"""
    Discover subjects, scan directories, and build spatiotemporal matrices using multiprocessing.

    Args:
        examples_dir (str): Root directory containing the 'outputs_mov1' and 'outputs_mov2' folders.
        pattern (str): Regex pattern to extract subject ID from folder names (e.g., r'(sub-[A-Z0-9]+)').
        n_pcs (int): Number of principal components to load per subject.
        sample_pct (float): Percentage of samples to sample (0 < sample_pct <= 1).
        skip_nm (bool): Whether to skip subjects matching specific patterns.
        jobs (int): Number of parallel jobs.
        cache_path (str, optional): Path to a .pkl or .joblib file to load/save the processed data.
        logger (logging.Logger, optional): Logger instance. Defaults to 'clustering' logger.

    Returns:
        dict: Dict mapping {subject_id: [list of (ROI x Time) matrices]}. Only subjects with
              all N requested PCs are included.
    """
    if logger is None:
        logger = logging.getLogger('clustering')

    # Check if cached data exists to avoid re-processing
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached recovered data from: {cache_path}")
        try:
            if platform.system() == 'Windows':
                pathlib.PosixPath = pathlib.WindowsPath
            # 1. Load the tuple
            cached_data = joblib.load(cache_path)
            data_dict = cached_data[0]
            atlas_labels = cached_data[1]
            mov1_times = cached_data[2]
            mov2_times = cached_data[3]
            atlas_labels = [label.decode('utf-8') if isinstance(label, bytes) else str(label) for label in atlas_labels]
            return data_dict, atlas_labels, mov1_times, mov2_times
        except Exception as e:
            logger.error(f"Failed to load cache from {cache_path}. Error: {e}")
            logger.info("Proceeding with full data processing...")

    logger.info("Scanning directories and building base matrices...")

    dataset = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    atlas_img = nib.load(dataset.maps)
    # atlas_labels = [label.decode('utf-8') for label in dataset.labels]
    atlas_labels = [label.decode('utf-8') if isinstance(label, bytes) else str(label) for label in dataset.labels]
    # Move one level up from the default 'outputs' directory to the parent 'group_run' directory

    # Define the main directories for each movement correctly
    group_dirs = {
        1: os.path.join(examples_dir, "outputs_mov1"),
        2: os.path.join(examples_dir, "outputs_mov2")
    }

    # Map available subjects in each movement directory
    subs_per_mov = {1: {}, 2: {}}
    for mov in [1, 2]:
        if not os.path.exists(group_dirs[mov]):
            logger.warning(f"Movement directory not found: {group_dirs[mov]}")
            continue

        for d in os.listdir(group_dirs[mov]):
            dir_path = os.path.join(group_dirs[mov], d)
            if os.path.isdir(dir_path):
                match = re.search(pattern, d)
                if match:
                    # Extract the subject ID based on the provided regex
                    sub_id = match.group(1)
                    subs_per_mov[mov][sub_id] = dir_path

    # Keep only subjects that exist in BOTH movement 1 and movement 2
    valid_subs = list(set(subs_per_mov[1].keys()).intersection(set(subs_per_mov[2].keys())))
    valid_subs.sort()

    # Apply the skip list if required
    if skip_nm:
        skip_folders = [
            "sub-FNCL43", "sub-FNCL47", "sub-FNCL49", "sub-FNCL50", "sub-FNCL51", "sub-FNCL52",
            "sub-FNCL54", "sub-FNCL60", "sub-FTNC09", "sub-FTNC11", "sub-GYML33",
            "sub-GYML38", "sub-MRTC18", "sub-YA1394", "sub-YA1396", "sub-YA1397",
            "sub-YA1402", "sub-YA1404", "sub-YA1410"
        ]

        filtered_subs = []
        for sub in valid_subs:
            if not any(skip in sub for skip in skip_folders):
                filtered_subs.append(sub)
            else:
                logger.info(f"Skipping subject {sub} as it matches the skip pattern")
        valid_subs = filtered_subs

    # Apply sampling according to sample_pct
    if sample_pct < 1.0 and len(valid_subs) > 0:
        num_samples = max(1, int(len(valid_subs) * sample_pct))
        valid_subs = list(np.random.choice(valid_subs, size=num_samples, replace=False))
        logger.info(f"Sampling {num_samples} out of {len(valid_subs)} subjects ({sample_pct * 100:.1f}%)")

    # Build the final raw_data dictionary mapping sub_id -> {mov_id -> path}
    raw_data = {sub: {1: subs_per_mov[1][sub], 2: subs_per_mov[2][sub]} for sub in valid_subs}

    # Calculate min lengths based on the GLOBAL temporal profiles (once per movement)
    min_len = {1: float('inf'), 2: float('inf')}
    for mov in [1, 2]:
        ts_path = os.path.join(group_dirs[mov], 'temporal_profile_pc_0_group.txt')
        if os.path.exists(ts_path):
            min_len[mov] = len(load_temporal_profile(ts_path))
        else:
            logger.warning(f"Global temporal profile missing for movement {mov}: {ts_path}")

    # Build the arguments list for multiprocessing (passing group_dirs down)
    args_list = [(sub, n_pcs, raw_data, group_dirs, min_len, atlas_img, atlas_labels, is_test) for sub in valid_subs]

    logger.info(f"Processing {len(valid_subs)} subjects with {n_pcs} PCs...")
    if jobs == 1:
        results = [_process_subject_pc(args) for args in args_list]
    else:
        # Import Parallel and delayed locally if jobs > 1
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=jobs)(delayed(_process_subject_pc)(args) for args in args_list)

    # Reconstruct data_dict from multiprocessing results
    subs, pcs_for_subject_list, mov1_times_list, mov2_times_list = zip(*results)
    valid_mov1 = next(t for t in mov1_times_list if t is not None)
    valid_mov2 = next(t for t in mov2_times_list if t is not None)
    mov1_times = valid_mov1
    mov2_times = valid_mov2
    data_dict = {}
    for sub, pcs_for_subject in zip(subs, pcs_for_subject_list):
        data_dict[sub] = pcs_for_subject

    # Keep only subjects that have all N requested PCs
    final_data = {k: v for k, v in data_dict.items() if len(v) == n_pcs}
    logger.info(f"Data loaded successfully for {len(final_data)} subjects.")

    # Save the processed data to the cache file if a path was provided
    if cache_path:
        if platform.system() == 'Windows':
            pathlib.PosixPath = pathlib.WindowsPath
        logger.info(f"Saving processed data to cache: {cache_path}")
        # Create the parent directory if it does not exist
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            joblib.dump((final_data, atlas_labels, mov1_times, mov2_times), cache_path)
            logger.info("Cache saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path}. Error: {e}")

    return final_data, atlas_labels, mov1_times, mov2_times


