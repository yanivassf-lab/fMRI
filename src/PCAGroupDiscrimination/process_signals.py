import os
import numpy as np
from sklearn.preprocessing import normalize

from .build_signals_from_files import build_subject_data, scan_input_folder, \
    build_subject_raw_data

class ProcessSignals:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def extract_sequence_features1(self, X_raw, n_regions, exp1_end_tp=None, n_windows_per_exp=3,
                                   include_connectivity=False):
        """
        Extracts features from a concatenated fMRI sequence matrix, correctly handling
        the time x regions structure and the boundary between two experiments.
        """
        # Convert input to a NumPy array if it is a list
        X_raw = np.array(X_raw)

        # Calculate the total number of subjects and total timepoints
        n_subjects = X_raw.shape[0]
        n_total_timepoints = X_raw.shape[1] // n_regions

        all_features = []

        for i in range(n_subjects):
            # Reshape: since the data is ordered by all regions per timepoint,
            # reshaping to (timepoints, regions) reconstructs the original matrix perfectly.
            subj_data = X_raw[i].reshape((n_total_timepoints, n_regions))

            # Split the data into separate experiments if a cutoff point is provided
            if exp1_end_tp is not None:
                exp_blocks = [
                    subj_data[:exp1_end_tp, :],  # Experiment 1
                    subj_data[exp1_end_tp:, :]  # Experiment 2
                ]
            else:
                # Treat as a single continuous experiment
                exp_blocks = [subj_data]

            subj_features = []

            # Extract features for each experiment block independently
            for exp_data in exp_blocks:
                n_exp_timepoints = exp_data.shape[0]

                # Skip empty blocks (just in case exp1_end_tp is misconfigured)
                if n_exp_timepoints == 0:
                    continue

                # 1. Global Statistics per region within this experiment
                subj_features.extend(np.mean(exp_data, axis=0))
                subj_features.extend(np.std(exp_data, axis=0))
                subj_features.extend(np.max(exp_data, axis=0))
                subj_features.extend(np.min(exp_data, axis=0))

                # 2. Windowed Statistics
                # Divide the current experiment into n_windows_per_exp
                window_size = n_exp_timepoints // n_windows_per_exp
                for w in range(n_windows_per_exp):
                    start_idx = w * window_size
                    # Ensure the last window captures any remaining timepoints
                    end_idx = start_idx + window_size if w < (n_windows_per_exp - 1) else n_exp_timepoints

                    window_data = exp_data[start_idx:end_idx, :]
                    subj_features.extend(np.mean(window_data, axis=0))

                # 3. Temporal Dynamics (Derivatives)
                # Calculated safely within the experiment, avoiding the cross-experiment boundary
                diff_data = np.diff(exp_data, axis=0)
                subj_features.extend(np.std(diff_data, axis=0))
                # 4. Functional Connectivity
                if include_connectivity:
                    # Transpose so regions are rows and timepoints are columns
                    corr_matrix = np.corrcoef(exp_data.T)

                    # Fix zero-variance regions: Replace NaNs and Infs with 0.0
                    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

                    # Take only the upper triangle to avoid redundant features
                    upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
                    subj_features.extend(corr_matrix[upper_tri_indices])

            all_features.append(subj_features)

        return np.array(all_features)

    def extract_sequence_features2(self, X_raw, n_regions, exp1_end_tp=None, include_connectivity=False):
        """
        Extracts fMRI features using exact experimental transitions, optimized for ML.
        It shifts windows by an HRF delay to prevent signal leakage across styles,
        and uses robust averages rather than noisy single-TR measurements.
        """
        X_raw = np.array(X_raw)
        n_subjects = X_raw.shape[0]
        n_total_timepoints = X_raw.shape[1] // n_regions

        all_features = []

        # HRF delay compensation (approx 4.5 seconds = 6 TRs if TR=0.75s)
        hrf_shift = 6
        # Margin for robust transition jumps
        margin = 5

        for i in range(n_subjects):
            subj_data = X_raw[i].reshape((n_total_timepoints, n_regions))

            if exp1_end_tp is not None:
                exp_blocks = [
                    (subj_data[:exp1_end_tp, :], self.args.mov1_transition_trs),
                    (subj_data[exp1_end_tp:, :], self.args.mov2_transition_trs)
                ]
            else:
                exp_blocks = [(subj_data, [])]

            subj_features = []

            for exp_data, trs in exp_blocks:
                n_exp_timepoints = exp_data.shape[0]
                if n_exp_timepoints == 0:
                    continue

                # 1. Global statistics (Robust baselines)
                subj_features.extend(np.mean(exp_data, axis=0))
                subj_features.extend(np.std(exp_data, axis=0))

                # 2. Style-Segment Averages (HRF Shifted)
                # We shift the start and end of each window to account for blood flow delay
                starts = [0] + trs
                ends = trs + [n_exp_timepoints]

                for start, end in zip(starts, ends):
                    # Shift window to ignore the bleeding signal from previous style
                    shifted_start = min(start + hrf_shift, n_exp_timepoints)
                    shifted_end = min(end + hrf_shift, n_exp_timepoints)

                    # Ensure the window is large enough to mean safely (at least 3 TRs)
                    if (shifted_end - shifted_start) >= 3:
                        window_data = exp_data[shifted_start:shifted_end, :]
                        subj_features.extend(np.mean(window_data, axis=0))
                    else:
                        # Fallback to the global mean of the experiment to prevent zero-trap
                        subj_features.extend(np.mean(exp_data, axis=0))

                # 3. Robust Transition Jumps (Difference of averages, not single TRs)
                for tr in trs:
                    if margin <= tr < (n_exp_timepoints - margin):
                        # Average activity slightly before the style changed
                        before_transition = np.mean(exp_data[tr - margin: tr, :], axis=0)

                        # Average activity slightly after, including the HRF build-up
                        after_transition = np.mean(exp_data[tr: tr + margin + hrf_shift, :], axis=0)

                        jump = after_transition - before_transition
                        subj_features.extend(jump)
                    else:
                        subj_features.extend(np.zeros(n_regions))

                # 4. Functional Connectivity (Optional)
                if include_connectivity:
                    corr_matrix = np.corrcoef(exp_data.T)
                    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                    upper_tri = np.triu_indices_from(corr_matrix, k=1)
                    subj_features.extend(corr_matrix[upper_tri])

            all_features.append(subj_features)

        return np.array(all_features)

    def read_signals(self):
        if self.args.use_raw_data:
            # 1. Package all the LoadData and cropping parameters
            load_params = {
                'TR': self.args.TR,
                'smooth_size': self.args.smooth_size,
                'highpass': self.args.highpass,
                'lowpass': self.args.lowpass,
                'processed': self.args.processed,
                'n_skip_vols_start': self.args.n_skip_vols_start,
                'n_skip_vols_end': self.args.n_skip_vols_start
            }

            # 2. Dynamically scan the drive to pair NIfTI files with their specific masks
            subject_files_map = scan_input_folder(
                input_dir=self.args.raw_data_path,
                file_name_pattern=self.args.file_name_pattern,
                logger=self.logger
            )

            cache_file_path = os.path.join(self.args.cache_file_pardir, "processed_raw_data_cache.joblib")
            # Add logger configuration to the parameters dictionary
            load_params['log_dir'] = self.args.output_dir

            raw_data_dict, atlas_labels, mov1_times, mov2_times = build_subject_raw_data(
                subject_files_map=subject_files_map,
                load_params=load_params,
                jobs=2,
                cache_path=cache_file_path,
                logger=self.logger
            )

            # Output structure of raw_data_dict is:
            # {'sub-YA1308': [st_matrix], 'sub-YA1314': [st_matrix], ...}
            # You can now proceed to align, stack, and plot these matrices just like you did with the PCs.
            subjects = list(raw_data_dict.keys())
            if not subjects:
                raise RuntimeError("No subjects loaded by build_subject_raw_data")
            # Extract the raw concatenated ROI x Time matrix for each subject and build the feature matrix
            X = []
            ids = []
            for sub in subjects:
                raw_matrices = raw_data_dict.get(sub)
                # Check if the subject has valid data (the list is not empty)
                if not raw_matrices or len(raw_matrices) == 0:
                    self.logger.warning(f"Missing raw data for subject {sub}, skipping.")
                    continue
                # Since we are using raw data, there is only one combined matrix per subject
                full_matrix = raw_matrices[0]
                # Flatten the matrix into a 1D feature vector (maintaining the original spatial-temporal order)
                vec = full_matrix.flatten(order='F')
                X.append(vec)
                ids.append(f"sub-{sub}" if not str(sub).startswith("sub-") else str(sub))
            X = np.array(X)
            self.logger.info(f"Built feature matrix X with shape: {X.shape}")
        else:  # Original PC-based logic
            if self.args.mode == "ml-test":
                cache_file_path = os.path.join(self.args.cache_file_pardir,
                                               f"recoverd_pcs_0-{self.args.n_pcs - 1}_data_cache_test_proj.joblib")
            else:
                cache_file_path = os.path.join(self.args.cache_file_pardir,
                                               f"recoverd_pcs_0-{self.args.n_pcs - 1}_data_cache.joblib")
            data_dict, atlas_labels, mov1_times, mov2_times = build_subject_data(
                self.args.input_dir,
                self.args.file_name_pattern,
                self.args.n_pcs,
                1.0,  # Read all signals
                False,  # Don't skip NM for ML analysis
                self.args.mode == "ml-test",
                self.args.jobs,
                cache_file_path,
                self.logger
            )

            subjects = list(data_dict.keys())
            if not subjects:
                raise RuntimeError("No subjects loaded by build_subject_data")

            # Extract the target PC 0 vector for each subject and build the feature matrix
            X = []
            ids = []
            for sub in subjects:
                pcs = data_dict[sub]
                if len(pcs) <= self.args.target_pc_index:
                    continue
                vec = pcs[self.args.target_pc_index].flatten(order='F')
                X.append(vec)
                ids.append(f"sub-{sub}" if not str(sub).startswith("sub-") else str(sub))

        X_features = np.array(X)
        if self.args.extra_features_set == 1:
            X_features = self.extract_sequence_features1(
                X_raw=X,
                n_regions=len(atlas_labels),
                exp1_end_tp=mov1_times,
            )
        elif self.args.extra_features_set == 2:
            X_features = self.extract_sequence_features2(
                X_raw=X,
                n_regions=len(atlas_labels),
                exp1_end_tp=mov1_times,
            )
        self.logger.info(f"Original shape: {X_features.shape}")
        self.logger.info(f"Built feature matrix X with shape: {X_features.shape}")

        if self.args.row_normalize_ml:
            X_features = normalize(X_features, norm="l2", axis=1)
        return X_features, ids, atlas_labels
