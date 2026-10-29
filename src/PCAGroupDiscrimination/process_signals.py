import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from .build_signals_from_files import build_subject_data, scan_input_folder, \
    build_subject_raw_data, load_temporal_profile

class ProcessSignals:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def extract_sequence_features1(self, X_raw, n_regions, exp1_end_tp=None, n_windows=6):
        """
        Extracts comprehensive features for ML classification.
        Calculates global magnitude (std, abs mean) over the entire movement,
        and splits the movement into local time windows for the standard mean
        to preserve temporal directionality without washing it out to zero.
        """
        # Convert input to a NumPy array if it is a list
        X_raw = np.array(X_raw)

        # Calculate the total number of subjects and total timepoints
        n_subjects = X_raw.shape[0]
        n_total_timepoints = X_raw.shape[1] // n_regions

        all_features = []

        for i in range(n_subjects):
            # Reshape to (timepoints, regions) to reconstruct the original matrix
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

                # Skip empty blocks
                if n_exp_timepoints == 0:
                    continue

                # 1. Global Magnitude Features (Entire Movement)
                # Standard deviation captures the variance peaks
                subj_features.extend(np.std(exp_data, axis=0))
                # Mean of absolute values captures the sustained energy
                subj_features.extend(np.mean(np.abs(exp_data), axis=0))

                # 2. Local Directional Features (Windowed Mean)
                # Split the movement into n_windows to prevent full cancellation
                windows = np.array_split(exp_data, n_windows, axis=0)

                for window_data in windows:
                    # Calculate the standard mean for this specific time window
                    # This preserves the +/- sign for this segment of the wave
                    subj_features.extend(np.mean(window_data, axis=0))

            all_features.append(subj_features)

        return np.array(all_features)
    # def extract_sequence_features1(self, X_raw, n_regions, exp1_end_tp=None):
    #     """
    #     Extracts mean and standard deviation features from a concatenated fMRI sequence matrix,
    #     handling the time x regions structure and the boundary between two experiments.
    #     """
    #     # Convert input to a NumPy array if it is a list
    #     X_raw = np.array(X_raw)
    #
    #     # Calculate the total number of subjects and total timepoints
    #     n_subjects = X_raw.shape[0]
    #     n_total_timepoints = X_raw.shape[1] // n_regions
    #
    #     all_features = []
    #
    #     for i in range(n_subjects):
    #         # Reshape to (timepoints, regions) to reconstruct the original matrix
    #         subj_data = X_raw[i].reshape((n_total_timepoints, n_regions))
    #
    #         # Split the data into separate experiments if a cutoff point is provided
    #         if exp1_end_tp is not None:
    #             exp_blocks = [
    #                 subj_data[:exp1_end_tp, :],  # Experiment 1
    #                 subj_data[exp1_end_tp:, :]  # Experiment 2
    #             ]
    #         else:
    #             # Treat as a single continuous experiment
    #             exp_blocks = [subj_data]
    #
    #         subj_features = []
    #
    #         # Extract features for each experiment block independently
    #         for exp_data in exp_blocks:
    #             n_exp_timepoints = exp_data.shape[0]
    #
    #             # Skip empty blocks
    #             if n_exp_timepoints == 0:
    #                 continue
    #
    #             # Global Statistics: Mean and STD per region across time
    #             subj_features.extend(np.mean(exp_data, axis=0))
    #             subj_features.extend(np.std(exp_data, axis=0))
    #
    #         all_features.append(subj_features)
    #
    #     return np.array(all_features)
    #
    # def extract_sequence_features1(self, X_raw, n_regions, exp1_end_tp=None, n_windows_per_exp=3,
    #                                include_connectivity=False):
    #     """
    #     Extracts features from a concatenated fMRI sequence matrix, correctly handling
    #     the time x regions structure and the boundary between two experiments.
    #     """
    #     # Convert input to a NumPy array if it is a list
    #     X_raw = np.array(X_raw)
    #
    #     # Calculate the total number of subjects and total timepoints
    #     n_subjects = X_raw.shape[0]
    #     n_total_timepoints = X_raw.shape[1] // n_regions
    #
    #     all_features = []
    #
    #     for i in range(n_subjects):
    #         # Reshape: since the data is ordered by all regions per timepoint,
    #         # reshaping to (timepoints, regions) reconstructs the original matrix perfectly.
    #         subj_data = X_raw[i].reshape((n_total_timepoints, n_regions))
    #
    #         # Split the data into separate experiments if a cutoff point is provided
    #         if exp1_end_tp is not None:
    #             exp_blocks = [
    #                 subj_data[:exp1_end_tp, :],  # Experiment 1
    #                 subj_data[exp1_end_tp:, :]  # Experiment 2
    #             ]
    #         else:
    #             # Treat as a single continuous experiment
    #             exp_blocks = [subj_data]
    #
    #         subj_features = []
    #
    #         # Extract features for each experiment block independently
    #         for exp_data in exp_blocks:
    #             n_exp_timepoints = exp_data.shape[0]
    #
    #             # Skip empty blocks (just in case exp1_end_tp is misconfigured)
    #             if n_exp_timepoints == 0:
    #                 continue
    #
    #             # 1. Global Statistics per region within this experiment
    #             subj_features.extend(np.mean(exp_data, axis=0))
    #             subj_features.extend(np.std(exp_data, axis=0))
    #             subj_features.extend(np.max(exp_data, axis=0))
    #             subj_features.extend(np.min(exp_data, axis=0))
    #
    #             # 2. Windowed Statistics
    #             # Divide the current experiment into n_windows_per_exp
    #             window_size = n_exp_timepoints // n_windows_per_exp
    #             for w in range(n_windows_per_exp):
    #                 start_idx = w * window_size
    #                 # Ensure the last window captures any remaining timepoints
    #                 end_idx = start_idx + window_size if w < (n_windows_per_exp - 1) else n_exp_timepoints
    #
    #                 window_data = exp_data[start_idx:end_idx, :]
    #                 subj_features.extend(np.mean(window_data, axis=0))
    #
    #             # 3. Temporal Dynamics (Derivatives)
    #             # Calculated safely within the experiment, avoiding the cross-experiment boundary
    #             diff_data = np.diff(exp_data, axis=0)
    #             subj_features.extend(np.std(diff_data, axis=0))
    #             # 4. Functional Connectivity
    #             if include_connectivity:
    #                 # Transpose so regions are rows and timepoints are columns
    #                 corr_matrix = np.corrcoef(exp_data.T)
    #
    #                 # Fix zero-variance regions: Replace NaNs and Infs with 0.0
    #                 corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    #
    #                 # Take only the upper triangle to avoid redundant features
    #                 upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
    #                 subj_features.extend(corr_matrix[upper_tri_indices])
    #
    #         all_features.append(subj_features)
    #
    #     return np.array(all_features)

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

    def extract_loadings_all_pcs(self, data_dict, subjects, n_regions, mov1_times):
        """
        Features = ROI loadings of ALL PCs for both movements.
        Each cached matrix is rank one, M = outer(w, t), with t shared by all
        subjects. We recover t (up to a common scale/sign) from the first
        subject, then w_s = M_s t / (t.t) for every subject, so the loadings
        are on a common scale and sign across subjects.
        Output per subject: n_pcs * 2 * n_regions values.
        """
        n_pcs = len(data_dict[subjects[0]])

        t_ref = [
            [
                load_temporal_profile(
                    os.path.join(
                        self.args.input_dir,
                        f"outputs_mov{mov}",
                        f"temporal_profile_pc_{pc}_group.txt",
                    )
                )
                for mov in (1, 2)
            ]
            for pc in range(n_pcs)
        ]
        X = []
        for sub in subjects:
            feats = []
            for pc in range(n_pcs):
                M = np.asarray(data_dict[sub][pc])
                for (a, b), t in zip([(0, mov1_times), (mov1_times, M.shape[1])], t_ref[pc]):
                    block = M[:, a:b]
                    if block.shape[1] != len(t) or t @ t <= 0:
                        raise ValueError(
                            "Invalid or incompatible group temporal profile"
                        )

                    w = block @ t / (t @ t)
                    resid = np.linalg.norm(block - np.outer(w, t)) / (np.linalg.norm(block) + 1e-12)
                    if resid > 1e-4:
                        self.logger.warning(f"fPCA-CRITICAL-WARNING: {sub} pc{pc}: rank-one residual {resid:.2e}")
                    feats.append(w)
            X.append(np.concatenate(feats))
        return np.array(X)

    def load_regressors(self, mov1_times, mov2_times):
        """
        Per-TR regressors for feature set 4, as {1: DataFrame, 2: DataFrame}.
        Source, in this order:
          --annotation-xlsx : built here from the orchestration annotation workbook,
                              with the movement lengths, TR and skipped volumes of
                              this run (PCAGroupDiscrimination.annotation_regressors);
                              the regressors are also written to
                              <output-dir>/0_regressors/ for inspection.
          --acoustic-features-dir : precomputed acoustic_mov1.csv / acoustic_mov2.csv
                              (e.g. from cli/acoustic_features.py).
        """
        xlsx = getattr(self.args, "annotation_xlsx", None)
        if xlsx:
            from .annotation_regressors import build_annotation_regressors
            res = build_annotation_regressors(
                xlsx, mov1_times, mov2_times, self.args.TR,
                n_skip_vols_start=getattr(self.args, "n_skip_vols_start", 0),
                split_solo=getattr(self.args, "annotation_split_solo", False),
                logger=self.logger)
            out_dir = os.path.join(os.path.dirname(self.args.output_dir), "0_regressors")
            os.makedirs(out_dir, exist_ok=True)
            for mov, (reg, raw) in res.items():
                reg.to_csv(os.path.join(out_dir, f"acoustic_mov{mov}.csv"), index=False)
                raw.to_csv(os.path.join(out_dir, f"annotation_raw_mov{mov}.csv"), index=False)
            self.logger.info(f"Regressors built from {xlsx} and written to {out_dir}")
            return {mov: reg for mov, (reg, raw) in res.items()}

        frames = {}
        for mov in (1, 2):
            path = os.path.join(self.args.acoustic_features_dir, f"acoustic_mov{mov}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing regressors: {path}")
            frames[mov] = pd.read_csv(path)
        if list(frames[1].columns) != list(frames[2].columns):
            raise ValueError("acoustic_mov1.csv and acoustic_mov2.csv must have the same columns")
        return frames

    def extract_encoding_betas(self, raw_data_dict, subjects, n_regions, mov1_times, mov2_times):
        """
        Feature set 4: encoding betas.
        For every subject, movement and ROI, the (filtered) ROI time series is
        regressed on the per-TR acoustic regressors of that movement
        (acoustic_mov1.csv / acoustic_mov2.csv in --acoustic-features-dir, already
        HRF-convolved and z-scored by cli/acoustic_features.py). The regression
        coefficients ("how much this ROI follows loudness, pulse clarity, ...")
        are the features: n_regressors * n_regions per movement.
        """
        regs = {}
        reg_frames = self.load_regressors(mov1_times, mov2_times)
        for mov, n_t in [(1, mov1_times), (2, mov2_times)]:
            R = reg_frames[mov].values.astype(float)
            if R.shape[0] < n_t:
                raise ValueError(f"regressors of movement {mov} have {R.shape[0]} TRs but the fMRI data has {n_t}")
            if R.shape[0] > n_t:
                self.logger.warning(f"regressors of movement {mov} have {R.shape[0]} TRs, fMRI has {n_t}: truncating")
            R = R[:n_t]
            R = (R - R.mean(0)) / (R.std(0) + 1e-12)
            regs[mov] = np.hstack([np.ones((n_t, 1)), R])  # intercept + regressors
        self.regressor_names = list(reg_frames[1].columns)
        self.n_acoustic_regressors = len(self.regressor_names)

        X = []
        for sub in subjects:
            M = np.asarray(raw_data_dict[sub][0])  # (n_regions, mov1_times + mov2_times)
            feats = []
            for mov, (a, b) in [(1, (0, mov1_times)), (2, (mov1_times, mov1_times + mov2_times))]:
                Y = M[:, a:b].T  # (T, n_regions)
                Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)  # z-score each ROI over time
                beta = np.linalg.lstsq(regs[mov], Y, rcond=None)[0]  # (1 + k, n_regions)
                feats.append(beta[1:].ravel())  # drop intercept; order: regressor-major, ROI-minor
            X.append(np.concatenate(feats))
        return np.array(X)

    def read_signals(self):
        if self.args.use_raw_data and self.args.extra_features_set == 3:
            raise ValueError("extra-features-set 3 (all-PC loadings) requires PC data, not raw data")
        encoding_source = getattr(self.args, "encoding_source", "fpca")
        if self.args.extra_features_set == 4 and encoding_source == "raw" and not self.args.use_raw_data:
            raise ValueError("extra-features-set 4 with --encoding-source raw requires --use-raw-data (ROI time series)")
        if self.args.extra_features_set == 4 and not (getattr(self.args, "annotation_xlsx", None)
                                                      or getattr(self.args, "acoustic_features_dir", None)):
            raise ValueError("extra-features-set 4 requires --annotation-xlsx or --acoustic-features-dir")

        if self.args.extra_features_set == 4 and encoding_source == "fpca":
            # ============================================================
            # Feature set 4 on the subject-level fPCA reconstruction: each
            # subject's ROI signal is rebuilt from its OWN n_pcs components
            # (pca-singles outputs), then regressed on the stimulus regressors.
            # ============================================================
            from .fpca_subject_signals import build_subject_fpca_signals
            cache_file_path = os.path.join(self.args.cache_file_pardir,
                                           f"fpca_subject_signals_{self.args.n_pcs}pcs_cache.joblib")
            fpca_dict, atlas_labels, mov1_times, mov2_times = build_subject_fpca_signals(
                self.args.input_dir, self.args.file_name_pattern, self.args.n_pcs,
                self.args.jobs, cache_file_path, self.logger)
            subjects = [s for s in fpca_dict if fpca_dict.get(s)]
            if not subjects:
                raise RuntimeError("No subjects reconstructed by build_subject_fpca_signals")
            ids = [f"sub-{s}" if not str(s).startswith("sub-") else str(s) for s in subjects]
            X_features = self.extract_encoding_betas(fpca_dict, subjects, len(atlas_labels), mov1_times, mov2_times)
            self.logger.info(f"Encoding-beta feature matrix (subject fPCA reconstruction): {X_features.shape}")
            if self.args.row_normalize_ml:
                X_features = normalize(X_features, norm="l2", axis=1)
            return X_features, ids, atlas_labels

        if self.args.use_raw_data:
            # 1. Package all the LoadData and cropping parameters
            load_params = {
                'TR': self.args.TR,
                'smooth_size': self.args.smooth_size,
                'highpass': self.args.highpass,
                'lowpass': self.args.lowpass,
                'use_nilearn_filter': self.args.use_nilearn_filter,
                'n_compcor_nilearn_filter': self.args.n_compcor_nilearn_filter,
                'smoothing_fwhm_nilearn_filter': self.args.smoothing_fwhm_nilearn_filter,
                'processed': self.args.processed,
                'n_skip_vols_start': self.args.n_skip_vols_start,
                'n_skip_vols_end': self.args.n_skip_vols_end,

                # Match the spatially centered representation of Set 4.
                'spatial_center': self.args.extra_features_set == 4
            }

            # 2. Dynamically scan the drive to pair NIfTI files with their specific masks
            subject_files_map = scan_input_folder(
                input_dir=self.args.raw_data_path,
                file_name_pattern=self.args.file_name_pattern,
                logger=self.logger
            )

            cache_suffix = "_test" if self.args.mode == "ml-test" else ""
            cache_file_path = os.path.join(self.args.cache_file_pardir, f"processed_raw_data_cache{cache_suffix}.joblib")
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

            # ====================================================================
            # Feature set 4: encoding betas on acoustic regressors (ROI time series
            # regressed on the music's per-TR features). Handled here, before the
            # flattening loop, which is not needed for it.
            # ====================================================================
            if self.args.extra_features_set == 4:
                subjects = [s for s in subjects if raw_data_dict.get(s)]
                ids = [f"sub-{s}" if not str(s).startswith("sub-") else str(s) for s in subjects]
                X_features = self.extract_encoding_betas(
                    raw_data_dict, subjects, len(atlas_labels), mov1_times, mov2_times)
                self.logger.info(f"Encoding-beta feature matrix (raw ROI time series): {X_features.shape}")
                if self.args.row_normalize_ml:
                    X_features = normalize(X_features, norm="l2", axis=1)
                return X_features, ids, atlas_labels
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

            # 1. Define and create the new output directory
            parent_folder = os.path.dirname(self.args.output_dir)
            prestep_dir = os.path.join(parent_folder, "0_extracted_pcs")
            os.makedirs(prestep_dir, exist_ok=True)

            subjects = list(data_dict.keys())
            if not subjects:
                raise RuntimeError("No subjects loaded by build_subject_data")

            # ====================================================================
            # Feature set 3: ROI loadings of all PCs (no sequence reconstruction).
            # Handled here, before the per-PC loop, which is not needed for it.
            # ====================================================================
            if self.args.extra_features_set == 3:
                ids = [f"sub-{s}" if not str(s).startswith("sub-") else str(s) for s in subjects]
                X_features = self.extract_loadings_all_pcs(
                    data_dict, subjects, len(atlas_labels), mov1_times)
                self.logger.info(f"Loadings feature matrix (all PCs): {X_features.shape}")
                if self.args.row_normalize_ml:
                    X_features = normalize(X_features, norm="l2", axis=1)
                return X_features, ids, atlas_labels

            # Extract the target PC 0 vector for each subject and build the feature matrix
            X = []
            ids = []
            for sub in subjects:
                pcs = data_dict[sub]
                if len(pcs) <= self.args.target_pc_index:
                    continue
                clean_sub_id = f"sub-{sub}" if not str(sub).startswith("sub-") else str(sub)
                target_pc_data = pcs[self.args.target_pc_index]
                # Ensure subject ID has the correct 'sub-' prefix
                # ====================================================================
                # Save ONLY the target PC to CSV (Without flatten, regions as columns, timepoints as rows)
                # ====================================================================
                if target_pc_data.shape[1] == len(atlas_labels):
                    df = pd.DataFrame(target_pc_data, columns=atlas_labels)
                elif target_pc_data.shape[0] == len(atlas_labels):
                    df = pd.DataFrame(target_pc_data.T, columns=atlas_labels)
                else:
                    df = pd.DataFrame(target_pc_data)

                file_name = f"{clean_sub_id}_pc-{self.args.target_pc_index}.csv"
                out_path = os.path.join(prestep_dir, file_name)
                df.to_csv(out_path, index=False)
                # ====================================================================
                # Build the feature matrix exactly as it was before
                # ====================================================================
                vec = target_pc_data.flatten(order='F')
                X.append(vec)
                ids.append(clean_sub_id)

        X_features = np.array(X)
        if self.args.extra_features_set == 1:
            X_features = self.extract_sequence_features1(
                X_raw=X,
                n_regions=len(atlas_labels),
                exp1_end_tp=mov1_times,
                n_windows=self.args.n_windows_per_exp,
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
