import numpy as np
from itertools import combinations
from dtaidistance import dtw
from scipy.stats import pearsonr

from fmri.find_best_lags import calculate_lags_durations, build_regressor_vectorized


class RepresentPC:
    ""  "Base class for similarity metrics."""

    def __init__(self, pcs_list, F_list, n_subs, n_movs, logger):
        """
        Initialize with a similarity matrix and parameters.

        Attributes:
        n_subs : int
            Number of subjects per movement/experiment
        n_movs : int
            Number of movements/experiments
        """
        self.pcs_list = pcs_list
        self.F_list = F_list
        self.n_subs = n_subs
        self.n_movs = n_movs
        self.logger = logger
        self.n_components = pcs_list[0].shape[1]
        self.n_files = n_subs * n_movs
        self.sim_matrix = np.ones((self.n_files, self.n_files))
        self.score = None
        self.matrix_op_pval = None

    def get_sub_mov(self, i: int) -> int:
        """Get the movement number of the i-th file (starts from 1)."""
        return int(i / self.n_subs) + 1

    @staticmethod
    def get_signals_from_seleted_pcs(pcs_list, F_list, rep_pcs):
        """
        Get signals reconstructed from selected principal components.

        Args:
            pcs_list: list of principal component matrices for each subject/movement.
            F_list:   list of signal matrices for each subject/movement.
            rep_pcs: list of selected principal component indices for each subject/movement.

        Returns: List of reconstructed signals for each subject/movement.
        """
        signals = []
        for i in range(len(pcs_list)):
            pc_i = pcs_list[i][rep_pcs[i]]
            signal_i = F_list[i] @ pc_i
            signals.append(signal_i)
        return signals

    def find_representative_pcs(self, pc_sim_auto_best_similar_pc, pc_sim_auto_weight_similar_pc) -> tuple[
        list[np.ndarray], list[str]]:
        """Calculates the most representative PC for each file.
        This method identifies a single "main" or "representative" PC for each
        file. It does this by summing the squared correlations of each PC with
        the PCs from all other files.

        Parameters
        ----------
        - pc_sim_auto_best_similar_pc (bool): If True, only the best matching PC from each
          other file is considered when calculate similarity, otherwise all PCs are
          averaged.
        - pc_sim_auto_weight_similar_pc (int): Exponent to which the absolute correlation values
          are raised. Higher values give more weight to stronger correlations.

        Returns
        -------
        rep_pcs_signals (np.ndarray)
            A list (n_files) each element rep_pcs_signals[i] is a 1D numpy array contains
            the most representative pc for file i.
        rep_pcs (list)
            A list of shape (n_files,). Each element rep_pcs[i] is the
            integer index (from 0 to n_components-1) of the most representative
            principal component for file i.
        """
        pc_corrs_sum = np.zeros((self.n_files, self.n_components))

        for i, j in combinations(range(self.n_files), 2):
            if self.get_sub_mov(i) == self.get_sub_mov(j):
                pcs_i = self.pcs_list[i]
                pcs_j = self.pcs_list[j]

                # Correlation between individual PCs
                corr_matrix = np.corrcoef(pcs_i.T, pcs_j.T)[:self.n_components, self.n_components:]
                corr_matrix = np.abs(
                    corr_matrix) ** pc_sim_auto_weight_similar_pc  # stronger weight to higher correlations

                # Sum correlations for each PC
                if pc_sim_auto_best_similar_pc:
                    # Only consider the best matching PC for each PC
                    pc_corrs_sum[i, :] += np.max(corr_matrix, axis=1)
                    pc_corrs_sum[j, :] += np.max(corr_matrix, axis=0)
                else:
                    pc_corrs_sum[i, :] += np.sum(corr_matrix, axis=1)
                    pc_corrs_sum[j, :] += np.sum(corr_matrix, axis=0)
            else:
                pc_corrs_sum[i, :] += 0.0
                pc_corrs_sum[j, :] += 0.0

        # Select the PC with the highest total correlation for each sample
        rep_pcs = np.argmax(pc_corrs_sum, axis=1)
        rep_pcs_signals = self.get_signals_from_seleted_pcs(self.pcs_list, self.F_list, rep_pcs)
        rep_pcs_names = [pcs.pc_name(i) for i, pcs in zip(rep_pcs, self.pcs_list)]
        return rep_pcs_signals, rep_pcs_names

    def auto_calc_lag_duartions(self):
        """Automatically calculates lags and durations structure for all PCs except PC0 and any skipped PCs."""
        # Determine PCs to test (skip PC0 and any additional skipped PCs)
        pcs_matrix = self.pcs_list[0]
        orig_n_components = pcs_matrix.get_orig_pcs().shape[1]
        skip_pc_num = pcs_matrix.get_skip_pc_num().copy()  # returns set of skipped pc numbers
        skip_pc_num.add(0)  # Also skip PC0
        pcs_to_test = [j for j in range(1, orig_n_components) if
                       j not in skip_pc_num]  # Skip PC0 and any additional skipped PCs
        pcs_to_test_dummy_idx = [pcs_matrix.get_dummy_idx(i) for i in pcs_to_test]
        return pcs_to_test, pcs_to_test_dummy_idx

    def build_lags_durations_structure(self, times_list, stim_times, lags, durs, pcs_to_test):
        """Builds a structure mapping PCs and movements to their corresponding lags, durations, and target regressors."""
        lags_durations = {}
        target_regressors = []
        for mov_i in range(self.n_movs):
            reg = build_regressor_vectorized(times_list[mov_i * self.n_subs], stim_times[mov_i], lags[mov_i],
                                             durs[mov_i])
            target_regressors.append(reg)
        for mov_i in range(self.n_movs):
            for j in pcs_to_test:
                lags_durations[(j, mov_i)] = (lags[mov_i], durs[mov_i], 0, target_regressors[mov_i])
        return lags_durations

    def combine_pcs(self, files_path, params, F_list, times_list, stimulus_times, bold_lag_seconds,
                    bold_dur_seconds, correlation_threshold):
        """
        Combines relevant PCs based on windowed correlation with stimulus times.

        This method replaces the DTW logic with a Pearson correlation against
        a "boxcar" regressor built from the stimulus times.
        """

        self.logger.info(
            f"Combining relevant PCs based on Windowed Correlation (Threshold r={correlation_threshold})...")

        pcs_to_test, pcs_to_test_dummy_idx = self.auto_calc_lag_duartions()
        if not bold_lag_seconds:
            lags_durations = calculate_lags_durations(files_path, params, stimulus_times, pcs_to_test,
                                                      logger=self.logger)

        else:
            lags_durations = self.build_lags_durations_structure(times_list, stimulus_times, bold_lag_seconds,
                                                                 bold_dur_seconds, pcs_to_test)

        combined_signals = []
        rep_pcs_names = []  # For logging
        for i in range(self.n_files):
            #  Get data for this subject
            pcs = self.pcs_list[i]  # (K, n_comp)
            F = F_list[i]  # (N, K)
            times = times_list[i]  # (N,)
            n_timepoints = len(times)
            mov_i = self.get_sub_mov(i) - 1  # movement index

            # Find and sum all relevant PCs (skip PC0)
            final_signal = np.zeros(n_timepoints)
            rep_pcs_names_i = ""
            all_pc_signals = F @ pcs  # (N, n_comp)

            for j, dj in zip(pcs_to_test, pcs_to_test_dummy_idx):
                _, _, _, target_regressor = lags_durations[(j, mov_i)]
                pc_j_signal = all_pc_signals[:, dj]
                try:
                    corr, pval = pearsonr(pc_j_signal, target_regressor)
                except ValueError:
                    corr = 0  # Handle zero-variance signals (e.g., if target is all zeros)

                pc_j_name = pcs.pc_name(j)
                # Log the result
                self.logger.info("Subject %d, PC %s: Correlation=%.4f, thre=%.2f", i, pc_j_name, corr,
                                 correlation_threshold)

                # If correlation is high, add it
                if abs(corr) > correlation_threshold:
                    # Use the correlation value as a weight.
                    # This automatically handles the sign (corr is +/-)
                    # and gives more weight to strongly correlated PCs.
                    corr_sign = '+' if np.sign(corr) > 0 else '-'
                    final_signal += np.sign(corr) * pc_j_signal
                    rep_pcs_names_i += f"{corr_sign}{pc_j_name}(c={corr:.2f}) "

            combined_signals.append(final_signal)
            rep_pcs_names.append(rep_pcs_names_i if rep_pcs_names_i else "No PCs Found")
            self.logger.info(
                f"Subject {i} combined PCs: {rep_pcs_names_i if rep_pcs_names_i else "No PCs Found"}")
        return combined_signals, rep_pcs_names

    # def combine_pcs(self, files_path, params, F_list, times_list, stimulus_times, bold_lag_seconds, correlation_threshold):
    #     """
    #     Combines relevant PCs based on windowed correlation with stimulus times.
    #
    #     This method replaces the DTW logic with a Pearson correlation against
    #     a "boxcar" regressor built from the stimulus times.
    #     """
    #
    #     self.logger.info(
    #         f"Combining relevant PCs based on Windowed Correlation (Threshold r={correlation_threshold})...")
    #
    #     n_components = self.pcs_list[0].shape[1]
    #     pcs_to_test = list(range(1, n_components)) # Skip PC0
    #     lags_durations = calculate_lags_durations(files_path, params, stimulus_times, pcs_to_test)
    #     combined_signals = []
    #     rep_pcs_names = []  # For logging
    #
    #     for i in range(self.n_files):
    #         #  Get data for this subject
    #         pcs = self.pcs_list[i]  # (K, n_comp)
    #         F = F_list[i]  # (N, K)
    #         times = times_list[i]  # (N,)
    #         n_timepoints = len(times)
    #         mov_i = self.get_sub_mov(i) - 1  # movement index
    #
    #         # Build the target regressor (the "ideal" signal)
    #         # This is now a "boxcar" or "window" regressor
    #         target_regressor = np.zeros(n_timepoints)
    #         # wins = [[27.0, 20.0, 20.0, 20.0], [17.0, 25.0, 0.0, 25.0]]
    #         # for t_stim, b_lag, win in zip(stimulus_times[mov_i], bold_lag_seconds[mov_i], wins[mov_i]):
    #         for t_stim in stimulus_times[mov_i]:
    #             b_lag = lags_durations[mov_i][stimulus_times[mov_i].index(t_stim)]
    #             win = lags_durations[(files_path[i], mov_i)][pcs_to_test.index(1)][stimulus_times[mov_i].index(t_stim)]
    #             # Create boxcar window
    #             t_start = t_stim + b_lag
    #             t_end = t_start + win
    #             self.logger.info(f"Subject {i}: Stimulus at {t_stim}s (lag {b_lag}s) -> window [{t_start}s, {t_end}s]")
    #             # Find indices within this window
    #             indices = np.where((times >= t_start) & (times <= t_end))[0]
    #             if len(indices) > 0:
    #                 target_regressor[indices] = 1.0  # Create a "box" of 1s
    #
    #         # Check if target regressor is all zeros (no stims in range)
    #         if np.sum(target_regressor) == 0:
    #             self.logger.warning(
    #                 f"Subject {i}: No stimulus times fell within the analysis window. Target regressor is empty.")
    #             combined_signals.append(np.zeros(n_timepoints))
    #             rep_pcs_names.append("No stims in range")
    #             continue
    #
    #         # Find and sum all relevant PCs (skip PC0)
    #         final_signal = np.zeros(n_timepoints)
    #         rep_pcs_names_i = ""
    #         all_pc_signals = F @ pcs  # (N, n_comp)
    #
    #         for j in range(1, n_components):
    #         # for j in range(n_components):
    #             pc_j_signal = all_pc_signals[:, j]
    #             try:
    #                 corr, pval = pearsonr(pc_j_signal, target_regressor)
    #             except ValueError:
    #                 corr = 0  # Handle zero-variance signals (e.g., if target is all zeros)
    #
    #             pc_j_name = pcs.pc_name(j)
    #             # Log the result
    #             self.logger.info("Subject %d, PC %s: Correlation=%.4f, thre=%.2f", i, pc_j_name, corr,
    #                              correlation_threshold)
    #
    #             # If correlation is high, add it
    #             if abs(corr) > correlation_threshold:
    #                 # Use the correlation value as a weight.
    #                 # This automatically handles the sign (corr is +/-)
    #                 # and gives more weight to strongly correlated PCs.
    #                 corr_sign = '+' if np.sign(corr) > 0 else '-'
    #                 final_signal += np.sign(corr) * pc_j_signal
    #                 rep_pcs_names_i += f"{corr_sign}{pc_j_name}(c={corr:.2f}) "
    #
    #
    #         combined_signals.append(final_signal)
    #         rep_pcs_names.append(rep_pcs_names_i if rep_pcs_names_i else "No PCs Found")
    #         self.logger.info(
    #             f"Subject {i} combined PCs: {rep_pcs_names_i if rep_pcs_names_i else "No PCs Found"}")
    #     return combined_signals, rep_pcs_names

    # Old DTW-based method
    # ====================

    # def combine_pcs(self, F_list, times_list, stimulus_times, bold_lag_seconds, distance_threshold):
    #     self.logger.info(f"Combining relevant PCs based on DTW distance to {len(stimulus_times)} stimulus times...")
    #
    #     combined_signals = []
    #     rep_pcs_names = []  # For logging
    #
    #     for i in range(self.n_files):
    #         # 1. Get data for this subject
    #         pcs = self.pcs_list[i]  # (K, n_comp)
    #         F = F_list[i]  # (N, K)
    #         times = times_list[i]  # (N,)
    #         n_timepoints = len(times)
    #         n_components = pcs.shape[1]
    #         mov_i = self.get_sub_mov(i) - 1 # movement index
    #
    #         # 2. Build the target regressor (the "ideal" signal)
    #         target_regressor = np.zeros(n_timepoints)
    #         for t_stim, b_lag in zip(stimulus_times[mov_i], bold_lag_seconds[mov_i]):
    #             target_time = t_stim + b_lag
    #             if times[0] <= target_time <= times[-1]:
    #                 idx = np.argmin(np.abs(times - target_time))
    #                 target_regressor[idx] = 1.0  # Create a "spike" at the expected peak
    #
    #         # 3. Find and sum all relevant PCs (skip PC0)
    #         final_signal = np.zeros(n_timepoints)
    #         rep_pcs_names_i = ""
    #         all_pc_signals = F @ pcs  # (N, n_comp)
    #
    #         for j in range(1, n_components):
    #             pc_j_signal = all_pc_signals[:, j]
    #             # Calculate distance for original signal
    #             dist_orig = dtw.distance(pc_j_signal, target_regressor)
    #             # Calculate distance for inverted signal
    #             dist_inv = dtw.distance(-pc_j_signal, target_regressor)
    #             self.logger.info("Subject %d, PC %s: DTW dist orig=%.4f, dist inv=%.4f, thre=%d", i, pcs.pc_name(j), dist_orig, dist_inv, distance_threshold)
    #             min_dist = min(dist_orig, dist_inv)
    #             pc_j_name = pcs.pc_name(j)
    #             # If distance is low, add it (and fix sign)
    #             if min_dist < distance_threshold:
    #                 if dist_orig < dist_inv:
    #                     final_signal += pc_j_signal
    #                     rep_pcs_names_i += f"+{pc_j_name}"
    #                 else:
    #                     final_signal -= pc_j_signal  # Add the inverted signal
    #                     rep_pcs_names_i += f"-{pc_j_name}"
    #
    #         combined_signals.append(final_signal)
    #         rep_pcs_names.append(rep_pcs_names_i)
    #         self.logger.info(f"Subject {i} combined PCs: {rep_pcs_names_i}")
    #     return combined_signals, rep_pcs_names
# 0.12, 0.28, lag=8, durs=15
# 0.13, 0.22, lags = [-5.0 -10.0 -6.0 -7.0], [-10.0 -13.0 -13.0 -12.0], durs=27,
# 0.0468, 0.6844, lags = [-5.0 -10.0 -6.0 -7.0], [-10.0 -13.0 0.0 -12.0], durs = [[20.0, 20.0, 25.0, 25.0], [20.0, 25.0, 0.0, 25.0]]
# 0.20, 0.065 ,lags = [-3.0 -10.0 -6.0 -8.0], [-8.0 -13.0 0.0 -13.0], durs = [[27.0, 20.0, 20.0, 20.0], [17.0, 25.0, 0.0, 25.0]]
# 0.19, 0.08, auto lags/durations no pc0. correlation threshold 0.1
# 0.334, 0.0028 , auto lags/durations no pc0 - new version. correlation threshold 0.1
# 0.01,0.87 , auto lags/durations no pc0 - new version. correlation threshold 0.15
# 0.01, 0.87 , auto lags/durations no pc0 - new version. correlation threshold 0.08
# 0.05, 0.62 , auto lags/durations no pc0 - new version. correlation threshold 0.11
# 0.15, 0.18 , auto lags/durations no pc0 - new version. correlation threshold 0.09
