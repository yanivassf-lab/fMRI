from xmlrpc.client import FastParser

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing
import matplotlib.gridspec as gridspec

import os
import sys
import glob
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed, CancelledError

from fmri.pc_similarity import PcSimilarity
from fmri.peaks_similarity import PeaksSimilarity
from fmri.utils import setup_logger, PCS
from fmri.represent_pc import RepresentPC


class ComparePCS:
    def __init__(self, files_path, output_folder, movements, pc_sim_auto: bool,
                 pc_sim_auto_best_similar_pc: bool, pc_sim_auto_weight_similar_pc: int,
                 pc_num_comp: int | None, fix_orientation: bool, peaks_abs: bool, peaks_dist: int, skip_timepoints: int,
                 skip_pc_num: list[int], combine_pcs: bool, combine_pcs_stimulus_times: list[list[float]],
                 combine_pcs_bold_lag_seconds: list[list[float]] | None,
                 combine_pcs_bold_dur_seconds: list[list[float]] | None,
                 combine_pcs_correlation_threshold: float,
                 peaks_all_cpus: bool, logger: logging.Logger = None):
        """
        Initialize the ComparePCS class.

        Parameters:
        - files_path (str): Path to the directory containing subject subdirectories.
        - output_folder (str): Path to the output folder for logs and figures.
        - movements (list[int]): List of movement identifiers to compare.
        - pc_sim_auto (bool): If True, find the best representative PC for each sample according to similarity to other samples's PCs.
        - pc_sim_auto_best_similar_pc (bool): If True, only the best matching PC from each other file is considered when calculate similarity, otherwise all PCs are averaged (relevant only if pc_sim_auto is True).
        - pc_sim_auto_weight_similar_pc (int): Exponent to which the absolute correlation values are raised. Higher values give more weight to stronger correlations (relevant only if pc_sim_auto is True).
        - pc_num_comp (int): Number of PCs to compare (relevant only if pc_sim_auto is False).
        - fix_orientation (bool): If True, corrects for signal orientation before peaks similarity calculation.
        - peaks_abs (bool): If True, uses absolute peak heights for similarity calculation.
        - peaks_dist (int): Minimum distance between peaks (in timepoints).
        - skip_timepoints (int): Number of timepoints to skip at the beginning and end when comparing peaks.
        - skip_pc_num (list[int]): If not [], exclude the PCs in the list from the analysis.
        - combine_pcs (bool): If True, combine PCs based on stimulus times and BOLD lag (overrides pc_sim_auto and pc_num_comp).
        - combine_pcs_stimulus_times (list[list[float]]): List of stimulus times for each movement (in seconds).
        - combine_pcs_bold_lag_seconds (list[list[float]]|None): List of BOLD lag times for each movement (in seconds).
        - combine_pcs_bold_dur_seconds (list[list[float]]|None): List of BOLD duration times for each movement (in seconds).
        - combine_pcs_correlation_threshold (float): DTW distance threshold for combining PCs.
        - peaks_all_cpus (bool): If True, use all available CPUs for DTW distance matrix calculation.
        - logger: Logger object for logging information.
        """
        self.files_path = files_path
        self.output_folder = output_folder
        self.movements = movements
        self.skip_timepoints = skip_timepoints
        self.pc_sim_auto = pc_sim_auto
        self.pc_sim_auto_best_similar_pc = pc_sim_auto_best_similar_pc
        self.pc_sim_auto_weight_similar_pc = pc_sim_auto_weight_similar_pc
        self.pc_num_comp = pc_num_comp
        self.combine_pcs = combine_pcs
        self.combine_pcs_stimulus_times = combine_pcs_stimulus_times
        self.combine_pcs_bold_lag_seconds = combine_pcs_bold_lag_seconds
        self.combine_pcs_bold_dur_seconds = combine_pcs_bold_dur_seconds
        self.combine_pcs_correlation_threshold = combine_pcs_correlation_threshold
        self.fix_orientation = fix_orientation
        self.skip_pc_num = skip_pc_num
        self.peaks_abs = peaks_abs
        self.peaks_dist = peaks_dist
        self.peaks_all_cpus = peaks_all_cpus
        self.params_comb = self.get_list_of_params(files_path)
        self.file_list = self.get_list_of_files(files_path, movements)
        self.n_subs = int(len(self.file_list) / len(movements))
        self.n_movs = len(movements)
        self.n_files = len(self.file_list)
        self.calc_pc_score = True  # If the length of PCs is not equal in all samples it became to False
        self.logger = logger
        self.logger.info(
            f"Found {len(self.file_list)} files for {self.n_movs} movements with {self.n_subs} subjects per movement:\n {'\n '.join(os.path.basename(s) for s in self.file_list)}")

    def load_pcs_and_signals(self, params: str):
        """
        Load principal components and signals for all subjects and movements.

        Args:
            params: Parameter combination string.

        Returns: Tuple of lists: (x_norm_all, pcs_list, F_list)
            - pcs_list: List of principal component matrices for each subject/movement.
            - F_list: List of signal matrices for each subject/movement.
        """
        pcs_list = []
        F_list = []
        times_list = []
        for i, sub in enumerate(self.file_list):
            store_data_file = os.path.join(sub, params, f"eigvecs_eigval_F.npz")
            try:
                data = np.load(store_data_file)
                eigvecs_sorted = data['eigvecs_sorted']
                # eigvals_sorted = data['eigvals_sorted']
                F = data['F']
                times = data['times']
            except Exception as e:
                raise Exception(f"Error reading file {store_data_file}: {e}")

            eigvecs_obj = PCS(eigvecs_sorted, self.skip_pc_num)
            pcs_list.append(eigvecs_obj)
            F_list.append(F)
            times_list.append(times)
            self.logger.info(
                f"Loaded sub-{self.get_sub_name(i)}_mov-{self.get_sub_mov(i)} with {eigvecs_obj.shape[1]} PCs, {eigvecs_obj.shape[0]} basis functions and {F.shape[0]} time points.")

        pcs_lengths = set([pcs_list[i].shape[0] for i in range(len(pcs_list))])
        if len(pcs_lengths) > 1:
            self.calc_pc_score = False
            self.logger.info("There are different sizes of PCs, cannot calculate score.")
        return pcs_list, F_list, times_list

    def load_original_signals(self):
        """
        Load original signals for all subjects and movements.

        Returns: List of original signals for each subject/movement.
        """
        org_signals = []
        for i, sub in enumerate(self.file_list):
            org_sig_file = os.path.join(sub, self.params_comb[0], f"original_averaged_signal_intensity.txt")
            try:
                with open(org_sig_file, 'r') as f:
                    lines = f.readlines()
                    signal_y = np.array(lines[3].strip().split(), dtype=float)
            except Exception as e:
                raise Exception(f"Error reading file {org_sig_file}: {e}")
            self.logger.info(
                f"Loaded original signal for sub-{self.get_sub_name(i)}_mov-{self.get_sub_mov(i)} with {len(signal_y)} time points.")
            org_signals.append(signal_y)
        return org_signals

    def extract_values_from_peaks(self, signals, peaks_sim, i):
        """Extract peak similarity values for subject/movement at index i."""
        peaks_height_i = peaks_sim.peaks_height[i]
        signal_i = signals[i]
        peaks_idx_i = peaks_sim.peaks_idx[i]
        revereted_i = peaks_sim.reverted[i]
        if revereted_i:
            signal_i *= -1
            peaks_height_i *= -1
        return signal_i, peaks_idx_i, peaks_height_i, revereted_i

    def plot_peaks_similarity(self, peaks_sim, rep_pcs_names_signals, axis, rep_pcs_names=None):
        """Plot peaks similarity for all subjects and movements."""
        for i in range(self.n_files):
            signal_i, peaks_idx_i, peaks_height_i, revereted_i = self.extract_values_from_peaks(
                rep_pcs_names_signals, peaks_sim, i)
            sub_name_i = self.get_sub_name(i)
            # i.e. for 3 movements with 4 subject per movement: 0,1,2,3 -> movement 1; 4,5,6,7 -> movement 2; 8,9,10,11 -> movement 3
            sub_movement = self.get_sub_mov(i)
            sub_num = self.get_sub_num(i)
            pc_text = f", pc: {rep_pcs_names[i]} " if rep_pcs_names is not None else ""
            axis[sub_num].plot(np.arange(len(signal_i)), signal_i,
                               label=f"mov-{sub_movement}{pc_text} {', Reverted Signal' if revereted_i else ', Original Signal'}")
            axis[sub_num].scatter(peaks_idx_i, peaks_height_i, color='red', s=10, zorder=3)  # mark peaks
            if sub_movement == 1:
                axis[sub_num].set_title(f"subj: {sub_name_i}")
                axis[sub_num].set_ylabel('Signal Intensity')
                if pc_text != "":
                    axis[sub_num].set_ylim(-0.2, 0.2)
            axis[sub_num].legend()

            axis[sub_num + self.n_subs].plot(np.arange(len(signal_i)), np.abs(signal_i),
                                             label=f"mov-{sub_movement}{pc_text}{', Reverted Signal' if revereted_i else ', Original Signal'}")
            axis[sub_num + self.n_subs].scatter(peaks_idx_i, np.abs(peaks_height_i), color='red', s=10,
                                                zorder=3)  # mark peaks
            if sub_movement == 1:
                axis[sub_num + self.n_subs].set_title(f"subj: {sub_name_i} abs signal")
                axis[sub_num + self.n_subs].set_ylabel('Signal Intensity')
                if pc_text != "":
                    axis[sub_num + self.n_subs].set_ylim(0.0, 0.4)
            axis[sub_num + self.n_subs].legend()

    def plot_similarity_matrices(self, sim_matrix, label, title, axis):
        np.fill_diagonal(sim_matrix, np.nan)
        im = axis.imshow(sim_matrix, aspect='equal', cmap='viridis')
        plt.colorbar(im, ax=axis, label=label)
        axis.set_title(title)

        # --- build labels in the same order as file_list (the order used to build sim_matrix) ---
        labels = []
        for i in range(self.n_files):
            labels.append(f"sub-{self.get_sub_name(i)}_mov-{self.get_sub_mov(i)}")

        # --- set ticks on the matrix axis (ax2) ---
        n = sim_matrix.shape[0]
        axis.set_xticks(np.arange(n))
        axis.set_yticks(np.arange(n))
        axis.set_xticklabels(labels, rotation=45, ha='right')
        axis.set_yticklabels(labels, rotation=45)
        axis.set_xlabel('Signal Index')
        axis.set_ylabel('Signal Index')

    def save_plot_data_to_txt(self, params, rep_pcs_names_signals, sim_objects_names, sim_objects, rep_pcs_names=None):
        txt_file = os.path.join(self.output_folder, f"similarity_{params}.txt")
        with open(txt_file, "w") as f:
            f.write(f"Parameters: {params}\n\n")

            # --- save peaks info ---
            f.write("Peaks similarity per subject and movement:\n")
            for i in range(self.n_files):
                signal_i, peaks_idx_i, peaks_height_i, revereted_i = self.extract_values_from_peaks(
                    rep_pcs_names_signals, sim_objects[0], i)
                sub_name_i = self.get_sub_name(i)
                sub_movement = self.get_sub_mov(i)
                pc_text = f", pc: {rep_pcs_names[i]}" if rep_pcs_names is not None else ""
                f.write(
                    f"Subject {sub_name_i}, movement {sub_movement}{pc_text}, {'Reverted' if revereted_i else 'Original'}:\n")
                f.write("Signal: " + ", ".join([f"{v:.4f}" for v in signal_i]) + "\n")
                f.write("Peaks idx: " + ", ".join(map(str, peaks_idx_i)) + "\n")
                f.write("Peaks height: " + ", ".join([f"{v:.4f}" for v in peaks_height_i]) + "\n\n")

            # --- save similarity matrices ---
            for sim_obj_name, sim_obj in zip(sim_objects_names, sim_objects):
                f.write(f"Similarity matrix: {sim_obj_name}\n")
                f.write("Matrix:\n")
                np.savetxt(f, sim_obj.sim_matrix, fmt="%.4f")
                f.write(f"Average score: {sim_obj.score:.4f}\n")
                f.write(f"Average P-value: {1 - sim_obj.matrix_op_pval:.4f}\n\n")

        self.logger.info(f"Saved plot data to {txt_file}")

    def plot_params(self, params, sim_objects_names, sim_objects, rep_pcs_names_signals, rep_pcs_names=None):
        """Plot results for a given parameter combination."""
        fig = plt.figure(figsize=(35, 1.5 * len(self.file_list)))
        gs = gridspec.GridSpec(len(self.file_list), 2, width_ratios=[2, 1])  # 2x wider second column
        axes_left = [fig.add_subplot(gs[i, 0]) for i in range(self.n_files)]
        gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 1])
        if rep_pcs_names is not None:
            axes_right = [fig.add_subplot(gs_right[i, 0]) for i in range(2)]
        else:  # original signals
            axes_right = [fig.add_subplot(gs_right[0, 0])]

        # --- plot peaks similarity ---
        self.plot_peaks_similarity(sim_objects[0], rep_pcs_names_signals, axes_left, rep_pcs_names)
        # --- plot similarity matrices ---
        for i, sim_obj_name, sim_obj in zip(range(len(sim_objects)), sim_objects_names, sim_objects):
            self.plot_similarity_matrices(sim_obj.sim_matrix, f'Similarity {sim_obj_name}',
                                          f'Similarity Matrix {sim_obj_name}', axes_right[i])
        # --- finalize and save figure ---
        title = "\n".join([f"Average Spearman correlation of {name} matrix: "
                           f"{sim_obj.score:.4f}, average P-value: {1 - sim_obj.matrix_op_pval:.4f}"
                           for name, sim_obj in zip(sim_objects_names, sim_objects)])
        fig.suptitle(f"Params {params}\n\n" + title + "\n", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
        fig.savefig(os.path.join(self.output_folder, f"similarity_{params}.png"))
        plt.close(fig)
        self.save_plot_data_to_txt(params, rep_pcs_names_signals, sim_objects_names, sim_objects, rep_pcs_names)

    def process_single_combination(self, params, comb_num):
        """
        Process a single parameter combination.

        Returns:
            tuple: (pc_sim.score, pc_sim.matrix_op_pval, peaks_sim.score, peaks_sim.matrix_op_pval, params)
        """
        # --- load pcs and signals ---
        pcs_list, F_list, times_list = self.load_pcs_and_signals(params)
        # --- compare pcs ---
        self.logger.info(f"Comparing pcs of {params} ({comb_num + 1} of {len(self.params_comb)} combinations)")
        if self.calc_pc_score:
            pc_sim = PcSimilarity(pcs_list, self.n_subs, self.n_movs, logger=self.logger)
            # --- calculate pc similarity score ---
            pc_sim.calculate_score()
        # --- get main pcs ---
        if self.pc_sim_auto or self.combine_pcs:
            represent_pc = RepresentPC(pcs_list, F_list, self.n_subs, self.n_movs, logger=self.logger)
            if self.pc_sim_auto:
                rep_pcs_names_signals, rep_pcs_names = represent_pc.find_representative_pcs(
                    self.pc_sim_auto_best_similar_pc,
                    self.pc_sim_auto_weight_similar_pc)
            elif self.combine_pcs:
                rep_pcs_names_signals, rep_pcs_names = represent_pc.combine_pcs(self.files_path, params, F_list,
                                                                                times_list,
                                                                                self.combine_pcs_stimulus_times,
                                                                                self.combine_pcs_bold_lag_seconds,
                                                                                self.combine_pcs_bold_dur_seconds,
                                                                                self.combine_pcs_correlation_threshold)
        else:
            rep_pcs = np.zeros(len(self.file_list), dtype=int) + self.pc_num_comp
            rep_pcs_names_signals = RepresentPC.get_signals_from_seleted_pcs(pcs_list, F_list, rep_pcs)
            rep_pcs_names = [pcs_list[0].pc_name(self.pc_num_comp)] * len(self.file_list)

        # --- get signals from selected pcs ---

        # --- calculate peaks similarity score on selected signals ---
        self.logger.info(f"Comparing peaks of {params} ({comb_num + 1} of {len(self.params_comb)} combinations)")
        peaks_sim = PeaksSimilarity(rep_pcs_names_signals, self.n_subs, self.n_movs,
                                    fix_orientation=self.fix_orientation,
                                    peaks_abs=self.peaks_abs, peaks_dist=self.peaks_dist,
                                    skip_timepoints=self.skip_timepoints, peaks_all_cpus=self.peaks_all_cpus,
                                    logger=self.logger)
        # -- calculate peaks similarity score ---
        peaks_sim.calculate_score()
        # --- plot results ---
        self.logger.info(f"Plotting similarity for {params} ({comb_num} of {len(self.params_comb)} combinations)")
        if self.calc_pc_score:
            sim_objects_names, sim_objects = ['Peaks Similarity', 'PCs Similarity'], [peaks_sim, pc_sim]
        else:
            sim_objects_names, sim_objects = ['Peaks Similarity'], [peaks_sim]
        self.plot_params(params, sim_objects_names, sim_objects, rep_pcs_names_signals, rep_pcs_names)
        # returns self.calc_pc_score for parallel run
        return self.calc_pc_score, pc_sim.score if self.calc_pc_score else None, pc_sim.matrix_op_pval if self.calc_pc_score else None, peaks_sim.score, peaks_sim.matrix_op_pval, params

    def process_original_signals(self):
        """
        Process and plot similarity for original signals across all subjects and movements.
        """
        self.logger.info("Processing original signals for comparison.")
        # --- load original signals ---
        org_signals = self.load_original_signals()
        # --- calculate peaks similarity score on original signals ---
        self.logger.info("Comparing peaks of original signals.")
        # The orientation correction is not relevant when comparing original signals
        peaks_sim = PeaksSimilarity(org_signals, self.n_subs, self.n_movs, fix_orientation=False,
                                    peaks_abs=self.peaks_abs, peaks_dist=self.peaks_dist,
                                    skip_timepoints=self.skip_timepoints, peaks_all_cpus=self.peaks_all_cpus,
                                    logger=self.logger)
        # -- calculate peaks similarity score ---
        peaks_sim.calculate_score()
        # --- plot results ---
        self.logger.info("Plotting similarity for original signals.")
        sim_objects_names, sim_objects = ['Peaks Similarity'], [peaks_sim]
        self.plot_params('original_signals', sim_objects_names, sim_objects, org_signals)

    def compare(self, num_scores=10, max_workers=0):
        """
        Compare principal components across different movements and subjects.

        - num_scores (int): Number of top scores to keep for each movement and subject.
        - max_workers (int): Maximum number of parallel workers. if 1, runs sequentially.
        """
        self.process_original_signals()

        pc_best_scores = np.array([-np.inf] * num_scores)
        pc_best_scores_params = ['.'] * num_scores
        pc_best_op_pvals = np.array([-np.inf] * num_scores)
        pc_best_op_pvals_params = ['.'] * num_scores
        peaks_best_scores = np.array([-np.inf] * num_scores)
        peaks_best_scores_params = ['.'] * num_scores
        peaks_best_op_pvals = np.array([-np.inf] * num_scores)
        peaks_best_op_pvals_params = ['.'] * num_scores

        # Handle sequential execution (avoid threading overhead when max_workers=1)
        if max_workers == 1:
            self.logger.info(f"Running sequentially without parallel processing.")
            for comb_num, params in enumerate(self.params_comb):
                try:
                    _, pc_score, pc_pval, peaks_score, peaks_pval, _ = self.process_single_combination(params, comb_num)
                    # --- update the parameters with the best scores ---
                    if self.calc_pc_score:
                        self.update_best_scores(pc_best_scores, pc_best_scores_params, abs(pc_score), params)
                        self.update_best_scores(pc_best_op_pvals, pc_best_op_pvals_params, pc_pval, params)
                    self.update_best_scores(peaks_best_scores, peaks_best_scores_params, abs(peaks_score), params)
                    self.update_best_scores(peaks_best_op_pvals, peaks_best_op_pvals_params, peaks_pval, params)
                except Exception as exc:
                    self.logger.exception(f'Parameter combination {params} generated an exception: {exc}')
        else:
            self.logger.info(f"Running parallel processing with {max_workers} workers.")
            # Parallel processing of parameter combinations
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_params = {
                    executor.submit(self.process_single_combination, params, comb_num): params
                    for comb_num, params in enumerate(self.params_comb)
                }

                # Process completed tasks as they finish
                for future in as_completed(future_to_params):
                    params = future_to_params[future]
                    try:
                        # self.calc_pc_score is updated in child process
                        calc_pc_score, pc_score, pc_pval, peaks_score, peaks_pval, _ = future.result()
                        # --- update the parameters with the best scores (in absolute value) ---
                        if calc_pc_score:
                            self.update_best_scores(pc_best_scores, pc_best_scores_params, abs(pc_score), params)
                            self.update_best_scores(pc_best_op_pvals, pc_best_op_pvals_params, pc_pval, params)
                        self.update_best_scores(peaks_best_scores, peaks_best_scores_params, abs(peaks_score), params)
                        self.update_best_scores(peaks_best_op_pvals, peaks_best_op_pvals_params, peaks_pval, params)
                    except KeyboardInterrupt:
                        self.logger.info("KeyboardInterrupt caught, shutting down executor...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    except CancelledError:
                        self.logger.info("Some tasks were cancelled due to shutdown")
                    except Exception as exc:
                        self.logger.exception(f'Parameter combination {params} generated an exception: {exc}')

        # --- print the parameters with the best scores ---
        self.print_best_scores(pc_best_scores, pc_best_scores_params, 'pc score')
        self.print_best_scores(pc_best_op_pvals, pc_best_op_pvals_params, 'P-value', oposite=True)
        self.print_best_scores(peaks_best_scores, peaks_best_scores_params, 'peaks score')
        self.print_best_scores(peaks_best_op_pvals, peaks_best_op_pvals_params, 'peaks P-value', oposite=True)

    def update_best_scores(self, best_scores: np.ndarray, best_params: list[str], new_score: float, params: str):
        """Update the list of best scores and corresponding parameters."""
        if new_score > np.min(best_scores):
            min_idx = np.argmin(best_scores)
            best_scores[min_idx] = new_score
            best_params[min_idx] = params

    def print_best_scores(self, best_scores: np.ndarray, best_params: list[str], score_type: str,
                          oposite: bool = False):
        """Print the best scores and their corresponding parameters."""
        sorted_idx = np.argsort(best_scores)[::-1]
        self.logger.info(
            f"Best {score_type}: {','.join([f'{1 - s if oposite else s:.4f}' for s in best_scores[sorted_idx]])} with params {','.join([best_params[i] for i in sorted_idx])}"
        )

    def get_list_of_params(self, files_path: str):
        """Get a list of unique parameter combinations from the directory structure."""
        subs = glob.glob(os.path.join(files_path, f'sub-*'))[0]
        params_comb = [os.path.basename(p) for p in glob.glob(os.path.join(subs, '*'))]
        return sorted(params_comb)

    def get_list_of_files(self, files_path: str, movements: list[int]):
        """
        Get a sorted list of files for each movement.
        Args:
            files_path: Path to the directory containing subject subdirectories.
            movements: List of movement identifiers to compare.

        Returns: A sorted list of file paths for each movement, sorted by movement number and subject ID.

        """
        subs = glob.glob(os.path.join(files_path, f'sub-*movement*'))
        if not subs:
            raise FileNotFoundError(f"No input files found in '{files_path}' matching 'sub-*movement*'.")
        sorted_subs = []
        for mov in movements:
            if not any(f'movement{mov}' in s for s in subs):
                raise ValueError(f"No input files found for movement {mov} in '{files_path}'.")
            sorted_subs.extend(sorted([f for f in subs if f'movement{mov}' in f]))
        return sorted_subs

    def get_sub_name(self, i: int) -> str:
        """Get the subject name from the file path at index i."""
        file_path = self.file_list[i]
        base_name = os.path.basename(file_path)
        sub_name = base_name.split('_')[0].split('-')[1]
        return sub_name

    def get_sub_mov(self, i: int) -> int:
        """Get the movement number of the i-th file (starts from 1)."""
        return int(i / self.n_subs) + 1

    def get_sub_num(self, i: int) -> int:
        """Get the subject number within its movement from the file path at index i."""
        return int(i % self.n_subs)


def main():
    parser = argparse.ArgumentParser(description="Compare peaks between movements")
    parser.add_argument("--files-path", type=str, required=True, help="Path to the subjects directory")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--movements", type=int, nargs='+', default=[1, 2],
                        help="List of movements to compare (from 1 to 9)")
    parser.add_argument("--num-scores", type=int, default=10,
                        help="Number of top scores to keep for each movement and subject")
    parser.add_argument('--pc-sim-auto', action='store_true',
                        help="If set, find the best representative PC for each sample according to similarity to other samples's PCs.")
    parser.add_argument('--pc-sim-auto-best-similar-pc', action='store_true',
                        help="If set, only the best matching PC from each other file is considered when calculate similarity, otherwise all PCs are averaged (relevant only if pc_sim_auto is True).")
    parser.add_argument('--pc-sim-auto-weight-similar-pc', type=int, default=2,
                        help="Exponent to which the absolute correlation values are raised. Higher values give more weight to stronger correlations (relevant only if pc_sim_auto is True).")
    parser.add_argument('--pc-num-comp', type=int, default=None,
                        help="PC to compare - starting from 0 (relevant only if pc_sim_auto is False).")
    parser.add_argument('--skip-pc-num', type=int, nargs='+', default=[],
                        help="List of number of PCs to exclude from the entire analysis (starting from 0). If set to None, all components are used.")
    parser.add_argument('--fix-orientation', action='store_true',
                        help="If set, corrects for signal orientation before peaks similarity calculation.")
    parser.add_argument('--peaks-abs', action='store_true',
                        help="If set, uses absolute peak heights for similarity calculation.")
    parser.add_argument('--peaks-dist', type=int, default=5, help="Minimum distance between peaks (in timepoints).")
    parser.add_argument("--skip-timepoints", type=int, default=100,
                        help="Number of timepoints to skip at the beginning and end when comparing peaks.")
    parser.add_argument('--combine-pcs', action='store_true',
                        help="If set, combine PCs based on stimulus times and BOLD lag (overrides pc_sim_auto and pc_num_comp).")
    parser.add_argument('--combine-pcs-stimulus-times', type=float, nargs='+', action='append', default=[],
                        help="List of stimulus times for each movement (in seconds). Provide a list of lists, one per movement.")
    parser.add_argument('--combine-pcs-bold-lag-seconds', type=float, nargs='+', action='append', default=[],
                        help="provide a list of BOLD lag values per movement; repeat this flag once for each movement. If not set, these values are determined automatically (recommended) (action='append', default=[]])")
    parser.add_argument('--combine-pcs-bold-dur-seconds', type=float, nargs='+', action='append', default=[],
                        help="provide a list of BOLD duration values per movement; repeat this flag once for each movement. If not set, these values are determined automatically (recommended) (action='append', default=[]])")
    parser.add_argument('--combine-pcs-correlation-threshold', type=float, default=0.1,
                        help="DTW distance threshold for combining PCs.")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Maximum number of parallel workers. 0=auto (CPU count), 1=sequential")
    parser.add_argument("--peaks-all-cpus", action='store_true',
                        help="If set, use all available CPUs for DTW distance matrix calculation.")
    args = parser.parse_args()


    if not all(1 <= m <= 9 for m in args.movements):
        raise ValueError("Movements should be between 1 and 9.")
    if args.combine_pcs:
        if args.pc_sim_auto or args.pc_num_comp is not None:
            raise ValueError("When --combine-pcs is set, --pc-sim-auto and pc-num-comp cannot be set.")
        if not args.combine_pcs_stimulus_times:
            raise ValueError("--pc-num-comp requires --combine-pcs-stimulus-times to be set.")
        if args.combine_pcs_bold_lag_seconds or args.combine_pcs_bold_dur_seconds:
            n_mov = len(args.combine_pcs_stimulus_times)
            if not (len(args.combine_pcs_bold_lag_seconds) == len(args.combine_pcs_bold_dur_seconds) == len(
                args.combine_pcs_stimulus_times) == n_mov):
                raise ValueError(
                    f"--combine-pcs-stimulus-times and --combine-pcs-bold-lag-seconds and --combine-pcs-bold-dur-seconds must match number of movements ({n_mov})")
            for i in range(n_mov):
                if not (len(args.combine_pcs_bold_lag_seconds[i]) == len(args.combine_pcs_bold_dur_seconds[i]) == len(
                    args.combine_pcs_stimulus_times[i])):
                    raise ValueError(
                        f"--combine-pcs-bold-lag-seconds and --combine-pcs-bold-dur-seconds and --combine-pcs-stimulus-times must have the same length for each movement.")
        else:
            args.combine_pcs_bold_lag_seconds = None
            args.combine_pcs_bold_dur_seconds = None
    if args.pc_sim_auto:
        if args.pc_num_comp is not None or args.combine_pcs:
            raise ValueError("When --pc-sim-auto is set, --pc-num-comp cannot and --combine-pcs cannot be specified.")
    if args.pc_num_comp is not None:
        if args.pc_sim_auto or args.combine_pcs:
            raise ValueError("When --pc-num-comp is set, --pc-sim-auto and --combine-pcs cannot be specified.")
    if args.pc_num_comp is None and not args.pc_sim_auto and not args.combine_pcs:
        raise ValueError("Either --pc-sim-auto, --pc-num-comp, or --combine-pcs must be specified.")

    if not os.path.exists(args.files_path):
        raise FileNotFoundError(
            f"Files folder '{args.files_path}' does not exist."
        )
    if os.path.exists(args.output_folder):
        raise FileExistsError(f"Output folder '{args.output_folder}' already exists.")
    else:
        os.makedirs(args.output_folder)

    args.skip_pc_num = sorted(args.skip_pc_num)
    setup_logger(output_folder=args.output_folder, file_name="compare_peaks_log.txt", loger_name="compare_peaks_logger",
                 log_level=logging.INFO)
    logger = logging.getLogger("compare_peaks_logger")
    logger.info(f"Command line: {' '.join(sys.argv)}")
    effective_workers = os.cpu_count() if args.max_workers == 0 else int(args.max_workers)
    logger.info(f"Run with {effective_workers} cpu's.")
    compare_pcs = ComparePCS(args.files_path, args.output_folder, args.movements,
                             args.pc_sim_auto, args.pc_sim_auto_best_similar_pc,
                             args.pc_sim_auto_weight_similar_pc, args.pc_num_comp,
                             args.fix_orientation, args.peaks_abs, args.peaks_dist, args.skip_timepoints,
                             args.skip_pc_num, args.combine_pcs, args.combine_pcs_stimulus_times,
                             args.combine_pcs_bold_lag_seconds, args.combine_pcs_bold_dur_seconds,
                             args.combine_pcs_correlation_threshold, args.peaks_all_cpus, logger)
    compare_pcs.compare(num_scores=args.num_scores, max_workers=effective_workers)


if __name__ == "__main__":
    main()

# Instructions to run the script:
# 1. The input files should be organized in the following structure:
#    /path/to/subjects/
#        ├── sub-01_movement1/
#        │   ├── no_penalty_nb100/
#        │   │   ├── eigvecs_eigval_F.npz
#        │   │   ├── original_averaged_signal_intensity.txt
#        │   │   ├── temporal_profile_pc_0.txt
#        │   │   ├── temporal_profile_pc_1.txt
#        │   │   └── ...
#        │   ├── p0_u1_t1e-6_l-4_0_nb100/
#        │   │   ├── eigvecs_eigval_F.npz
#        │   │   ├── original_averaged_signal_intensity.txt
#        │   │   ├── temporal_profile_pc_0.txt
#        │   │   ├── temporal_profile_pc_1.txt
#        │   │   └── ...
#        │   └── ...
#        ├── sub-01_movement2/
#        │   ├── no_penalty_nb100/
#        │   │   ├── eigvecs_eigval_F.npz
#        │   │   ├── original_averaged_signal_intensity.txt
#        │   │   ├── temporal_profile_pc_0.txt
#        │   │   ├── temporal_profile_pc_1.txt
#        │   │   └── ...
#        │   ├── p0_u1_t1e-6_l-4_0_nb100/
#        │   │   ├── eigvecs_eigval_F.npz
#        │   │   ├── original_averaged_signal_intensity.txt
#        │   │   ├── temporal_profile_pc_0.txt
#        │   │   ├── temporal_profile_pc_1.txt
#        │   │   └── ...
#        │   └── ...
#        └── ...
#
# 2. Run the script from the command line:
# compare-pcs --files-path /path/to/subjects/ --output-folder /path/to/non-exists/output/folder --movements 1 2 --num-scores 10 --max-workers 1
#   - Replace /path/to/subjects/ with the actual path to the directory containing subject subdirectories.
#   - Replace /path/to/non-exists/output/folder with the desired output folder path (it should not exist prior to running).
# 3. The output will be saved in the specified output folder, including logs and figures for each parameter combination and principal component.
