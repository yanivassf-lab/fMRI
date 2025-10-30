import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import resample

matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing
import matplotlib.gridspec as gridspec

import os
import sys
import glob
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from fmri.pc_similarity import PcSimilarity
from fmri.peaks_similarity import PeaksSimilarity
from fmri.utils import setup_logger



class ComparePCS:
    def __init__(self, files_path, output_folder, movements, skip_timepoints, logger):
        """
        Initialize the ComparePCS class.

        Parameters:
        - files_path (str): Path to the directory containing subject subdirectories.
        - output_folder (str): Path to the output folder for logs and figures.
        - movements (list[int]): List of movement identifiers to compare.
        - skip_timepoints (int): Number of timepoints to skip at the beginning and end when comparing peaks.
        - logger: Logger object for logging information.
        """
        self.files_path = files_path
        self.output_folder = output_folder
        self.movements = movements
        self.skip_timepoints = skip_timepoints
        self.params_comb = self.get_list_of_params(files_path)
        self.file_list = self.get_list_of_files(files_path, movements)
        self.n_subs = int(len(self.file_list) / len(movements))
        self.n_movs = len(movements)
        self.n_files = len(self.file_list)
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
        for i, sub in enumerate(self.file_list):
            store_data_file = os.path.join(sub, params, f"eigvecs_eigval_F.npz")
            try:
                data = np.load(store_data_file)
                eigvecs_sorted = data['a']
                # eigvals_sorted = data['b']
                F = data['c']
            except Exception as e:
                raise Exception(f"Error reading file {store_data_file}: {e}")

            pcs_list.append(eigvecs_sorted)
            F_list.append(F)
            n_basis, n_pcs = eigvecs_sorted.shape
            n_times = F.shape[0]
            self.logger.info(
                f"Loaded sub-{self.get_sub_name(i)}_mov-{self.get_sub_mov(i)} with {n_pcs} PCs, {n_basis} basis functions and {n_times} time points.")
        return pcs_list, F_list

    def load_original_signals(self):
        """
        Load original signals for all subjects and movements.

        Returns: List of original signals for each subject/movement.
        """
        org_signals = []
        average_len = 0
        for i, sub in enumerate(self.file_list):
            org_sig_file = os.path.join(sub, self.params_comb[0], f"original_averaged_signal_intensity.txt")
            try:
                with open(org_sig_file, 'r') as f:
                    lines = f.readlines()
                    signal_y = np.array(lines[3].strip().split(), dtype=float)
            except Exception as e:
                raise Exception(f"Error reading file {org_sig_file}: {e}")
            n_times = len(signal_y)
            average_len += n_times
            self.logger.info(
                f"Loaded original signal for sub-{self.get_sub_name(i)}_mov-{self.get_sub_mov(i)} with {n_times} time points.")
            org_signals.append(signal_y)

        average_len = int(average_len / self.n_files)
        self.logger.info(f"Resampling original signals to average length of {average_len} time points.")
        for i in range(self.n_files):
            org_signals[i] = resample(org_signals[i], average_len)
        return org_signals

    def get_signals_from_seleted_pcs(self, pcs_list, F_list, main_pcs):
        """
        Get signals reconstructed from selected principal components.

        Args:
            pcs_list: list of principal component matrices for each subject/movement.
            F_list:   list of signal matrices for each subject/movement.
            main_pcs: list of selected principal component indices for each subject/movement.

        Returns: List of reconstructed signals for each subject/movement.
        """
        signals = []
        average_len = 0
        for i in range(self.n_files):
            pc_i = pcs_list[i][:, main_pcs[i]]
            signal_i  = F_list[i] @ pc_i
            average_len += signal_i.shape[0]
            signals.append(signal_i)

        average_len = int(average_len / self.n_files)
        self.logger.info(f"Resampling signals to average length of {average_len} time points.")
        for i in range(self.n_files):
            signals[i] = resample(signals[i], average_len)

        return signals


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

    def plot_peaks_similarity(self, peaks_sim, selected_signals, axis, main_pcs=None):
        """Plot peaks similarity for all subjects and movements."""
        for i in range(self.n_files):
            signal_i, peaks_idx_i, peaks_height_i, revereted_i = self.extract_values_from_peaks(
                selected_signals, peaks_sim, i)
            sub_name_i = self.get_sub_name(i)
            # i.e. for 3 movements with 4 subject per movement: 0,1,2,3 -> movement 1; 4,5,6,7 -> movement 2; 8,9,10,11 -> movement 3
            sub_movement = self.get_sub_mov(i)
            sub_num = self.get_sub_num(i)
            pc_text = f", pc: {main_pcs[i]} " if main_pcs is not None else ""
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

    def plot_params(self, params, sim_objects_names, sim_objects, selected_signals, main_pcs=None):
        """Plot results for a given parameter combination."""
        fig = plt.figure(figsize=(35, 1.5 * len(self.file_list)))
        gs = gridspec.GridSpec(len(self.file_list), 2, width_ratios=[2, 1])  # 2x wider second column
        axes_left = [fig.add_subplot(gs[i, 0]) for i in range(self.n_files)]
        gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 1])
        if main_pcs is not None:
            axes_right = [fig.add_subplot(gs_right[i, 0]) for i in range(2)]
        else:
            axes_right = [fig.add_subplot(gs_right[0, 0])]

        # --- plot peaks similarity ---
        self.plot_peaks_similarity(sim_objects[0], selected_signals, axes_left, main_pcs)
        # --- plot similarity matrices ---
        for i, sim_obj_name, sim_obj in zip(range(len(sim_objects)), sim_objects_names, sim_objects):
            self.plot_similarity_matrices(sim_obj.sim_matrix, f'Similarity {sim_obj_name}', f'Similarity Matrix {sim_obj_name}', axes_right[i])
        # --- finalize and save figure ---
        title = "\n".join(
            [f"Average Spearman correlation of {name} matrix: {sim_obj.score:.4f}, average P-value: {1 - sim_obj.matrix_op_pval:.4f}"
             for name, sim_obj in zip(sim_objects_names, sim_objects)])

        fig.suptitle(
            f"Params {params}\n\n" + title + "\n",
            fontsize=16
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
        fig.savefig(os.path.join(self.output_folder, f"similarity_{params}.png"))
        plt.close(fig)

    def process_single_combination(self, params, comb_num):
        """
        Process a single parameter combination.

        Returns:
            tuple: (pc_sim.score, pc_sim.matrix_op_pval, peaks_sim.score, peaks_sim.matrix_op_pval, params)
        """
        # --- load pcs and signals ---
        pcs_list, F_list = self.load_pcs_and_signals(params)
        # --- compare pcs ---
        self.logger.info(f"Comparing pcs of {params} ({comb_num} of {len(self.params_comb)} combinations)")
        pc_sim = PcSimilarity(pcs_list, self.n_subs, self.n_movs)
        # --- get main pcs ---
        main_pcs = pc_sim.compare_and_plot_top_pc_avg_corr()
        # --- calculate pc similarity score ---
        pc_sim.calculate_score()
        # --- get signals from selected pcs ---
        selected_signals = self.get_signals_from_seleted_pcs(pcs_list, F_list, main_pcs)
        # --- calculate peaks similarity score on selected signals ---
        self.logger.info(f"Comparing peaks of {params} ({comb_num} of {len(self.params_comb)} combinations)")
        peaks_sim = PeaksSimilarity(selected_signals, self.n_subs, self.n_movs, fix_orientation=True, skip_timepoints=self.skip_timepoints)
        # -- calculate peaks similarity score ---
        peaks_sim.calculate_score()
        # --- plot results ---
        self.logger.info(f"Plotting similarity for {params} ({comb_num} of {len(self.params_comb)} combinations)")
        sim_objects_names, sim_objects = ['Peaks Similarity', 'PCs Similarity'], [peaks_sim, pc_sim]
        self.plot_params(params, sim_objects_names, sim_objects, selected_signals, main_pcs)

        return pc_sim.score, pc_sim.matrix_op_pval, peaks_sim.score, peaks_sim.matrix_op_pval, params

    def process_original_signals(self):
        """
        Process and plot similarity for original signals across all subjects and movements.
        """
        self.logger.info("Processing original signals for comparison.")
        # --- load original signals ---
        org_signals = self.load_original_signals()
        # --- calculate peaks similarity score on original signals ---
        self.logger.info("Comparing peaks of original signals.")
        peaks_sim = PeaksSimilarity(org_signals, self.n_subs, self.n_movs, fix_orientation=False, skip_timepoints=self.skip_timepoints)
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
                    pc_score, pc_pval, peaks_score, peaks_pval, _ = self.process_single_combination(params, comb_num)
                    # --- update the parameters with the best scores ---
                    self.update_best_scores(pc_best_scores, pc_best_scores_params, abs(pc_score), params)
                    self.update_best_scores(pc_best_op_pvals, pc_best_op_pvals_params, pc_pval, params)
                    self.update_best_scores(peaks_best_scores, peaks_best_scores_params, abs(peaks_score), params)
                    self.update_best_scores(peaks_best_op_pvals, peaks_best_op_pvals_params, peaks_pval, params)
                except Exception as exc:
                    self.logger.error(f'Parameter combination {params} generated an exception: {exc}')
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
                        pc_score, pc_pval, peaks_score, peaks_pval, _ = future.result()
                        # --- update the parameters with the best scores (in absolute value) ---
                        self.update_best_scores(pc_best_scores, pc_best_scores_params, abs(pc_score), params)
                        self.update_best_scores(pc_best_op_pvals, pc_best_op_pvals_params, pc_pval, params)
                        self.update_best_scores(peaks_best_scores, peaks_best_scores_params, abs(peaks_score), params)
                        self.update_best_scores(peaks_best_op_pvals, peaks_best_op_pvals_params, peaks_pval, params)
                    except Exception as exc:
                        self.logger.error(f'Parameter combination {params} generated an exception: {exc}')

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
        """Get the movement number from the file path at index i."""
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
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Maximum number of parallel workers. 0=auto (CPU count), 1=sequential")
    parser.add_argument("--skip-timepoints", type=int, default=100, help="Number of timepoints to skip at the beginning and end when comparing peaks.")

    args = parser.parse_args()

    if not os.path.exists(args.files_path):
        raise FileNotFoundError(
            f"Files folder '{args.files_path}' does not exist."
        )
    if os.path.exists(args.output_folder):
        raise FileExistsError(f"Output folder '{args.output_folder}' already exists.")
    else:
        os.makedirs(args.output_folder)

    if not all(1 <= m <= 9 for m in args.movements):
        raise ValueError("Movements should be between 1 and 9.")

    setup_logger(output_folder=args.output_folder, file_name="compare_peaks_log.txt", loger_name="compare_peaks_logger",
                 log_level=logging.INFO)
    logger = logging.getLogger("compare_peaks_logger")
    logger.info(f"Command line: {' '.join(sys.argv)}")
    effective_workers = os.cpu_count() if args.max_workers == 0 else int(args.max_workers)
    logger.info(f"Run with {effective_workers} cpu's.")
    compare_pcs = ComparePCS(args.files_path, args.output_folder, args.movements, args.skip_timepoints, logger)
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
#   - Adjust --movements, --num-scores, --skip-timepoints and --max-workers as needed.

# 3. The output will be saved in the specified output folder, including logs and figures for each parameter combination and principal component.
