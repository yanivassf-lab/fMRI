import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import sys
import glob
import argparse
import logging

from fmri.pc_similarity import PcSimilarity
from fmri.peaks_similarity import PeaksSimilarity
from fmri.utils import setup_logger
from nibabel.externals.netcdf import netcdf_file

SKIP_EDGES = 100  # number of timepoints to skip at the beginning and end when looking for peaks, TODO: make it a parameter

class ComparePCS:
    def __init__(self, files_path, output_folder, movements, logger):
        """
        Initialize the ComparePCS class.

        Parameters:
        - files_path (str): Path to the directory containing subject subdirectories.
        - output_folder (str): Path to the output folder for logs and figures.
        - movements (list[int]): List of movement identifiers to compare.
        - logger: Logger object for logging information.
        """
        self.files_path = files_path
        self.output_folder = output_folder
        self.movements = movements
        self.params_comb = self.get_list_of_params(files_path)
        self.file_list = self.get_list_of_files(files_path, movements)
        self.n_subs = int(len(self.file_list) / len(movements))
        self.n_movs = len(movements)
        self.n_files = len(self.file_list)
        self.logger = logger
        self.logger.info(f"Found {len(self.file_list)} files for {self.n_movs} movements with {self.n_subs} subjects per movement:\n {'\n '.join(os.path.basename(s) for s in self.file_list)}")

    def compare(self, num_scores=5):
        """
        Compare principal components across different movements and subjects.

        - num_scores (int): Number of top scores to keep for each movement and subject.
        """
        pc_best_scores = np.array([-np.inf] * num_scores)
        pc_best_scores_params = ['.'] * num_scores
        pc_best_op_pvals = np.array([-np.inf] * num_scores)
        pc_best_op_pvals_params = ['.'] * num_scores
        peaks_best_scores = np.array([-np.inf] * num_scores)
        peaks_best_scores_params = ['.'] * num_scores
        peaks_best_op_pvals = np.array([-np.inf] * num_scores)
        peaks_best_op_pvals_params = ['.'] * num_scores
        for comb_num, params in enumerate(self.params_comb):
            self.logger.info(f"comparing peaks of {params} ({comb_num} of {len(self.params_comb)} combinations)")
            fig = plt.figure(figsize=(35, 15))
            gs = gridspec.GridSpec(len(self.file_list), 2, width_ratios=[2, 1])  # 2x wider second column
            axes_left = [fig.add_subplot(gs[i, 0]) for i in range(self.n_files)]
            gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 1])
            axes_right = [fig.add_subplot(gs_right[i, 0]) for i in range(2)]
            x_norm_all = []
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
                self.logger.info(f"Loaded sub-{self.get_sub_name(i)}_mov-{self.get_sub_mov(i)} with {n_pcs} PCs, {n_basis} basis functions and {n_times} time points.")
                x_norm = np.linspace(0, n_times - 1, n_times) / (n_times - 1)
                x_norm_all.append(x_norm)

            pc_sim = PcSimilarity(pcs_list, self.n_subs, self.n_movs)

            main_pcs = pc_sim.compare_and_plot_top_pc_avg_corr()
            signals = []
            for i in range(self.n_files):
                pc_i = pcs_list[i][:, main_pcs[i]]
                signals.append(F_list[i] @ pc_i)

            peaks_sim = PeaksSimilarity(signals, self.n_subs, self.n_movs, skip_edges=SKIP_EDGES)
            for i in range(self.n_files):
                pc_num_i = main_pcs[i]
                peaks_height_i = peaks_sim.peaks_height[i]
                signal_i = signals[i]
                x_norm_i = x_norm_all[i]
                peaks_idx_i = peaks_sim.peaks_idx[i]
                revereted_i = peaks_sim.reverted[i]
                if revereted_i:
                    signal_i *= -1
                    peaks_height_i *= -1

                sub_name_i = self.get_sub_name(i)
                # i.e. for 3 movements with 4 subject per movement: 0,1,2,3 -> movement 1; 4,5,6,7 -> movement 2; 8,9,10,11 -> movement 3
                sub_movement = self.get_sub_mov(i)
                sub_num = self.get_sub_num(i)
                axes_left[sub_num].plot(x_norm_i, signal_i,
                                        label=f"mov-{sub_movement}, pc: {pc_num_i}, {', Reverted Signal' if revereted_i else 'Original Signal'}")
                axes_left[sub_num].scatter(x_norm_i[peaks_idx_i], peaks_height_i, color='red', s=10, zorder=3)  # mark peaks
                if sub_movement == 1:
                    axes_left[sub_num].set_title(f"subj: {sub_name_i}")
                    axes_left[sub_num].set_ylabel('Signal Intensity')
                    axes_left[sub_num].set_ylim(-0.2, 0.2)
                axes_left[sub_num].legend()

                axes_left[sub_num + self.n_subs].plot(x_norm_i, np.abs(signal_i),
                                                 label=f"mov-{sub_movement}, pc: {pc_num_i}, {', Reverted Signal' if revereted_i else 'Original Signal'}")
                axes_left[sub_num + self.n_subs].scatter(x_norm_i[peaks_idx_i], np.abs(peaks_height_i), color='red', s=10,
                                                    zorder=3)  # mark peaks
                if sub_movement == 1:
                    axes_left[sub_num + self.n_subs].set_title(f"subj: {sub_name_i} abs signal")
                    axes_left[sub_num + self.n_subs].set_ylabel('Signal Intensity')
                    axes_left[sub_num + self.n_subs].set_ylim(-0.2, 0.2)
                axes_left[sub_num + self.n_subs].legend()

            im_pcs = axes_right[0].imshow(pc_sim.sim_matrix, aspect='equal', cmap='viridis')
            im_peaks = axes_right[1].imshow(peaks_sim.sim_matrix, aspect='equal', cmap='viridis')
            plt.colorbar(im_pcs, ax=axes_right[0], label='Similarity PCs')
            plt.colorbar(im_peaks, ax=axes_right[1], label='Similarity Peaks')
            axes_right[0].set_title('Similarity Matrix PCs')
            axes_right[1].set_title('Similarity Matrix Peaks')

            # --- build labels in the same order as file_list (the order used to build sim_matrix) ---
            labels = []
            for i in range(self.n_files):
                labels.append(f"sub-{self.get_sub_name(i)}_mov-{self.get_sub_mov(i)}")

            # --- set ticks on the matrix axis (ax2) ---
            n = peaks_sim.sim_matrix.shape[0]
            axes_right[0].set_xticks(np.arange(n))
            axes_right[0].set_yticks(np.arange(n))
            axes_right[0].set_xticklabels(labels, rotation=45, ha='right')
            axes_right[0].set_yticklabels(labels, rotation=45)
            axes_right[0].set_xlabel('Signal Index')
            axes_right[0].set_ylabel('Signal Index')

            axes_right[1].set_xticks(np.arange(n))
            axes_right[1].set_yticks(np.arange(n))
            axes_right[1].set_xticklabels(labels, rotation=45, ha='right')
            axes_right[1].set_yticklabels(labels, rotation=45)
            axes_right[1].set_xlabel('Signal Index')
            axes_right[1].set_ylabel('Signal Index')

            pc_sim.calculate_matrix_score()
            peaks_sim.calculate_matrix_score()

            fig.suptitle(
                f"Params {params}\n"
                f"Average score of pc matrix: {pc_sim.matrix_score:.4f}, average P-value: {1-pc_sim.matrix_op_pval:.4f}\n"
                f"Average score of peaks matrix: {peaks_sim.matrix_score:.4f}, average P-value: {1-peaks_sim.matrix_op_pval:.4f}\n",
                fontsize=16
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
            fig.savefig(os.path.join(self.output_folder, f"peaks_{params}_pc.png"))
            plt.close(fig)

            self.update_best_scores(pc_best_scores, pc_best_scores_params, pc_sim.matrix_score, params)
            self.update_best_scores(pc_best_op_pvals, pc_best_op_pvals_params, pc_sim.matrix_op_pval, params)
            self.update_best_scores(peaks_best_scores, peaks_best_scores_params, peaks_sim.matrix_score, params)
            self.update_best_scores(peaks_best_op_pvals, peaks_best_op_pvals_params, peaks_sim.matrix_op_pval, params)

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

    def print_best_scores(self, best_scores: np.ndarray, best_params: list[str], score_type: str, oposite: bool = False):
        """Print the best scores and their corresponding parameters."""
        sorted_idx = np.argsort(best_scores)[::-1]
        self.logger.info(
            f"Best {score_type}: {','.join([f'{1-s if oposite else s:.4f}' for s in best_scores[sorted_idx]])} with params {','.join([best_params[i] for i in sorted_idx])}"
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
    parser.add_argument("--num-scores", type=int, default=5,
                        help="Number of top scores to keep for each movement and subject")
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

    compare_pcs = ComparePCS(args.files_path, args.output_folder, args.movements, logger)
    compare_pcs.compare(num_scores=args.num_scores)


if __name__ == "__main__":
    main()

# Instructions to run the script:
# 1. The input files should be organized in the following structure:
#    /path/to/subjects/
#        ├── sub-01_movement1/
#        │   ├── no_penalty_nb100/
#        │   │   ├── temporal_profile_pc_0.txt
#        │   │   ├── temporal_profile_pc_1.txt
#        │   │   └── ...
#        │   ├── p0_u1_t1e-6_l-4_0_nb100/
#        │   │   ├── temporal_profile_pc_0.txt
#        │   │   ├── temporal_profile_pc_1.txt
#        │   │   └── ...
#        │   └── ...
#        ├── sub-01_movement2/
#        │   ├── no_penalty_nb100/
#        │   │   ├── temporal_profile_pc_0.txt
#        │   │   ├── temporal_profile_pc_1.txt
#        │   │   └── ...
#        │   ├── p0_u1_t1e-6_l-4_0_nb100/
#        │   │   ├── temporal_profile_pc_0.txt
#        │   │   ├── temporal_profile_pc_1.txt
#        │   │   └── ...
#        │   └── ...
#        └── ...
#
# 2. Run the script from the command line:
# compare-peaks --files-path /path/to/subjects/ --output-folder /path/to/non-exists/output/folder --movements 1 2 --num-scores 5
#   - Replace /path/to/subjects/ with the actual path to the directory containing subject subdirectories.
#   - Replace /path/to/non-exists/output/folder with the desired output folder path (it should not exist prior to running).
#   - Adjust --movements, and --num-scores as needed.

# 3. The output will be saved in the specified output folder, including logs and figures for each parameter combination and principal component.
