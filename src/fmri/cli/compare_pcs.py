import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import sys
import glob
import argparse
import logging

from src.fmri.peaks_similarity import PeaksSimilarity
from src.fmri.utils import setup_logger
from src.fmri.pc_similarity import PcSimilarity

SKIP_EDGES = 100  # number of timepoints to skip at the beginning and end when looking for peaks, TODO: make it a parameter


def compare_pcs(files_path: str, output_folder: str, pc_num: int, movements: list[int],
                alpha: float = 0.5, num_scores=5, logger=None):
    """
    Compare peaks between different movements by plotting the centered average of absolute signal intensities.

    Parameters:
    - files_path (str): Path to the directory containing subject subdirectories.
    - output_folder (str): Path to the output folder for logs and figures.
    - pc_num (int): Number of components/functions to consider.
    - movements (list[int]): List of movement identifiers to compare.
    - alpha (float): Alpha parameter for combined score calculation between 0 and 1.
                    0 <= alpha <= 1, where alpha=1 gives full weight to the between movements similarity
                    and alpha=0 gives full weight to the within movements similarity (between subs).
    - num_scores (int): Number of top scores to keep for each movement and subject.
    """

    params_comb = get_list_of_params(files_path)
    file_list = get_list_of_files(files_path, movements)
    sub_names = [os.path.basename(s).split('_')[0].split('-')[1] for s in file_list]

    n_subs = int(len(file_list) / len(movements))
    n_movs = len(movements)
    logger.info(f"Found {len(file_list)} files for {n_movs} movements with {n_subs} subjects per movement.")

    logger.info(f"Processing PC: {pc_num}")
    os.makedirs(os.path.join(output_folder, f"pc_{pc_num}"), exist_ok=True)
    pc_best_scores = np.array([-np.inf] * num_scores)
    pc_best_params = ['.'] * num_scores
    peaks_best_scores = np.array([-np.inf] * num_scores)
    peaks_best_params = ['.'] * num_scores
    for comb_num, params in enumerate(params_comb):
        logger.info(f"PC:{pc_num}, comparing peaks of {params} ({comb_num} of {len(params_comb)} combinations)")
        fig = plt.figure(figsize=(35, 15))
        gs = gridspec.GridSpec(len(file_list), 2, width_ratios=[2, 1])  # 2x wider second column
        axes_left = [fig.add_subplot(gs[i, 0]) for i in range(len(file_list))]
        gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 1])
        axes_right = [fig.add_subplot(gs_right[i, 0]) for i in range(2)]
        x_norm_all = []
        F = None  # to be loaded from the first file
        pcs_list = []

        for i, sub in enumerate(file_list):
            file = os.path.join(sub, params, f"temporal_profile_pc_{pc_num}.txt")
            try:
                with open(file, 'r') as f:
                    lines = f.readlines()
                    F = np.array(lines[4].strip().split(), dtype=float)
                    pcs = np.array(lines[5].strip().split(), dtype=float)
            except Exception as e:
                raise Exception(f"Error reading file {file}: {e}")

            pcs_list.append(pcs)
            n_basis, pc_num = pcs.shape
            n_times = F.shape[0]
            x_norm = np.linspace(0, n_times - 1, n_times) / (n_times - 1)
            x_norm_all.append(x_norm)

        pc_sim = PcSimilarity(pcs_list, F, n_subs, n_movs)

        subspace_sim_matrix, main_pcs = pc_sim.compare_and_plot_top_pc_avg_corr()
        signals = []
        for i, sub in enumerate(file_list):
            pc_i = pcs_list[i][:, main_pcs[i]]
            signals.append(F @ pc_i)

        peaks_sim = PeaksSimilarity(signals, n_subs, n_movs, skip_edges=SKIP_EDGES)
        for i, sub in enumerate(file_list):
            pc_num_i = main_pcs[i]
            peaks_height_i = peaks_sim.peaks_height[i]
            signal_i = signals[i]
            x_norm_i = x_norm_all[i]
            peaks_idx_i = peaks_sim.peaks_idx[i]
            revereted_i = peaks_sim.reverted[i]
            if revereted_i:
                signal_i *= -1
                peaks_height_i *= -1

            sub_name_i = os.path.basename(sub).split('_')[0].split('-')[1]
            # i.e. for 3 movements with 4 subject per movement: 0,1,2,3 -> movement 1; 4,5,6,7 -> movement 2; 8,9,10,11 -> movement 3
            sub_movement = int(i / n_subs)
            sub_num = int(i % n_subs)
            axes_left[sub_num].plot(x_norm_i, signal_i,
                                    label=f"mov-{sub_movement}, pc: {pc_num_i}, {'Reverted Signal' if revereted_i else 'Original Signal'}")
            axes_left[sub_num].scatter(x_norm_i[peaks_idx_i], peaks_height_i, color='red', s=10, zorder=3)  # mark peaks
            if sub_movement == 1:
                axes_left[sub_num].set_title(f"subj: {sub_name_i}")
                axes_left[sub_num].set_ylabel('Signal Intensity')
                axes_left[sub_num].set_ylim(-0.2, 0.2)
            axes_left[sub_num].legend()

            axes_left[sub_num + n_subs].plot(x_norm_i, np.abs(signal_i), label=f"mov-{sub_movement}, pc: {pc_num_i}")
            axes_left[sub_num + n_subs].scatter(x_norm_i[peaks_idx_i], np.abs(peaks_height_i), color='red', s=10,
                                                zorder=3)  # mark peaks
            if sub_movement == 1:
                axes_left[sub_num + n_subs].set_title(f"subj: {sub_name_i} abs signal")
                axes_left[sub_num + n_subs].set_ylabel('Signal Intensity')
                axes_left[sub_num + n_subs].set_ylim(-0.2, 0.2)
            axes_left[sub_num + n_subs].legend()

        im_pcs = axes_right[0].imshow(subspace_sim_matrix, aspect='equal', cmap='viridis')
        im_peaks = axes_right[1].imshow(peaks_sim.sim_matrix, aspect='equal', cmap='viridis')
        plt.colorbar(im_pcs, ax=axes_right[0], label='Similarity PCs')
        plt.colorbar(im_peaks, ax=axes_right[1], label='Similarity Peaks')
        axes_right[0].set_title('Similarity Matrix PCs')
        axes_right[1].set_title('Similarity Matrix Peaks')

        # --- build labels in the same order as file_list (the order used to build sim_matrix) ---
        labels = []
        for i, s in enumerate(file_list):
            sub_name = os.path.basename(s).split('_')[0].split('-')[1]
            sub_movement = int(i / (len(file_list) / len(movements)))
            labels.append(f"sub-{sub_name}_mov-{sub_movement + 1}")

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

        pc_sim.get_matrix_score()
        peaks_sim.get_matrix_score()

        fig.suptitle(
            f"Params {params}\n"
            f"Average score of pc matrix: {pc_sim.matrix_score:.4f}\n"
            f"Average score of peaks matrix: {peaks_sim.matrix_score:.4f}\n",
            fontsize=16
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
        fig.savefig(os.path.join(output_folder, f"pc_{pc_num}", f"peaks_{params}_pc_{pc_num}.png"))
        plt.close(fig)

        if pc_sim.matrix_score > np.min(pc_best_scores):
            min_idx = np.argmin(pc_best_scores)
            pc_best_scores[min_idx] = pc_sim.matrix_score
            pc_best_params[min_idx] = params
        if peaks_sim.matrix_score > np.min(peaks_best_scores):
            min_idx = np.argmin(peaks_best_scores)
            peaks_best_scores[min_idx] = pc_sim.matrix_score
            peaks_best_params[min_idx] = params

    sorted_idx = np.argsort(pc_best_scores)[::-1]
    logger.info(
        f"Best pc score: {','.join([f'{s:.4f}' for s in pc_best_scores[sorted_idx]])} with params {','.join([pc_best_params[i] for i in sorted_idx])}"
    )
    sorted_idx = np.argsort(peaks_best_scores)[::-1]
    logger.info(
        f"Best peaks score: {','.join([f'{s:.4f}' for s in peaks_best_scores[sorted_idx]])} with params {','.join([peaks_best_params[i] for i in sorted_idx])}"
    )


def get_list_of_params(files_path: str):
    """Get a list of unique parameter combinations from the directory structure."""
    subs = glob.glob(os.path.join(files_path, f'sub-*'))[0]
    params_comb = [os.path.basename(p) for p in glob.glob(os.path.join(subs, '*'))]
    return sorted(params_comb)


def get_list_of_files(files_path: str, movements: list[int]):
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


def main():
    parser = argparse.ArgumentParser(description="Compare peaks between movements")
    parser.add_argument("--files-path", type=str, required=True, help="Path to the subjects directory")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--pc_num", type=int, required=True, help="Number of components/functions")
    parser.add_argument("--movements", type=int, nargs='+', default=[1, 2],
                        help="List of movements to compare (from 1 to 9)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha parameter for combined score calculation between 0 and 1")
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
    if not (0 <= args.alpha <= 1):
        raise ValueError("Alpha should be between 0 and 1.")

    setup_logger(output_folder=args.output_folder, file_name="compare_peaks_log.txt", loger_name="compare_peaks_logger",
                 log_level=logging.INFO)
    logger = logging.getLogger("compare_peaks_logger")
    logger.info(f"Command line: {' '.join(sys.argv)}")

    compare_pcs(args.files_path, args.output_folder, args.pc_num, args.movements, args.alpha, args.num_scores, logger)


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
# compare-peaks --files-path /path/to/subjects/ --output-folder /path/to/non-exists/output/folder --pc_num 7 --movements 1 2 --alpha 0.9 --num-scores 5
#   - Replace /path/to/subjects/ with the actual path to the directory containing subject subdirectories.
#   - Replace /path/to/non-exists/output/folder with the desired output folder path (it should not exist prior to running).
#   - Adjust --pc_num, --movements, --alpha, and --num-scores as needed.

# 3. The output will be saved in the specified output folder, including logs and figures for each parameter combination and principal component.
