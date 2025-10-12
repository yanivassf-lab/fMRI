import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import argparse
import logging
from fmri.peaks_similarity import get_peaks, calculate_similarity_score, peaks_to_signal
from fmri.utils import setup_logger


def compare_peaks(files_path: str, output_folder: str, pc_num: int, movements: list[int],
                                    alpha: float = 0.5, num_scores=5, logger=None):
    """
    Compare peaks between different movements by plotting the centered average of absolute signal intensities.

    Parameters:
    - files_path (str): Path to the directory containing subject subdirectories.
    - output_folder (str): Path to the output folder for logs and figures.
    - pc_num (int): Number of components/functions to consider.
    - movements (list[int]): List of movement identifiers to compare.
    - alpha (float): Alpha parameter for combined score calculation between 0 and 1.
                    0 <= alpha <= 1, where alpha=1 gives full weight to the minimum similarity
                    and alpha=0 gives full weight to the mean similarity.
    - num_scores (int): Number of top scores to keep for each movement and subject.
    """

    params_comb = get_list_of_params(files_path)
    file_list, subs_names = get_list_of_files(files_path, movements)
    subs_num = int(len(file_list)/len(movements))
    logger.info(f"Found {len(file_list)} files for {len(movements)} movements with {subs_num} subjects per movement.")


    for pc_num in range(pc_num):
        logger.info(f"Processing PC: {pc_num}")
        os.makedirs(os.path.join(output_folder, f"pc_{pc_num}"), exist_ok=True)
        best_scores_mov = np.array([[-np.inf] * num_scores] * len(movements))
        best_params_mov = [['.'] * num_scores] * len(movements)
        best_scores_sub = np.array([[-np.inf] * num_scores] * subs_num)
        best_params_sub = [['.'] * num_scores] * subs_num

        for comb_num, params in enumerate(params_comb):
            logger.info(f"PC:{pc_num}, comparing peaks of {params} ({comb_num} of {len(params_comb)} combinations)")
            peaks_signal_all = []
            plt.figure(figsize=(20, 10))
            scores_sub = []
            scores_mov = []
            for i, sub in enumerate(file_list):
                sub_name = os.path.basename(sub).split('_')[0]
                sub_movement = int(i/(len(file_list) / len(movements))) # i.e. for 3 movements with 4 subject per movement: 0,1,2,3 -> movement 1; 4,5,6,7 -> movement 2; 8,9,10,11 -> movement 3
                file = os.path.join(sub, params, f"temporal_profile_pc_{pc_num}.txt")
                try:
                    with open(file, 'r') as f:
                        lines = f.readlines()
                        signal_y = np.array(lines[4].strip().split(), dtype=float)
                except Exception as e:
                    raise Exception(f"Error reading file {file}: {e}")

                length_signal = len(signal_y)
                x_norm = np.linspace(0, length_signal - 1, length_signal) / (length_signal - 1)

                plt.subplot(len(file_list), 1, (i + 1))
                peaks_idx, peaks_height = get_peaks(signal_y)
                peaks_signal = peaks_to_signal(peaks_idx, peaks_height, length_signal)
                peaks_signal_all.append(peaks_signal)
                plt.plot(x_norm, signal_y)
                plt.scatter(x_norm[peaks_idx], peaks_height, color='red', s=10, zorder=3)  # mark peaks
                # for peak_idx in peaks_idx:
                #     plt.annotate(f'{peak_idx}', (peak_idx, signal_y[peak_idx]),
                #                 textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)
                #     plt.annotate(f'{signal_y[peak_idx]:.2f}', (peak_idx, signal_y[peak_idx]),
                #                 textcoords="offset points", xytext=(0, 1), ha='center', fontsize=6)

                plt.title(f'subj: {sub_name}, movement: {sub_movement}')
                plt.ylabel('Signal Intensity')
                plt.ylim(-0.2, 0.2)
            for mov in movements:
                start = (mov-1)*subs_num
                end = start + subs_num
                score_mov = calculate_similarity_score(peaks_signal_all[start:end], alpha=alpha)
                scores_mov.append(score_mov)
                if score_mov > np.min(best_scores_mov[mov - 1]):
                    min_idx = np.argmin(best_scores_mov[mov - 1])
                    best_scores_mov[mov - 1][min_idx] = score_mov
                    best_params_mov[mov - 1][min_idx] = params
            for sub in range(subs_num):
                score_sub = calculate_similarity_score(peaks_signal_all[sub::subs_num], alpha=alpha)
                scores_sub.append(score_sub)
                if score_sub > np.min(best_scores_sub[sub]):
                    min_idx = np.argmin(best_scores_sub[sub])
                    best_scores_sub[sub][min_idx] = score_sub
                    best_params_sub[sub][min_idx] = params

            plt.suptitle(
                f"Params {params} for pc {pc_num}\n \
                Score for {','.join([f'movement:  {m} = {scores_mov[m - 1]:.4f}' for m in movements])} \n \
                Score for {', '.join([f'subject: {subs_names[s]} = {scores_sub[s]:.4f}' for s in range(subs_num)])}",
                fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"pc_{pc_num}", f"peaks_{params}_pc_{pc_num}.png"))
            plt.close()



        for movement in movements:
            sorted_idx = np.argsort(best_scores_mov[movement - 1])[::-1]
            logger.info(
                f"Best score for pc {pc_num} movement {movement}: {','.join([f'{s:.4f}' for s in best_scores_mov[movement - 1][sorted_idx]])} with params {','.join([best_params_mov[movement - 1][i] for i in sorted_idx])}"
            )

        for sub_num in range(subs_num):
            sorted_idx = np.argsort(best_scores_sub[sub_num])[::-1]
            logger.info(
                f"Best score for pc {pc_num} subject {subs_names[sub_num]}: {','.join([f'{s:.4f}' for s in best_scores_sub[sub_num][sorted_idx]])} with params {','.join([best_params_sub[sub_num][i] for i in sorted_idx])}"
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
    subs_names = list(set([os.path.basename(s).split('_')[0] for s in sorted_subs]))
    return sorted_subs, subs_names

def main():
    parser = argparse.ArgumentParser(description="Compare peaks between movements")
    parser.add_argument("--files-path", type=str, required=True, help="Path to the subjects directory")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--pc_num", type=int, required=True, help="Number of components/functions")
    parser.add_argument("--movements", type=int, nargs='+', default=[1, 2], help="List of movements to compare (from 1 to 9)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha parameter for combined score calculation between 0 and 1")
    parser.add_argument("--num-scores", type=int, default=5, help="Number of top scores to keep for each movement and subject")
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

    compare_peaks(args.files_path, args.output_folder, args.pc_num, args.movements, args.alpha, args.num_scores, logger)


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
