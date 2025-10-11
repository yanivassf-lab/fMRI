import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import glob
import argparse
from fmri.peaks_similarity import get_peaks, calculate_similarity_score


def compare_peaks_between_movements(files_path: str, n_comp: int, movements: list[int], alpha: float = 0.5):
    """
    Compare peaks between different movements by plotting the centered average of absolute signal intensities.

    Parameters:
    - files_path (str): Path to the directory containing subject subdirectories.
    - n_comp (int): Number of components/functions to consider.
    - movements (list[int]): List of movement identifiers to compare.
    - alpha (float): Alpha parameter for combined score calculation between 0 and 1.
                    0 <= alpha <= 1, where alpha=1 gives full weight to the minimum similarity
                    and alpha=0 gives full weight to the mean similarity.
    """

    subs = glob.glob(os.path.join(files_path, f'sub-*'))[0]
    params_comb = [os.path.basename(p) for p in glob.glob(os.path.join(subs, '*'))]

    for pc_num in range(n_comp):
        best_score = [-np.inf] * len(movements)
        best_params = ['.'] * len(movements)
        for params in params_comb:
            plt.figure(figsize=(20, 10))
            for movement in movements:
                subs_params = sorted(glob.glob(os.path.join(files_path, f'sub-*movement{movement}*', params)))
                subject_averages = []
                for sub in subs_params:
                    file = os.path.join(sub, f"temporal_profile_pc_{pc_num}.txt")
                    try:
                        with open(file, 'r') as f:
                            lines = f.readlines()
                            signal_comp = np.array(lines[4].strip().split(), dtype=float)
                            subject_averages.append(signal_comp)
                    except Exception as e:
                        raise Exception(f"Error reading file {file}: {e}")

                subject_averages = np.array(subject_averages)  # shape (n_subjects, n_timepoints)
                # grand_mean = np.mean(subject_averages, axis=0)
                # subject_centered = subject_averages - grand_mean # TODO: It is not in use, check if needed
                x_norm = np.linspace(0, subject_averages.shape[1] - 1, subject_averages.shape[1]) / (
                    subject_averages.shape[1] - 1)

                peaks_idx_subs = []
                peaks_heights_subs = []
                for i, sub in enumerate(subs_params):
                    plt.subplot(len(subs_params) * len(movements), 1, (i + 1) + len(subs_params) * (movement - 1))
                    signal_x = x_norm
                    signal_y = subject_averages[i]
                    # peaks, _ = find_peaks(np.abs(signal_y), height=0.3, distance=5)
                    peaks_idx, peaks_height = get_peaks(signal_y)
                    peaks_idx_subs.append(peaks_idx)
                    peaks_heights_subs.append(peaks_height)
                    plt.plot(signal_x, signal_y)
                    plt.scatter(signal_x[peaks_idx], peaks_height, color='red', s=10, zorder=3)  # mark peaks
                    # for peak_idx in peaks_idx:
                    #     plt.annotate(f'{peak_idx}', (peak_idx, signal_y[peak_idx]),
                    #                 textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)
                    #     plt.annotate(f'{signal_y[peak_idx]:.2f}', (peak_idx, signal_y[peak_idx]),
                    #                 textcoords="offset points", xytext=(0, 1), ha='center', fontsize=6)

                    sub_name = os.path.basename(os.path.dirname(sub)).split('_')[0]
                    plt.title(
                        f'subj: {sub_name}, movement: {movement}')  # TODO: It not centered average but average of abs values
                    plt.ylabel('Signal Intensity')
                    plt.ylim(-0.2, 0.2)
                score = calculate_similarity_score(peaks_idx_subs, peaks_heights_subs, len(subject_averages[0]),
                                                   alpha=alpha)
                if score > best_score[movement - 1]:
                    best_score[movement - 1] = score
                    best_params[movement - 1] = params
                    print(f"New best score for pc {pc_num} movement {movement} with params {params}, pc {pc_num}: {score:.4f}")

            plt.suptitle(f"Best score for {','.join([f'movement:  {m} = {best_score[m - 1]:.4f}' for m in movements])} with params {params} for pc {pc_num}", fontsize=16)
            plt.tight_layout()
            os.makedirs(os.path.join(files_path, "compare_peaks_figs", f"pc_{pc_num}"), exist_ok=True)
            plt.savefig(os.path.join(files_path, "compare_peaks_figs", f"pc_{pc_num}",
                                     f"peaks_{params}_pc_{pc_num}.png"))
            plt.close()
        for movement in movements:
            print(
                f"Best score for pc {pc_num} movement {movement}: {best_score[movement - 1]:.4f} with params {best_params[movement - 1]}")


def main():
    parser = argparse.ArgumentParser(description="Compare peaks between movements")
    parser.add_argument("--files-path", type=str, default='./subjects/', help="Path to the subjects directory")
    parser.add_argument("--n-comp", type=int, required=True, help="Number of components/functions")
    parser.add_argument("--movements", type=int, nargs='+', default=[1, 2], help="List of movements to compare")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha parameter for combined score calculation between 0 and 1")
    args = parser.parse_args()

    if not os.path.exists(args.files_path):
        raise FileNotFoundError(
            f"Files folder '{args.files_path}' does not exist."
        )

    compare_peaks_between_movements(args.files_path, args.n_comp, args.movements, args.alpha)


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
#    compare-peaks-between-movements --files-path /path/to/subjects/ --n-comp 7 --movements 1 2 --alpha 0.5
#
# 3. The output figure will be saved in /path/to/subjects/figs/both_movements_compare_peaks_abs_values_xnorm.png
