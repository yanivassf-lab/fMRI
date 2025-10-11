import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import glob
import argparse

def compare_peaks_between_movements(files_path: str, n_comp: int, movements: list[int]):
    """
    Compare peaks between different movements by plotting the centered average of absolute signal intensities.

    Parameters:
    - files_path (str): Path to the directory containing subject subdirectories.
    - n_comp (int): Number of components/functions to consider.
    - movements (list[int]): List of movement identifiers to compare.
    """



    plt.figure(figsize=(20, 10))
    for movement in movements:
        subs = glob.glob(os.path.join(files_path, f'sub-*movement{movement}*'))

        subject_averages = []
        for sub in subs:
            all_comp_abs = []
            for comp_num in range(n_comp):
                file = os.path.join(sub, f"temporal_profile_pc_{comp_num}.txt")
                with open(file, 'r') as f:
                    lines = f.readlines()
                    signal_comp = np.array(lines[4].strip().split(), dtype=float)
                all_comp_abs.append(np.abs(signal_comp)) # TODO: abs or not?

            all_comp_abs = np.array(all_comp_abs)
            subject_avg = np.mean(all_comp_abs, axis=0) # TODO: Why average across components? each component is different function
            subject_averages.append(subject_avg)

        subject_averages = np.array(subject_averages) # shape (n_subjects, n_timepoints)
        grand_mean = np.mean(subject_averages, axis=0)
        subject_centered = subject_averages - grand_mean # TODO: It is not in use, check if needed
        x_norm = np.linspace(0, subject_centered.shape[1]-1, subject_centered.shape[1]) / (subject_centered.shape[1]-1)


        for i, sub in enumerate(subs):
            plt.subplot(len(subs), 1, i + 1 + movement)

            signal_x = x_norm
            signal_y = subject_averages[i]
            peaks, _ = find_peaks(np.abs(signal_y), height=0.3, distance=5)

            plt.plot(signal_x, signal_y)
            plt.scatter(signal_x[peaks], signal_y[peaks], color='red', s=10, zorder=3)  # mark peaks
            for peak in peaks:
                plt.annotate(f'{peak}', (peak, signal_y[peak]),
                            textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)
                plt.annotate(f'{signal_y[peak]:.2f}', (peak, signal_y[peak]),
                            textcoords="offset points", xytext=(0, 1), ha='center', fontsize=6)

            sub_name = os.path.basename(sub).split('_')[0]
            plt.title(f'Centered average of {n_comp} functions - {sub_name} - movement - {movement}') # TODO: It not centered average but average of abs values
            plt.ylabel('Signal Intensity')
            plt.ylim(-0.2, 0.2)

    plt.tight_layout()
    os.makedirs(files_path + 'figs', exist_ok=True)
    plt.savefig(f"{files_path}/figs/both_movements_compare_peaks_abs_values_xnorm.png")


def main():
    parser = argparse.ArgumentParser(description="Compare peaks between movements")
    parser.add_argument("--files-path", type=str, default='./subjects/', help="Path to the subjects directory")
    parser.add_argument("--n-comp", type=int, required=True, help="Number of components/functions")
    parser.add_argument("--movements", type=int, nargs='+', default=[1, 2], help="List of movements to compare")
    args = parser.parse_args()

    if not os.path.exists(args.files_path):
        raise FileNotFoundError(
            f"Files folder '{args.files_path}' does not exist."
        )

    compare_peaks_between_movements(args.files_path, args.n_comp, args.movements)



if __name__ == "__main__":
    main()

# Instructions to run the script:
# 1. The input files should be organized in the following structure:
#    /path/to/subjects/
#        ├── sub-01/
#        │   ├── movement1/
#        │   │   ├── eigenfunction_0_signal_intensity.txt
#        │   │   ├── eigenfunction_1_signal_intensity.txt
#        │   │   └── ...
#        │   └── movement2/
#        │       ├── eigenfunction_0_signal_intensity.txt
#        │       ├── eigenfunction_1_signal_intensity.txt
#        │       └── ...
#        ├── sub-02/
#        │   ├── movement1/
#        │   │   ├── eigenfunction_0_signal_intensity.txt
#        │   │   ├── eigenfunction_1_signal_intensity.txt
#        │   │   └── ...
#        │   └── movement2/
#        │       ├── eigenfunction_0_signal_intensity.txt
#        │       ├── eigenfunction_1_signal_intensity.txt
#        │       └── ...
#        └── ...
#
# 2. Run the script from the command line:
#    python compare_peaks_between_movements.py --files-path /path/to/subjects/ --n-comp 7 --movements 1 2
#
# 3. The output figure will be saved in /path/to/subjects/figs/both_movements_compare_peaks_abs_values_xnorm.png
