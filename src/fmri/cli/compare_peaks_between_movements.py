
# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import glob

FILES_PATH = './subjects/'
subs = glob.glob(os.path.join(FILES_PATH, 'sub-*'))
subs = [os.path.basename(sub) for sub in subs]
N_COMP = 7
MOVEMENTS = [1,2]

plt.figure(figsize=(20, 10))
for MOVEMENT in MOVEMENTS:
    subject_averages = []
    for sub in subs:
        all_func_abs = []
        for function_name in range(N_COMP):
            file = f"{FILES_PATH}/{sub}/output{MOVEMENT}/eigenfunction_{function_name}_signal_intensity.txt"
            with open(file, 'r') as f:
                lines = f.readlines()
                signal_func = np.array(lines[4].strip().split(), dtype=float)
            all_func_abs.append(np.abs(signal_func))
        
        all_func_abs = np.array(all_func_abs)
        subject_avg = np.mean(all_func_abs, axis=0)
        subject_averages.append(subject_avg)

    subject_averages = np.array(subject_averages)
    grand_mean = np.mean(subject_averages, axis=0)
    subject_centered = subject_averages - grand_mean
    x_norm = np.linspace(0, subject_centered.shape[1]-1, subject_centered.shape[1]) / (subject_centered.shape[1]-1)


    for i, sub in enumerate(subs):
        plt.subplot(len(subs), 1, i + 1)

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

        plt.title(f'Centered average of 7 functions - {sub}')
        plt.ylabel('Signal Intensity')
        plt.ylim(-0.2, 0.2)

plt.tight_layout()
os.makedirs(FILES_PATH + 'figs', exist_ok=True)
plt.savefig(f"{FILES_PATH}/figs/both_movements_compare_peaks_abs_values_xnorm.png")
