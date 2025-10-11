import numpy as np
from scipy.signal import find_peaks
from dtaidistance import dtw


# ====== Step 1: Find peaks ======
def get_peaks(signal):
    """Return indices and heights of both maxima and minima"""
    peaks_up, _ = find_peaks(signal)  # , height=0.3, distance=5)
    heights_up = signal[peaks_up]
    peaks_down, _ = find_peaks(-signal)
    heights_down = signal[peaks_down]  # keep original sign
    return np.concatenate([peaks_up, peaks_down]), np.concatenate([heights_up, heights_down])


def peaks_to_signal(peaks_idx, peaks_height, length):
    """Convert peaks to a full 1D signal with zeros elsewhere"""
    signal = np.zeros(length)
    signal[peaks_idx] = peaks_height
    return signal


# ====== Step 2: DTW similarity ======
def dtw_similarity(sig1, sig2):
    """Compute similarity between two signals based on DTW (0-1)"""
    distance = dtw.distance(sig1, sig2)
    return 1 / (1 + distance)  # similarity 0-1


# ====== Step 3: Compute similarity matrix for a triple ======
def compute_similarity_matrix(peaks_idx_subs, peaks_heights_subs, length):
    """
    This function computes a n x n similarity matrix for three signals based on their peaks.
    Args:
        peaks_idx_subs: list of n lists, each containing indices of peaks for a signal
        peaks_heights_subs: list of n lists, each containing heights of peaks for a signal
        length: length of the signal
    Returns:
        sim_matrix: n x n similarity matrix where sim_matrix[i][j] is the similarity between signal i and signal j
    """
    n = len(peaks_idx_subs)
    peak_signals = []
    for idx, height in zip(peaks_idx_subs, peaks_heights_subs):
        peak_sig = peaks_to_signal(idx, height, length)
        peak_signals.append(peak_sig)

    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = dtw_similarity(peak_signals[i], peak_signals[j])
            sim_matrix[i, j] = sim_matrix[j, i] = sim
    return sim_matrix


# ====== Step 4: Score a triple ======
def triple_score_combined(sim_matrix, alpha=0.9):
    """Combined score of a 3x3 similarity matrix (min + mean)"""
    off_diag = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
    return alpha * np.max(off_diag) + (1 - alpha) * np.mean(off_diag)


def calculate_similarity_score(peaks_idx_subs, peaks_heights_subs, length, alpha=0.5):
    """
    This function calculates the similarity score between two signals based on DTW (0-1)
    and combines the minimum and mean similarity using a weighted average controlled by alpha.

    Args:
        peaks_idx_subs:       Indices of peaks for the subjects.
        peaks_heights_subs:   Heights of peaks for the subjects.
        length:               Length of each signal.
        alpha:                Weighting factor between 0 and 1 for combining min and mean similarity.

    Returns:       Combined similarity score.

    """
    sim_matrix = compute_similarity_matrix(peaks_idx_subs, peaks_heights_subs, length)
    score = triple_score_combined(sim_matrix, alpha)
    return score
