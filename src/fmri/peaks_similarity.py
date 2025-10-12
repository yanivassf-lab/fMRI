import numpy as np
from scipy.signal import find_peaks
from dtaidistance import dtw


class PeaksSimilarity:
    def __init__(self, subs_num, movements, alpha=0.5):
        """
        Initialize the PeaksSimilarity class.

        Args:
            subs_num:   Number of subjects per movement
            movements:  List of movement indices (1-based)
            alpha:      Weighting factor between 0 and 1 for combining scores between and within movements.
                        alpha=1 gives full weight to between-movement similarity,
                        alpha=0 gives full weight to within-movement similarity.
        """
        self.subs_num = subs_num
        self.movements = movements
        self.alpha = alpha
        self.peaks_signal_all = []
        self.sim_matrix = None
        self.score_between_mov_avg = None
        self.score_within_mov_avg = None
        self.weighted_score = None

    # ====== Step 1: Find peaks ======
    def _get_peaks(self, signal):
        """Return indices and heights of both maxima and minima"""
        peaks_up, _ = find_peaks(signal)  # , height=0.3, distance=5)
        heights_up = signal[peaks_up]
        peaks_down, _ = find_peaks(-signal)
        heights_down = signal[peaks_down]  # keep original sign
        return np.concatenate([peaks_up, peaks_down]), np.concatenate([heights_up, heights_down])

    def _peaks_to_signal(self, peaks_idx, peaks_height, length):
        """Convert peaks to a full 1D signal with zeros elsewhere"""
        signal = np.zeros(length)
        signal[peaks_idx] = peaks_height
        return signal

    def extract_peaks_signal(self, signal):
        peaks_idx, peaks_height = self._get_peaks(signal)
        peak_signal = self._peaks_to_signal(peaks_idx, peaks_height, len(signal))
        self.peaks_signal_all.append(peak_signal)
        return peaks_idx, peaks_height

    # ====== Step 2: DTW similarity ======
    def _dtw_similarity(self, sig1, sig2):
        """Compute similarity between two signals based on DTW (0-1)"""
        distance = dtw.distance(sig1, sig2)
        return 1 / (1 + distance)  # similarity 0-1

    # ====== Step 3: Compute similarity matrix for a triple ======
    def _compute_similarity_matrix(self):
        """
        This function computes a n x n similarity matrix for three signals based on their peaks.

        Returns:
            sim_matrix: n x n similarity matrix where sim_matrix[i][j] is the similarity between signal i and signal j
        """
        dist_matrix = dtw.distance_matrix(self.peaks_signal_all)
        sim_matrix = 1 / (1 + dist_matrix)

        return sim_matrix

    # ====== Step 4: Score a triple ======
    def _score_combined(self):
        """
        This function calculates a combined similarity score from a similarity matrix.
        It computes the average similarity between movements (diagonal elements of sub-matrices)
        and the average similarity within movements (off-diagonal elements of sub-matrices),
        and combines them using a weighted average controlled by alpha.

        Args:
            sim_matrix: n x n similarity matrix

        Returns:      Tuple of (score_between_mov_avg, score_within_mov_avg, combined_score)
        """
        movements = [mov - 1 for mov in self.movements]
        score_within_mov = []
        score_between_mov = []
        for mov_row in movements:
            for mov_col in movements[mov_row:]:
                start_row = (mov_row) * self.subs_num
                end_row = start_row + self.subs_num
                start_col = (mov_col) * self.subs_num
                end_col = start_col + self.subs_num
                sim_matrix_mov = self.sim_matrix[start_row:end_row, start_col:end_col]
                if mov_row == mov_col:
                    off_diag = sim_matrix_mov[np.triu_indices_from(sim_matrix_mov, k=1)]
                    score_within_mov.extend(off_diag.tolist())
                else:
                    diag = np.diag(sim_matrix_mov)
                    score_between_mov.extend(diag.tolist())
        # Weighted average of mean scores
        self.score_between_mov_avg = sum(score_between_mov) / len(score_between_mov)
        self.score_within_mov_avg = sum(score_within_mov) / len(score_within_mov)
        self.weighted_score = self.alpha * self.score_between_mov_avg + (1 - self.alpha) * self.score_within_mov_avg

    def calculate_similarity_score(self):
        """
        This function calculates the similarity score between two signals based on DTW (0-1)
        and combines the minimum and mean similarity using a weighted average controlled by alpha.

        Returns:       Combined similarity score.

        """
        self.sim_matrix = self._compute_similarity_matrix()
        self._score_combined()
