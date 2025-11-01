import numpy as np
from scipy.signal import find_peaks
from dtaidistance import dtw
from .similarity import Similarity


class PeaksSimilarity(Similarity):
    def __init__(self, signals, n_subs, n_movs, fix_orientation=True, peaks_abs=False, skip_timepoints=100):
        super().__init__(n_subs, n_movs)
        """
        Initialize the PeaksSimilarity class.

        Args:
            signals (list): list of signals
            n_subs:      Number of subjects/samples
            n_movs:      Number of movements/experiments
            fix_orientation: If True, corrects for signal orientation before similarity calculation.
            peaks_abs: If True, uses absolute peak heights for similarity calculation (not relevant if fix_orientation is True).
            skip_timepoints: Number of timesteps to skip at the start and end of each signal to avoid edge effects.
        """
        self.signals = signals
        self.skip_timepoints = skip_timepoints
        self.fix_orientation = fix_orientation
        self.peaks_abs = peaks_abs
        self.peaks_idx = []
        self.peaks_height = []
        self.peaks_signals = []
        self.reverted = None
        self._extract_peaks_signal()
        if fix_orientation:
            self._calculate_similarity_score_correct_orientation()
        else:
            self._calculate_similarity_score_orig_orientation()



    # ====== Step 1: Find peaks ======
    def _get_peaks(self, signal):
        """Return indices and heights of both maxima and minima"""
        peaks_up, _ = find_peaks(signal , height=0.3, distance=5)
        heights_up = signal[peaks_up]
        peaks_down, _ = find_peaks(-signal)
        heights_down = signal[peaks_down]  # keep original sign
        return np.concatenate([peaks_up, peaks_down]), np.concatenate([heights_up, heights_down])

    def _peaks_to_signal(self, peaks_idx, peaks_height, length):
        """Convert peaks to a full 1D signal with zeros elsewhere"""
        p_signal = np.zeros(length)
        p_signal[peaks_idx] = peaks_height
        self.peaks_signals.append(p_signal)

    def _extract_peaks_signal(self):
        for signal in self.signals:
            peaks_idx, peaks_height = self._get_peaks(signal)
            self.peaks_idx.append(peaks_idx)
            self.peaks_height.append(peaks_height)
            self._peaks_to_signal(peaks_idx, peaks_height, len(signal))

    # ====== Step 2: DTW similarity ======
    def _calculate_similarity_score_correct_orientation(self):
        # 1. Cut edges
        cut_peaks_signals = [p_sig[self.skip_timepoints:-self.skip_timepoints] for p_sig in self.peaks_signals]

        # 2. Compute distance matrix for reversed signals
        cut_peaks_signals_rev = [np.negative(p_sig) for p_sig in cut_peaks_signals]  # invert each signal
        cut_peaks_signals_all = cut_peaks_signals_rev + cut_peaks_signals

        dist_mat_all = dtw.distance_matrix(cut_peaks_signals_all, block=((0, 2*self.n_files),(self.n_files,2*self.n_files)))#,parallel=True, use_mp=True)
        dist_mat_org = dist_mat_all[self.n_files:2*self.n_files, self.n_files:2*self.n_files]
        dist_mat_rev = dist_mat_all[:self.n_files, self.n_files:]
        # 3. Compute minimal distances and reversed flags
        dist_min_mat = np.minimum(dist_mat_org, dist_mat_rev)
        reversed_mat = dist_mat_rev < dist_mat_org

        # 5. Find optimal orientations using the spectral method
        self.reverted = self.find_optimal_orientations(reversed_mat)

        # 6. Similarity matrix (higher = more similar)
        self.sim_matrix = 1 / (1 + dist_min_mat)

    def find_optimal_orientations(self, M, num_iter=100, tol=1e-6):
        """
        Computes approximate optimal orientations for a set of samples
        using a spectral method with power iteration (faster than full eigen decomposition).

        Args:
            M (array-like): Symmetric (N x N) matrix of 0s and 1s.
                            M[i, j] = 0 means i and j have the same orientation.
                            M[i, j] = 1 means i and j have opposite orientations.
            num_iter (int): Maximum number of power iteration steps.
            tol (float): Convergence tolerance.

        Returns:
            list: List of length N of 0s and 1s.
                  0 = keep original orientation.
                  1 = flip orientation.
        """
        W = 1 - 2 * M
        np.fill_diagonal(W, 0)

        # --- Power iteration to approximate principal eigenvector ---
        n = W.shape[0]
        b = np.random.rand(n)
        b /= np.linalg.norm(b)

        for _ in range(num_iter):
            b_next = W @ b
            b_next /= np.linalg.norm(b_next)
            if np.linalg.norm(b_next - b) < tol:
                break
            b = b_next

        principal_eigenvector = b

        # Determine orientations based on sign
        orientations = (principal_eigenvector < 0).astype(int)
        return orientations.tolist()

    def _calculate_similarity_score_orig_orientation(self):
        if self.peaks_abs:
            cut_peaks_signals = [np.abs(p_sig[self.skip_timepoints:-self.skip_timepoints]) for p_sig in self.peaks_signals]
        else:
            cut_peaks_signals = [p_sig[self.skip_timepoints:-self.skip_timepoints] for p_sig in self.peaks_signals]

        dist_min_mat = dtw.distance_matrix(cut_peaks_signals, parallel=True, use_mp=True)
        self.reverted = np.zeros(self.n_files)
        self.sim_matrix = 1 / (1 + dist_min_mat)

