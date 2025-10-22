import numpy as np
from scipy.signal import find_peaks
from dtaidistance import dtw
from itertools import combinations
from .similarity import Similarity


class PeaksSimilarity(Similarity):
    def __init__(self, signals, n_subs, n_movs, skip_edges=100):
        super().__init__(n_subs, n_movs)
        """
        Initialize the PeaksSimilarity class.

        Args:
            signals (list): list of signals
            n_subs:      Number of subjects/samples
            n_movs:      Number of movements/experiments
            skip_edges: Number of samples to skip at the start and end of each signal to avoid edge effects.
        """
        self.signals = signals
        self.skip_edges = skip_edges
        self.peaks_idx = []
        self.peaks_height = []
        self.peaks_signals = []
        self.reverted = []
        self._extract_peaks_signal()
        self._find_correct_orientation()

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
    def _find_correct_orientation(self):
        dist_min = np.zeros((self.n_subs_tot, self.n_subs_tot))

        for i, p_sig in enumerate(self.peaks_signals):
            dist_org = np.zeros(self.n_subs_tot)
            dist_rev = np.zeros(self.n_subs_tot)
            for j, p_ref_sig in enumerate(self.peaks_signals):
                if np.array_equal(p_sig, p_ref_sig):
                    dist_org[j] = 0.0
                    continue
                dist_org[j] = dtw.distance(p_sig[self.skip_edges:-self.skip_edges],
                                           p_ref_sig[self.skip_edges:-self.skip_edges])
                dist_rev[j] = dtw.distance(-p_sig[self.skip_edges:-self.skip_edges],
                                           p_ref_sig[self.skip_edges:-self.skip_edges])
            dist_org_avg = np.mean(dist_org)
            dist_rev_avg = np.mean(dist_rev)
            if dist_rev_avg < dist_org_avg:
                dist_min[i, :] = dist_rev
                self.reverted.append(True)
            else:
                dist_min[i, :] = dist_org
                self.reverted.append(False)
        self.sim_matrix = 1 / (1 + dist_min)
