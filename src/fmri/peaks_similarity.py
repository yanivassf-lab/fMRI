import numpy as np
from scipy.signal import find_peaks
from dtaidistance import dtw
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist


class PeaksSimilarity:
    def __init__(self, subs_num, movements, alpha=0.5, skip_edges=100):
        """
        Initialize the PeaksSimilarity class.

        Args:
            subs_num:   Number of subjects per movement
            movements:  List of movement indices (1-based)
            alpha:      Weighting factor between 0 and 1 for combining scores between and within movements.
                        alpha=1 gives full weight to between-movement similarity,
                        alpha=0 gives full weight to within-movement similarity.
            skip_edges: Number of samples to skip at the start and end of each signal to avoid edge effects.
        """
        self.subs_num = subs_num
        self.movements = movements
        self.mov_num = len(movements)
        self.alpha = alpha
        self.skip_edges = skip_edges
        self.peaks_signal_all = []
        self.sim_matrix = None
        self.score_between_mov_avg = None
        self.score_within_mov_avg = None
        self.weighted_score = None
        self.hierarchical_scores = np.zeros((self.mov_num, self.mov_num))
        self.hierarchical_scores_avg = None
        self.replaced_negatives = []  # indices of signals that were inverted

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
    def _find_correct_orientation(self, sig):
        dist_plus = np.zeros(len(self.peaks_signal_all))
        dist_minus = np.zeros(len(self.peaks_signal_all))
        j = 0
        for i, ref_sig in enumerate(self.peaks_signal_all):
            if np.array_equal(sig, ref_sig):
                j = i
                continue
            dist_plus[i] = dtw.distance(sig[self.skip_edges:-self.skip_edges], ref_sig[self.skip_edges:-self.skip_edges])
            dist_minus[i] = dtw.distance(-sig[self.skip_edges:-self.skip_edges], ref_sig[self.skip_edges:-self.skip_edges])
        dist_plus_avg = np.mean(dist_plus)
        dist_minus_avg = np.mean(dist_minus)
        if dist_plus_avg > dist_minus_avg:
            self.replaced_negatives.append(j)
            return -sig
        else:
            return sig

    # ====== Step 3: Compute similarity matrix for a triple ======
    def _compute_similarity_matrix(self):
        """
        This function computes a n x n similarity matrix for three signals based on their peaks.

        Returns:
            sim_matrix: n x n similarity matrix where sim_matrix[i][j] is the similarity between signal i and signal j
        """
        peaks_signal_no_edges = [self.peaks_signal_all[i][self.skip_edges:-self.skip_edges] for i in range(len(self.peaks_signal_all))]
        dist_matrix = dtw.distance_matrix(peaks_signal_no_edges)
        sim_matrix = 1 / (1 + dist_matrix)
        self.sim_matrix = sim_matrix
        return sim_matrix


    def extract_experiment_blocks(self):
        """
        Extracts the diagonal blocks (each experiment) from the large similarity matrix.
        """
        blocks = []
        for i in range(self.mov_num):
            start = i * self.subs_num
            end = (i + 1) * self.subs_num
            block = self.sim_matrix[start:end, start:end]
            blocks.append(block)
        return np.array(blocks)

    def hierarchical_similarity(self):
        """
        Builds a hierarchical clustering tree for each experiment separately
        from a similarity matrix of size (m*n, m*n),
        and computes a hierarchical similarity score between experiments.
        """
        # Step 1: extract per-experiment similarity matrices
        similarity_matrices = self.extract_experiment_blocks()

        linkage_matrices = []
        distance_vectors = []

        # Step 2: build hierarchical tree for each experiment
        for i in range(self.mov_num):
            sim = similarity_matrices[i]
            dist = 1 - sim  # convert similarity to distance
            condensed = pdist(dist)  # condensed upper-triangle form
            Z = linkage(condensed, method='average')
            linkage_matrices.append(Z)
            distance_vectors.append(condensed)

        # Step 3: compute cophenetic correlation between all pairs of experiments
        for i in range(self.mov_num):
            for j in range(i + 1, self.mov_num):
                # cophenet returns both the coefficient and the cophenetic distances
                coph_i, _ = cophenet(linkage_matrices[i], distance_vectors[i])
                coph_j, _ = cophenet(linkage_matrices[j], distance_vectors[j])
                score = np.corrcoef(coph_i, coph_j)[0, 1]
                self.hierarchical_scores[i, j] = self.hierarchical_scores[j, i] = score

        # Step 4: compute the overall average similarity
        self.hierarchical_scores_avg = np.mean(self.hierarchical_scores[np.triu_indices(self.mov_num, k=1)])



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

    def calculate_similarity_score(self, fix_orient):
        """
        This function calculates the similarity score between two signals based on DTW (0-1)
        and combines the minimum and mean similarity using a weighted average controlled by alpha.

        Args:
            fix_orient: True or False, flag to fix the orientation of signals

        Returns:       Combined similarity score.

        """
        if fix_orient:
            for i, sig in enumerate(self.peaks_signal_all):
                self.peaks_signal_all[i] = self._find_correct_orientation(sig)


        self._compute_similarity_matrix()
        self.hierarchical_similarity()
        self._score_combined()
