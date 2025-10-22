import numpy as np
from itertools import combinations
from scipy.linalg import subspace_angles

from .similarity import Similarity


class PcSimilarity(Similarity):
    """
    Class to compute similarity between PCA components of multiple samples.
    """

    def __init__(self, PCs_list, n_subs, n_movs):
        super().__init__(n_subs, n_movs)
        """
        Initialize with a list of PCA components and basis matrix.

        Parameters
        ----------
        PCs_list : list of np.ndarray
            Each element is (n_bases, n_components)
        n_subs     : int Number of subjects/samples
        n_movs     : int Number of movements/experiments
        """

        self.PCs_list = PCs_list
        self.n_components = PCs_list[0].shape[1]

    def compare_and_plot_top_pc_avg_corr(self):
        """
        Compare PCA components between samples using:
          (1) Subspace angles (global similarity)
          (2) Correlation between individual PCs
        - Update the similarity matrix with subspace similarities.
        - Determine the main PC per sample based on average correlations:
            The calculation is:
            1) For each pair of samples (i, j):
               - For each pair of PCs (ci, cj):
                 - Compute the correlation between PC ci of sample i and PC cj of sample j.
               - Accumulate these correlations in a 3D matrix pc_corrs_sum[i, ci, cj].
            2) After processing all pairs, sum the correlations across the third axis (cj) to get total correlations for each PC ci of sample i.
            3) The main PC for sample i is the one with the highest total correlation.


        Returns
        -------
        main_pcs : np.ndarray
            Index of the top PC per sample
        """

        # Matrices to store similarities
        pc_corrs_sum = np.zeros((self.n_subs_tot, self.n_components, self.n_components))

        # Compute pairwise similarities
        for i, j in combinations(range(self.n_subs_tot), 2):
            pcs_i = self.PCs_list[i]
            pcs_j = self.PCs_list[j]

            # Subspace angles
            angles = subspace_angles(pcs_i, pcs_j)
            subspace_similarity = np.mean(np.cos(angles))
            self.sim_matrix[i, j] = self.sim_matrix[j, i] = subspace_similarity

            for ci, cj in combinations(range(self.n_components), 2):
                pc_corrs = np.corrcoef(pcs_i[:, ci], pcs_j[:, cj])[0, 1]
                # Accumulate PC correlations for average
                pc_corrs_sum[i, ci, cj] += pc_corrs
                pc_corrs_sum[j, cj, ci] += pc_corrs

        main_pcs = np.argmax(np.sum(pc_corrs_sum, axis=2), axis=1)
        # Determine main PC per sample
        return main_pcs
