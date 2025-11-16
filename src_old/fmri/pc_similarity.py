import numpy as np
from itertools import combinations
from scipy.linalg import subspace_angles

from .similarity import Similarity


class PcSimilarity(Similarity):
    """
    Class to compute similarity between PCA components of multiple samples.
    """

    def __init__(self, pcs_list, n_subs, n_movs, calc_pc_score):
        super().__init__(n_subs, n_movs)
        """Initialize with a list of PCA components and basis matrix.

        Parameters
        ----------
        pcs_list : list of np.ndarray
            Each element is (n_bases, n_components)
        n_subs     : int Number of subjects/samples
        n_movs     : int Number of movements/experiments
        calc_pc_score: bool If calculate PC similarity score
        """

        self.pcs_list = pcs_list
        self.n_components = pcs_list[0].shape[1]
        if calc_pc_score:
            self._calculate_similarity_score()

    def _calculate_similarity_score(self):
        """Calculates the similarity matrix between all pairs of samples based on their PCA components.

        The similarity between two samples is computed using the RV coefficient,
        which measures the similarity between two sets of multivariate data.

        The method iterates through every unique pair of samples (i, j),
        computes the subspace angles between their PCA components, and
        calculates the RV similarity. The resulting similarity values
        are stored in a symmetric similarity matrix.
        """
        # Compute pairwise similarities
        for i, j in combinations(range(self.n_files), 2):
            pcs_i = self.pcs_list[i]
            pcs_j = self.pcs_list[j]

            # Subspace angles
            angles = subspace_angles(pcs_i, pcs_j)
            # subspace_similarity = np.mean(np.cos(angles))
            # min_angle_cos = np.cos(angles[0])

            k_i = pcs_i.shape[1]
            k_j = pcs_j.shape[1]
            sum_sq_cos = np.sum(np.cos(angles) ** 2)
            rv_similarity = sum_sq_cos / np.sqrt(k_i * k_j)
            self.sim_matrix[i, j] = self.sim_matrix[j, i] = rv_similarity

    def get_sub_mov(self, i: int) -> int:
        """Get the movement number from the file path at index i."""
        return int(i / self.n_subs) + 1

    def find_representative_pcs(self, pc_sim_auto_best_similar_pc, pc_sim_auto_weight_similar_pc) -> np.ndarray:
        """Calculates the most representative PC for each file.
        This method identifies a single "main" or "representative" PC for each
        file. It does this by summing the squared correlations of each PC with
        the PCs from all other files.

        Parameters
        ----------
        - pc_sim_auto_best_similar_pc (bool): If True, only the best matching PC from each
          other file is considered when calculate similarity, otherwise all PCs are
          averaged.
        - pc_sim_auto_weight_similar_pc (int): Exponent to which the absolute correlation values
          are raised. Higher values give more weight to stronger correlations.

        Returns
        -------
        np.ndarray
            A 1D numpy array of shape (n_files,). Each element main_pcs[i] is the
            integer index (from 0 to n_components-1) of the most representative
            principal component for file i.
        """
        pc_corrs_sum = np.zeros((self.n_files, self.n_components))

        for i, j in combinations(range(self.n_files), 2):
            if self.get_sub_mov(i) == self.get_sub_mov(j):
                pcs_i = self.pcs_list[i]
                pcs_j = self.pcs_list[j]

                # Correlation between individual PCs
                corr_matrix = np.corrcoef(pcs_i.T, pcs_j.T)[:self.n_components, self.n_components:]
                corr_matrix = np.abs(corr_matrix) ** pc_sim_auto_weight_similar_pc  # stronger weight to higher correlations

                # Sum correlations for each PC
                if pc_sim_auto_best_similar_pc:
                    # Only consider the best matching PC for each PC
                    pc_corrs_sum[i, :] += np.max(corr_matrix, axis=1)
                    pc_corrs_sum[j, :] += np.max(corr_matrix, axis=0)
                else:
                    pc_corrs_sum[i, :] += np.sum(corr_matrix, axis=1)
                    pc_corrs_sum[j, :] += np.sum(corr_matrix, axis=0)
            else:
                pc_corrs_sum[i, :] += 0.0
                pc_corrs_sum[j, :] += 0.0

        # Select the PC with the highest total correlation for each sample
        main_pcs = np.argmax(pc_corrs_sum, axis=1)
        return main_pcs
