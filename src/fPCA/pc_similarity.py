import numpy as np
from itertools import combinations
from scipy.linalg import subspace_angles

from .similarity import Similarity


class PcSimilarity(Similarity):
    """
    Class to compute similarity between PCA components of multiple samples.
    """

    def __init__(self, pcs_list, n_subs, n_movs, logger):
        super().__init__(n_subs, n_movs, logger)
        """Initialize with a list of PCA components and basis matrix.

        Parameters
        ----------
        pcs_list : list of np.ndarray
            Each element is (n_bases, n_components)
        n_subs     : int Number of subjects/samples
        n_movs     : int Number of movements/experiments
        """

        self.pcs_list = pcs_list
        self.n_components = pcs_list[0].shape[1]
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


