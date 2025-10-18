import numpy as np
from itertools import combinations
from scipy.linalg import subspace_angles

from src.fmri.similarity import Similarity


class PcSimilarity(Similarity):
    """
    Class to compute similarity between PCA components of multiple samples.
    """

    def __init__(self, PCs_list, F, n_subs, n_movs):
        super().__init__(n_subs, n_movs)
        """
        Initialize with a list of PCA components and basis matrix.

        Parameters
        ----------
        PCs_list : list of np.ndarray
            Each element is (n_bases, n_components)
        F : np.ndarray
            Matrix (timepoints, n_bases)
        n_subs     : int Number of subjects/samples
        n_movs     : int Number of movements/experiments
        """

        self.PCs_list = PCs_list
        self.F = F
        self.n_subs = len(PCs_list)
        self.n_movs = n_movs
        self.n_components = PCs_list[0].shape[1]
        self.sim_matrix = np.zeros((self.n_subs, self.n_subs))

    def compare_and_plot_top_pc_avg_corr(self):
        """
        Compare PCA components between samples using:
          (1) Subspace angles (global similarity)
          (2) Component-wise correlation (to find top contributing PCs)
        Select the main PC per sample based on average correlation across all other samples,
        then plot one signal per sample.

        Parameters
        ----------
        PCs_list : list of np.ndarray
            Each element is (basis, n_components)
        F : np.ndarray
            Matrix (timepoints, basis)
        sample_names : list of str, optional

        Returns
        -------
        subspace_sim_matrix : np.ndarray
            Global similarity matrix (subspace angles)
        corr_sim_matrix : np.ndarray
            Mean component-wise correlation matrix
        main_pcs : np.ndarray
            Index of the top PC per sample
        avg_pc_corrs : np.ndarray
            Average correlation of each PC per sample across all other samples
        """

        # Matrices to store similarities
        corr_sim_matrix = np.zeros((self.n_subs, self.n_subs))
        pc_corrs_sum = np.zeros((self.n_subs, self.n_components))
        pc_counts = np.zeros((self.n_subs, self.n_components))

        # Compute pairwise similarities
        for i, j in combinations(range(self.n_subs), 2):
            PCs1 = self.PCs_list[i]
            PCs2 = self.PCs_list[j]

            # Subspace angles
            angles = subspace_angles(PCs1, PCs2)
            subspace_similarity = np.mean(np.cos(angles))
            self.sim_matrix[i, j] = self.sim_matrix[j, i] = subspace_similarity

            # Component-wise correlation ----
            pc_corrs = [np.corrcoef(PCs1[:, k], PCs2[:, k])[0, 1] for k in range(self.n_components)]
            mean_corr = np.nanmean(pc_corrs)
            corr_sim_matrix[i, j] = corr_sim_matrix[j, i] = mean_corr

            # Accumulate PC correlations for average
            pc_corrs_sum[i, :] += pc_corrs
            pc_corrs_sum[j, :] += pc_corrs
            pc_counts[i, :] += 1
            pc_counts[j, :] += 1

        # Average correlation per PC
        avg_pc_corrs = pc_corrs_sum / pc_counts

        # Determine main PC per sample
        main_pcs = np.argmax(avg_pc_corrs, axis=1)
        return main_pcs
