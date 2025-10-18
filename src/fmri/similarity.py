import numpy as np
from itertools import combinations
from scipy.stats import spearmanr


class Similarity:
    ""  "Base class for similarity metrics."""
    def __init__(self, n_subs, n_movs):
        """
        Initialize with a similarity matrix and parameters.

        Parameters
        ----------
        sim_matrix : np.ndarray
            Large similarity matrix (subjects * movements, subjects * movements)
        subs_num : int
            Number of subjects per movement/experiment
        mov_num : int
            Number of movements/experiments
        """
        self.n_subs = n_subs
        self.n_movs = n_movs
        self.sim_matrix = None # to be set in derived classes
        self.matrix_score = None

    def get_matrix_score(self):
        """
        Computes the overall consistency score across experiments.

        Returns
        -------
        float
            Consistency score.
        """
        blocks = self.extract_experiment_blocks()
        self.matrix_score = self.consistency_score(blocks)

    def extract_experiment_blocks(self):
        """
        Extracts the diagonal blocks (each experiment) from the large similarity matrix.
        """
        blocks = []
        for i in range(self.n_movs):
            start = i * self.n_subs
            end = (i + 1) * self.n_subs
            block = self.sim_matrix[start:end, start:end]
            blocks.append(block)
        return np.array(blocks)

    def consistency_score(self, blocks):
        """
        Computes an overall consistency score across multiple similarity blocks.

        Each block is assumed to be a square similarity matrix (subjects × subjects)
        representing one experiment or condition.

        The function measures how consistent the relative distances/similarities
        are across all blocks — i.e., whether pairs that are close in one experiment
        also tend to be close in others.

        The metric used is the average Spearman correlation between
        the upper-triangular (non-redundant) parts of all blocks.

        Parameters
        ----------
        blocks : list or np.ndarray of shape (n_blocks, n, n)
            List of similarity matrices for each experiment.

        Returns
        -------
        float
            Average Spearman correlation across all pairs of blocks.
            Higher = more consistent structure between experiments.
        """
        n_blocks = len(blocks)
        corrs = []

        # Compute Spearman correlation for every pair of blocks
        for i, j in combinations(range(n_blocks), 2):
            a = blocks[i]
            b = blocks[j]

            # Take only the upper triangle (no redundancy)
            iu = np.triu_indices_from(a, k=1)
            vec_a = a[iu]
            vec_b = b[iu]

            # Spearman correlation measures relative ranking similarity
            r, _ = spearmanr(vec_a, vec_b)
            corrs.append(r)

        # Final score = average correlation across all pairs
        return np.mean(corrs)

