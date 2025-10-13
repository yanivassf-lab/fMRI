import unittest
import numpy as np
from fmri.peaks_similarity import PeaksSimilarity

class TestPeaksSimilarity(unittest.TestCase):
    def test_score_combined(self):
        # Create a dummy PeaksSimilarity instance

        dummy = PeaksSimilarity(3, [1,2,3], alpha=0.5)
        dummy.subs_num = 3
        dummy.movements = [1, 2,3]  # 2 movements
        dummy.alpha = 0.5

        # Construct a 9x9 similarity matrix (3 subs per movement)
        dummy.sim_matrix = np.array([
            # Movement 1 (subs 0-2)
            [1.0, 0.8, 0.7, 0.2, 0.3, 0.1, 0.3, 0.2, 0.2],
            [0.8, 1.0, 0.6, 0.2, 0.1, 0.3, 0.3, 0.2, 0.1],
            [0.7, 0.6, 1.0, 0.3, 0.2, 0.2, 0.1, 0.3, 0.2],

            # Movement 2 (subs 3-5)
            [0.2, 0.2, 0.3, 1.0, 0.7, 0.8, 0.1, 0.2, 0.3],
            [0.3, 0.1, 0.2, 0.7, 1.0, 0.6, 0.2, 0.1, 0.2],
            [0.1, 0.3, 0.2, 0.8, 0.6, 1.0, 0.3, 0.2, 0.1],

            # Movement 3 (subs 6-8)
            [0.3, 0.3, 0.1, 0.1, 0.2, 0.3, 1.0, 0.8, 0.7],
            [0.2, 0.2, 0.3, 0.2, 0.1, 0.2, 0.8, 1.0, 0.6],
            [0.2, 0.1, 0.2, 0.3, 0.2, 0.1, 0.7, 0.6, 1.0]
        ])
        dummy._score_combined()
        # Expected values for test
        # Within-movement (off-diagonal inside each movement block)
        expected_within_mov = 0.7  # average of all off-diagonal elements within movements

        # Between-movement (diagonal of off-diagonal blocks)
        expected_between_mov = 0.1666666666  # average of diagonals in off-diagonal blocks

        # Weighted score with alpha=0.5
        alpha = 0.5
        expected_weighted = alpha * expected_between_mov + (1 - alpha) * expected_within_mov
        # expected_weighted ≈ 0.4333

        # print("sim_matrix:\n", sim_matrix)
        print("Expected within-movement avg:", expected_within_mov)
        print("Expected between-movement avg:", expected_between_mov)
        print("Expected weighted score:", expected_weighted)
        self.assertAlmostEqual(dummy.score_within_mov_avg, expected_within_mov)
        self.assertAlmostEqual(dummy.score_between_mov_avg, expected_between_mov)
        self.assertAlmostEqual(dummy.weighted_score, expected_weighted)

if __name__ == "__main__":
    unittest.main()
