import logging
import os
import numpy as np


def setup_logger(output_folder, file_name, loger_name, log_level=logging.INFO):
    log_file = os.path.join(output_folder, file_name)
    logger = logging.getLogger(loger_name)
    logger.setLevel(log_level)
    logger.handlers = []
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file, mode='w')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


class PCS:
    __array_priority__ = 1000.0  # ensure our __array__ / methods are preferred

    def __init__(self, eigvecs_sorted, skip_pc_num):
        self._pcs = eigvecs_sorted
        self._skip_pc_num = set(skip_pc_num)  # faster lookups

    def __len__(self):
        return self._pcs.shape[1] - len(self._skip_pc_num)

    def get_orig_pcs(self):
        return self._pcs

    def get_skip_pc_num(self):
        return self._skip_pc_num

    def get_dummy_idx(self, real_idx):
        # get the index in the filtered pcs corresponding to the original index
        dummy_idx = real_idx
        for j in self._skip_pc_num:
            if j == real_idx:
                return None  # this pc is skipped
            if j < real_idx:
                dummy_idx -= 1
        return dummy_idx

    def _real_idx(self, idx):
        # handle negative indices
        if idx < 0:
            idx = len(self) + idx
        real_idx = idx
        for j in sorted(self._skip_pc_num):
            if j <= real_idx:
                real_idx += 1
        return real_idx

    def __getitem__(self, idx):
        r_idx = self._real_idx(idx)
        return self._pcs[:, r_idx]

    def __iter__(self):
        for j, pc in enumerate(self._pcs.T):
            if j not in self._skip_pc_num:
                yield pc

    def pc_name(self, idx) -> str:
        return f"{self._real_idx(idx)}"

    @property
    def shape(self):
        n_rows = self._pcs.shape[0]
        n_cols = self._pcs.shape[1] - len(self._skip_pc_num)
        return (n_rows, n_cols)

    def transpose(self):
        kept_cols = [j for j in range(self._pcs.shape[1]) if j not in self._skip_pc_num]
        return self._pcs[:, kept_cols].T

    @property
    def T(self):
        return self.transpose()

    def _filtered(self):
        """Return the filtered matrix (kept columns only)."""
        kept_cols = [j for j in range(self._pcs.shape[1]) if j not in self._skip_pc_num]
        return self._pcs[:, kept_cols]

    # make numpy view of this object behave like the filtered matrix
    def __array__(self, dtype=None):
        arr = self._filtered()
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __matmul__(self, other):
        """Right multiplication: PCS @ other"""
        return self._filtered() @ other

    def __rmatmul__(self, other):
        """Left multiplication: other @ PCS"""
        return other @ self._filtered()


if __name__ == "__main__":
    eigvecs_sorted = np.array([[0, 0, 0],
                               [1, 1, 1],
                               [2, 2, 2],
                               [3, 3, 3],
                               [4, 4, 4]]).T
    skip_pc_num = [1]

    pcs = PCS(eigvecs_sorted, skip_pc_num)

    print("len:", len(pcs))  # 4
    print("shape:", pcs.shape)  # (3, 4)
    print("shape 0:", pcs.shape[0])  # (3)
    print("shape 1:", pcs.shape[1])  # (4)

    print(pcs[0])  # [0 0 0]
    print(pcs[1])  # [2 2 2]
    print(pcs[2])  # [3 3 3]
    print(pcs[-1])  # [4 4 4]
    print(pcs[-2])  # [3 3 3]
    # print(pcs[:, :])  # all kept columns

    print("pc_name(1):", pcs.pc_name(1))  # '2'

    print("\nIteration on the columns:")
    for pc in pcs:
        print(pc)
    # Output:
    # (0,0,0)
    # (2,2,2)
    # (4,4,4)

    print("\nTranspose:")
    print(pcs.transpose())
    print(pcs.T)

    print("\nList of PCS")
    pcs2 = PCS(eigvecs_sorted, skip_pc_num)
    pcs_list = [pcs, pcs2]
    print("\nFirst item, pcs2:")
    print(pcs_list[0][2])

    # Matrix multiplication
    print("\nMatrix multiplication left:")
    other = np.ones((pcs.shape[1], 2))  # 3 kept columns × 2
    result = pcs @ other
    print(result.shape)  # (3 rows, 2 columns)

    print("\nMatrix multiplication right:")
    other = np.ones((2, pcs.shape[0]))  # 3 kept columns × 2
    result = other @ pcs
    print(result)
    print(result.shape)  # (2 rows, 3 columns)

    print(pcs)  # [0 0 0]
    print(pcs[0])  # [0 0 0]
    print(pcs[:,0])  # [0 0 0]
