import logging

import numpy as np

logger = logging.getLogger("fmri_logger")


def compute_hat_matrix(F: np.ndarray, FtF: np.ndarray, P: np.ndarray, lambda_const: float) -> np.ndarray:
    """
    Compute the penalized regression hat matrix (dim: (n_timepoints, n_timepoints)).

    Parameters:
    - F: (n_timepoints x n_basis) matrix of basis functions evaluated at timepoints.
    - FtF: (n_basis x n_basis) = F.T @ F. recomputed for efficiency.
    - P: (n_basis x n_basis) penalty matrix.
    - lambda_const: Regularization constant for penalization.

    Returns:
    - H_lambda: (n_timepoints x n_basis) hat matrix.
    """
    H_lambda = F @ np.linalg.inv(FtF + lambda_const * P) @ F.T  # np.linalg.pinv
    # H_lambda = F @ np.linalg.pinv(FtF + lambda_const * P, rcond=1e-10) @ F.T
    return H_lambda


def compute_hat_matrices_all_lambda(F, FtF, P, n_timepoints, lambda_values):
    """
    Compute a set of hat matrices for a range of regularization parameters (lambda values).

    The hat matrix H maps the observed data to the fitted values in spline regression:
        H = F @ inv(F.T @ F + lambda * P) @ F.T
    This function computes H for each lambda in the provided list.

    Parameters:
        F (ndarray): Basis matrix of shape (n_timepoints, n_basis).
        FtF (ndarray): Precomputed matrix product F.T @ F of shape (n_basis, n_basis).
        P (ndarray): Penalty matrix of shape (n_basis, n_basis).
        n_timepoints (int): Number of timepoints (length of time series).
        lambda_values (ndarray): 1D array of regularization parameters.

    Returns:
        H_all_lambda (ndarray): Array of shape (n_lambda, n_timepoints, n_timepoints),
                                where each slice [i, :, :] is the hat matrix for lambda_values[i].
    """
    n_lambda = len(lambda_values)
    H_all_lambda = np.zeros((n_lambda, n_timepoints, n_timepoints))
    for i, lambda_const in enumerate(lambda_values):
        # logger.info(f"\t\tCompute hat matrix for lambda value: {lambda_const}")
        H_all_lambda[i, :, :] = compute_hat_matrix(F, FtF, P, lambda_const)  # (n_lambda, n_timepoints, n_timepoints)
    return H_all_lambda


def select_lambda(I_minus_H: np.ndarray, voxel_data_batch: np.ndarray,
                  n_timepoints: int, lambda_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized GCV selection over a batch of voxels.

    For each voxel in voxel_data_batch (shape: (n_voxels, n_timepoints)),
    compute the GCV score for each lambda candidate using:

        GCV(λ) = [n_timepoints * ||(I-H(λ)) y||^2] / [trace(I-H(λ))]^2,

    where I_minus_H has shape (n_lambda, n_timepoints, n_timepoints).

    Parameters:
      I_minus_H: 3D array of shape (n_lambda, n_timepoints, n_timepoints)
                 containing I - H for each candidate lambda.
      voxel_data_batch: 2D array of shape (n_voxels, n_timepoints), where each row is a voxel’s time series.
      n_timepoints: Integer, the length of the time series.
     lambda_values: 1D array of candidate lambda values (length = n_lambda).

    Returns:
      best_lambda: 1D array (n_voxels,) of the best lambda for each voxel.
      best_score: 1D array (n_voxels,) of the best (minimum) GCV score for each voxel.
      scores: 2D array of shape (n_lambda, n_voxels) with the computed GCV score for each candidate lambda and each voxel.
    """

    # Compute residuals for all lambda candidates and voxels.
    # We need to compute (I_minus_H @ voxel) for each lambda and each voxel.
    # Rearrange: let voxel_data_batch have shape (n_voxels, n_timepoints).
    # Then, using np.matmul:
    #   R = I_minus_H @ voxel_data_batch.T gives shape (n_lambda, n_timepoints, n_voxels)
    R = np.matmul(I_minus_H, voxel_data_batch.T)  # shape: (n_lambda, n_timepoints, n_voxels)

    # Compute the squared norm along the time axis for each lambda and voxel:
    # Resulting shape: (n_lambda, n_voxels)
    res_norm_sq = np.sum(R ** 2, axis=1)

    # Compute trace of I_minus_H for each lambda candidate (independent of voxel)
    # Shape: (n_lambda,)
    traces = np.trace(I_minus_H, axis1=1, axis2=2)

    # Now compute the GCV score for each lambda and each voxel:
    # Using broadcasting, reshape traces to (n_lambda, n_voxels)
    gcv_scores = (n_timepoints * res_norm_sq) / (traces[:, None] ** 2)

    # For each voxel, find the candidate (axis 0) with the minimal score.
    best_idx = np.argmin(gcv_scores, axis=0)  # shape: (n_voxels,)

    best_lambda = lambda_values[best_idx]  # shape: (n_voxels,)
    best_gcv_scores = gcv_scores[best_idx, np.arange(gcv_scores.shape[1])]  # shape: (n_voxels,)

    return best_lambda, best_gcv_scores, gcv_scores
