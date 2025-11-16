import logging

import numpy as np

from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve
from time import time
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def gcv_score_one_voxel_one_lambda(voxel_data, H, I, n_timepoints):
    """
    Compute the generalized cross-validation (GCV) score.

    GCV(λ) = [n_timepoints * ||(I-H) voxel_data||^2] / [trace(I-H)]^2

    Parameters:
    - voxel_data: (n_timepoints,) array of observed time series data for a voxel.
    - H: (n_timepoints x n_timepoints) hat matrix.
    - I: (n_timepoints x n_timepoints) eye function
    - n_timepoints: the length of the time series data


    Returns:
    - score: The computed GCV score (float).
    """
    I_minus_H = I - H
    residual_norm_sq = norm(I_minus_H @ voxel_data) ** 2
    trace_I_minus_H = np.trace(I_minus_H)
    score = (n_timepoints * residual_norm_sq) / (trace_I_minus_H ** 2)
    return score

# Used by calculate_coeff_loop_H_loop_voxels_loop_lambda and calculate_coeff_vectorized_H_loop_voxels_loop_lamba functions
def select_lambda(F, FtF, P, voxel_data, lambda_values, I, n_timepoints, H_all_lambda):
    """
    For an array of lambda candidates, compute the GCV score
    and select the lambda with the lowest GCV.

    Parameters:
    - F: (n_timepoints x n_basis) matrix of basis functions evaluated at timepoints.
    - FtF: (n_basis x n_basis) = F.T @ F. recomputed for efficiency.
    - P: (n_basis x n_basis) penalty matrix.
    - voxel_data: (n_timepoints,) array of observed time series data for a voxel.
    - lambda_values: (N_lambda,) array of candidate lambda values.
    - I: (n_timepoints x n_timepoints) eye function
    - n_timepoints: the length of the time series data

    Returns:
    - best_lambda: The lambda value that minimizes the GCV score.
    - scores: (N_lambda,) array of computed GCV scores for each lambda value.
    """

    # batch_size = 100
    # for i in range(0, n_voxels, batch_size):
    #     batch_voxels = fmri_data[i:i + batch_size]  # shape: (batch_size, n_timepoints)
    #
    #     # Loop over lambda values
    #     for lambda_const in lambda_values:
    #         H = compute_hat_matrix(F, P, lambda_const)
    #         I_minus_H = np.eye(H.shape[0]) - H
    #         residuals = batch_voxels @ I_minus_H.T  # shape: (batch_size, n_timepoints)
    #         residual_norm_sq = np.sum(residuals ** 2, axis=1)
    #         trace = np.trace(I_minus_H)
    #         scores = (F.shape[0] * residual_norm_sq) / (trace ** 2)

    scores = []
    for i, lambda_const in enumerate(lambda_values):
        score = gcv_score_one_voxel_one_lambda(voxel_data, H_all_lambda[i, :, :], I, n_timepoints)
        scores.append(score)
    scores = np.array(scores)
    best_idx = np.argmin(scores)
    best_lambda = lambda_values[best_idx]
    best_score = scores[best_idx]
    return best_lambda, best_score, scores

################################################################################################

# Used by calculate_coeff_vectorized_H_loop_voxels_vectorized_lamba and calculate_coeff_vectorized_H_loop_voxels_loop_lamba functions
def select_lambda_at_once(I_minus_H, voxel_data, n_timepoints, lambda_values):
    """
    Compute the generalized cross-validation (GCV) score.

    GCV(λ) = [n_timepoints * ||(I-H) voxel_data||^2] / [trace(I-H)]^2

    Parameters:
    - voxel_data: (n_timepoints,) array of observed time series data for a voxel.
    - H: (n_lambda_values x n_timepoints x n_timepoints) hat matrix.
    - I: (n_lambda_values x n_timepoints x n_timepoints) eye function
    - n_timepoints: the length of the time series data


    Returns:
    - score: The computed GCV score (float).
    """
    residual_norm_sq = norm(I_minus_H @ voxel_data, axis=1)**2 # dim: (n_lambda_values, n_timepoints)
    trace_I_minus_H = np.trace(I_minus_H, axis1=1, axis2=2) # dim: (n_lambda_values, )
    scores = (n_timepoints * residual_norm_sq) / (trace_I_minus_H ** 2) # dim: (n_lambda_values, )

    best_idx = np.argmin(scores)
    best_lambda = lambda_values[best_idx]
    best_score = scores[best_idx]
    return best_lambda, best_score, scores



################################################################################################


def compute_hat_matrix_efficient(F, P, lambda_const):
    """
    Compute the penalized regression hat matrix (dim: (n_timepoints, n_timepoints)).

    Parameters:
    - F: (n_timepoints x n_basis) matrix of basis functions evaluated at timepoints.
    - P: (n_basis x n_basis) penalty matrix.
    - lambda_const: Regularization constant for penalization.

    Returns:
    - H_lambda: (n_timepoints x n_timepoints) hat matrix.
    """
    # More efficient way to compute:
    # H_lambda = F @ np.linalg.inv(FtF + lambda_const * P) @ F.T  # np.linalg.pinv

    # Precompute matrix A = F^T F + lambda_const * P
    A = F.T @ F + lambda_const * P

    # Perform Cholesky factorization of A.
    c, lower = cho_factor(A, lower=True)

    # Solve for X in A X = F.T. Each column of F.T is solved independently.
    # This effectively computes X = A^{-1} F.T without forming the inverse explicitly.
    X = cho_solve((c, lower), F.T)

    # Now compute the hat matrix
    H_lambda = F @ X
    return H_lambda

def compute_hat_matrix(F, FtF, P, lambda_const):
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
    return H_lambda


# Used by calculate_coeff_vectorized_H_loop_voxels_loop_lamba and calculate_coeff_vectorized_H_loop_voxels_vectorized_lamba fucntions
def compute_hat_matrices_all_lambda(F, FtF, P, n_timepoints, n_basis, lambda_values):
    H_all_lambda = np.zeros((len(lambda_values), n_timepoints, n_timepoints)) # dim: (n_lambda_values, n_timepoints, n_timepoints)
    for i, lambda_const in enumerate(lambda_values):
        # logging.info(f"\t\tCompute hat matrix for lambda value: {lambda_const}")
        # H = compute_hat_matrix(F, FtF, P, lambda_const) # less efficient version
        H_all_lambda[i, :, :] = compute_hat_matrix(F, P, lambda_const) # (len(lambda_values), n_timepoints, n_timepoints)
        # H_all_lambda[i, :, :] = compute_hat_matrix(F, FtF, P, lambda_const)
    return H_all_lambda
