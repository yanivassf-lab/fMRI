import numpy as np
from scipy.linalg import eigh


def solve_batch_gcv(voxel_data_batch: np.ndarray,
                    F_V: np.ndarray,
                    VT_FT: np.ndarray,
                    V: np.ndarray,
                    evals: np.ndarray,
                    lambda_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves the penalized regression problem for a batch of voxels using
    Generalized Cross Validation (GCV) via Generalized Eigenvalue Decomposition.

    Mathematical Logic:
    Instead of inverting (F.T@F + lambda*P) for every lambda, we use the
    decomposition: (F.T@F)V = (P)V@Lambda.
    This allows us to diagonalize the problem.

    Parameters:
        voxel_data_batch (ndarray): Shape (n_voxels, n_timepoints). Observed data.
        F_V (ndarray): Shape (n_timepoints, n_basis). Precomputed F @ V.
        VT_FT (ndarray): Shape (n_basis, n_timepoints). Precomputed V.T @ F.T.
        V (ndarray): Shape (n_basis, n_basis). Eigenvectors from GEVD.
        evals (ndarray): Shape (n_basis,). Generalized eigenvalues.
        lambda_values (ndarray): Array of candidate regularization parameters.

    Returns:
        coeffs (ndarray): Shape (n_voxels, n_basis). Fitted B-spline coefficients.
        best_lambdas (ndarray): Shape (n_voxels,). Optimal lambda per voxel.
        best_gcv_scores (ndarray): Shape (n_voxels,). Minimum GCV score per voxel.
    """
    n_voxels, n_timepoints = voxel_data_batch.shape
    n_lambdas = len(lambda_values)

    # 1. Project data onto the generalized eigen-basis
    # We compute: Z = V.T @ F.T @ y
    # Shape: (n_basis, n_voxels)
    Z = VT_FT @ voxel_data_batch.T

    # Initialize arrays to store scores
    # Shape: (n_lambdas, n_voxels)
    gcv_scores_all = np.zeros((n_lambdas, n_voxels))

    # 2. Iterate over lambda values to calculate GCV scores
    # This loop is fast because it involves only diagonal operations (vector scaling)
    for i, lam in enumerate(lambda_values):
        # Calculate the scaling vector for the inverse: 1 / (evals + lambda)
        # Shape: (n_basis, 1) to allow broadcasting
        D_inv = 1.0 / (evals + lam)
        D_inv = D_inv[:, np.newaxis]

        # Calculate fitted coefficients in the eigen-basis for this lambda
        # C_eig = D_inv * Z
        C_eig = Z * D_inv

        # Calculate fitted values (y_hat)
        # y_hat = F @ V @ C_eig = F_V @ C_eig
        # Shape: (n_timepoints, n_voxels)
        Y_hat = F_V @ C_eig

        # Calculate Residual Sum of Squares (RSS) per voxel
        # Shape: (n_voxels,)
        rss = np.sum((voxel_data_batch.T - Y_hat) ** 2, axis=0)

        # Calculate Effective Degrees of Freedom (Trace of Hat Matrix)
        # tr(H) = sum( evals / (evals + lambda) )
        tr_H = np.sum(evals / (evals + lam))

        # Calculate GCV Score
        # GCV = (RSS / n) / (1 - tr_H / n)^2
        denominator = (1 - tr_H / n_timepoints) ** 2
        gcv_scores_all[i, :] = (rss / n_timepoints) / denominator

    # 3. Select optimal Lambda
    # Find the index of the minimum GCV score for each voxel
    best_indices = np.argmin(gcv_scores_all, axis=0)  # Shape: (n_voxels,)

    best_lambdas = lambda_values[best_indices]
    best_gcv_scores = gcv_scores_all[best_indices, np.arange(n_voxels)]

    # 4. Reconstruct the final coefficients for the optimal lambdas
    # We need to compute C = V @ (1/(evals + best_lambda) * Z) specific to each voxel

    # Create a matrix of the optimal scaling factors per voxel
    # Shape: (n_basis, n_voxels)
    # broadcasting: (n_basis, 1) + (n_voxels,) -> (n_basis, n_voxels)
    optimal_D_inv = 1.0 / (evals[:, np.newaxis] + best_lambdas[np.newaxis, :])

    # Scale the projected data Z by the optimal inverse term
    Z_scaled = Z * optimal_D_inv

    # Transform back to the original B-spline basis
    # C = V @ Z_scaled
    # Shape: (n_basis, n_basis) @ (n_basis, n_voxels) -> (n_basis, n_voxels)
    coeffs_transposed = V @ Z_scaled

    return coeffs_transposed.T, best_lambdas, best_gcv_scores
