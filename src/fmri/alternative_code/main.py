import numpy as np
import logging

from evaluate_lambda import select_lambda,compute_hat_matrices_all_lambda, select_lambda_at_once
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_coeff(F, FtF, P, lambda_const, y):
    """
    Compute the regularized coefficients for basis function approximation.
    """
    coeff = np.linalg.inv(FtF + lambda_const * P) @ F.T @ y
    return coeff

# We use an alternative function that computes the integral using the quad method
def penalty_matrix_discretization(basis_funs, T_min, T_max, num_points=1000, derivative_order=0):
    """
    Approximate the penalty matrix using discrete integration (trapezoidal rule).

    Used as a faster alternative to `penalty_matrix()` when slight approximation is acceptable.

    Parameters:
    - basis_funs: BSpline object containing basis functions.
    - T_min, T_max: Time interval over which to integrate.
    - num_points: Number of discretization points.
    - derivative_order: Order of derivative (0 for identity, 2 for curvature penalty).

    Returns:
    - G: (n_basis x n_basis) approximated penalty matrix.
    """
    # Compute the derivative of the basis functions
    deriv_funs = basis_funs.derivative(nu=derivative_order)
    x_vals = np.linspace(T_min, T_max, num_points)
    weights = np.ones_like(x_vals)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    dx = (T_max - T_min) / (num_points - 1)

    # Evaluate the basis functions at the discretized points
    basis_vals = np.nan_to_num(deriv_funs(x_vals))  # (num_points x n_basis)
    W = weights[:, None] * basis_vals
    G = basis_vals.T @ W * dx  # shape: (n_basis x n_basis)

    return G


# This function is not referenced in the paper and is not utilized in the current implementation
def reconstruct_from_pca(F, pca_coeff, eigvecs_sorted, C_mean, basis_funs, T_eval):
    """
    Reconstruct smoothed voxel-wise functions from PCA components.
    """
    n_voxels, num_pca_comp = pca_coeff.shape
    n_timepoints = len(T_eval)

    G = F @ eigvecs_sorted[:, :num_pca_comp]  # (n_timepoints x num_pca_comp)
    f_mean = F @ C_mean.reshape(-1)  # (n_timepoints,)
    f_hat = f_mean[np.newaxis, :] + pca_coeff @ G.T  # (n_voxels x n_timepoints)
    return f_hat

# Different versions of calculate_coeff function

def calculate_coeff_loop_H_loop_voxels_loop_lambda(C, times, n_timepoints, n_basis, basis_funs, n_voxels, P, fmri_data):
    logging.info("Fitting basis to each voxel with optimal λ...")
    F = np.nan_to_num(basis_funs(times))  # (n_timepoints x n_basis)
    FtF = F.T @ F # (n_basis x n_basis)
    I = np.eye(n_timepoints) # (n_timepoints x n_timepoints)
    lambda_values = np.linspace(0.01, 1.0, 100)
    for v in range(n_voxels):
        y = fmri_data[v]
        optimal_lambda, _, _ = select_lambda(F, FtF, P, y, lambda_values,I, n_timepoints, None)
        optimal_lambda = 0.1
        C[v, :] = compute_coeff(F, FtF, P, optimal_lambda, y)

def calculate_coeff_vectorized_H_loop_voxels_loop_lamba(C, times, n_timepoints, n_basis, basis_funs, n_voxels, P, fmri_data):
    logging.info("Fitting basis to each voxel with optimal λ...")
    F = np.nan_to_num(basis_funs(times))  # (n_timepoints x n_basis)
    FtF = F.T @ F # (n_basis x n_basis)
    I = np.eye(n_timepoints) # (n_timepoints x n_timepoints)
    lambda_values = np.linspace(0.01, 1.0, 100)
    H_all_lambda = compute_hat_matrices_all_lambda(F, FtF, P, n_timepoints, n_basis, lambda_values)
    for v in range(n_voxels):
        y = fmri_data[v]
        optimal_lambda, _, _ = select_lambda(F, FtF, P, y, lambda_values,I, n_timepoints, H_all_lambda)
        optimal_lambda = 0.1
        C[v, :] = compute_coeff(F, FtF, P, optimal_lambda, y)

def calculate_coeff_vectorized_H_loop_voxels_vectorized_lamba(C, times, n_timepoints, n_basis, basis_funs, n_voxels, P, fmri_data):
    logging.info("Fitting basis to each voxel with optimal λ...")
    F = np.nan_to_num(basis_funs(times))  # (n_timepoints x n_basis)
    FtF = F.T @ F # (n_basis x n_basis)
    lambda_values = np.linspace(0.01, 1.0, 100)
    H_all_lambda = compute_hat_matrices_all_lambda(F, FtF, P, n_timepoints, n_basis, lambda_values) #(n_lambda_values, n_timepoints, n_timepoints)
    I = np.eye(n_timepoints)[None,:,:].repeat(len(lambda_values), axis=0) # (n_lambda_values,n_timepoints,n_timepoints)
    I_minus_H = I - H_all_lambda
    for v in range(n_voxels):
        y = fmri_data[v]
        optimal_lambda, _, _ = select_lambda_at_once(I_minus_H, y, n_timepoints, lambda_values)
        optimal_lambda = 0.1
        C[v, :] = compute_coeff(F, FtF, P, optimal_lambda, y)


