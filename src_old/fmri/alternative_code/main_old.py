"""
Module for performing functional PCA on fMRI data.

This module defines an fMRI class that:
  1. Loads and logs fMRI data from NIfTI files and a mask.
  2. Constructs B-spline basis functions.
  3. Computes penalty matrices for smoothing.
  4. Estimates voxel-wise spline coefficients via regularized regression.
  5. Performs functional PCA on the coefficient matrix.
  6. Generates and saves voxel importance maps and eigenfunction intensity plots.
"""

import logging
from time import time

import matplotlib
import nibabel as nib
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BSpline

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from preprocess import LoadData
from b_spline_skfda import spline_base_funs
from evaluate_lambda import select_lambda, compute_hat_matrices_all_lambda

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# parameters:
# ===========
BATCH_SIZE = 200
DEGREE = 3
NUM_PCA_COMP = 3


class fMRI:
    """
    Class to perform functional PCA on fMRI data.

    Attributes:
        degree (int): degree of the B-spline basis.
        num_pca_comp (int): number of principal components to compute.
        fmri_data (ndarray): voxel-wise time series after masking.
        mask (ndarray): binary mask of voxels to include.
        nii_affine (ndarray): affine transform of the NIfTI image.
        times (ndarray): array of time indices.
        T_min (float): start time for basis functions.
        T_max (float): end time (scan duration).
        n_basis (int): number of B-spline basis functions.
        n_voxels (int): number of voxels after masking.
        n_timepoints (int): number of time points per voxel.
    """

    def __init__(self, nii_file: str, mask_file: str, degree: int = DEGREE, num_pca_comp: int = NUM_PCA_COMP,
                 batch_size: int = BATCH_SIZE) -> None:
        """
        Initialize the fMRI processing pipeline.

        Loads data, sets up time and basis parameters, and runs the analysis.

        Parameters:
            nii_file (str): path to the 4D fMRI NIfTI file.
            mask_file (str): path to the 3D mask NIfTI file.
            degree (int): degree of the B-spline basis.
            num_pca_comp (int): number of principal components to compute.
            batch_size (int): batch size of voxels
        """
        logging.info("Load data...")
        self.degree = degree
        self.num_pca_comp = num_pca_comp
        self.batch_size = batch_size
        self.fmri_data, _, _, self.mask, self.nii_affine, TR = LoadData(nii_file, mask_file)
        self.orig_n_voxels = self.mask.shape
        self.n_voxels = self.fmri_data.shape[0]
        self.n_timepoints = self.fmri_data.shape[1]
        self.times = np.arange(self.n_timepoints)
        self.T_min = 0
        self.T_max = self.n_timepoints * TR
        self.n_basis = min(20, self.n_timepoints // 10)
        self.log_data()
        self.run()

    def log_data(self) -> None:
        """
        Log dataset dimensions and basis parameters.
        """
        logging.info(f"# original voxels: {self.orig_n_voxels}")
        logging.info(f"# voxels after mask: {self.n_voxels}")
        logging.info(f"# timepoints: {self.n_timepoints}")
        logging.info(f"first time: {self.T_min}, last time: {self.T_max}")
        logging.info(f"# basis functions: {self.n_basis}")

    def run(self) -> None:
        """
        Execute the main analysis pipeline:
        1. Construct B-spline basis.
        2. Build penalty matrix and fit voxel coefficients.
        3. Perform functional PCA and generate outputs.
        """
        logging.info("Constructing B-spline basis...")
        basis_funs, _ = spline_base_funs(self.T_min, self.T_max, self.degree, self.n_basis)

        # Build penalty matrix
        P = self.penalty_matrix(basis_funs, derivative_order=2)
        F = np.nan_to_num(basis_funs(self.times))  # (n_timepoints, n_basis)
        start = time()
        logging.info("Calculate Coefficients...")
        C = self.calculate_coeff(F, P)
        end = time()
        print("Elapsed time:", end - start, "seconds")

        logging.info("Performing functional PCA...")
        scores, eigvecs_sorted = self.fPCA(C, basis_funs)
        self.draw_graphs(F, scores, eigvecs_sorted)

    def penalty_matrix(self, basis_funs: BSpline, derivative_order: int = 0) -> np.ndarray:
        """
        Compute the penalty matrix by integrating products of k-th derivatives of basis functions.

        Parameters:
            basis_funs: BSpline object for basis functions.
            derivative_order (int): order of derivative to penalize.

        Returns:
            G (ndarray): penalty matrix of shape (n_basis, n_basis).
        """
        deriv_funs = basis_funs.derivative(nu=derivative_order)
        n_basis = deriv_funs.c.shape[1]
        G = np.zeros((n_basis, n_basis))

        for i in range(n_basis):
            for j in range(i, n_basis):
                f = lambda t: (
                    np.nan_to_num(deriv_funs(np.atleast_1d(t))[0, i]) *
                    np.nan_to_num(deriv_funs(np.atleast_1d(t))[0, j])
                )
                integral, _ = quad(f, self.T_min, self.T_max)
                G[i, j] = integral
                G[j, i] = integral

        return G

    def fPCA(self, C: np.ndarray, basis_funs: BSpline) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform functional PCA on the coefficient matrix.

        Parameters:
            C (ndarray): coefficient matrix (n_voxels, n_basis).
            basis_funs: BSpline object to compute Gram matrix.

        Returns:
            scores (ndarray): voxel scores on principal components (n_voxels, num_pca_comp).
            eigvecs_sorted (ndarray): basis-space eigenvectors sorted by importance (n_basis, num_pca_comp).
        """
        n_voxels, n_basis = C.shape
        C_tilde = C - C.mean(axis=1, keepdims=True)

        U = self.penalty_matrix(basis_funs, derivative_order=0)
        cov_mat = (C_tilde.T @ C_tilde @ U) / n_voxels
        cov_mat = (cov_mat + cov_mat.T) / 2

        eigvals, eigvecs = np.linalg.eigh(cov_mat)
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvecs_sorted = eigvecs[:, sorted_indices]

        scores = np.zeros((n_voxels, self.num_pca_comp))
        for i in range(self.num_pca_comp):
            scores[:, i] = C_tilde @ U @ eigvecs_sorted[:, i]

        return scores, eigvecs_sorted

    def compute_coeff_by_regularized_regression(self, F: np.ndarray, FtF: np.ndarray, P: np.ndarray,
                                                lambda_vec: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        """
        Compute regularized spline coefficients for a batch of voxels.

        Solves coefficient c_i by (FtF + λ_i P) c_i = F.T y_i for each voxel i in the batch.
        (FtF + lambda_vec[i] * P) * coeff_i = F.T @ y_batch[i]

        Parameters:
            F (ndarray): basis matrix (n_timepoints, n_basis).
            FtF (ndarray): precomputed F.T @ F (n_basis, n_basis).
            P (ndarray): penalty matrix (n_basis, n_basis).
            lambda_vec (ndarray): per-voxel optimal λ values (n_voxels,).
            y_batch (ndarray): voxel time series (n_voxels, n_timepoints).

        Returns:
            coeffs (ndarray): per-voxel estimated coefficients (n_voxels, n_basis).
        """
        # Construct the system matrices in batch:
        A_batch = FtF[None, :, :] + lambda_vec[:, None, None] * P[None, :, :]  # shape: (n_voxels, n_basis, n_basis)

        # Compute the right-hand side for each voxel:
        # For voxel i, RHS[i] = F.T @ y_batch[i]. Compute for all voxels at once:
        # F.T is (n_basis, n_timepoints) and y_batch.T is (n_timepoints, n_voxels),
        # so the product has shape (n_basis, n_voxels); then transpose to get (n_voxels, n_basis).
        RHS = (F.T @ y_batch.T).T  # shape: (n_voxels, n_basis)
        RHS = RHS[..., None]  # shape: (n_voxels, n_basis, 1)

        # Solve the linear system for each voxel in the batch.
        # np.linalg.solve supports batched matrices if A_batch.shape is (n_voxels, n_basis, n_basis)
        # and RHS is (n_voxels, n_basis).
        coeffs_exp = np.linalg.solve(A_batch, RHS)  # shape: (n_voxels, n_basis)
        coeffs = coeffs_exp.squeeze(-1)
        return coeffs

    def calculate_coeff(self, F: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Fit coefficients for spline basis functions to each voxel time series with optimal λ, in batches.

        Parameters:
            C (ndarray): output coefficient matrix to fill (n_voxels, n_basis).
            F (ndarray): basis matrix (n_timepoints, n_basis).
            P (ndarray): penalty matrix (n_basis, n_basis).
            batch_size (int): number of voxels per batch.

        Returns:
            C (ndarray): filled coefficient matrix.
        """

        logging.info("Fitting basis to each voxel with optimal λ in batches...")

        # Coeff matrix
        C = np.zeros((self.n_voxels, self.n_basis))  # (n_voxels x n_basis)

        # Precompute the basis matrix and related matrices for efficiency.
        FtF = F.T @ F  # (n_basis, n_basis)
        lambda_values = np.linspace(0.01, 1.0, 100)

        # Compute H matrices for all lambda values.
        # H_all_lambda: (n_lambda, n_timepoints, n_timepoints)
        H_all_lambda = compute_hat_matrices_all_lambda(F, FtF, P, self.n_timepoints, lambda_values)

        # Create a batch of identity matrices for computing I - H.
        I = np.eye(self.n_timepoints)[None, :, :].repeat(len(lambda_values), axis=0)
        I_minus_H = I - H_all_lambda  # (n_lambda, n_timepoints, n_timepoints)

        # Process voxels in batches.
        for i in range(0, self.n_voxels, self.batch_size):
            end = min(i + self.batch_size, self.n_voxels)
            voxel_data_batch = self.fmri_data[i:end]  # (batch_size, n_timepoints)

            # Use a vectorized function to select the best lambda for all voxels in the batch.
            best_lambda_batch, _, _ = select_lambda(I_minus_H, voxel_data_batch, self.n_timepoints, lambda_values)

            # Compute coefficients for the entire batch in a vectorized way.
            coeff_batch = self.compute_coeff_by_regularized_regression(F, FtF, P, best_lambda_batch, voxel_data_batch)

            # Store the computed coefficients in the overall array.
            C[i:end, :] = coeff_batch
        return C

    def draw_graphs(self, F: np.ndarray, scores: np.ndarray, eigvecs_sorted: np.ndarray) -> None:
        """
        Generate and save voxel importance maps and eigenfunction intensity plots.

        Parameters:
            F (ndarray): basis matrix (n_timepoints, n_basis).
            scores (ndarray): voxel scores (n_voxels, num_pca_comp).
            eigvecs_sorted (ndarray): principal component vectors.
        """

        for i in range(self.num_pca_comp):
            logging.info(f"Saving voxel-wise importance map for first eigenfunction {i}...")
            importance_map = np.zeros(self.orig_n_voxels)
            importance_map[self.mask > 0] = scores[:, i]
            importance_nii = nib.Nifti1Image(importance_map, affine=self.nii_affine)
            nib.save(importance_nii, f"eigenfunction_{i}_importance_map.nii.gz")

            # Display the slice
            logging.info(f"voxel-wise importance map for eigenfunction {i}...")
            # Pick the middle slice along Z-axis
            z_middle = importance_map.shape[2] // 2
            slice_img = importance_map[:, :, z_middle]
            plt.figure(figsize=(6, 6))
            plt.imshow(slice_img.T, cmap='hot', origin='lower')
            plt.title(f'Middle Slice of Importance Map Eigenfunction {i}')
            plt.colorbar(label='Importance')
            plt.axis('off')
            plt.savefig(f"eigenfunction_{i}_importance_map.png", dpi=300, bbox_inches='tight')
            plt.show()

            logging.info(f"Plotting signal intensity for eigenfunction {i}...")
            signal_intensity = F @ eigvecs_sorted[:, i]
            plt.figure(figsize=(10, 4))
            plt.plot(self.times, signal_intensity, color='blue')
            plt.title(f'Signal Intensity: Eigenfunction {i}')
            plt.xlabel('Time (scans)')
            plt.ylabel('Intensity')
            plt.grid(True)
            plt.savefig(f"eigenfunction_{i}_signal_intensity.png", dpi=300, bbox_inches='tight')
            plt.show()
