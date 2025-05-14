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
import os.path
from time import time

import matplotlib
import nibabel as nib
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BSpline

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from .preprocess import LoadData
from .b_spline import spline_base_funs
from .evaluate_lambda import select_lambda, compute_hat_matrices_all_lambda

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class FunctionalMRI:
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

    def __init__(self, nii_file: str, mask_file: str, degree: int, threshold: float, num_pca_comp: int,
                 batch_size: int, output_folder: str) -> None:
        """
        Initialize the fMRI processing pipeline.

        Loads data, sets up time and basis parameters, and runs the analysis.

        Parameters:
            nii_file (str): path to the 4D fMRI NIfTI file.
            mask_file (str): path to the 3D mask NIfTI file.
            degree (int): degree of the B-spline basis.
            threshold (float): Maximum allowed absolute error for interpolation.
            num_pca_comp (int): number of principal components to compute.
            batch_size (int): batch size of voxels
            output_folder (str): existing folder for output files
        """
        logging.info("Load data...")
        self.degree = degree
        self.threshold = threshold
        self.num_pca_comp = num_pca_comp
        self.batch_size = batch_size
        self.output_folder = output_folder

        data = LoadData(nii_file, mask_file)
        self.fmri_data, _, _, _, self.mask, self.nii_affine, TR = data.run_preprocessing()
        self.orig_n_voxels = self.mask.shape
        self.n_voxels = self.fmri_data.shape[0]
        self.n_timepoints = self.fmri_data.shape[1]
        self.times = np.arange(self.n_timepoints)
        self.T_min = 0
        self.T_max = self.n_timepoints * TR
        # self.n_basis = min(20, self.n_timepoints // 10)

        ##--- שלב 1: טעינת הקובץ ---
        # print("Loading NIfTI file...")
        # img = nib.load(nii_file)
        # data = img.get_fdata()  # צורת הנתונים: (X, Y, Z, Time)
        # X, Y, Z, T = data.shape
        # print(f"Data shape: {data.shape}")
        #
        ##--- שלב 2: בניית מסכת מח פשוטה ---
        # print("Building brain mask...")
        # self.mask = np.mean(data, axis=3) > 100  # ערך סף פשוט
        # self.n_voxels = np.sum(self.mask)
        # print(f"Number of brain voxels: {self.n_voxels}")
        #

    ## --- שלב 3: פריסת המידע למטריצה ---
    # print("Reshaping data...")
    ##   data_2d = data[brain_mask].T  # צורת הנתונים: (Time, Voxels)
    # data_2d = data[self.mask]  # צורת הנתונים: (Voxels, Time)
    #
    ##--- שלב 4: ניקוי בסיסי (למשל Detrend) ---
    # print("Preprocessing signals...")
    # from scipy.signal import detrend
    #    # self.fmri_data = detrend(data_2d, axis=0)
    # self.fmri_data = detrend(data_2d, axis=1)
    #

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        return instance

    def log_data(self) -> None:
        """
        Log dataset dimensions and basis parameters.
        """
        logging.info(f"# original voxels: {self.orig_n_voxels}")
        logging.info(f"# voxels after mask: {self.n_voxels}")
        logging.info(f"# timepoints: {self.n_timepoints}")
        logging.info(f"first time: {self.T_min}, last time: {self.T_max}")
        # logging.info(f"# basis functions: {self.n_basis}")

    def run(self) -> None:
        """
        Execute the main analysis pipeline:
        1. Construct B-spline basis.
        2. Build penalty matrix and fit voxel coefficients.
        3. Perform functional PCA and generate outputs.
        """
        logging.info("Constructing B-spline basis...")
        for n_basis in range(self.degree + 1, self.n_timepoints + 20):  # try increasing
            basis_funs, _ = spline_base_funs(self.T_min, self.T_max, self.degree, n_basis)

            # Build penalty matrix for regularized regression (second derivative)
            P = self.penalty_matrix(basis_funs, derivative_order=2)
            F = np.nan_to_num(basis_funs(self.times))  # (n_timepoints, n_basis)
            start = time()
            logging.info("Calculate Coefficients...")
            C = self.calculate_coeff(F, P)
            end = time()
            print("Elapsed time:", end - start, "seconds")

        logging.info("Performing functional PCA...")
        # Build penalty matrix for fPCA (no derivative)
        U = self.penalty_matrix(basis_funs, derivative_order=0)
        scores, eigvecs_sorted, eigvals_sorted = self.fPCA(C, U)
        self.draw_graphs(F, scores, eigvecs_sorted, eigvals_sorted)

    def find_min_n_basis_to_interpolate(self):
        """
        Find the minimum number of B-spline basis functions required to interpolate a time series.

        This function incrementally increases the number of basis functions until the spline fit
        can reconstruct the original signal within a specified error threshold.

        Returns:
            int: The smallest number of basis functions such that the fitted signal approximates
                 the original data within the given threshold. Returns None if no such value is found.
        """
        for n_basis in range(self.degree + 1, self.n_timepoints + 20):  # try increasing
            basis_funs, _ = spline_base_funs(self.T_min, self.T_max, self.degree, n_basis)
            F = np.nan_to_num(basis_funs(self.times))
            c = np.linalg.pinv(F) @ y
            y_hat = F @ c
            error = np.max(np.abs(y - y_hat))
            if error < self.threshold:
                return n_basis
        return None  # if not found

    def penalty_matrix_old(self, basis_funs: BSpline, derivative_order: int) -> np.ndarray:
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

    def penalty_matrix(self, basis_funs: BSpline, derivative_order: int) -> np.ndarray:
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
        x_vals = np.linspace(self.T_min, self.T_max, 1000)
        weights = np.ones_like(x_vals)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        dx = (self.T_max - self.T_min) / (1000 - 1)

        # Evaluate the basis functions at the discretized points
        basis_vals = np.nan_to_num(deriv_funs(x_vals))  # (num_points x n_basis)
        W = weights[:, None] * basis_vals
        G = basis_vals.T @ W * dx  # shape: (n_basis x n_basis)

        return G

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
            F (ndarray): basis matrix (n_timepoints, n_basis).
            P (ndarray): penalty matrix (n_basis, n_basis).

        Returns:
            C (ndarray): filled coefficient matrix (n_voxels, n_basis).
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

    def fPCA(self, C: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform functional PCA on the coefficient matrix.

        Parameters:
            C (ndarray): coefficient matrix (n_voxels, n_basis).
            U (ndarray): penalty matrix (n_basis, n_basis).

        Returns:
            scores (ndarray): voxel scores on principal components (n_voxels, num_pca_comp).
            eigvecs_sorted (ndarray): basis-space eigenvectors sorted by importance (n_basis, num_pca_comp).
            eigvals_sorted (ndarray): Corresponding sorted eigenvalues.
        """
        n_voxels, n_basis = C.shape
        C_tilde = C - C.mean(axis=1, keepdims=True)

        # cov_mat = (C_tilde.T @ C_tilde @ U) / n_voxels
        cov_mat = (C_tilde.T @ C_tilde) / n_voxels
        cov_mat = (cov_mat + cov_mat.T) / 2

        eigvals, eigvecs = np.linalg.eigh(cov_mat)
        sorted_indices = np.argsort(eigvals)[::-1][:self.num_pca_comp]
        eigvecs_sorted = eigvecs[:, sorted_indices]
        eigvals_sorted = eigvals[sorted_indices]

        scores = np.zeros((n_voxels, self.num_pca_comp))
        for i in range(self.num_pca_comp):
            # scores[:, i] = C_tilde @ U @ eigvecs_sorted[:, i]
            scores[:, i] = C_tilde @ eigvecs_sorted[:, i]

        return scores, eigvecs_sorted, eigvals_sorted

    def draw_graphs(self, F: np.ndarray, scores: np.ndarray, eigvecs_sorted: np.ndarray,
                    eigvals_sorted: np.ndarray) -> None:
        """
        Generate and save voxel importance maps and eigenfunction intensity plots.

        Parameters:
            F (ndarray): basis matrix (n_timepoints, n_basis).
            scores (ndarray): voxel scores (n_voxels, num_pca_comp).
            eigvecs_sorted (ndarray): principal component vectors.
            eigvals_sorted (ndarray): Corresponding sorted eigenvalues.
        """

        for i in range(self.num_pca_comp):
            explained_vairance_i = (eigvals_sorted[i] * 100) / np.sum(eigvals_sorted)
            logging.info(f"Saving voxel-wise importance map for first eigenfunction {i}...")
            importance_map = np.zeros(self.orig_n_voxels)
            print("shape of scores ", scores[:, i].shape)
            importance_map[self.mask > 0] = scores[:, i]
            importance_nii = nib.Nifti1Image(importance_map, affine=self.nii_affine)
            impmap_file = os.path.join(self.output_folder, f"eigenfunction_{i}_importance_map.nii.gz")
            nib.save(importance_nii, impmap_file)

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
            impmap_fig = os.path.join(self.output_folder, f"eigenfunction_{i}_importance_map.png")
            plt.savefig(impmap_fig, dpi=300, bbox_inches='tight')
            plt.show()

            logging.info(f"Plotting signal intensity for eigenfunction {i}...")
            signal_intensity = F @ eigvecs_sorted[:, i]
            plt.figure(figsize=(10, 4))
            plt.plot(self.times, signal_intensity, color='blue')
            plt.title(f'Signal Intensity: Eigenfunction {i} ({explained_vairance_i}% var)')
            plt.xlabel('Time (scans)')
            plt.ylabel('Intensity')
            plt.grid(True)
            intence_fig = os.path.join(self.output_folder, f"eigenfunction_{i}_signal_intensity.png")
            plt.savefig(intence_fig, dpi=300, bbox_inches='tight')
            plt.show()

        logging.info(f"Plotting original average signal intensity ...")
        signal_intensity = np.average(self.fmri_data, axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(self.times, signal_intensity, color='blue')
        plt.title(f'Original average Signal Intensity')
        plt.xlabel('Time (scans)')
        plt.ylabel('Intensity')
        plt.grid(True)
        intence_fig = os.path.join(self.output_folder, f"original_averaged_signal_intensity.png")
        plt.savefig(intence_fig, dpi=300, bbox_inches='tight')
        plt.show()
