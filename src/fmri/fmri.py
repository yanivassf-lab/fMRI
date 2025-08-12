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
        n_basis (int or 'auto'): Number of basis functions. Use 'auto' to determine it automatically based on the
                                 interpolation threshold (default: 'auto').
        threshold (float): maximum allowed mean error for interpolation.
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

    def __init__(self, nii_file: str, mask_file: str, degree: int, n_basis: int, threshold: float, num_pca_comp: int,
                 batch_size: int, output_folder: str, TR: float, processed: bool, calc_penalty_accurately: bool) -> None:
        """
        Initialize the fMRI processing pipeline.

        Loads data, sets up time and basis parameters, and runs the analysis.

        Parameters:
            nii_file (str): path to the 4D fMRI NIfTI file.
            mask_file (str): path to the 3D mask NIfTI file.
            degree (int): degree of the B-spline basis.
            n_basis (int): Number of basis functions. Use 0 to determine it automatically based on the
                                 interpolation threshold.
            n_basis (int): number of basis functions.
            threshold (float): maximum allowed absolute error for interpolation.
            num_pca_comp (int): number of principal components to compute.
            batch_size (int): batch size of voxels
            output_folder (str): existing folder for output files
            TR (float): repetition time in seconds.
            processed (bool): if True, the data is already preprocessed (e.g., filtered, smoothed).
            calc_penalty_accurately (bool): if True, the penalty matrix will be calculated using an accurate method.
                                          If False, an approximate method will be used.
        """
        logging.info("Load data...")
        self.degree = degree
        self.threshold = threshold
        self.num_pca_comp = num_pca_comp
        self.batch_size = batch_size
        self.output_folder = output_folder
        self.calc_penalty_accurately = calc_penalty_accurately
        data = LoadData(nii_file, mask_file)
        self.fmri_data, self.mask, self.nii_affine, TR = data.load_data(TR=TR, processed=processed)  # Load fMRI data and mask
        self.orig_n_voxels = self.mask.shape
        self.n_voxels = self.fmri_data.shape[0]
        self.n_timepoints = self.fmri_data.shape[1]
        self.times = np.arange(self.n_timepoints)
        self.T_min = 0
        self.T_max = self.n_timepoints * TR
        self.n_basis = n_basis  # min(20, self.n_timepoints // 10)

        ## step 1: load the NIfTI file and mask
        # print("Loading NIfTI file...")
        # img = nib.load(nii_file)
        # data = img.get_fdata()  # shape: (X, Y, Z, Time)
        # X, Y, Z, T = data.shape
        # print(f"Data shape: {data.shape}")
        #
        ## step 2: build a simple brain mask
        # print("Building brain mask...")
        # self.mask = np.mean(data, axis=3) > 100  # simple threshold to create a mask
        # self.n_voxels = np.sum(self.mask)
        # print(f"Number of brain voxels: {self.n_voxels}")
        #

    # print("Reshaping data...")
    ##   data_2d = data[brain_mask].T  # צורת הנתונים: (Time, Voxels)
    # data_2d = data[self.mask]  # צורת הנתונים: (Voxels, Time)
    #
    ## step 4: basic preprocessing (e.g., Detrend)
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
        logging.info(f"# basis functions: {self.n_basis if self.n_basis else 'find according the threshold'}")

    def run(self) -> None:
        """
        Execute the main analysis pipeline:
        1. Construct B-spline basis.
        2. Build penalty matrix and fit voxel coefficients.
        3. Perform functional PCA and generate outputs.
        """
        logging.info("Constructing B-spline basis...")
        C, F, basis_funs, best_lambdas = self.calculate_interpolation()
        logging.info("Performing functional PCA...")
        # Build penalty matrix for fPCA (no derivative)
        U = self.penalty_matrix(basis_funs, derivative_order=0) if not self.calc_penalty_accurately else self.penalty_matrix_accurate(basis_funs, derivative_order=0)
        scores, eigvecs_sorted, eigvals_sorted, v_max_scores_pos = self.fPCA(C, U)
        self.draw_graphs(C, F, scores, eigvecs_sorted, eigvals_sorted, v_max_scores_pos, best_lambdas)

    def calculate_interpolation(self):
        if self.n_basis:
            C, F, basis_funs, mean_error, best_lambdas = self.calculate_n_basis_interpolation(self.n_basis)
            return C, F, basis_funs, best_lambdas
        else:  # if user set threshold instead fixed number of n_basis
            n_basis_errors = []
            range_n_basis = range(self.degree + 1, self.n_timepoints + 20, 10)
            n_basis_errors = np.zeros(len(range_n_basis))
            for i, n_basis in enumerate(range_n_basis):  # try increasing
                C, F, basis_funs, mean_error, best_lambdas = self.calculate_n_basis_interpolation(n_basis)
                n_basis_errors[i] = mean_error
                if mean_error <= self.threshold:
                    logging.info(f"The minimum number of basis functions that interporate the data is {n_basis}.")
                    self.n_basis = n_basis
                    return C, F, basis_funs, best_lambdas

            self.n_basis = range_n_basis[np.argmin(n_basis_errors)]
            logging.info(
                f"Cannot achieve interpolation threshold, continue with {self.n_basis} basis functions with {np.min(n_basis_errors):.2f} mean error.")
            C, F, basis_funs, mean_error, best_lambdas = self.calculate_n_basis_interpolation(self.n_basis)
            return C, F, basis_funs, best_lambdas

    def calculate_n_basis_interpolation(self, n_basis):
        basis_funs, _ = spline_base_funs(self.T_min, self.T_max, self.degree, n_basis)
        # Build penalty matrix for regularized regression (second derivative)
        P = self.penalty_matrix(basis_funs, derivative_order=2) if not self.calc_penalty_accurately else self.penalty_matrix_accurate(basis_funs, derivative_order=2)
        F = np.nan_to_num(basis_funs(self.times))  # (n_timepoints, n_basis)
        start = time()
        logging.info(f"Calculate coefficients for {n_basis} basis functions...")
        C, best_lambdas = self.calculate_coeff(F, P, n_basis)  # shape: (n_voxels, n_basis)
        end = time()
        logging.info(f"Elapsed time: {end - start} seconds")
        Y_hat = C @ F.T  # (n_voxels, n_timepoints)
        mean_error = np.mean(np.abs(self.fmri_data - Y_hat))
        logging.info(f"The mean interpolation error for {n_basis} basis functions is {mean_error:.2f}.")
        return C, F, basis_funs, mean_error, best_lambdas

    def penalty_matrix_accurate(self, basis_funs: BSpline, derivative_order: int) -> np.ndarray:
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

    def calculate_coeff(self, F: np.ndarray, P: np.ndarray, n_basis: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit coefficients for spline basis functions to each voxel time series with optimal λ, in batches.

        Parameters:
            F (ndarray): basis matrix (n_timepoints, n_basis).
            P (ndarray): penalty matrix (n_basis, n_basis).

        Returns:
            C (ndarray): filled coefficient matrix (n_voxels, n_basis).
            best_labmdas (ndarray): best lambdas for each voxel
        """

        logging.info("Fitting basis to each voxel with optimal λ in batches...")

        # Coeff matrix
        C = np.zeros((self.n_voxels, n_basis))  # (n_voxels x n_basis)

        # Precompute the basis matrix and related matrices for efficiency.
        FtF = F.T @ F  # (n_basis, n_basis)
        lambda_values = np.linspace(0.01, 1.0, 100)

        # Compute H matrices for all lambda values.
        # H_all_lambda: (n_lambda, n_timepoints, n_timepoints)
        H_all_lambda = compute_hat_matrices_all_lambda(F, FtF, P, self.n_timepoints, lambda_values)

        # Create a batch of identity matrices for computing I - H.
        I = np.eye(self.n_timepoints)[None, :, :].repeat(len(lambda_values), axis=0)
        I_minus_H = I - H_all_lambda  # (n_lambda, n_timepoints, n_timepoints)

        best_lambdas = np.zeros(self.n_voxels)
        # Process voxels in batches.
        for i in range(0, self.n_voxels, self.batch_size):
            end = min(i + self.batch_size, self.n_voxels)
            voxel_data_batch = self.fmri_data[i:end]  # (batch_size, n_timepoints)

            # Use a vectorized function to select the best lambda for all voxels in the batch.
            best_lambdas_batch, _, _ = select_lambda(I_minus_H, voxel_data_batch, self.n_timepoints, lambda_values)

            # Compute coefficients for the entire batch in a vectorized way.
            coeff_batch = self.compute_coeff_by_regularized_regression(F, FtF, P, best_lambdas_batch, voxel_data_batch)

            # Store the computed coefficients in the overall array.
            C[i:end, :] = coeff_batch
            best_lambdas[i:end] = best_lambdas_batch
        return C, best_lambdas

    def fPCA(self, C: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform functional PCA on the coefficient matrix.

        Parameters:
            C (ndarray): coefficient matrix (n_voxels, n_basis).
            U (ndarray): penalty matrix (n_basis, n_basis).

        Returns:
            scores (ndarray): voxel scores on principal components (n_voxels, num_pca_comp).
            eigvecs_sorted (ndarray): basis-space eigenvectors sorted by importance (n_basis, num_pca_comp).
            eigvals_sorted (ndarray): corresponding sorted eigenvalues.
            v_max_scores_pos (ndarray): positions of voxels with maximum score in each component (num_pca_comp, )
        """
        n_voxels, n_basis = C.shape
        C_tilde = C - C.mean(axis=0, keepdims=True)  # mean along the voxels.

        cov_mat = (C_tilde.T @ C_tilde @ U) / n_voxels
        cov_mat = (cov_mat + cov_mat.T) / 2

        eigvals, eigvecs = np.linalg.eigh(cov_mat)
        sorted_indices = np.argsort(eigvals)[::-1][:self.num_pca_comp]
        eigvecs_sorted = eigvecs[:, sorted_indices]
        eigvals_sorted = eigvals[sorted_indices]

        scores = np.zeros((n_voxels, self.num_pca_comp))
        for i in range(self.num_pca_comp):
            scores_i = C_tilde @ U @ eigvecs_sorted[:, i]  # shape: (n_voxels,)
            if np.sum(scores_i) < 0:  # if most of the scores are negative replace the sign
                scores_i = -scores_i
                eigvecs_sorted[:, i] = -eigvecs_sorted[:, i]
            scores[:, i] = scores_i
        v_max_scores_pos = np.argmax(scores, axis=0)  # maximum score in each component
        print(v_max_scores_pos, np.max(scores, axis=0))
        return scores, eigvecs_sorted, eigvals_sorted, v_max_scores_pos

    def voxel_index_to_coord(self, k):
        flat_indices = np.flatnonzero(self.mask)
        flat_index = flat_indices[k]
        return np.unravel_index(flat_index, self.mask.shape)

    def draw_graphs(self, C: np.ndarray, F: np.ndarray, scores: np.ndarray, eigvecs_sorted: np.ndarray,
                    eigvals_sorted: np.ndarray, v_max_scores_pos: np.ndarray, best_lambdas: np.ndarray) -> None:
        """
        Generate and save voxel importance maps and eigenfunction intensity plots.

        Parameters:
            F (ndarray): basis matrix (n_timepoints, n_basis).
            scores (ndarray): voxel scores (n_voxels, num_pca_comp).
            eigvecs_sorted (ndarray): principal component vectors.
            eigvals_sorted (ndarray): corresponding sorted eigenvalues.
            v_max_scores_pos (ndarray): positions of voxels with maximum score in each component (num_pca_comp, )
            best_lambdas (ndarray): best lambdas for each voxel (n_voxels,)
        """

        for i in range(self.num_pca_comp):
            explained_vairance_i = (eigvals_sorted[i] * 100) / np.sum(eigvals_sorted)
            logging.info(f"Saving voxel-wise importance map for first eigenfunction {i}...")
            importance_map = np.zeros(self.orig_n_voxels)
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
            # plt.show()

            logging.info(f"Plotting signal intensity for eigenfunction {i}...")
            signal_intensity = F @ eigvecs_sorted[:, i]
            plt.figure(figsize=(10, 4))
            plt.plot(self.times, signal_intensity, color='blue')
            plt.title(f'Signal Intensity: Eigenfunction {i} ({explained_vairance_i}% var)')
            plt.xlabel('Time (scans)')
            plt.ylabel('Intensity')
            plt.grid(True)
            intence_txt = os.path.join(self.output_folder, f"eigenfunction_{i}_signal_intensity.txt")
            with open(intence_txt, 'w') as f:
                f.write(f"#Eigenfunction {i}:\n")
                f.write(f"#Times:\n{' '.join(map(str,self.times))}\n")
                f.write(f"#Signal intensity:\n{' '.join(map(str,signal_intensity))}\n")
            intence_fig = os.path.join(self.output_folder, f"eigenfunction_{i}_signal_intensity.png")
            plt.savefig(intence_fig, dpi=300, bbox_inches='tight')
            # plt.show()

            v_max_scores_pos_i = v_max_scores_pos[i]
            Y_v_max_score = self.fmri_data[v_max_scores_pos_i]
            Y_hat_v_max_score = C[v_max_scores_pos_i] @ F.T  # (1, n_timepoints)
            mean_error_max_score = np.mean(np.abs(Y_v_max_score - Y_hat_v_max_score))
            best_lambda = best_lambdas[v_max_scores_pos_i]
            logging.info(f"Plotting signal intensity for best voxel in eigenfunction {i}...")
            plt.figure(figsize=(10, 4))
            plt.plot(self.times, Y_hat_v_max_score, color='blue')
            plt.scatter(self.times, Y_v_max_score, color='red')
            plt.title(f'Signal and best voxel\'s fitted function: Eigenfunction {i}')
            plt.xlabel('Time (scans)')
            plt.ylabel('Intensity')
            plt.grid(True)
            ax = plt.gca()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            # Compute position near bottom-right
            x = xlim[1] - 0.01 * (xlim[1] - xlim[0])  # Slightly inside from right
            y = ylim[0] + 0.01 * (ylim[1] - ylim[0])  # Slightly above bottom
            x_c,y_c,z_c = self.voxel_index_to_coord(v_max_scores_pos_i)
            plt.text(x, y, rf"$\lambda={best_lambda}, score: {scores[v_max_scores_pos_i, i]:.2f}, x={x_c};y={y_c};z={z_c}$", fontsize=12, ha='right', va='bottom')
            # Compute position near upper-left
            # x_c,y_c,z_c = self.voxel_index_to_coord(v_max_scores_pos_i)
            # x = xlim[0] + 0.01 * (xlim[1] - xlim[0])  # Slightly inside from the left
            # y = ylim[1] - 0.01 * (ylim[1] - ylim[0])  # Slightly under top
            # plt.text(x, y, f"$x={x_c};y={y_c};z={z_c}$", fontsize=12, ha='left', va='top')
            best_voxel_txt = os.path.join(self.output_folder, f"eigenfunction_{i}_best_voxel.txt")
            with open(best_voxel_txt, 'w') as f:
                f.write(f"#Eigenfunction {i}:\n")
                f.write(f"#Times:\n{' '.join(map(str,self.times))}\n")
                f.write(f"#Y_estimated:\n{' '.join(map(str,Y_hat_v_max_score))}\n")
                f.write(f"#Y_real:\n{' '.join(map(str,Y_v_max_score))}\n")
            best_voxel_fig = os.path.join(self.output_folder, f"eigenfunction_{i}_best_voxel.png")
            plt.savefig(best_voxel_fig, dpi=300, bbox_inches='tight')
            # plt.show()

        logging.info(f"Plotting original average signal intensity ...")
        signal_intensity = np.average(self.fmri_data, axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(self.times, signal_intensity, color='blue')
        plt.title(f'Original average Signal Intensity')
        plt.xlabel('Time (scans)')
        plt.ylabel('Intensity')
        plt.grid(True)
        orig_intence_txt = os.path.join(self.output_folder, f"original_averaged_signal_intensity.txt")
        with open(orig_intence_txt, 'w') as f:
            f.write(f"#Times:\n{' '.join(map(str, self.times))}\n")
            f.write(f"#Signal intensity:\n{' '.join(map(str, signal_intensity))}\n")
        orig_intence_fig = os.path.join(self.output_folder, f"original_averaged_signal_intensity.png")
        plt.savefig(orig_intence_fig, dpi=300, bbox_inches='tight')
        # plt.show()
