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

import os.path
from time import time

import matplotlib
import nibabel as nib
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import quad
from scipy.interpolate import BSpline
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.operators import gram_matrix
from skfda.representation.basis import BSplineBasis

from .b_spline_skfda import spline_base_funs as spline_base_funs_skfda
from .b_spline_bspline import spline_base_funs as spline_base_funs_bspline

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .preprocess import LoadData
from .evaluate_lambda import select_lambda, compute_hat_matrices_all_lambda
from .gcv_solver import solve_batch_gcv

import logging

logger = logging.getLogger("fmri_logger")
FPCA_CRITICAL_WARNING = 'fPCA-CRITICAL-WARNING'

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

    def __init__(self, nii_file: str, mask_file: str, degree: int, n_basis: list[int], threshold: float,
                 num_pca_comp: int,
                 batch_size: int, output_folder: str, TR: float, smooth_size: int, lambda_min: float, lambda_max: float,
                 derivatives_num_p: int, derivatives_num_u: int, processed: bool, bad_margin_size: int,
                 no_penalty: bool = False, calc_penalty_bspline_accurately: bool = False,
                 calc_penalty_skfda: bool = False, n_skip_vols_start: int = 0, n_skip_vols_end: int = 0,
                 highpass: float = 0.01, lowpass: float = 0.08, low_mem:bool = False,
                 use_nilearn_filter: bool = False, n_compcor_nilearn_filter: int = 5,
                 smoothing_fwhm_nilearn_filter: float = 6.0) -> None:
        """
        Initialize the fMRI processing pipeline.

        Loads data, sets up time and basis parameters, and runs the analysis.

        Parameters:
            nii_file (str): path to the 4D fMRI NIfTI file.
            mask_file (str): path to the 3D mask NIfTI file.
            degree (int): degree of the B-spline basis.
            n_basis (int): Number of basis functions. Use 0 to determine it automatically based on the
                                 interpolation threshold.
            threshold (float): maximum allowed absolute error for interpolation.
            num_pca_comp (int): number of principal components to compute.
            batch_size (int): batch size of voxels
            output_folder (str): existing folder for output files
            TR (float): repetition time in seconds.
            smooth_size (int): box size of smoothing kernel.
            lambda_min (float): minimum lambda value.
            lambda_max (float): maximum lambda value.
            derivatives_num_p (int): number of derivatives in calculation of penalty matrix P.
            derivatives_num_u (int): number of derivatives in calculation of penalty matrix U.
            processed (bool): if True, the data is already preprocessed (e.g., filtered, smoothed).
            bad_margin_size (int): Size of the margin to ignore in calculating direction of eigvecs.
            no_penalty (bool): if True, no penalty will be used.
            calc_penalty_bspline_accurately (bool): if True, the penalty matrix will be calculated using spline package with an accurate method.
                                                    If False, an approximate method will be used also with spline package.
            calc_penalty_skfda (bool): if True, the penalty matrix will be calculated using skfda.gram_matrix.
            n_skip_vols_start (int): Number of initial fMRI volumes to discard from the beginning of the signal.
            n_skip_vols_end (int): Number of initial fMRI volumes to discard from the end of the signal.
            highpass (float): High-pass filter cutoff frequency in Hz. Filters out slow drifts below this frequency.
            lowpass (float): Low-pass filter cutoff frequency in Hz. Filters out high-frequency noise above this frequency.
            low_mem (bool): if True, only output text files without saving any images.
            use_nilearn_filter (bool): if True, use Nilearn for filtering.
            n_compcor_nilearn_filter (int): number of compcor components for Nilearn filtering.
            smoothing_fwhm_nilearn_filter (float): FWHM for smoothing in Nilearn filtering.
        """
        # Store all parameters
        self.nii_file = nii_file
        self.mask_file = mask_file
        self.degree = degree
        self.n_basis_list = n_basis
        self.threshold = threshold
        self.num_pca_comp = num_pca_comp
        self.batch_size = batch_size
        self.output_folder = output_folder
        self.TR = TR
        self.smooth_size = smooth_size
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.derivatives_num_p = derivatives_num_p
        self.derivatives_num_u = derivatives_num_u
        self.processed = processed
        self.bad_margin_size = bad_margin_size
        self.no_penalty = no_penalty
        self.calc_penalty_bspline_accurately = calc_penalty_bspline_accurately
        self.calc_penalty_skfda = calc_penalty_skfda
        self.highpass = highpass
        self.lowpass = lowpass
        self.low_mem = low_mem
        self.use_nilearn_filter = use_nilearn_filter
        self.n_compcor_nilearn_filter = n_compcor_nilearn_filter
        self.smoothing_fwhm_nilearn_filter = smoothing_fwhm_nilearn_filter

        # Load fMRI data
        logger.info("Load data...")
        data = LoadData(nii_file, mask_file, TR=TR, smooth_size=smooth_size, highpass=self.highpass, lowpass=self.lowpass, use_nilearn=self.use_nilearn_filter, n_compcor=self.n_compcor_nilearn_filter, nilearn_smoothing_fwhm=self.smoothing_fwhm_nilearn_filter)
        fmri_data_all, self.mask, self.nii_affine = data.load_data(processed=processed, save_filtered_file=False)

        self.fmri_data = fmri_data_all[:, n_skip_vols_start:fmri_data_all.shape[1] - n_skip_vols_end]
        # Derived values
        self.orig_n_voxels = self.mask.shape
        self.n_voxels = self.fmri_data.shape[0]
        self.n_timepoints = self.fmri_data.shape[1]
        self.times = np.arange(n_skip_vols_start, self.n_timepoints + n_skip_vols_start) * self.TR
        self.T_min = self.times[0]
        self.T_max = self.times[-1]  # Assuming time points are indexed from 0 to n_timepoints-1
        self.n_basis = None

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        return instance

    def log_data(self, argv) -> None:
        """
        Log all dataset parameters and pipeline configuration.
        """
        logger.info("======== fMRI Processing Configuration ========")
        logger.info(f"Command line: {' '.join(argv)}")

        # Input
        logger.info(f"NIfTI file: {self.nii_file}")
        logger.info(f"Mask file: {self.mask_file}")
        logger.info(f"Output folder: {self.output_folder}")

        # Data
        logger.info(f"Original voxel dimensions: {self.orig_n_voxels}")
        logger.info(f"fmri_data shape (voxels × timepoints): {self.fmri_data.shape}")
        logger.info(f"Voxels after mask: {self.n_voxels}")
        logger.info(f"Number of timepoints: {self.n_timepoints}, starting from {self.T_min} to {self.T_max}")
        logger.info(f"Time range: {self.T_min} to {self.T_max}")
        logger.info(f"Repetition time (TR): {self.TR}")
        logger.info(f"Smoothing kernel size: {self.smooth_size}")
        logger.info(f"Preprocessed input: {self.processed}")

        # Basis / PCA
        logger.info(f"B-spline degree: {self.degree}")
        logger.info(f"# basis functions: {self.n_basis_list if self.n_basis_list != [0] else self.n_timepoints} ")
        logger.info(f"Interpolation threshold: {self.threshold}")
        logger.info(f"# PCA components: {self.num_pca_comp}")

        # Regularization / Penalty
        logger.info(f"Lambda range: {self.lambda_min} – {self.lambda_max}")
        logger.info(f"Penalty: {'disabled' if self.no_penalty else 'enabled'}")
        if not self.no_penalty:
            logger.info(f"Penalty matrix (derivatives P): {self.derivatives_num_p}")
            logger.info(f"Penalty matrix (derivatives U): {self.derivatives_num_u}")
            logger.info(f"Penalty (accurate B-spline): {self.calc_penalty_bspline_accurately}")
            logger.info(f"Penalty (skfda): {self.calc_penalty_skfda}")

        # Misc
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Bad margin size: {self.bad_margin_size}")
        logger.info(f"High-pass filter cutoff: {self.highpass} Hz")
        logger.info(f"Low-pass filter cutoff: {self.lowpass} Hz")
        logger.info(f"Low memory mode (only text outputs): {self.low_mem}")

        logger.info("================================================")

    def run(self) -> None:
        """
        Execute the main analysis pipeline:
        1. Construct B-spline basis.
        2. Build penalty matrix and fit voxel coefficients.
        3. Perform functional PCA and generate outputs.
        """
        logger.info("Constructing B-spline basis...")
        C, F, basis_funs, best_lambdas = self.calculate_interpolation()
        logger.info("Performing functional PCA...")
        # Build penalty matrix for fPCA (no derivative)
        # U is the L2 Gram matrix, even when smoothing is disabled.
        if self.derivatives_num_u != 0:
            logger.warning(f"{FPCA_CRITICAL_WARNING}: The paper requires derivatives_num_u = 0")

        if self.no_penalty:
            U = np.eye(self.n_basis)
        elif self.calc_penalty_skfda:
            U = self.penalty_matrix_skfda(basis_funs, derivative_order=self.derivatives_num_u)
        elif self.calc_penalty_bspline_accurately:
            U = self.penalty_matrix_bspline_accurate(basis_funs, derivative_order=self.derivatives_num_u)
        else:
            U = self.penalty_matrix_bspline(basis_funs, derivative_order=self.derivatives_num_u)

        scores, eigvecs_sorted, eigvals_sorted, v_max_scores_pos, pc_temporal_profiles, total_variance = self.fPCA(C, U,
                                                                                                                   F)
        # save F,C in the instance for group running.
        self.F = F
        self.U = U
        self.draw_graphs(C, F, scores, eigvecs_sorted, eigvals_sorted, v_max_scores_pos, best_lambdas,
                         pc_temporal_profiles, total_variance)

    def calculate_interpolation(self):
        if self.n_basis_list != [0] and len(self.n_basis_list) == 1:
            self.n_basis = self.n_basis_list[0]
            C, F, basis_funs, mean_error, best_lambdas = self.calculate_n_basis_interpolation(self.n_basis)
            return C, F, basis_funs, best_lambdas
        elif self.n_basis_list == [0]:
            self.n_basis = self.n_timepoints
            C, F, basis_funs, mean_error, best_lambdas = self.calculate_n_basis_interpolation(self.n_basis)
            return C, F, basis_funs, best_lambdas
        else:  # if user set list instead fixed number of n_basis
            n_basis_errors = np.zeros(len(self.n_basis_list))
            for i, n_basis in enumerate(self.n_basis_list):  # try increasing
                C, F, basis_funs, mean_error, best_lambdas = self.calculate_n_basis_interpolation(n_basis)
                n_basis_errors[i] = mean_error
                if mean_error <= self.threshold:
                    logger.info(f"The minimum number of basis functions that interporate the data is {n_basis}.")
                    self.n_basis = n_basis
                    return C, F, basis_funs, best_lambdas

            self.n_basis = self.n_basis_list[np.argmin(n_basis_errors)]
            logger.info(
                f"Cannot achieve interpolation threshold, continue with {self.n_basis} basis functions with {np.min(n_basis_errors):.2f} mean error.")
            C, F, basis_funs, mean_error, best_lambdas = self.calculate_n_basis_interpolation(self.n_basis)
            # Save the coefficient matrix for group-level analysis
            return C, F, basis_funs, best_lambdas

    def calculate_n_basis_interpolation(self, n_basis):
        if self.calc_penalty_skfda:
            basis_funs, _ = spline_base_funs_skfda(self.T_min, self.T_max, self.degree, n_basis)
        else:
            basis_funs, _ = spline_base_funs_bspline(self.T_min, self.T_max, self.degree, n_basis)
        # Build penalty matrix for regularized regression (second derivative)
        if self.no_penalty:
            P = np.eye(n_basis)
        else:
            if self.calc_penalty_skfda:
                P = self.penalty_matrix_skfda(basis_funs, derivative_order=self.derivatives_num_p)
            elif self.calc_penalty_bspline_accurately:
                P = self.penalty_matrix_bspline_accurate(basis_funs, derivative_order=self.derivatives_num_p)
            else:
                P = self.penalty_matrix_bspline(basis_funs, derivative_order=self.derivatives_num_p)
        if self.calc_penalty_skfda:
            F = np.nan_to_num(basis_funs(
                self.times)).squeeze().T  # before transpose: (n_basis, n_timepoints, 1), after transpose: (n_timepoints, n_basis)
        else:
            # BSpline package:
            F = np.nan_to_num(basis_funs(self.times))  # (n_timepoints, n_basis)

        start = time()
        logger.info(f"Calculate coefficients for {n_basis} basis functions...")
        C, best_lambdas = self.calculate_coeff(F, P, n_basis)  # shape: (n_voxels, n_basis)
        end = time()
        logger.info(f"Elapsed time: {end - start} seconds")
        Y_hat = C @ F.T  # (n_voxels, n_timepoints)
        mean_error = np.mean(np.abs(self.fmri_data - Y_hat))
        logger.info(f"The mean interpolation error for {n_basis} basis functions is {mean_error:.2f}.")
        return C, F, basis_funs, mean_error, best_lambdas

    def penalty_matrix_skfda(self, basis_funs: BSplineBasis, derivative_order: int) -> np.ndarray:
        """
        Compute the penalty matrix using scikit-fda (analytical).

        Parameters:
            basis_funs: BSplineBasis object for basis functions.
            derivative_order (int): order of derivative to penalize.

        Returns:
            G (ndarray): penalty matrix of shape (n_basis, n_basis).
        """
        logger.info(f"Computing (accurately) penalty matrix for {derivative_order}-th derivative using scikit-fda...")

        # n_basis = basis_funs.n_basis

        # Create a LinearDifferentialOperator for the specified derivative order
        weights = np.zeros(derivative_order + 1)
        weights[derivative_order] = 1
        operator = LinearDifferentialOperator(weights=weights)

        # Use gram_matrix function to compute the penalty matrix
        penalty_matrix = gram_matrix(operator, basis_funs)

        # Convert to numpy array if it's not already
        if hasattr(penalty_matrix, 'toarray'):
            penalty_matrix = penalty_matrix.toarray()

        return penalty_matrix

    def penalty_matrix_bspline_accurate(self, basis_funs: BSpline, derivative_order: int) -> np.ndarray:
        """
        Compute the penalty matrix by integrating products of k-th derivatives of basis functions.

        Parameters:
            basis_funs: BSpline object for basis functions.
            derivative_order (int): order of derivative to penalize.

        Returns:
            G (ndarray): penalty matrix of shape (n_basis, n_basis).
        """
        logger.info(f"Computing (accurately) penalty matrix for {derivative_order}-th derivative...")
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

    # work only with BSpline package: (don't work for scikit-fda)
    def penalty_matrix_bspline(self, basis_funs: BSpline, derivative_order: int) -> np.ndarray:
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
        logger.info(f"Computing (approximately) penalty matrix for {derivative_order}-th derivative...")
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
        coeffs_exp = np.linalg.solve(A_batch, RHS)  # shape: (n_voxels, n_basis, 1)
        coeffs = coeffs_exp.squeeze(-1)
        return coeffs

    def calculate_coeff(self, F, P, n_basis):
        """
        Fit spline coefficients to each voxel time series, selecting an
        optimal regularization parameter lambda using Generalized Cross
        Validation (GCV).

        Mathematical Logic:
            For each voxel, the penalized regression problem is:

                minimize_c  ||y - F @ c||^2 + lambda * c.T @ P @ c

            Define the discrete Gram matrix A = F.T @ F.
            The generalized eigenvalue decomposition solves:

                P @ V = A @ V @ diag(d)

            with the normalization:

                V.T @ A @ V = I
                V.T @ P @ V = diag(d)

            Writing c = V @ a diagonalizes the regression problem.
            Each transformed coordinate is multiplied by:

                1 / (1 + lambda * d_k)

            This formulation permits a singular penalty matrix P.
            Its expected polynomial nullspace is explicitly preserved:
            directions with d_k = 0 remain unpenalized for every lambda.

        Optimization:
            Compute the generalized eigenvalue decomposition once and
            reuse it across all voxel batches and candidate lambdas.
            The batch solver evaluates GCV using diagonal shrinkage,
            avoiding a separate matrix inversion for every lambda.

            When self.no_penalty is enabled, fit ordinary least squares
            directly and return zero lambdas without running GCV.

        Parameters:
            F (ndarray):
                Shape (n_timepoints, n_basis). Sampled spline basis matrix.
            P (ndarray):
                Shape (n_basis, n_basis). Derivative penalty matrix.
            n_basis (int):
                Number of spline basis functions.

        Returns:
            C (ndarray):
                Shape (n_voxels, n_basis). Fitted spline coefficients.
            best_lambdas (ndarray):
                Shape (n_voxels,). Selected lambda for each voxel.
                Contains zeros when self.no_penalty is enabled.

        Assumptions:
            The penalized path requires A = F.T @ F to be positive definite.
            For derivative order q, the penalty is expected to have exactly
            q null directions corresponding to polynomials of degree
            below q, and to be positive on the remaining directions.
        """
        # 1. Prepare the voxel data and check dimensional compatibility.
        # Each row contains one voxel's observed time series.
        # Shape: (n_voxels, n_timepoints)
        Y = np.asarray(self.fmri_data, dtype=float)

        # The basis must have the requested number of columns and the
        # same number of time samples as each voxel time series.
        if F.shape[1] != n_basis or Y.shape[1] != F.shape[0]:
            logger.warning(f"{FPCA_CRITICAL_WARNING}: Incompatible data and basis dimensions")

        # Reject NaN or Inf in the observed data before fitting.
        if not np.isfinite(Y).all():
            logger.warning(f"{FPCA_CRITICAL_WARNING}: Non-finite fMRI data")

        # 2. Handle the unpenalized case with ordinary least squares.
        # Solve F @ C ≈ Y.T for all voxels simultaneously.
        # Here C initially has shape (n_basis, n_voxels).
        if self.no_penalty:
            C, _, rank, _ = np.linalg.lstsq(F, Y.T, rcond=None)

            # Full column rank is required to identify all basis coefficients.
            if rank != n_basis:
                logger.warning(f"{FPCA_CRITICAL_WARNING}: Rank-deficient basis; reduce n_basis")

            # Return one coefficient row per voxel and lambda = 0.
            return C.T, np.zeros(Y.shape[0])

        # 3. Precompute the generalized eigenvalue decomposition.
        # A is the discrete Gram matrix of the sampled basis functions.
        # Shape: (n_basis, n_basis)
        A = F.T @ F

        # P may be singular; A must be positive definite.
        # Passing P first and A second solves P @ v = d * A @ v.
        # The symmetric parts are supplied explicitly to the solver.
        #
        # d: (n_basis,), eigenvalues in ascending order.
        # V: (n_basis, n_basis), eigenvectors normalized so V.T @ A @ V = I.
        d, V = eigh(
            (P + P.T) / 2,
            (A + A.T) / 2,
        )

        # 4. Define a numerical tolerance for the expected zero eigenvalues.
        # Scale the tolerance by the largest eigenvalue magnitude,
        # machine precision, and the number of basis functions.
        # tiny supplies a positive lower bound for the scale.
        scale = max(
            float(np.max(np.abs(d))),
            np.finfo(float).tiny,
        )
        tol = 100 * np.finfo(float).eps * n_basis * scale

        # The nullspace of the derivative penalty consists
        # of polynomials of degree below the derivative order.
        # For example, a second-derivative penalty leaves constant and
        # linear functions unpenalized, giving q = 2 null directions.
        q = int(self.derivatives_num_p)

        # This implementation requires a nonnegative derivative order
        # smaller than the number of basis functions.
        if not 0 <= q < n_basis:
            logger.warning(f"{FPCA_CRITICAL_WARNING}: Invalid derivative order for this basis")

        # Since d is sorted, the first q eigenvalues are the expected
        # nullspace eigenvalues. Verify that they are close to zero.
        # For q = 0, there are no expected null directions to check.
        if q and np.max(np.abs(d[:q])) > tol:
            logger.warning(
                f"{FPCA_CRITICAL_WARNING}: Penalty does not have the expected polynomial nullspace"
            )

        # Set the verified nullspace eigenvalues to exactly zero.
        # Their shrinkage factors then remain exactly 1, even for large
        # lambdas, preserving the intended unpenalized directions.
        d[:q] = 0.0

        # All remaining directions must have strictly positive penalties.
        # This check rejects additional zero or negative eigenvalues.
        if np.any(d[q:] <= 0):
            logger.warning(
                f"{FPCA_CRITICAL_WARNING}: Penalty is not positive on the penalized subspace"
            )

        # 5. Set up the lambda search space.
        # Generate 100 logarithmically spaced candidates between
        # 10**self.lambda_min and 10**self.lambda_max.
        # The configuration values are base-10 exponents.
        lambdas = np.logspace(
            self.lambda_min,
            self.lambda_max,
            100,
        )

        # Transform the sampled basis once for reuse across batches.
        # F_V.T @ F_V = V.T @ A @ V = I, so its columns are orthonormal
        # up to numerical precision.
        # Shape: (n_timepoints, n_basis)
        F_V = F @ V

        # Allocate the final output arrays.
        # Every voxel's entries are filled by its corresponding batch.
        C = np.empty((Y.shape[0], n_basis))
        best_lambdas = np.empty(Y.shape[0])

        # 6. Fit voxel batches using the same precomputed decomposition.
        # Batching limits the size of the solver's temporary voxel arrays.
        for start in range(0, Y.shape[0], self.batch_size):
            stop = min(start + self.batch_size, Y.shape[0])

            # Batch data shape: (stop - start, n_timepoints).
            #
            # F_V.T equals V.T @ F.T and provides the projection matrix
            # expected by solve_batch_gcv as its VT_FT argument.
            #
            # The solver selects a lambda independently for each voxel,
            # reconstructs its coefficients, and returns its best GCV score.
            # Store coefficients and lambdas directly in the output slices;
            # the returned GCV scores are discarded here.
            C[start:stop], best_lambdas[start:stop], _ = solve_batch_gcv(
                Y[start:stop],
                F_V,
                F_V.T,
                V,
                d,
                lambdas,
            )

        return C, best_lambdas

    def calculate_coeff_old(self, F: np.ndarray, P: np.ndarray, n_basis: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit coefficients for spline basis functions to each voxel time series with optimal λ, in batches.

        Parameters:
            F (ndarray): basis matrix (n_timepoints, n_basis).
            P (ndarray): penalty matrix (n_basis, n_basis).

        Returns:
            C (ndarray): filled coefficient matrix (n_voxels, n_basis).
            best_labmdas (ndarray): best lambdas for each voxel
        """

        logger.info("Fitting basis to each voxel with optimal λ in batches...")

        # Coeff matrix
        C = np.zeros((self.n_voxels, n_basis))  # (n_voxels x n_basis)

        # Precompute the basis matrix and related matrices for efficiency.
        FtF = F.T @ F  # (n_basis, n_basis)
        # 100 logarithmically spaced values between 10^-lambda_min and 10^lambda_max
        lambda_values = np.unique(np.logspace(self.lambda_min, self.lambda_max, 100))
        # lambda_values = np.linspace(self.lambda_min, self.lambda_max, 100) # 100 linear spaced values between 10^-4 and 10^3

        # Compute H matrices for all lambda values.
        # H_all_lambda: (n_lambda, n_timepoints, n_timepoints)
        H_all_lambda = compute_hat_matrices_all_lambda(F, FtF, P, self.n_timepoints, lambda_values)

        # Create a batch of identity matrices for computing I - H.
        I = np.eye(self.n_timepoints)[None, :, :].repeat(len(lambda_values), axis=0)
        I_minus_H = I - H_all_lambda  # (n_lambda, n_timepoints, n_timepoints)

        best_lambdas = np.zeros(self.n_voxels)
        best_scores = np.zeros(self.n_voxels)
        # Process voxels in batches.
        for i in range(0, self.n_voxels, self.batch_size):
            end = min(i + self.batch_size, self.n_voxels)
            voxel_data_batch = self.fmri_data[i:end]  # (batch_size, n_timepoints)

            # Use a vectorized function to select the best lambda for all voxels in the batch.
            best_lambdas_batch, best_scores_batch, _ = select_lambda(I_minus_H, voxel_data_batch, self.n_timepoints,
                                                                     lambda_values)

            # Compute coefficients for the entire batch in a vectorized way.
            coeff_batch = self.compute_coeff_by_regularized_regression(F, FtF, P, best_lambdas_batch, voxel_data_batch)

            # Store the computed coefficients in the overall array.
            C[i:end, :] = coeff_batch
            best_lambdas[i:end] = best_lambdas_batch
            best_scores[i:end] = best_scores_batch
        logger.info(
            f"Average best lambdas: {np.average(best_lambdas)}, average best GCV scores: {np.average(best_scores)} (over all voxels) for number of {n_basis} basis fucntion.")

        return C, best_lambdas

    def solve_eigen_problem(self, cov_mat, F, U):
        S = (cov_mat + cov_mat.T) / 2
        U = (U + U.T) / 2

        if not 1 <= self.num_pca_comp <= S.shape[0]:
            raise ValueError("num_pca_comp must be between 1 and n_basis")

        # S @ U @ G = G @ diag(eigenvalues)
        # G.T @ U @ G = I
        eigvals, eigvecs = eigh(S, U, type=2)

        if not np.isfinite(eigvals).all():
            logger.warning("fPCA-CRITICAL-WARNING: Eigenvalues contain NaN or Inf; continuing unchanged.")

        scale = max(float(np.max(np.abs(eigvals))), np.finfo(float).tiny)
        tol = 100 * np.finfo(np.float64).eps * len(eigvals) * scale

        if np.min(eigvals) < -tol:
            logger.warning("fPCA-CRITICAL-WARNING: Functional covariance has a significant negative eigenvalue")

        eigvals = np.maximum(eigvals, 0.0)
        total_variance = float(eigvals.sum())

        if total_variance <= 0:
            logger.warning("fPCA-CRITICAL-WARNING: No variance remains after spatial centering")

        idx = np.argsort(eigvals)[::-1][:self.num_pca_comp]
        G = eigvecs[:, idx]

        return G, eigvals[idx], F @ G, total_variance

    def calcualte_scores(self, n_voxels, U, C_tilde, eigvecs_sorted):
        scores = np.zeros((n_voxels, self.num_pca_comp))
        for i in range(self.num_pca_comp):
            scores_i = C_tilde @ U @ eigvecs_sorted[:, i]  # shape: (n_voxels,)
            # if np.sum(scores_i) < 0:  # if most of the scores are negative, replace the sign
            #     scores_i *= -1
            # eigvecs_sorted[:, i] = -eigvecs_sorted[:, i]
            scores[:, i] = scores_i
        v_max_scores_pos = np.argmax(scores, axis=0)  # maximum score in each component
        return scores, v_max_scores_pos

    def fPCA(self, C: np.ndarray, U: np.ndarray, F: np.ndarray) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Perform functional PCA on the coefficient matrix.

        Parameters:
            C (ndarray): coefficient matrix (n_voxels, n_basis).
            U (ndarray): penalty matrix (n_basis, n_basis).
            F (ndarray): basis matrix (n_timepoints, n_basis).

        Returns:
            scores (ndarray): voxel scores on principal components (n_voxels, num_pca_comp).
            eigvecs_sorted (ndarray): basis-space eigenvectors sorted by importance (n_basis, num_pca_comp).
            eigvals_sorted (ndarray): corresponding sorted eigenvalues.
            v_max_scores_pos (ndarray): positions of voxels with maximum score in each component (num_pca_comp, )
            pc_temporal_profiles (ndarray): temporal profiles of the principal components (n_timepoints, num_pca_comp).
            total_variance (int): total variance in the data.
        """
        n_voxels, n_basis = C.shape
        C_tilde = C - C.mean(axis=0, keepdims=True)  # mean along the voxels.

        cov_mat = (C_tilde.T @ C_tilde) / n_voxels

        eigvecs_sorted, eigvals_sorted, pc_temporal_profiles, total_variance = (
            self.solve_eigen_problem(cov_mat, F, U)
        )
        scores, v_max_scores_pos = self.calcualte_scores(n_voxels, U, C_tilde, eigvecs_sorted)
        # print(v_max_scores_pos, np.max(scores, axis=0))
        return scores, eigvecs_sorted, eigvals_sorted, v_max_scores_pos, pc_temporal_profiles, total_variance

    def project_test_group_fpca(self, test_c_file_paths: list, test_mask_files: list) -> None:
        """
        Projects unseen test subjects onto the pre-calculated global Principal Components.
        Extracts subject-specific spatial scores without leaking data into the PCA space.

        Parameters:
            test_c_file_paths (list): List of paths to C_matrix.npy files for the TEST set.
            test_mask_files (list): List of paths to mask NIfTI files for the TEST set.
        """
        logger.info("Starting projection of TEST subjects onto TRAIN PCA space...")

        # Define where global group results go based on the first test subject's path hierarchy
        train_global_dir = os.path.dirname(os.path.dirname(test_c_file_paths[0]))

        # ---------------------------------------------------------
        # Load the required trained parameters from disk
        # ---------------------------------------------------------
        global_mean_path = os.path.join(train_global_dir, "global_mean.npy")
        if not os.path.exists(global_mean_path):
            raise FileNotFoundError(f"Missing train global mean file: {global_mean_path}. Did the train PCA complete?")
        train_global_mean = np.load(global_mean_path)
        logger.info("Successfully loaded TRAIN global mean from disk.")

        global_eigvecs_path = os.path.join(train_global_dir, "global_eigvecs.npy")
        if not os.path.exists(global_eigvecs_path):
            raise FileNotFoundError(f"Missing train global eigenvectors file: {global_eigvecs_path}")
        train_eigvecs_sorted = np.load(global_eigvecs_path)
        logger.info("Successfully loaded TRAIN eigenvectors from disk.")

        # ---------------------------------------------------------
        # Process each test subject
        # ---------------------------------------------------------
        for idx, c_file in enumerate(test_c_file_paths):
            # Load the C matrix for the test subject
            C_i_test = self.load_checked_coefficients(c_file)
            n_voxels = C_i_test.shape[0]

            # CRITICAL STEP: Center the test data using the TRAIN global mean!
            C_tilde_i_test = C_i_test - train_global_mean

            # Calculate subject-specific scores (spatial weights) using the TRAIN eigenvectors
            scores_i_test, v_max_scores_pos_i_test = self.calcualte_scores(
                n_voxels, self.U, C_tilde_i_test, train_eigvecs_sorted
            )

            # Load mask for 3D reconstruction and NIfTI saving
            subj_mask_img = nib.load(test_mask_files[idx])
            self.mask = subj_mask_img.get_fdata()
            self.nii_affine = subj_mask_img.affine
            self.orig_n_voxels = self.mask.shape

            subject_dir = os.path.dirname(c_file)

            # Optional: save the raw NumPy array if needed for quick ML loading later
            np.save(os.path.join(subject_dir, "global_pca_scores_proj.npy"), scores_i_test)

            # ---------------------------------------------------------
            # Loop over each PC to create and save its specific 3D NIfTI map
            # ---------------------------------------------------------
            num_components = scores_i_test.shape[1]
            for pc_idx in range(num_components):
                # PC numbers are typically 1-indexed for filenames
                pc_num = pc_idx

                # Extract the spatial weights for this specific PC
                pc_scores = scores_i_test[:, pc_idx]

                # Reconstruct the 3D brain volume using the mask
                vol = np.zeros(self.orig_n_voxels)
                vol[self.mask > 0] = pc_scores

                # Construct the specific output filename requested
                nii_filename = f"eigenfunction_{pc_num}_importance_map_group_test_proj.nii.gz"
                nii_filepath = os.path.join(subject_dir, nii_filename)

                # Save as NIfTI
                nii_img = nib.Nifti1Image(vol, self.nii_affine)
                nib.save(nii_img, nii_filepath)
                logger.info(f"Saved projected importance map: {nii_filename}")

            # If you still want to generate PNGs or other graphs, call your drawing function
            # self.draw_graphs_group(scores_i_test, output_dir=subject_dir)

        logger.info("Test set projection completed successfully.")

    def load_checked_coefficients(self, path):
        with np.load(path, allow_pickle=False) as data:
            for key, expected in (
                ("F", self.F),
                ("U", self.U),
                ("times", self.times),
            ):
                actual = data[key]

                if actual.shape != expected.shape or not np.allclose(
                    actual,
                    expected,
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise ValueError(f"Incompatible {key} in {path}")

            C = data["C"]

        if (
            C.ndim != 2
            or C.shape[1] != self.U.shape[0]
            or not len(C)
            or not np.isfinite(C).all()
        ):
            raise ValueError(f"Invalid coefficient matrix in {path}")

        return C

    def run_group_fpca(self, c_file_paths: list, mask_files: list):
        """
        Performs out-of-core Group fPCA by accumulating covariance over multiple subjects.

        Parameters:
            c_file_paths (list): List of paths to individual C_matrix.npy files.
            U (ndarray): Penalty/Gram matrix.
            F (ndarray): Basis matrix.
            fmri_instance: Instance of FunctionalMRI to reuse the eigenproblem solver.
        """
        K = self.U.shape[0]

        # ==========================================
        # Pass 1: Compute Global Mean of Coefficients
        # ==========================================
        logger.info("Pass 1: Computing global mean across all subjects...")
        global_sum = np.zeros(K)
        total_voxels = 0

        if not c_file_paths or len(c_file_paths) != len(mask_files):
            raise ValueError("Missing or unmatched subject archives and masks")

        for c_file in c_file_paths:
            C_i = self.load_checked_coefficients(c_file)
            global_sum += C_i.sum(axis=0)
            total_voxels += len(C_i)
            del C_i

        global_mean = global_sum / total_voxels
        global_mean = global_mean[None, :]  # Reshape for broadcasting

        # ==========================================
        # Pass 2: Accumulate Group Covariance Matrix
        # ==========================================
        logger.info("Pass 2: Accumulating group covariance matrix...")
        cov_sum = np.zeros((K, K))

        for i in range(len(c_file_paths)):

            # Center the subject's data using the GLOBAL mean
            C_tilde_i = (
                self.load_checked_coefficients(c_file_paths[i])
                - global_mean
            )
            cov_sum += C_tilde_i.T @ C_tilde_i

        # Finalize covariance calculation (same logic as single-subject fPCA)
        cov_mat = cov_sum / total_voxels

        # ==========================================
        # Solve Eigenproblem (Reusing your function)
        # ==========================================
        logger.info("Solving global eigenproblem...")
        eigvecs_sorted, eigvals_sorted, pc_temporal_profiles, total_variance = self.solve_eigen_problem(cov_mat, self.F)

        # ==========================================
        # Pass 3: Project subjects onto global PCs
        # ==========================================
        logger.info("Pass 3: Extracting subject-specific spatial scores...")

        # Pass 3: Project subjects onto global PCs
        for idx, c_file in enumerate(c_file_paths):
            # Load the archive

            # Extract the actual 'C' matrix from the archive
            C_i = self.load_checked_coefficients(c_file)
            n_voxels = C_i.shape[0]

            # Center the data using the global_mean (calculated in Pass 1)
            C_tilde_i = C_i - global_mean

            # Calculate subject-specific scores (using the centered data)
            scores_i, v_max_scores_pos_i = self.calcualte_scores(n_voxels, self.U, C_tilde_i, eigvecs_sorted)

            subj_mask_img = nib.load(mask_files[idx])
            self.mask = subj_mask_img.get_fdata()
            self.nii_affine = subj_mask_img.affine
            self.orig_n_voxels = self.mask.shape

            # Save the specific scores for this subject to be used as ML features
            subject_dir = os.path.dirname(c_file)
            np.save(os.path.join(subject_dir, "global_pca_scores.npy"), scores_i)

            # Generate the spatial NIfTI/PNG maps specifically for this subject
            self.draw_graphs_group(scores_i, output_dir=subject_dir)

        global_output_dir = os.path.dirname(os.path.dirname(c_file_paths[0]))  # Define where global group results go
        global_mean_path = os.path.join(global_output_dir, "global_mean.npy")
        np.save(global_mean_path, global_mean)
        # Save the global eigenvectors (Crucial for projecting the test set later!)
        global_eigvecs_path = os.path.join(global_output_dir, "global_eigvecs.npy")
        np.save(global_eigvecs_path, eigvecs_sorted)
        logger.info(f"Saved TRAIN global mean and eigvecs to shared directory: {global_mean_path}")

        self.draw_graphs_group_global(eigvals_sorted, pc_temporal_profiles, total_variance, global_output_dir)
        logger.info("Group fPCA completed successfully.")
        return eigvecs_sorted, eigvals_sorted, pc_temporal_profiles

    def voxel_index_to_coord(self, k):
        flat_indices = np.flatnonzero(self.mask)
        flat_index = flat_indices[k]
        return np.unravel_index(flat_index, self.mask.shape)

    def _export_importance_map(self, scores_i: np.ndarray, comp_idx: int, output_dir: str, group: bool=False) -> None:
        """
        Save the importance map as a NIfTI file and plot its middle slice.
        """
        logger.info(f"Saving voxel-wise importance map for eigenfunction {comp_idx} in {output_dir}...")
        importance_map = np.zeros(self.orig_n_voxels)
        importance_map[self.mask > 0] = scores_i

        # Save NIfTI
        importance_nii = nib.Nifti1Image(importance_map, affine=self.nii_affine)
        if not group:
            impmap_file = os.path.join(output_dir, f"eigenfunction_{comp_idx}_importance_map.nii.gz")
        else:
            impmap_file = os.path.join(output_dir, f"eigenfunction_{comp_idx}_importance_map_group.nii.gz")
        nib.save(importance_nii, impmap_file)

        # Save PNG if memory allows
        if not self.low_mem:
            z_middle = importance_map.shape[2] // 2
            slice_img = importance_map[:, :, z_middle]
            plt.figure(figsize=(6, 6))
            plt.imshow(slice_img.T, cmap='hot', origin='lower')
            plt.title(f'Middle Slice of Importance Map Eigenfunction {comp_idx}')
            plt.colorbar(label='Importance')
            plt.axis('off')
            if not group:
                impmap_fig = os.path.join(output_dir, f"eigenfunction_{comp_idx}_importance_map.png")
            else:
                impmap_fig = os.path.join(output_dir, f"eigenfunction_{comp_idx}_importance_map_group.png")
            plt.savefig(impmap_fig, dpi=300, bbox_inches='tight')
            plt.close()

    def _export_temporal_profile(self, profile_i: np.ndarray, comp_idx: int, explained_var: float,
                                 output_dir: str, group: bool = False) -> None:
        """
        Save the temporal profile of a principal component as a plot and text file.
        """
        logger.info(f"Plotting temporal profile of eigenfunction {comp_idx}...")

        # Save PNG if memory allows
        if not self.low_mem:
            plt.figure(figsize=(10, 4))
            max_time = int(max(self.times))
            ticks = np.arange(0, max_time + 20, 20)
            plt.xticks(ticks, rotation=45)
            plt.plot(self.times, profile_i, color='blue')
            plt.title(f'Temporal profile of eigenfunction {comp_idx} ({explained_var:.1f}% var)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Intensity')
            plt.grid(True)
            if not group:
                intence_fig = os.path.join(output_dir, f"temporal_profile_pc_{comp_idx}.png")
            else:
                intence_fig = os.path.join(output_dir, f"temporal_profile_pc_{comp_idx}_group.png")
            plt.savefig(intence_fig, dpi=300, bbox_inches='tight')
            plt.close()

        # Save TXT
        if not group:
            intence_txt = os.path.join(output_dir, f"temporal_profile_pc_{comp_idx}.txt")
        else:
            intence_txt = os.path.join(output_dir, f"temporal_profile_pc_{comp_idx}_group.txt")
        with open(intence_txt, 'w') as f:
            f.write(f"#Eigenfunction {comp_idx}:\n")
            f.write(f"#Times:\n{' '.join(map(str, self.times))}\n")
            f.write(f"#PC temporal profile:\n{' '.join(map(str, profile_i))}\n")

    def _export_best_voxel_signal(self, comp_idx: int, v_max_pos: int, C: np.ndarray, F: np.ndarray, best_lambda: float,
                                  score: float, output_dir: str) -> None:
        """
        Plot and save the signal intensity for the voxel with the maximum score in a specific principal component.
        """
        logger.info(f"Plotting signal intensity for best voxel in eigenfunction {comp_idx}...")

        Y_v_max_score = self.fmri_data[v_max_pos]
        Y_hat_v_max_score = C[v_max_pos] @ F.T  # (1, n_timepoints)

        if not self.low_mem:
            plt.figure(figsize=(10, 4))
            plt.plot(self.times, Y_hat_v_max_score, color='blue')
            plt.scatter(self.times, Y_v_max_score, color='red')
            plt.title(f'Signal and best voxel\'s fitted function: Eigenfunction {comp_idx}')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Intensity')
            plt.grid(True)

            ax = plt.gca()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Compute position near bottom-right
            x = xlim[1] - 0.01 * (xlim[1] - xlim[0])  # Slightly inside from right
            y = ylim[0] + 0.01 * (ylim[1] - ylim[0])  # Slightly above bottom
            x_c, y_c, z_c = self.voxel_index_to_coord(v_max_pos)

            plt.text(x, y,
                     rf"$\lambda={best_lambda}, score: {score:.2f}, x={x_c};y={y_c};z={z_c}$",
                     fontsize=12, ha='right', va='bottom')

            best_voxel_fig = os.path.join(output_dir, f"eigenfunction_{comp_idx}_best_voxel.png")
            plt.savefig(best_voxel_fig, dpi=300, bbox_inches='tight')
            plt.close()

        best_voxel_txt = os.path.join(output_dir, f"eigenfunction_{comp_idx}_best_voxel.txt")
        with open(best_voxel_txt, 'w') as f:
            f.write(f"#Eigenfunction {comp_idx}:\n")
            f.write(f"#Times:\n{' '.join(map(str, self.times))}\n")
            f.write(f"#Y_estimated:\n{' '.join(map(str, Y_hat_v_max_score))}\n")
            f.write(f"#Y_real:\n{' '.join(map(str, Y_v_max_score))}\n")

    def _export_average_signal(self, output_dir: str) -> None:
        """
        Plot and save the original averaged signal intensity across all voxels.
        """
        logger.info(f"Plotting original average signal intensity ...")
        signal_intensity = np.average(self.fmri_data, axis=0)

        if not self.low_mem:
            plt.figure(figsize=(10, 4))
            max_time = int(max(self.times))
            ticks = np.arange(0, max_time + 20, 20)
            plt.xticks(ticks, rotation=45)
            plt.plot(self.times, signal_intensity, color='blue')
            plt.title('Original average Signal Intensity')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Intensity')
            plt.grid(True)

            orig_intence_fig = os.path.join(output_dir, "original_averaged_signal_intensity.png")
            plt.savefig(orig_intence_fig, dpi=300, bbox_inches='tight')
            plt.close()

        orig_intence_txt = os.path.join(output_dir, "original_averaged_signal_intensity.txt")
        with open(orig_intence_txt, 'w') as f:
            f.write(f"#Times:\n{' '.join(map(str, self.times))}\n")
            f.write(f"#Signal intensity:\n{' '.join(map(str, signal_intensity))}\n")

    def draw_graphs(self, C: np.ndarray, F: np.ndarray, scores: np.ndarray, eigvecs_sorted: np.ndarray,
                    eigvals_sorted: np.ndarray, v_max_scores_pos: np.ndarray, best_lambdas: np.ndarray,
                    pc_temporal_profiles: np.ndarray, total_variance: float) -> None:
        """
        Generate and save voxel importance maps and eigenfunction intensity plots.

        Parameters:
            F (ndarray): basis matrix (n_timepoints, n_basis).
            scores (ndarray): voxel scores (n_voxels, num_pca_comp).
            eigvecs_sorted (ndarray): principal component vectors.
            eigvals_sorted (ndarray): corresponding sorted eigenvalues.
            v_max_scores_pos (ndarray): positions of voxels with maximum score in each component (num_pca_comp, )
            best_lambdas (ndarray): best lambdas for each voxel (n_voxels,)
            pc_temporal_profiles (ndarray): temporal profiles of the principal components.
            total_variance (int): total variance in the data.
        """
        # Save main arrays
        store_data_file = os.path.join(self.output_folder, "eigvecs_eigval_F.npz")
        np.savez_compressed(
            store_data_file,
            eigvecs_sorted=eigvecs_sorted,
            eigvals_sorted=eigvals_sorted,
            F=F,
            C=C,
            U=self.U,
            times=self.times,
            mask=self.mask > 0,
            affine=self.nii_affine,
            mean_function=F @ C.mean(axis=0),
        )
        """
        For loading the arrays back, use:
            with np.load(store_data_file, allow_pickle=False) as data:
                eigvecs_sorted = data['eigvecs_sorted']
                eigvals_sorted = data['eigvals_sorted']
                F = data['F']
                C = data['C']
                U = data['U']
                times = data['times']
                mask = data['mask']
                affine = data['affine']
                mean_function = data['mean_function']

        Additional saved arrays:
            U: Basis inner-product matrix used for fPCA.
            mask: Boolean spatial mask identifying included voxels.
            affine: NIfTI affine mapping voxel indices to spatial coordinates.
            mean_function: Mean fitted time series across voxels,
                computed as F @ C.mean(axis=0).
        """
        logger.info("Eigenvectors, eigenvalues, F, and C matrices are saved to 'eigvecs_eigval_F.npz'")

        # Generate outputs per component
        for i in range(self.num_pca_comp):
            explained_variance_i = (eigvals_sorted[i] * 100) / total_variance

            # Export spatial importance map
            self._export_importance_map(scores[:, i], i, self.output_folder)

            # Export temporal profile
            self._export_temporal_profile(pc_temporal_profiles[:, i], i, explained_variance_i, self.output_folder)

            self._export_best_voxel_signal(
                comp_idx=i,
                v_max_pos=v_max_scores_pos[i],
                C=C,
                F=F,
                best_lambda=best_lambdas[v_max_scores_pos[i]],
                score=scores[v_max_scores_pos[i], i],
                output_dir=self.output_folder
            )

        self._export_average_signal(output_dir=self.output_folder)

    def draw_graphs_group(self, scores: np.ndarray, output_dir: str) -> None:
        """
        Generates and saves the subject-specific spatial maps during a group analysis.
        Does NOT plot global temporal profiles or best raw voxels to avoid duplication and loading full raw data.
        """
        for i in range(self.num_pca_comp):
            # Export spatial importance map using the shared helper
            self._export_importance_map(scores[:, i], i, output_dir, group=True)

    def draw_graphs_group_global(self, eigvals_sorted: np.ndarray, pc_temporal_profiles: np.ndarray,
                                 total_variance: float, output_dir: str) -> None:
        """
        Generates and saves the global temporal profiles and overall variance metrics for the entire group.
        Should be called ONCE after the group loop finishes.
        """
        for i in range(self.num_pca_comp):
            explained_variance_i = (eigvals_sorted[i] * 100) / total_variance
            self._export_temporal_profile(pc_temporal_profiles[:, i], i, explained_variance_i, output_dir, group=True)


