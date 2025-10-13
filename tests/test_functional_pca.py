import numpy as np
import pytest
# from scipy.interpolate import BSpline
from src.fmri.fmri import FunctionalMRI
from src.fmri.b_spline_bspline import spline_base_funs as spline_base_funs_bspline
from src.fmri.b_spline_skfda import spline_base_funs as spline_base_funs_skfda

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@pytest.fixture
def dummy_fmri():
    """
    Create a FunctionalMRI instance without running init, and set necessary attributes.
    """
    # Create a dummy instance bypassing init
    dummy = FunctionalMRI.__new__(FunctionalMRI)  # bypasses init

    # dummy = FunctionalMRI.new('str', 'str',  3,   3,3,'str')
    dummy.T_min = 0
    dummy.T_max = 100
    dummy.num_pca_comp = 2
    dummy.n_voxels = 10
    dummy.degree = 3
    dummy.n_basis = 5
    dummy.processed = True
    dummy.derivatives_num_p = 2
    dummy.derivatives_num_u = 0
    dummy.bad_margin_size = 0
    dummy.calc_penalty_bspline_accurately = False
    dummy.calc_penalty_skfda = False
    return dummy


def create_dummy_bspline(dummy_fmri):
    """
    Create a dummy BSpline object.
    The BSpline requires a knot vector, coefficient array, and degree.
    We choose knots uniformly in [0,1]. For a BSpline there are num_basis + degree + 1 knots.
    The coefficient vector is chosen arbitrarily.
    """
    basis_funs_bspline, _ = spline_base_funs_bspline(dummy_fmri.T_min, dummy_fmri.T_max, dummy_fmri.degree,
                                                     dummy_fmri.n_basis)
    basis_funs_skfda, _ = spline_base_funs_skfda(dummy_fmri.T_min, dummy_fmri.T_max, dummy_fmri.degree,
                                                 dummy_fmri.n_basis)
    # knots = np.linspace(dummy_fmri.T_min, dummy_fmri.T_max, dummy_fmri.num_basis + dummy_fmri.degree + 1-2*dummy_fmri.degree)
    # Create coefficients; here we choose ones (or random values)
    # coeffs = np.ones(dummy_fmri.num_basis)
    # return BSpline(knots, coeffs, dummy_fmri.degree)
    return basis_funs_bspline, basis_funs_skfda


def test_penalty_matrix_methods(dummy_fmri):
    """
    Test that the penalty matrix computed via the two methods (quad integration versus
    discrete trapezoidal rule) have the same shape and similar values.
    """
    basis_funs_bspline, basis_funs_skfda = create_dummy_bspline(dummy_fmri)
    # Compute the penalty matrices (here derivative_order=0 is used)
    G_accurate_bspline = dummy_fmri.penalty_matrix_bspline_accurate(basis_funs_bspline, derivative_order=0)
    G_accurate_skfda = dummy_fmri.penalty_matrix_skfda(basis_funs_skfda, derivative_order=0)
    # Cannot work with the new pacakge:
    G_bspline = dummy_fmri.penalty_matrix_bspline(basis_funs_bspline, derivative_order=0)

    # Check same shape.
    assert G_accurate_bspline.shape == G_bspline.shape, "Penalty matrix shapes should match."

    # Since the two integration methods differ, they won’t be exactly identical.
    # We check that they are close within a reasonable relative tolerance.
    np.testing.assert_allclose(G_accurate_bspline, G_bspline, rtol=0.00005,
                               err_msg="Integration methods bspline - accurate and not accurate, (derivative_order=0) differ more than expected.")
    np.testing.assert_allclose(G_accurate_bspline, G_accurate_skfda, rtol=0.00005,
                               err_msg="Integration methods bspline accurate and skfda, (derivative_order=0) differ more than expected.")

    # Compute the penalty matrices (here derivative_order=2 is used)
    G_accurate_bspline = dummy_fmri.penalty_matrix_bspline_accurate(basis_funs_bspline, derivative_order=2)
    G_accurate_skfda = dummy_fmri.penalty_matrix_skfda(basis_funs_skfda, derivative_order=2)
    G_bspline = dummy_fmri.penalty_matrix_bspline(basis_funs_bspline, derivative_order=2)
    np.testing.assert_allclose(G_accurate_bspline, G_bspline, rtol=0.00005,
                               err_msg="Integration methods bspline - accurate and not accurate,  derivative_order=2) differ more than expected.")
    np.testing.assert_allclose(G_accurate_bspline, G_accurate_skfda, rtol=0.00005,
                               err_msg="Integration methods bspline accurate and skfda, derivative_order=2) differ more than expected.")


def test_fPCA(dummy_fmri):
    """
    Test the fPCA method using a small randomly generated coefficient matrix
    and a dummy basis.
    """
    # Create a dummy coefficient matrix C (n_voxels x n_basis)
    np.random.seed(0)
    n_voxels = 10
    n_basis = 5
    n_timepoints = 20
    C = np.random.rand(n_voxels, n_basis)
    F = np.random.rand(n_timepoints, n_basis)
    # Create dummy BSpline basis function
    basis_funs_bspline, _ = create_dummy_bspline(dummy_fmri)
    # Run fPCA
    U = dummy_fmri.penalty_matrix_bspline(basis_funs_bspline, derivative_order=0)
    scores, eigvecs_sorted, eigvals_sorted, v_max_scores_pos, pc_temporal_profiles, total_variance = dummy_fmri.fPCA(C,
                                                                                                                     U,
                                                                                                                     F)

    # Verify shapes
    assert scores.shape == (n_voxels, dummy_fmri.num_pca_comp), "Scores shape mismatch."
    assert eigvecs_sorted.shape == (n_basis, dummy_fmri.num_pca_comp), "Eigenvectors shape mismatch."


# Check that the covariance matrix reconstructed from the scores has the expected properties.
# (Eigenvectors are defined up to sign, so checking for exact values is not necessary.)
def test_compute_coeff_by_regularized_regression(dummy_fmri):
    """
    Test the batched coefficient computation using a small linear system.
    We solve for coefficients in a system:
    (F.T @ F + lambda * I) * coeff = F.T @ y
    where the penalty matrix P will be chosen as the identity.
    """

    np.random.seed(0)
    n_timepoints = 20
    n_basis = 4
    # Create F: (n_timepoints x n_basis) and compute FtF.
    F = np.random.rand(n_timepoints, n_basis)
    FtF = F.T @ F
    # Use P = I for simplicity.
    P = np.eye(n_basis)
    # For 3 voxels, take lambda = 0.1 for each.
    lambda_vec = np.array([0.1, 0.1, 0.1])
    # Create dummy y_batch: (n_voxels, n_timepoints)
    y_batch = np.random.rand(3, n_timepoints)

    coeffs = dummy_fmri.compute_coeff_by_regularized_regression(F, FtF, P, lambda_vec, y_batch)
    # Expect coeffs shape to be (3, n_basis)
    assert coeffs.shape == (3, n_basis), "Coefficient shape mismatch."

    # Validate the first voxel by solving the linear system manually.
    A = FtF + 0.1 * P  # (n_basis, n_basis)
    RHS = F.T @ y_batch[0]
    coeff_expected = np.linalg.solve(A, RHS)
    np.testing.assert_allclose(coeffs[0], coeff_expected, rtol=1e-5,
                               err_msg="Coefficient computation error for first voxel.")


if __name__ == "__main__":
    # Allows the tests to be run using: python -m pytest tests/test_functional_pca.py
    import pytest

    pytest.main()
