import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skfda.representation.basis import BSplineBasis
from skfda import FDataBasis
import logging

logger = logging.getLogger("fmri_logger")


def validate_spline_basis_funs(basis_funs: BSplineBasis, domain_range):
    """
    Validate if the B-spline basis functions sum to 1 at all x values within the specified tolerance.

    Parameters:
    - basis_funs: BSplineBasis object
    - domain_range: tuple of (min, max) values for the domain

    Returns:
    - None: Prints a message indicating whether the basis functions sum to 1 at all x values.
    """
    values = np.linspace(domain_range[0], domain_range[1], 400)

    # Evaluate the basis functions using the evaluation matrix
    basis_matrix = basis_funs(values).squeeze().T

    # Sum across basis functions
    sum_vals = np.sum(basis_matrix, axis=1)

    tolerance = 1e-10
    if np.all(np.abs(sum_vals - 1) < tolerance):
        logger.info("Success: The basis functions sum to 1 (within tolerance) at all x values.")
    else:
        max_error = np.max(np.abs(sum_vals - 1))
        logger.info(f"Warning: The basis functions do not sum to 1 at some x values. Max error: {max_error}")
        # For B-splines, the sum should be 1 in the interior of the domain
        # Check only the interior (exclude 5% at each boundary)
        interior_start = int(len(values) * 0.05)
        interior_end = int(len(values) * 0.95)
        interior_sum = sum_vals[interior_start:interior_end]
        interior_max_error = np.max(np.abs(interior_sum - 1))
        logger.info(f"Interior max error (excluding boundaries): {interior_max_error}")


def spline_base_funs(T_min, T_max, degree, n_basis):
    """
    Generate B-spline basis functions with specified parameters.

    Parameters:
    - T_min: The minimum value of the domain (start of the range).
    - T_max: The maximum value of the domain (end of the range).
    - degree: The degree of the spline (e.g., 3 for cubic splines).
    - n_basis: The number of basis functions to generate.

    Returns:
    - basis_funs: A BSplineBasis object containing the generated B-spline basis functions.
    - knots: The knot sequence
    """
    # Define spline parameters
    order = degree + 1
    domain_range = (T_min, T_max)

    # Create BSplineBasis with uniform knots
    basis_funs = BSplineBasis(
        n_basis=n_basis,
        order=order,
        domain_range=domain_range,
    )

    validate_spline_basis_funs(basis_funs, domain_range)

    # Get the knot sequence from the basis object
    knots = basis_funs.knots

    return basis_funs, knots


# def evaluate_basis_functions(basis_funs, x_vals):
#     """
#     Evaluate all basis functions at given x values.
#
#     Parameters:
#     - basis_funs: BSplineBasis object
#     - x_vals: array of x values
#
#     Returns:
#     - basis_matrix: (n_x_vals, n_basis) matrix of basis function values
#     """
#     # Use the evaluate method of BSplineBasis
#     # It returns a 3D array of shape (n_samples, n_basis, n_outputs)
#     # For basis evaluation, n_outputs = 1
#     basis_matrix = basis_funs(x_vals).squeeze().T  # (n_basis, n_points)
#
#     # If x_vals is a single point, ensure we have the right shape
#     if basis_matrix.ndim == 1:
#         basis_matrix = basis_matrix.reshape(1, -1)
#
#     return basis_matrix
#
#
# def create_bspline_with_custom_knots(T_min, T_max, degree, interior_knots):
#     """
#     Create B-spline basis with custom interior knots.
#
#     Parameters:
#     - T_min: The minimum value of the domain.
#     - T_max: The maximum value of the domain.
#     - degree: The degree of the spline.
#     - interior_knots: List of interior knots (not including boundary repetitions).
#
#     Returns:
#     - basis_funs: BSplineBasis object
#     """
#     order = degree + 1
#
#     # Create the full knot sequence with repeated boundaries
#     knots = np.concatenate([
#         [T_min] * order,
#         interior_knots,
#         [T_max] * order
#     ])
#     domain_range = (T_min, T_max)
#
#     # Create BSplineBasis by specifying the full knot sequence
#     basis_funs = BSplineBasis(
#         knots=knots,
#         order=order,
#         domain_range=domain_range
#     )
#
#     return basis_funs
#
#
# def main():
#     """
#     Main function to demonstrate the creation of cubic B-spline basis functions and plot them.
#     """
#     # Define spline parameters: cubic spline (degree 3, order 4)
#     degree = 3
#     order = degree + 1
#     domain_range = (0, 10)
#
#     logger.info("Constructing B-spline basis...")
#
#     # Method 1: Create with custom knots
#     interior_knots = [2.5, 5.0, 7.5]
#     basis_funs = create_bspline_with_custom_knots(0, 10, degree, interior_knots)
#
#     # Get info
#     n_basis = basis_funs.n_basis
#     full_knots = basis_funs.knots
#     actual_domain = basis_funs.domain_range[0]
#
#     logger.info(f"Number of basis functions: {n_basis}")
#     logger.info(f"Interior knots: {interior_knots}")
#     logger.info(f"Full knot sequence: {full_knots}")
#     logger.info(f"Actual domain range: {actual_domain}")
#
#     # Create a grid of x values over the actual domain
#     # Use the actual domain from the basis object
#     x_vals = np.linspace(actual_domain[0], actual_domain[1], 400)
#
#     # Evaluate all basis functions
#     basis_matrix = evaluate_basis_functions(basis_funs, x_vals)
#
#     # Calculate the sum of basis functions
#     sum_vals = np.sum(basis_matrix, axis=1)
#
#     # Check partition of unity
#     tolerance = 1e-10
#     if np.all(np.abs(sum_vals - 1) < tolerance):
#         logger.info("Success: The basis functions sum to 1 (within tolerance) at all x values.")
#     else:
#         max_error = np.max(np.abs(sum_vals - 1))
#         logger.info(f"Warning: Max deviation from 1: {max_error}")
#
#     # Plot individual basis functions
#     plt.figure(figsize=(12, 8))
#
#     # Plot 1: Individual basis functions
#     plt.subplot(2, 1, 1)
#     for i in range(n_basis):
#         plt.plot(x_vals, basis_matrix[:, i], label=f'B_{i}', linewidth=2)
#     plt.title('B-spline Basis Functions using scikit-fda')
#     plt.xlabel('x')
#     plt.ylabel('Basis Function Value')
#     plt.legend()
#     plt.grid(True)
#
#     # Add vertical lines for interior knots
#     for knot in interior_knots:
#         plt.axvline(x=knot, color='gray', linestyle='--', alpha=0.5)
#
#     # Plot 2: Sum of basis functions (partition of unity check)
#     plt.subplot(2, 1, 2)
#     plt.plot(x_vals, sum_vals, color='blue', lw=2, label='Sum of Basis Functions')
#     plt.axhline(1, color='red', linestyle='--', label='1 (Exact Sum)')
#     plt.title('Partition of Unity Check for B-spline Basis Functions using scikit-fda')
#     plt.xlabel('x')
#     plt.ylabel('Sum')
#     plt.legend()
#     plt.grid(True)
#     # Show actual range
#     plt.ylim(min(0.95, min(sum_vals)), max(1.05, max(sum_vals)))
#
#     plt.tight_layout()
#     plt.show()
#
#     # Method 2: Test the spline_base_funs function (uniform knots)
#     logger.info("\nTesting spline_base_funs function (uniform knots)...")
#     basis_funs2, knots2 = spline_base_funs(0, 10, 3, 7)
#     logger.info(f"Generated basis with {basis_funs2.n_basis} functions")
#     logger.info(f"Knot sequence: {knots2}")
#
#     # Evaluate and check
#     x_vals2 = np.linspace(0, 10, 400)
#     basis_matrix2 = evaluate_basis_functions(basis_funs2, x_vals2)
#     sum_vals2 = np.sum(basis_matrix2, axis=1)
#     max_error2 = np.max(np.abs(sum_vals2 - 1))
#     logger.info(f"Uniform knots - Max deviation from 1: {max_error2}")
#
#     # Example: Create FDataBasis objects if needed
#     logger.info("\nExample: Creating FDataBasis with identity coefficients...")
#     # Identity matrix coefficients - each row represents one basis function
#     coefficients = np.eye(n_basis)
#     fd_basis = FDataBasis(basis=basis_funs, coefficients=coefficients)
#     # Evaluate at a few points
#     test_points = np.array([2.5, 5.0, 7.5])
#     fd_values = fd_basis(test_points).T
#     logger.info(f"FDataBasis evaluation shape: {fd_values.shape}")
#
#
# if __name__ == '__main__':
#     main()
