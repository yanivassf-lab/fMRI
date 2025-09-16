import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import logging

logger = logging.getLogger("fmri_logger")


def validate_spline_basis_funs(splines_funs, knots):
    """
    Validate if the B-spline basis functions sum to 1 at all x values within the specified tolerance.

    Parameters:
    - splines_funs: Callable function for B-spline basis functions. The function takes a set of x values and returns
                    the corresponding basis function values.
    - knots: (array-like) Knot vector defining the B-splines.

    Returns:
    - None: Prints a message indicating whether the basis functions sum to 1 at all x values.
    """
    values = np.linspace(knots[0], knots[-1], 400)

    basis_values_matrix = np.nan_to_num(splines_funs(values))
    sum_vals = np.sum(basis_values_matrix, axis=1)
    tolerance = 1e-10
    if np.all(np.abs(sum_vals - 1) < tolerance):
        logger.info("Success: The basis functions sum to 1 (within tolerance) at all x values.")
    else:
        logger.info("Warning: The basis functions do not sum to 1 at some x values.")
        raise ValueError("The basis functions do not sum to 1 at some x values.")


def spline_base_funs(T_min, T_max, degree, n_basis):
    """
    Generate B-spline basis functions with specified parameters.

    Parameters:
    - T_min: The minimum value of the domain (start of the range).
    - T_max: The maximum value of the domain (end of the range).
    - degree: The degree of the spline (e.g., 3 for cubic splines).
    - n_basis: The number of basis functions to generate.

    Returns:
    - splines_funs: A BSpline object containing the generated B-spline basis functions.
    """
    # Define spline parameters: cubic spline (degree 3, order 4)
    order = degree + 1
    # Define a clamped knot vector (repeated knots at the boundaries)

    # Create knot vector (uniform B-spline)
    n_knots = n_basis + degree + 1
    knots = np.linspace(T_min, T_max, n_knots - 2 * degree)
    knots = np.concatenate(([T_min] * degree, knots, [T_max] * degree))  # Add boundary knots

    # The number of B-spline basis functions is len(knots) - order.
    num_basis = len(knots) - order

    # Create the B-spline basis functions
    coeffs = np.eye(num_basis)
    splines_funs = BSpline(knots, coeffs.T, degree, extrapolate=False)
    validate_spline_basis_funs(splines_funs, knots)
    return splines_funs, knots


def main():
    """
    Main function to demonstrate the creation of cubic B-spline basis functions and plot them.

    Generates and visualizes the B-spline basis functions with a given knot vector, and checks the sum of the basis
    functions at all x values.
    """
    # Define spline parameters: cubic spline (degree 3, order 4)
    degree = 3
    order = degree + 1
    logger.info("Constructing B-spline basis...")
    knots = [0, 0, 0, 0, 2.5, 5.0, 7.5, 10, 10, 10, 10]
    # The number of B-spline basis functions is len(knots) - order.
    num_basis = len(knots) - order
    # Create a grid of x values over the domain.
    x_vals = np.linspace(knots[0], knots[-1], 400)

    # Build B-spline basis functions
    coeffs = np.eye(num_basis)
    spline = BSpline(knots, coeffs.T, degree, extrapolate=False)
    basis_matrix = np.nan_to_num(spline(x_vals))
    sum_vals = np.sum(basis_matrix, axis=1)
    validate_spline_basis_funs(spline, knots)

    # Check that the sum is approximately 1 at all evaluation points.
    tolerance = 1e-10
    if np.all(np.abs(sum_vals - 1) < tolerance):
        logger.info("Success: The basis functions sum to 1 (within tolerance) at all x values.")
    else:
        logger.info("Warning: The basis functions do not sum to 1 at some x values.")

    # Plot the sum of the basis functions to visually verify the partition of unity.
    plt.figure(figsize=(10, 4))
    plt.plot(x_vals, sum_vals, color='blue', lw=2, label='Sum of Basis Functions')
    plt.axhline(1, color='red', linestyle='--', label='1 (Exact Sum)')
    plt.title('Partition of Unity Check for B-spline Basis Functions using SciPy')
    plt.xlabel('x')
    plt.ylabel('Sum')
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    main()
