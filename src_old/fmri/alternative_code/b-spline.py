#!/usr/bin/env python
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

"""
B-Spline Theory
===============

B-splines, or basis splines, are a family of piecewise polynomial functions used in approximation theory, computer 
graphics, and data fitting. They are defined over a sequence of knots and have several desirable properties, such as 
local support and partition of unity.

Definition:

A B-spline of degree p is defined over a non-decreasing sequence of knots t = { t_0, t_1, ..., t_{n+p+1} }. The 
B-spline basis functions B_{i,p}(x) are defined recursively using the Cox-de Boor recursion formula:

1. Base Case: For p = 0 (piecewise constant functions),
   B_{i,0}(x) = 
   { 
     1, if t_i <= x < t_{i+1} 
     0, otherwise 
   }

2. Recursive Case: For p > 0,
   B_{i,p}(x) = ((x - t_i) / (t_{i+p} - t_i)) * B_{i,p-1}(x) + ((t_{i+p+1} - x) / (t_{i+p+1} - t_{i+1})) * B_{i+1,p-1}(x)

Properties:

- Local Support: Each B-spline basis function B_{i,p}(x) is non-zero only over the interval [t_i, t_{i+p+1}). This 
  means that for any given x, only a small number of basis functions are non-zero, typically p+1, due to their 
  overlapping support.

- Partition of Unity: The B-spline basis functions sum to one for any x in the domain:
  sum(B_{i,p}(x)) = 1

- Smoothness: B-splines are p-1 times continuously differentiable at the knots, assuming the knots are distinct.

Combination of Basis Functions:

A spline function S(x) is constructed as a linear combination of B-spline basis functions:
S(x) = sum(c_i * B_{i,p}(x))
where c_i are the coefficients. Due to the local support property, for any given x, only a few terms contribute to the 
sum, making the evaluation efficient. 
"""



def bspline_basis(i, k, knots, x):
    """
    Compute the value of the i-th B-spline basis function of order k at x
    using the Cox-de Boor recursion formula.

    Parameters:
      i     : index of the B-spline basis function
      k     : order of the spline (order=1 gives piecewise constants, degree = k - 1)
      knots : list or array of knot positions (must be non-decreasing)
      x     : evaluation point(s) (scalar or numpy array)

    Returns:
      Value(s) of the B-spline basis function at x.
    """
    # Ensure x is a numpy array for vectorized operations
    x = np.atleast_1d(x)

    # Base case: for k == 1, the function is 1 if x is within [knots[i], knots[i+1])
    if k == 1:
        return np.where((x >= knots[i]) & (x < knots[i + 1]), 1.0,
                        np.where((x == knots[-1]) & (knots[i + 1] == knots[-1]), 1.0, 0.0))

    # Recursive step
    denom1 = knots[i + k - 1] - knots[i]
    denom2 = knots[i + k] - knots[i + 1]

    term1 = np.zeros_like(x, dtype=float)
    term2 = np.zeros_like(x, dtype=float)

    if denom1 != 0:
        term1 = ((x - knots[i]) / denom1) * bspline_basis(i, k - 1, knots, x)
    if denom2 != 0:
        term2 = ((knots[i + k] - x) / denom2) * bspline_basis(i + 1, k - 1, knots, x)

    return term1 + term2


def __mane__():
    # Define spline parameters: here we use a cubic spline (degree 3, order 4)
    degree = 3  # cubic spline => degree 3
    order = degree + 1  # order = degree + 1, hence 4
    # Define a clamped knot vector (repeated knots at the boundaries)
    knots = [0, 0, 0, 0, 2.5, 5.0, 7.5, 10, 10, 10, 10]

    # The number of B-spline basis functions is len(knots) - order.
    num_basis = len(knots) - order

    # Create a grid of x values over the domain.
    x_vals = np.linspace(knots[0], knots[-1], 400)

    # Plot each B-spline basis function.
    plt.figure(figsize=(10, 6))
    for i in range(num_basis):
        y_vals = bspline_basis(i, order, knots, x_vals)
        plt.plot(x_vals, y_vals, label=f'B_{i}')
    plt.title('B-spline Basis Functions (Cubic, Order=4)')
    plt.xlabel('x')
    plt.ylabel('Basis Function Value')
    plt.legend()
    plt.grid(True)

    # Sum all basis functions at each x value.
    sum_vals = np.zeros_like(x_vals)
    for i in range(num_basis):
        sum_vals += bspline_basis(i, order, knots, x_vals)

    # Check that the sum is approximately 1 at all evaluation points.
    tolerance = 1e-10
    if np.all(np.abs(sum_vals - 1) < tolerance):
        print("Success: The basis functions sum to 1 (within tolerance) at all x values.")
    else:
        print("Warning: The basis functions do not sum to 1 at some x values.")

    # Plot the sum of the basis functions to visually verify the partition of unity.
    plt.figure(figsize=(10, 4))
    plt.plot(x_vals, sum_vals, color='blue', lw=2, label='Sum of Basis Functions')
    plt.axhline(1, color='red', linestyle='--', label='1 (Exact Sum)')
    plt.title('Partition of Unity Check for B-spline Basis Functions')
    plt.xlabel('x')
    plt.ylabel('Sum')
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    __mane__()
