import numpy as np


def solve_batch_gcv(
    voxel_data_batch,
    F_V,
    VT_FT,
    V,
    evals,
    lambda_values,
):
    """
    Solves the penalized regression problem for a batch of voxels using
    Generalized Cross Validation (GCV) and a precomputed generalized
    eigenvalue decomposition.

    Mathematical Logic:
        For each voxel, we solve the penalized least-squares problem:

            minimize_c  ||y - F @ c||^2 + lambda * c.T @ P @ c

        The precomputed transformation V satisfies:

            V.T @ F.T @ F @ V = I
            V.T @ P @ V = diag(evals)

        Thus, writing c = V @ a diagonalizes the problem:

            a_k = z_k / (1 + lambda * evals_k)

        where z = V.T @ F.T @ y.

        Unlike a decomposition that requires P to be invertible, this
        formulation allows zero penalty eigenvalues. A direction with
        evals_k = 0 has shrinkage factor 1 and remains unpenalized.

        Since F_V = F @ V has orthonormal columns, the residual sum of
        squares can be calculated in these coordinates without rebuilding
        the full fitted time series for every candidate lambda.

    Parameters:
        voxel_data_batch (ndarray):
            Shape (n_voxels, n_timepoints). Observed voxel time series.
        F_V (ndarray):
            Shape (n_timepoints, n_basis). Precomputed F @ V, with
            orthonormal columns.
        VT_FT (ndarray):
            Shape (n_basis, n_timepoints). Precomputed V.T @ F.T,
            which equals F_V.T.
        V (ndarray):
            Shape (n_basis, n_basis). Generalized eigenvectors normalized
            with respect to F.T @ F, as specified above.
        evals (ndarray):
            Shape (n_basis,). Penalty eigenvalues in the transformed
            coordinates; assumed nonnegative.
        lambda_values (iterable):
            Candidate nonnegative regularization parameters, evaluated
            in the supplied order.

    Returns:
        coefficients (ndarray):
            Shape (n_voxels, n_basis). Fitted B-spline coefficients,
            using the selected lambda for each voxel.
        best_lambdas (ndarray):
            Shape (n_voxels,). Selected lambda per voxel.
        best_scores (ndarray):
            Shape (n_voxels,). Minimum accepted GCV score per voxel.

    Raises:
        ValueError:
            If at least one voxel has no finite accepted GCV score.
    """
    # 1. Arrange the data with one voxel per column.
    # Shape: (n_timepoints, n_voxels)
    Y = np.asarray(voxel_data_batch, dtype=float).T
    n_timepoints, n_voxels = Y.shape

    # Project all voxel time series onto the transformed basis:
    # Z = V.T @ F.T @ Y = F_V.T @ Y.
    # Shape: (n_basis, n_voxels)
    Z = VT_FT @ Y

    # Cache the squared projected coordinates for the RSS calculation.
    # Shape: (n_basis, n_voxels)
    Z2 = Z * Z

    # 2. Calculate the part of the residual that is independent of lambda.
    # F_V has orthonormal columns.
    # Compute the residual outside their span only once.
    # F_V @ Z is the orthogonal projection of Y onto that span.
    # Shape of residual_ss: (n_voxels,)
    residual_ss = np.sum((Y - F_V @ Z) ** 2, axis=0)

    # 3. Initialize the best result for each voxel.
    # Keep only the current best scores and lambdas, rather than storing
    # a complete (n_lambdas, n_voxels) table.
    # A voxel's first finite accepted score replaces its initial infinity.
    best_scores = np.full(n_voxels, np.inf)
    best_lambdas = np.full(n_voxels, np.nan)

    # 4. Evaluate each candidate lambda for all voxels simultaneously.
    for lam in lambda_values:
        # Shrinkage factors in the transformed coordinates:
        # shrink_k = 1 / (1 + lambda * evals_k).
        # Shape: (n_basis,)
        # Zero penalty eigenvalues retain shrinkage factor 1.
        shrink = 1.0 / (1.0 + lam * evals)

        # The fitted-value operator (hat matrix) is:
        # H = F_V @ diag(shrink) @ F_V.T.
        # Since F_V has orthonormal columns:
        # tr(H) = sum(shrink).
        # GCV uses the remaining degrees of freedom n - tr(H).
        remaining_df = n_timepoints - shrink.sum()

        # Skip candidates whose GCV denominator is numerically too small.
        # This skips the current lambda and proceeds to the next one.
        if remaining_df <= np.finfo(float).eps * n_timepoints:
            continue

        # Split the residual into two orthogonal parts:
        #   - The component outside the span of F_V.
        #   - The component left inside that span by shrinkage.
        #
        # Therefore, for each voxel:
        # RSS = residual_ss + sum_k((1 - shrink_k)^2 * Z_k^2).
        #
        # The matrix product sums over basis directions for every voxel:
        # (n_basis,) @ (n_basis, n_voxels) -> (n_voxels,)
        rss = residual_ss + ((1.0 - shrink) ** 2) @ Z2

        # Generalized Cross Validation:
        # GCV = (RSS / n) / (1 - tr(H) / n)^2
        #     = n * RSS / (n - tr(H))^2.
        # Shape: (n_voxels,)
        scores = n_timepoints * rss / remaining_df ** 2

        # Update only voxels whose GCV score strictly improves.
        # Exact ties retain the earlier candidate in lambda_values.
        better = scores < best_scores
        best_scores[better] = scores[better]
        best_lambdas[better] = lam

    # 5. Verify that every voxel obtained a finite accepted GCV score.
    # If any voxel did not, the function raises the existing error.
    if not np.isfinite(best_scores).all():
        raise ValueError("No finite GCV score; check basis and lambda range")

    # 6. Reconstruct coefficients using each voxel's selected lambda.
    # Broadcasting combines:
    #   evals[:, None]:         (n_basis, 1)
    #   best_lambdas[None, :]:  (1, n_voxels)
    # Resulting shrink shape:  (n_basis, n_voxels)
    shrink = 1.0 / (
        1.0 + evals[:, None] * best_lambdas[None, :]
    )

    # Z * shrink contains the fitted coefficients in transformed coordinates.
    # Multiplying by V returns them to the original B-spline basis:
    # C = V @ (Z * shrink).
    # Transpose to return one row per voxel.
    # Final shape: (n_voxels, n_basis)
    coefficients = (V @ (Z * shrink)).T
    return coefficients, best_lambdas, best_scores
