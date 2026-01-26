import numpy as np
from scipy.linalg import eigh
import time

# ==========================================
# Compare the old method for selection of
# lambda to the new one.
# ==========================================



# ==========================================
# 1. OLD IMPLEMENTATION
# ==========================================

def compute_coeff_by_regularized_regression(F: np.ndarray, FtF: np.ndarray, P: np.ndarray,
                                            lambda_vec: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
    A_batch = FtF[None, :, :] + lambda_vec[:, None, None] * P[None, :, :]
    RHS = (F.T @ y_batch.T).T
    RHS = RHS[..., None]
    coeffs_exp = np.linalg.solve(A_batch, RHS)
    coeffs = coeffs_exp.squeeze(-1)
    return coeffs


def compute_hat_matrix_old(F, FtF, P, lambda_const):
    return F @ np.linalg.inv(FtF + lambda_const * P) @ F.T


def compute_hat_matrices_all_lambda_old(F, FtF, P, n_timepoints, lambda_values):
    n_lambda = len(lambda_values)
    H_all_lambda = np.zeros((n_lambda, n_timepoints, n_timepoints))
    for i, lambda_const in enumerate(lambda_values):
        H_all_lambda[i, :, :] = compute_hat_matrix_old(F, FtF, P, lambda_const)
    return H_all_lambda


def select_lambda_old(I_minus_H, voxel_data_batch, n_timepoints, lambda_values):
    R = np.matmul(I_minus_H, voxel_data_batch.T)
    res_norm_sq = np.sum(R ** 2, axis=1)
    traces = np.trace(I_minus_H, axis1=1, axis2=2)
    gcv_scores = (n_timepoints * res_norm_sq) / (traces[:, None] ** 2)

    best_idx = np.argmin(gcv_scores, axis=0)
    best_lambda = lambda_values[best_idx]
    best_gcv_scores = gcv_scores[best_idx, np.arange(gcv_scores.shape[1])]

    return best_lambda, best_gcv_scores


# ==========================================
# 2. NEW IMPLEMENTATION (Optimized)
# ==========================================
def solve_batch_gcv_new(voxel_data_batch, F_V, VT_FT, V, evals, lambda_values):
    n_voxels, n_timepoints = voxel_data_batch.shape

    # Projection
    Z = VT_FT @ voxel_data_batch.T  # (n_basis, n_voxels)

    gcv_scores_all = np.zeros((len(lambda_values), n_voxels))

    for i, lam in enumerate(lambda_values):
        D_inv = 1.0 / (evals + lam)
        tr_H = np.sum(evals * D_inv)

        # Coefficients in eigen-space for this lambda
        # Broadcasting: (n_basis, 1) * (n_basis, n_voxels)
        C_eig = Z * D_inv[:, np.newaxis]

        # Predictions
        Y_hat = F_V @ C_eig

        # RSS
        rss = np.sum((voxel_data_batch.T - Y_hat) ** 2, axis=0)

        # GCV
        denominator = (1 - tr_H / n_timepoints) ** 2
        gcv_scores_all[i, :] = (rss / n_timepoints) / denominator

    # Selection
    best_indices = np.argmin(gcv_scores_all, axis=0)
    best_lambdas = lambda_values[best_indices]
    best_scores = gcv_scores_all[best_indices, np.arange(n_voxels)]

    # Reconstruction (Replaces your old function logic)
    optimal_D_inv = 1.0 / (evals[:, np.newaxis] + best_lambdas[np.newaxis, :])
    Z_scaled = Z * optimal_D_inv
    coeffs = (V @ Z_scaled).T

    return coeffs, best_lambdas, best_scores


# ==========================================
# 3. TEST COMPARISON
# ==========================================
def run_comparison_test():
    np.random.seed(42)
    print("Generating Data...")
    n_timepoints = 50
    n_basis = 10
    n_voxels = 100
    n_lambdas = 20

    F = np.random.randn(n_timepoints, n_basis)
    P_raw = np.random.randn(n_basis, n_basis)
    P = P_raw.T @ P_raw
    FtF = F.T @ F
    Y = np.random.randn(n_voxels, n_timepoints)
    lambda_values = np.logspace(-3, 3, n_lambdas)

    # --- OLD METHOD ---
    print("\nRunning OLD Method (Brute Force + linalg.solve)...")
    t0 = time.time()

    # Step 1: Find Lambdas
    H_all = compute_hat_matrices_all_lambda_old(F, FtF, P, n_timepoints, lambda_values)
    I_minus_H = np.eye(n_timepoints)[None, :, :] - H_all
    lam_old, score_old = select_lambda_old(I_minus_H, Y, n_timepoints, lambda_values)

    # Step 2: Compute Coefficients (Using YOUR function)
    coeffs_old = compute_coeff_by_regularized_regression(F, FtF, P, lam_old, Y)

    t_old = time.time() - t0
    print(f"Old Time: {t_old:.4f}s")

    # --- NEW METHOD ---
    print("\nRunning NEW Method (GEVD Optimized)...")
    t1 = time.time()

    eps = 1e-12
    evals, V = eigh(FtF, P + eps * np.eye(n_basis))
    F_V = F @ V
    VT_FT = V.T @ F.T

    coeffs_new, lam_new, score_new = solve_batch_gcv_new(Y, F_V, VT_FT, V, evals, lambda_values)

    t_new = time.time() - t1
    print(f"New Time: {t_new:.4f}s")
    print(f"Speedup: {t_old / t_new:.2f}x")

    # --- CHECKS ---
    print("\nVerifying Accuracy...")

    diff_coeffs = np.max(np.abs(coeffs_old - coeffs_new))
    print(f"Max Coeff Diff: {diff_coeffs:.2e}")

    diff_scores = np.max(np.abs(score_old - score_new))
    print(f"Max Score Diff: {diff_scores:.2e}")

    if diff_coeffs < 1e-5:
        print("✅ Coefficients Match!")
    else:
        print("❌ Coefficients Mismatch (Check P singularity or epsilon)")


if __name__ == "__main__":
    run_comparison_test()
