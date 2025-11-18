import os

import numpy as np
import argparse
import sys
from fmri.utils import setup_logger

# --- Configuration ---

# Define stimulus times for each movement
STIMULUS_TIMES = [[30.0, 150.75, 324.75, 360.75, 427.5], [85.5, 228.0, 234.75, 337.5]]

# Define lags and durations to test
LAGS_TO_TEST = np.arange(-15, 10, 1.0)  # From -15 to +10 seconds
DURATIONS_TO_TEST = np.arange(10, 35, 1.0)  # within 10 to 35 seconds

# Define which PCs to test
PCS_TO_TEST = [1, 2, 3, 4, 5, 6]

PARAMS = "p2_u0_sk100"  # Default parameter folder to search for


def load_pc_signal_from_txt(txt_file, logger=None):
    """
    Loads the time and signal data from one of your temporal_profile_pc_X.txt files.
    """
    try:
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        # Assuming the file structure is as follows:
        # line 1: #Eigenfunction...
        # line 2: #Times:
        # line 3: 75.0 75.75 ...
        # line 4: #PC temporal profile:
        # line 5: 0.0278 0.0220 ...

        times_str = lines[2].strip().split()
        signal_str = lines[4].strip().split()

        times = np.array(times_str, dtype=float)
        signal = np.array(signal_str, dtype=float)

        return times, signal
    except Exception as e:
        logger.exception(f"  [Error] Could not read file {txt_file}: {e}")
        return None, None


def fast_corr_matrix(X, y):
    """
    X: matrix (#files , T)
    y: vector length T
    returns |corr| for each row
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    yc = y - y.mean()
    num = np.sum(Xc * yc, axis=1)
    den = np.sqrt(np.sum(Xc ** 2, axis=1) * np.sum(yc ** 2))
    return np.abs(num / den)


def build_regressor_vectorized(times, stim_times, lags, durs):
    reg = np.zeros_like(times)
    for t, lg, dr in zip(stim_times, lags, durs):
        start = t + lg
        end = start + dr
        idx = np.where((times >= start) & (times <= end))[0]
        reg[idx] = 1
    return reg


def find_best_params_vectorized(m_id, file_list, stimulus_times, pc_index,
                                       lags_to_test, durations_to_test,
                                       coarse_step=0.5, logger=None):
    """
    Vectorized, deterministic optimization of lags and durations.
    Coarse grid + hill climbing per event, no randomness.
    """
    logger.info(f"\n=== Optimizing PC{pc_index} Movement: {m_id + 1} ===")

    # ---------- Load signals ----------
    signals = []
    for fp, _ in file_list:
        t, sig = load_pc_signal_from_txt(fp, logger)
        signals.append(sig)
    signals = np.vstack(signals)
    times, _ = load_pc_signal_from_txt(file_list[0][0], logger)
    n_events = len(stimulus_times)

    # ---------- 1. Coarse grid search per event ----------
    best_lags = np.zeros(n_events)
    best_durs = np.zeros(n_events)

    lags_coarse = np.arange(lags_to_test.min(), lags_to_test.max() + coarse_step, coarse_step)
    durs_coarse = np.arange(durations_to_test.min(), durations_to_test.max() + coarse_step, coarse_step)

    for ev, t_stim in enumerate(stimulus_times):
        # Optimize lag
        corr_lags = []
        for lag in lags_coarse:
            reg = build_regressor_vectorized(times, [t_stim], [lag], [1.0])  # tiny duration
            corr = fast_corr_matrix(signals, reg).mean()
            corr_lags.append((lag, corr))
        best_lags[ev] = max(corr_lags, key=lambda x: x[1])[0]

        # Optimize duration
        corr_durs = []
        for dur in durs_coarse:
            reg = build_regressor_vectorized(times, [t_stim], [best_lags[ev]], [dur])
            corr = fast_corr_matrix(signals, reg).mean()
            corr_durs.append((dur, corr))
        best_durs[ev] = max(corr_durs, key=lambda x: x[1])[0]

    # ---------- 2. Hill climbing refinement ----------
    improved = True
    while improved:
        improved = False
        base_reg = build_regressor_vectorized(times, stimulus_times, best_lags, best_durs)
        base_corr = fast_corr_matrix(signals, base_reg).mean()

        for i in range(n_events):
            for delta_lag, delta_dur in [(coarse_step, 0), (-coarse_step, 0), (0, coarse_step), (0, -coarse_step)]:
                new_lags = best_lags.copy()
                new_durs = best_durs.copy()
                new_lags[i] += delta_lag
                new_durs[i] += delta_dur

                reg = build_regressor_vectorized(times, stimulus_times, new_lags, new_durs)
                corr = fast_corr_matrix(signals, reg).mean()

                if corr > base_corr:
                    best_lags, best_durs = new_lags, new_durs
                    base_corr = corr
                    improved = True

    # ---------- 3. Final regressor ----------
    final_reg = build_regressor_vectorized(times, stimulus_times, best_lags, best_durs)
    final_corr = fast_corr_matrix(signals, final_reg).mean()

    logger.info(f"  → Best mean correlation: {final_corr:.4f}")
    logger.info(f"  ✔ Lags: {best_lags}")
    logger.info(f"  ✔ Durations: {best_durs}")

    return best_lags, best_durs, final_corr, final_reg


def fast_corr_matrix_vectorized(X, Y):
    """
    X: shape = (n_signals, T)
    Y: shape = (n_candidates, T)
    returns: correlations |corr| shape = (n_signals, n_candidates)
    """
    Xc = X - X.mean(axis=1, keepdims=True)  # (n_signals, T)
    Yc = Y - Y.mean(axis=1, keepdims=True)  # (n_candidates, T)

    # חישוב מכפלה פנימית לכל זוג שורה
    num = Xc @ Yc.T  # (n_signals, n_candidates)
    den = np.sqrt(np.sum(Xc ** 2, axis=1)[:, None] * np.sum(Yc ** 2, axis=1)[None, :])
    corr = np.abs(num / den)
    return corr  # (n_signals, n_candidates)


def calculate_lags_durations(root_folder, params=None, stimulus_times=None, pcs_to_test=None, lags_to_test=LAGS_TO_TEST,
                             durations_to_test=DURATIONS_TO_TEST, logger=None):
    # We will store tuples: (file_path, movement_id)
    num_movements = len(stimulus_times)
    movement_ids = list(range(num_movements))  # or any iterable of movement IDs

    files_to_process = {
        pc_index: {m: [] for m in movement_ids}
        for pc_index in pcs_to_test
    }

    logger.info(f"Scanning for '{params}' files in {root_folder}...")

    for root, dirs, files in os.walk(root_folder, topdown=True):
        # We only care about folders matching our criteria
        if params not in os.path.basename(root):
            continue

        # Determine movement type from parent folder
        import re
        parent_folder = os.path.basename(os.path.dirname(root))

        movement_id = 0
        match = re.search(r"movement(\d+)", parent_folder, re.IGNORECASE)
        if match:
            movement_id = int(match.group(1)) - 1  # Convert to 0-based index
        else:
            continue  # Not a movement folder we care about

        # Find the PC text files we want to test
        for pc_index in pcs_to_test:
            pc_filename = f"temporal_profile_pc_{pc_index}.txt"
            if pc_filename in files:
                file_path = os.path.join(root, pc_filename)
                files_to_process[pc_index][movement_id].append((file_path, movement_id))

    logger.info("File scan complete. Found:")
    for pc_index, movements in files_to_process.items():
        counts = ", ".join(f"{len(files)} files (Mov{m + 1})" for m, files in movements.items())
        logger.info(f"  PC{pc_index}: {counts}")

    # --- 2. Run optimization for each PC and Movement ---
    final_results = {}
    for pc_index, movements in files_to_process.items():
        for m_id, files in movements.items():
            if files:
                best_lag, best_dur, best_corr, best_regressor = find_best_params_vectorized(m_id,
                                                                                                   files,
                                                                                                   stimulus_times[m_id],
                                                                                                   pc_index,
                                                                                                   lags_to_test,
                                                                                                   durations_to_test,
                                                                                                   coarse_step=0.01,
                                                                                                   logger=logger)
                final_results[(pc_index, m_id)] = (best_lag, best_dur, best_corr, best_regressor)
    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Search for best BOLD lags and durations for fMRI analysis per PC/movement.")
    parser.add_argument("root_folder", type=str,
                        help="The root folder containing the .../sub-XXX_movementY/p2_u0_skZ/ structure.")
    parser.add_argument("--params", type=str, default=PARAMS,
                        help=f"Specific parameter folder to search for (default: {PARAMS}).")
    parser.add_argument("--stimulus-times", type=float, nargs='+', action='append', default=STIMULUS_TIMES,
                        help="List of stimulus times for each movement (default: predefined).")
    parser.add_argument("--lags-to-test", type=float, nargs='+', default=LAGS_TO_TEST,
                        help="Array of lags to test (default: -15 to 10 by 1).")
    parser.add_argument("--durations-to-test", type=float, nargs='+', default=DURATIONS_TO_TEST,
                        help="Array of durations to test (default: 10 to 35 by 1).")
    parser.add_argument("--pcs-to-test", type=int, nargs='+', default=PCS_TO_TEST,
                        help="List of principal components to test (default: 0-6).")
    args = parser.parse_args()

    if not os.path.isdir(args.root_folder):
        print(f"Error: Path provided is not a valid directory: {args.root_folder}")
        sys.exit(1)
    logger = setup_logger(output_folder=args.root_folder, file_name="find_best_lags_log.txt", loger_name="find_best_lags_logger")
    # Call your calculation function
    lags_durations = calculate_lags_durations(args.root_folder, args.params, args.stimulus_times, args.pcs_to_test,
                                              args.lags_to_test, args.durations_to_test, logger=logger)

    # --- 3. Print final recommendations ---
    print("\n\n--- FInal Parameter Recommendations ---")
    for (pc_index, m_id), (lag, dur, corr, regressor) in lags_durations.items():
        print(f"\nAnalysis: PC{pc_index}_Movement{m_id + 1}")
        print(f"  Best Avg. Correlation: {corr:.4f}")
        print(f"  RECOMMENDED LAG:       {lag} seconds")
        print(f"  RECOMMENDED DURATION: {dur} seconds")
        print(f"  RECOMMENDED REGRESSION LEN: {len(regressor)} time points")


if __name__ == "__main__":
    main()
    sys.exit(0)

# ----- Old code for reference -----

# def build_regressor(times, stimulus_times, lag, duration):
#     """
#     Builds a "boxcar" regressor based on the given parameters.
#     """
#     target_regressor = np.zeros_like(times)
#     for t_stim in stimulus_times:
#         t_start = t_stim + lag
#         t_end = t_start + duration
#
#         indices = np.where((times >= t_start) & (times <= t_end))[0]
#         if len(indices) > 0:
#             target_regressor[indices] = 1.0
#     return target_regressor


# def find_best_params(file_list, stimulus_times, pc_index, lags_to_test, durations_to_test):
#     """
#     Iterates through all lags and durations to find the combination
#     with the highest average correlation.
#     """
#     print(f"\n--- Optimizing for PC{pc_index} (Movement {file_list[0][1] + 1}) ---")
#
#     # Store results here: results[(lag, duration)] = [list_of_correlations]
#     results = defaultdict(list)
#
#     for file_path, movement_id in file_list:
#         times, pc_signal = load_pc_signal_from_txt(file_path)
#         if times is None:
#             continue
#
#         for lag in lags_to_test:
#             for duration in durations_to_test:
#
#                 # Build the target regressor (boxcar) for these params
#                 target_regressor = build_regressor(times, stimulus_times, lag, duration)
#
#                 # Check if target is all zeros (e.g., stim times were cut off)
#                 if np.sum(target_regressor) == 0:
#                     continue
#
#                 try:
#                     corr, _ = pearsonr(pc_signal, target_regressor)
#                     results[(lag, duration)].append(abs(corr))
#                 except ValueError:
#                     continue  # Skip if constant signal
#
#     if not results:
#         print("  [Error] No valid correlations found.")
#         return None, None, 0
#
#     # Calculate average correlation for each param set
#     avg_correlations = {params: np.mean(corrs) for params, corrs in results.items()}
#
#     # Find the best one
#     best_params = max(avg_correlations, key=avg_correlations.get)
#     best_avg_corr = avg_correlations[best_params]
#
#     best_lag, best_duration = best_params
#     times, _ = load_pc_signal_from_txt(file_list[0][0])
#     best_regressor = build_regressor(times, stimulus_times, best_lag, best_duration)
#     return best_lag, best_duration, best_avg_corr, best_regressor
