import os
import sys
import argparse

from fPCA.utils import setup_logger
from fPCA.lags_durations import calculate_lags_durations, PARAMS, STIMULUS_TIMES, LAGS_TO_TEST, DURATIONS_TO_TEST, \
    PCS_TO_TEST


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
    logger = setup_logger(output_folder=args.root_folder, file_name="find_best_lags_log.txt",
                          loger_name="find_best_lags_logger")
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
