import os
import re
import numpy as np
import argparse
import sys


# ---------------------

def parse_parameters_from_path(file_path):
    """
    Parses n_timepoints and n_skip_vols_start from the file's path.

    Path structure is assumed to be:
    .../subfolder(contains movement)/subsubfolder(contains skips)/file.npz
    """
    try:
        # Get the subsubfolder (e.g., "p0_u0_sk100")
        subsubfolder_path = os.path.dirname(file_path)
        subsubfolder_name = os.path.basename(subsubfolder_path)

        # Get the subfolder (e.g., "sub-01_ses-01_task-movement1...")
        subfolder_path = os.path.dirname(subsubfolder_path)
        subfolder_name = os.path.basename(subfolder_path)

        # 2. Find n_skip_vols_start
        match = re.search(r'sk(\d+)', subsubfolder_name)
        if not match:
            print(f"Warning: Could not parse skip vols from {subsubfolder_name}. Skipping file.")
            return None, None

        n_skip_vols_start = int(match.group(1))

        return n_skip_vols_start

    except Exception as e:
        print(f"Error parsing path {file_path}: {e}")
        return None


def process_file(file_path, save_negative):
    """
    Converts a single .npz file from the old format to the new format.
    """
    print(f"\n--- Processing file: {file_path}")

    # 1. Parse parameters from the path
    n_skip_vols_start = parse_parameters_from_path(file_path)

    print(f"  Found params: n_skip_vols_start={n_skip_vols_start}")

    # 2. Load old data
    try:
        data = np.load(file_path)
        eigvecs_sorted = data['eigvecs_sorted']
        F = data['F']
        eigvals_sorted = data['eigvals_sorted']
    except Exception as e:
        eigvecs_sorted = data['a']
        F = data['c']
        eigvals_sorted = data['b']
    except Exception as e:
        print(f"  > ERROR: Could not load data from {file_path}: {e}")
        return

    negative_eigvals = np.any(eigvals_sorted <= 0.0)
    negative_eigvals_tol = np.any(eigvals_sorted <= -1e-6)
    mean_eigval = np.mean(eigvals_sorted)
    txt1 = f" Loaded data: eigvecs_sorted shape={eigvecs_sorted.shape}, F shape={F.shape}, eigvals_sorted shape={eigvals_sorted.shape}"
    txt2 = f"File {file_path}, Eigvals: mean={mean_eigval:.4f}, negative_present={negative_eigvals}, negative_eigvals_tol={negative_eigvals_tol}"
    print(txt1)
    print(txt2)

    if negative_eigvals:
        save_negative.append(txt1)
        save_negative.append(txt2)



def main(root_folder):
    """
    Walks the root_folder and converts all found .npz files.
    """
    print(f"Starting statistics in root folder: {root_folder}")

    file_count = 0
    save_negative = []
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(root_folder, topdown=True):
        if "eigvecs_eigval_F.npz" in files:
            file_path = os.path.join(root, "eigvecs_eigval_F.npz")
            process_file(file_path, save_negative)
            file_count += 1
    if save_negative:
        print("\n--- Files with negative eigenvalues detected: ---\n")
        for line in save_negative:
            print(line + "\n")

    print(f"\n--- Conversion complete. Processed {file_count} files. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Statistics Outputs NPZ Format Converter",
        epilog="Example: python statistics_outputs.py /path/to/your/data_folder"
    )
    parser.add_argument(
        "folder",
        type=str,
        help="The root folder containing the .../subfolder/subsubfolder/file.npz structure."
    )

    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: Path provided is not a valid directory: {args.folder}")
        sys.exit(1)

    main(args.folder)
