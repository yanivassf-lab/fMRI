import os
import re
import numpy as np
import argparse
import sys

# --- Configuration ---
TR = 0.75
TIMEPOINTS_MAP = {
    "movement1": 655,
    "movement2": 774
}


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

        # 1. Find n_timepoints
        n_timepoints = None
        if "movement1" in subfolder_name:
            n_timepoints = TIMEPOINTS_MAP["movement1"]
        elif "movement2" in subfolder_name:
            n_timepoints = TIMEPOINTS_MAP["movement2"]
        else:
            print(f"Warning: Could not determine movement type for {file_path}. Skipping.")
            return None, None

        # 2. Find n_skip_vols_start
        match = re.search(r'sk(\d+)', subsubfolder_name)
        if not match:
            print(f"Warning: Could not parse skip vols from {subsubfolder_name}. Skipping file.")
            return None, None

        n_skip_vols_start = int(match.group(1))

        return n_timepoints, n_skip_vols_start

    except Exception as e:
        print(f"Error parsing path {file_path}: {e}")
        return None, None


def process_file(file_path):
    """
    Converts a single .npz file from the old format to the new format.
    """
    print(f"\n--- Processing file: {file_path}")

    # 1. Parse parameters from the path
    n_timepoints, n_skip_vols_start = parse_parameters_from_path(file_path)
    if n_timepoints is None:
        return  # Error already printed by parser

    print(f"  > Found params: n_timepoints={n_timepoints}, n_skip_vols_start={n_skip_vols_start}")

    # 2. Load old data
    try:
        data = np.load(file_path)
        eigvecs_sorted = data['a']
        F = data['c']
        eigvals_sorted = data['d']
    except Exception as e:
        print(f"  > ERROR: Could not load data from {file_path}. Is it in the old 'a', 'b', 'c' format? {e}")
        return

    # 3. Calculate new 'times' array
    try:
        times = np.arange(n_skip_vols_start, n_timepoints) * TR
        print(f"  > Calculated 'times' array (shape: {times.shape})")
    except Exception as e:
        print(f"  > ERROR: Could not calculate 'times' array: {e}")
        return

    # 4. Rename old file
    old_file_path = file_path + ".old"
    try:
        os.rename(file_path, old_file_path)
        print(f"  > Renamed original file to: {old_file_path}")
    except Exception as e:
        print(f"  > ERROR: Could not rename old file: {e}")
        return

    # 5. Save new file
    try:
        np.savez_compressed(
            file_path,
            eigvecs_sorted=eigvecs_sorted,
            F=F,
            eigvals_sorted=eigvals_sorted,
            times=times
        )
        print(f"  > Successfully created new file: {file_path}")
    except Exception as e:
        print(f"  > ERROR: Could not save new .npz file: {e}")
        # Attempt to roll back the rename
        try:
            os.rename(old_file_path, file_path)
            print(f"  > Rollback: Restored original file.")
        except Exception as rb_e:
            print(f"  > CRITICAL ERROR: Could not save new file AND could not restore old file. {rb_e}")


def main(root_folder):
    """
    Walks the root_folder and converts all found .npz files.
    """
    print(f"Starting conversion in root folder: {root_folder}")
    print(f"Using TR={TR} and Timepoints Map={TIMEPOINTS_MAP}")

    file_count = 0
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(root_folder, topdown=True):
        if "eigvecs_eigval_F.npz" in files:
            file_path = os.path.join(root, "eigvecs_eigval_F.npz")
            process_file(file_path)
            file_count += 1

    print(f"\n--- Conversion complete. Processed {file_count} files. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert fMRI .npz files to a new format with a calculated 'times' array.",
        epilog="Example: python convert_npz_format.py /path/to/your/data_folder"
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
