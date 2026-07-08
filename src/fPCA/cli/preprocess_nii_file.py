#!/usr/bin/env python
"""
Console script for preprocess NIfTI file
"""

import argparse
import os
from pathlib import Path

from fPCA.preprocess import LoadData


def main():
    parser = argparse.ArgumentParser(description="Perform Functional PCA on fMRI data using B-spline basis expansion.")
    parser.add_argument("--nii-file", type=str, help="Path to the 4D fMRI NIfTI file.")
    parser.add_argument("--mask-file", type=str, help="Path to the 3D mask NIfTI file.")
    parser.add_argument("--output-folder", type=str, required=True,
                        help="Path to a existing folder where output files will be saved.")
    parser.add_argument("--TR", type=float, default=None,
                        help="Repetition time (TR) in seconds. If not provided, it will be extracted from the NIfTI header.")
    parser.add_argument("--smooth-size", type=int, default=5,
                        help="Box size of smoothing kernel. Relevant only if --processed is not set (default: 5).")
    parser.add_argument("--highpass", type=float, default=0.01,
                        help="High-pass filter cutoff frequency in Hz. Filters out slow drifts below this frequency.")
    parser.add_argument("--lowpass", type=float, default=0.08,
                        help="Low-pass filter cutoff frequency in Hz. Filters out high-frequency noise above this frequency.")
    parser.add_argument("--use-nilearn-filter", action='store_true', help="Use Nilearn for filtering.")
    parser.add_argument("--n-compcor-nilearn-filter", type=int, default=5, help="Number of compcor components for Nilearn filtering.")
    parser.add_argument("--smoothing-fwhm-nilearn-filter", type=float, default=6.0, help="FWHM for smoothing in Nilearn filtering.")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        raise FileNotFoundError(
            f"Output folder '{args.output_folder}' does not exist. Please create it before running."
        )

    data = LoadData(
        args.nii_file,
        args.mask_file,
        TR=args.TR,
        smooth_size=args.smooth_size,
        highpass=args.highpass,
        lowpass=args.lowpass,
        use_nilearn=args.use_nilearn_filter,
        n_compcor_nilearn_filter=args.n_compcor_nilearn_filter,
        smoothing_fwhm_nilearn_filter=args.smoothing_fwhm_nilearn_filter
    )
    _, _, _ = data.load_data(processed=False, save_filtered_file=True)

    # Build output filename in compressed NIfTI format (.nii.gz),
    # regardless of whether the input was .nii or .nii.gz.
    input_name = Path(args.nii_file).name
    if input_name.endswith(".nii.gz"):
        base = input_name[:-7]
    elif input_name.endswith(".nii"):
        base = input_name[:-4]
    else:
        base = input_name

    filtered_name = base + "_filtered.nii.gz"
    full_output_path = Path(args.output_folder) / filtered_name

    data.save_filtered_data(full_output_path)

    print(f"Filtered NIfTI file saved to {full_output_path}")


if __name__ == "__main__":
    main()

# preprocess_nii_file --output-folder /path/to/existing/folder --nii-file /path/to/nii/file --mask-file /path/to/mask/file
