#!/usr/bin/env python
"""
Console script for fmri
"""

import argparse

from fmri.fmri import FunctionalMRI

import logging
import sys
import os
from fmri.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Perform Functional PCA on fMRI data using B-spline basis expansion.")
    parser.add_argument("--nii-file", type=str, help="Path to the 4D fMRI NIfTI file.")
    parser.add_argument("--mask-file", type=str, help="Path to the 3D mask NIfTI file.")
    parser.add_argument("--degree", type=int, default=3, help="Degree of the B-spline basis (default: 3).")
    parser.add_argument("--n-basis", type=int, default=[0], nargs="+",
                        help="Number of B-spline basis functions. Use 0 to determine it as number of timepoints or use several values for finding the best automatically based on the interpolation threshold (default: 0).")
    parser.add_argument("--threshold", type=float, default=1e-6, help="Interpolation error threshold (default: 1e-6)")
    parser.add_argument("--num-pca-comp", type=int, default=3, help="Number of principal components (default: 3).")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size for processing voxels (default: 200).")
    parser.add_argument("--output-folder", type=str, required=True,
                        help="Path to a existing folder where output files will be saved.")
    parser.add_argument("--TR", type=float, default=None,
                        help="Repetition time (TR) in seconds. If not provided, it will be extracted from the NIfTI header.")
    parser.add_argument("--smooth-size", type=int, default=5,
                        help="Box size of smoothing kernel. Relevant only if --processed is not set (default: 5).")
    parser.add_argument("--lambda-min", type=float, default=-4,
                        help="Minimum value of lambda in log10 scale (i.e., 10^-4) (default: -4)")
    parser.add_argument("--lambda-max", type=float, default=3,
                        help="Maximum value of lambda in log10 scale (i.e., 10^3) (default: 3)")
    parser.add_argument("--derivatives-num-p", type=int, default=2,
                        help="Number of derivatives in calculation of penalty matrix P (default: 2)")
    parser.add_argument("--derivatives-num-u", type=int, default=0,
                        help="Number of derivatives used in calculating the penalty matrix (U) — not recommended to change from the default value (default: 0).")
    parser.add_argument("--processed", action='store_true',
                        help="If set, the data is already preprocessed. If not set, preprocessing will be applied.")
    parser.add_argument("--bad-margin-size", type=int, default=50,
                        help="Size of the margin to ignore in calculating direction of eigvecs (default: 50).")
    parser.add_argument("--no-penalty", action='store_true',
                        help="If set, no penalty will be used.")
    parser.add_argument("--calc-penalty-bspline-accurately", action='store_true',
                        help="If set, the penalty matrix will be calculated using bspline package with an accurate method. If not set, an approximate method of bspline will be used.")
    parser.add_argument("--calc-penalty-skfda", action='store_true',
                        help="If set, the penalty matrix will be calculated using skfda package an accurate method. If not set, an approximate method of bsplie will be used.")
    parser.add_argument("--n-skip-vols-start", type=int, default=0,
                        help="Number of initial fMRI volumes to discard from the beginning of the signal (default: 0).")
    parser.add_argument("--n-skip-vols-end", type=int, default=0,
                        help="Number of initial fMRI volumes to discard from the end of the signal (default: 0).")
    parser.add_argument("--highpass", type=float, default=0.01,
                        help="High-pass filter cutoff frequency in Hz. Filters out slow drifts below this frequency (default: 0.01).")
    parser.add_argument("--lowpass", type=float, default=0.08,
                        help="Low-pass filter cutoff frequency in Hz. Filters out high-frequency noise above this frequency (default: 0.08).")

    args = parser.parse_args()

    if os.path.exists(args.output_folder):
        raise FileExistsError(
            f"Output folder '{args.output_folder}' already exist. Please delete it before running."
        )
    else:
        os.makedirs(args.output_folder)

    setup_logger(output_folder=args.output_folder, file_name="fmri_log.txt", loger_name="fmri_logger",
                 log_level=logging.INFO)

    # Create an instance of fMRI and set the output folder.
    fmri_instance = FunctionalMRI(nii_file=args.nii_file, mask_file=args.mask_file, degree=args.degree,
                                  n_basis=args.n_basis, threshold=args.threshold, num_pca_comp=args.num_pca_comp,
                                  batch_size=args.batch_size, output_folder=args.output_folder, TR=args.TR,
                                  smooth_size=args.smooth_size, lambda_min=args.lambda_min, lambda_max=args.lambda_max,
                                  derivatives_num_p=args.derivatives_num_p, derivatives_num_u=args.derivatives_num_u,
                                  processed=args.processed, bad_margin_size=args.bad_margin_size,
                                  no_penalty=args.no_penalty,
                                  calc_penalty_bspline_accurately=args.calc_penalty_bspline_accurately,
                                  calc_penalty_skfda=args.calc_penalty_skfda,
                                  n_skip_vols_start=args.n_skip_vols_start, n_skip_vols_end=args.n_skip_vols_end,
                                  highpass=args.highpass, lowpass=args.lowpass)

    # Run the analysis.
    fmri_instance.log_data(sys.argv)
    fmri_instance.run()


if __name__ == "__main__":
    main()

# fmri_main --output-folder ../output_files/disc --nii-file tests/test_input/toy50-53_drc144images.nii --mask-file tests/test_input/toy50-53_mask.nii --num-pca-comp 2 --n-basis 300
