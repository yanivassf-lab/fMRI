#!/usr/bin/env python
"""
Console script for fmri
"""

import argparse
import os

from fmri.fmri import FunctionalMRI


def main():
    parser = argparse.ArgumentParser(description="Perform Functional PCA on fMRI data using B-spline basis expansion.")
    parser.add_argument("--nii-file", type=str, help="Path to the 4D fMRI NIfTI file.")
    parser.add_argument("--mask-file", type=str, help="Path to the 3D mask NIfTI file.")
    parser.add_argument("--degree", type=int, default=3, help="Degree of the B-spline basis (default: 3).")
    parser.add_argument("--n-basis", type=int, default=0, nargs="+",
                        help="Number of basis functions. Use 0 to determine it automatically based on the interpolation threshold (default: 0).")
    parser.add_argument("--threshold", type=float, default=1e-6, help="Interpolation error threshold (default: 1e-6)")
    parser.add_argument("--num-pca-comp", type=int, default=3, help="Number of principal components (default: 3).")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size for processing voxels (default: 200).")
    parser.add_argument("--output-folder", type=str, required=True,
                        help="Path to a existing folder where output files will be saved.")
    parser.add_argument("--TR", type=float, default=None,
                        help="Repetition time (TR) in seconds. If not provided, it will be extracted from the NIfTI header.")
    parser.add_argument("--processed", action='store_true',
                        help="If set, the data is already preprocessed. If not set, preprocessing will be applied.")
    parser.add_argument("--calc-penalty-accurately", action='store_true',
                        help="If set, the penalty matrix will be calculated using an accurate method. If not set, an approximate method will be used.")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        raise FileNotFoundError(
            f"Output folder '{args.output_folder}' does not exist. Please create it before running."
        )

    # Create an instance of fMRI and set the output folder.
    fmri_instance = FunctionalMRI(nii_file=args.nii_file, mask_file=args.mask_file, degree=args.degree,
                                  n_basis=args.n_basis, threshold=args.threshold, num_pca_comp=args.num_pca_comp,
                                  batch_size=args.batch_size, output_folder=args.output_folder, TR=args.TR,
                                  processed=args.processed, calc_penalty_accurately=args.calc_penalty_accurately)

    # Run the analysis.
    fmri_instance.log_data()
    fmri_instance.run()


if __name__ == "__main__":
    main()

# fmri_main --output-folder ../output_files/disc --nii-file tests/test_input/toy50-53_drc144images.nii --mask-file tests/test_input/toy50-53_mask.nii --num-pca-comp 2 --n-basis 300
