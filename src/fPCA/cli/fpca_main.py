#!/usr/bin/env python
"""
Console script for fmri
"""

import argparse
import logging
import sys
import os
import subprocess
import concurrent.futures
import numpy as np
from pathlib import Path

from fPCA.fpca import FunctionalMRI
from fPCA.utils import setup_logger


# ==============================================================================
# Route 1: Single Subject Worker (Runs on P-Cores as an independent process)
# ==============================================================================
def run_single_subject(args):
    """
    Executes the FunctionalMRI analysis for a single subject.
    Runs in a completely separate OS process to bypass macOS E-Core demotion.
    """
    base_name = os.path.basename(args.nii_files[0]).replace('.nii.gz', '').replace('.nii', '')
    subj_output_folder = args.internal_output_dir
    unique_log_file = f"fmri_log_{base_name}.txt"

    # Setup logger specific to this child process
    setup_logger(output_folder=subj_output_folder, file_name=unique_log_file, loger_name="fmri_logger",
                 log_level=logging.INFO)
    logger = logging.getLogger("fmri_logger")

    idx = args.internal_idx
    total = args.internal_total
    logger.info(f"--- Starting Subject {idx + 1}/{total}: {base_name} ---")

    try:
        fmri_instance = FunctionalMRI(
            nii_file=args.nii_files[0], mask_file=args.mask_files[0], degree=args.degree,
            n_basis=args.n_basis, threshold=args.threshold, num_pca_comp=args.num_pca_comp,
            batch_size=args.batch_size, output_folder=subj_output_folder, TR=args.TR,
            smooth_size=args.smooth_size, lambda_min=args.lambda_min, lambda_max=args.lambda_max,
            derivatives_num_p=args.derivatives_num_p, derivatives_num_u=args.derivatives_num_u,
            processed=args.processed, bad_margin_size=args.bad_margin_size,
            no_penalty=args.no_penalty,
            calc_penalty_bspline_accurately=args.calc_penalty_bspline_accurately,
            calc_penalty_skfda=args.calc_penalty_skfda,
            n_skip_vols_start=args.n_skip_vols_start, n_skip_vols_end=args.n_skip_vols_end,
            highpass=args.highpass, lowpass=args.lowpass, low_mem=args.low_mem,
            use_nilearn_filter=args.use_nilearn_filter, n_compcor_nilearn_filter=args.n_compcor_nilearn_filter,
            smoothing_fwhm_nilearn_filter=args.smoothing_fwhm_nilearn_filter
        )

        fmri_instance.log_data(sys.argv)

        # Execute heavy math using optimized libraries
        fmri_instance.run()

        # For the very first subject, export the global F and U matrices
        # so the main Manager script can use them later for Group PCA.
        if idx == 0:
            group_mats_path = os.path.join(os.path.dirname(subj_output_folder), "global_F_U_matrices.npz")
            np.savez_compressed(group_mats_path, F=fmri_instance.F, U=fmri_instance.U)

        logger.info(f"--- Finished Subject {idx + 1}/{total}: {base_name} ---")

    except Exception as e:
        logger.error(f"Error processing subject {base_name}: {str(e)}")
        sys.exit(1)


# ==============================================================================
# Helper: Subprocess Command Builder
# ==============================================================================
def process_subject_subprocess(idx, total_subjects, nii_file, mask_file, subj_output_folder, args):
    """
    Constructs a command line string to call this exact same script in 'single run' mode.
    """
    # Use the current Python interpreter and the path to this script
    cmd = [sys.executable, sys.argv[0]]

    # Inject hidden internal flags
    cmd.extend([
        "--internal-single-run",
        "--internal-idx", str(idx),
        "--internal-total", str(total_subjects),
        "--internal-output-dir", subj_output_folder,
        "--nii-files", nii_file,
        "--mask-files", mask_file,

        # Propagate configuration parameters
        "--degree", str(args.degree),
        "--threshold", str(args.threshold),
        "--num-pca-comp", str(args.num_pca_comp),
        "--batch-size", str(args.batch_size),
        "--smooth-size", str(args.smooth_size),
        "--lambda-min", str(args.lambda_min),
        "--lambda-max", str(args.lambda_max),
        "--derivatives-num-p", str(args.derivatives_num_p),
        "--derivatives-num-u", str(args.derivatives_num_u),
        "--bad-margin-size", str(args.bad_margin_size),
        "--n-skip-vols-start", str(args.n_skip_vols_start),
        "--n-skip-vols-end", str(args.n_skip_vols_end),
        "--highpass", str(args.highpass),
        "--lowpass", str(args.lowpass)
    ])

    if args.TR is not None:
        cmd.extend(["--TR", str(args.TR)])

    if args.n_basis:
        cmd.append("--n-basis")
        cmd.extend([str(x) for x in args.n_basis])

    # Propagate boolean flags
    if args.processed: cmd.append("--processed")
    if args.no_penalty: cmd.append("--no-penalty")
    if args.calc_penalty_bspline_accurately: cmd.append("--calc-penalty-bspline-accurately")
    if args.calc_penalty_skfda: cmd.append("--calc-penalty-skfda")
    if args.low_mem: cmd.append("--low-mem")

    # Propagate Nilearn filter arguments
    if args.use_nilearn_filter: cmd.append("--use-nilearn-filter")
    cmd.extend(["--n-compcor-nilearn-filter", str(args.n_compcor_nilearn_filter)])
    cmd.extend(["--smoothing-fwhm-nilearn-filter", str(args.smoothing_fwhm_nilearn_filter)])

    # Run the OS command and capture its output
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed for subject {idx + 1}:\n{result.stderr}\n{result.stdout}")

    return os.path.join(subj_output_folder, "eigvecs_eigval_F.npz")


# ==============================================================================
# Route 2: Main Application Manager
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Perform Functional PCA on fMRI data using B-spline basis expansion.")

    parser.add_argument(
        "--mode",
        default='pca-singles',
        nargs="+",
        choices=["pca-singles", "train-pca-group", "test-pca-project"],
        help=(
            "Specify the execution mode(s) for the pipeline: "
            "'pca-singles' - Process individual training subjects, run subject-level PCA, and extract their C matrices. "
            "'train-pca-group' - Aggregate C matrices from training subjects to build the shared global PCA space (requires pca-singles output). "
            "'test-pca-project' - Project unseen test subjects onto the pre-calculated global PCA space (requires train-pca-group output)."
        )
    )
    # Public arguments
    parser.add_argument("--nii-files", type=str, nargs="+", required=True, help="Paths to the 4D fMRI NIfTI files.")
    parser.add_argument("--mask-files", type=str, nargs="+", required=True, help="Paths to the 3D mask NIfTI files.")
    parser.add_argument("--output-folder", type=str, required=False, help="Path to save output files.")
    parser.add_argument("--degree", type=int, default=3, help="Degree of the B-spline basis.")
    parser.add_argument("--n-basis", type=int, default=[100], nargs="+", help="Number of B-spline basis functions.")
    parser.add_argument("--threshold", type=float, default=1e-6, help="Interpolation error threshold.")
    parser.add_argument("--num-pca-comp", type=int, default=7, help="Number of principal components.")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size for processing voxels.")
    parser.add_argument("--TR", type=float, default=0.75, help="Repetition time (TR).")
    parser.add_argument("--smooth-size", type=int, default=5, help="Box size of smoothing kernel.")
    parser.add_argument("--lambda-min", type=float, default=-6, help="Minimum value of lambda.")
    parser.add_argument("--lambda-max", type=float, default=12, help="Maximum value of lambda.")
    parser.add_argument("--derivatives-num-p", type=int, default=2, help="Derivatives in penalty P.")
    parser.add_argument("--derivatives-num-u", type=int, default=0, help="Derivatives in penalty U.")
    parser.add_argument("--processed", action='store_true', help="Data is already preprocessed.")
    parser.add_argument("--no-penalty", action='store_true', help="No penalty will be used.")
    parser.add_argument("--calc-penalty-bspline-accurately", action='store_true', help="Accurate bspline penalty.")
    parser.add_argument("--calc-penalty-skfda", action='store_true', help="Accurate skfda penalty.")
    parser.add_argument("--n-skip-vols-start", type=int, default=0, help="Vols to discard from start.")
    parser.add_argument("--n-skip-vols-end", type=int, default=0, help="Vols to discard from end.")
    parser.add_argument("--highpass", type=float, default=0.01, help="High-pass filter cutoff.")
    parser.add_argument("--lowpass", type=float, default=0.08, help="Low-pass filter cutoff.")
    parser.add_argument("--use-nilearn-filter", action='store_true', help="Use Nilearn for filtering.")
    parser.add_argument("--n-compcor-nilearn-filter", type=int, default=5, help="Number of compcor components for Nilearn filtering.")
    parser.add_argument("--smoothing-fwhm-nilearn-filter", type=float, default=6.0, help="FWHM for smoothing in Nilearn filtering.")
    parser.add_argument("--low-mem", action='store_true', help="If set, only NPZ/TXT/NIfTI files are written (no PNG plots).")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel processes to run. Use -1 for all cores.")

    # Internal hidden arguments used for the subprocess routing
    parser.add_argument("--internal-single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--internal-idx", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--internal-total", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--internal-output-dir", type=str, default="", help=argparse.SUPPRESS)
    # For tests
    parser.add_argument("--bad-margin-size", type=int, default=50, help="Size of margin to ignore (not in use for now).")

    args = parser.parse_args()

    # Route subprocess directly to the worker function, bypassing manager setup
    if args.internal_single_run:
        run_single_subject(args)
        return

    # Manager execution flow
    print(f"Running in mode(s): {args.mode}")
    print(f"Output folder: {args.output_folder}")

    if not args.output_folder:
        raise ValueError("Error: --output-folder is required for the main run.")

    # Validate output folder existence for the main process
    if os.path.exists(
        args.output_folder) and "train-pca-group" not in args.mode and "test-pca-project" not in args.mode:
        raise FileExistsError(f"Output folder '{args.output_folder}' already exists. Please delete it before running.")
    else:
        os.makedirs(args.output_folder, exist_ok=True)

    # Initialize the main manager logger
    setup_logger(output_folder=args.output_folder, file_name="fmri_group_log.txt", loger_name="fmri_logger",
                 log_level=logging.INFO)
    logger = logging.getLogger("fPCA_logger")

    # --------------------------------------------------------------------------
    # Manager Flow: Spawn background tasks
    # --------------------------------------------------------------------------
    nii_files = args.nii_files
    mask_files = args.mask_files
    if len(mask_files) == 1:
        mask_files = mask_files * len(nii_files)
    elif len(mask_files) != len(nii_files):
        raise ValueError(f"Error: Provided {len(nii_files)} nii files but {len(mask_files)} mask files.")



    total_subjects = len(nii_files)
    max_workers = os.cpu_count() if args.n_jobs == -1 else args.n_jobs
    logger.info(f"Starting pipeline for {total_subjects} subjects using {max_workers} subprocesses (P-Core optimized).")

    tasks = []
    for idx, (nii_file, mask_file) in enumerate(zip(nii_files, mask_files)):
        # Extract just the filename without the path
        filename = os.path.basename(nii_file)
        filename_noext = str(Path(filename).stem)
        # Split the string by '-preproc_bold' and take the first part
        # to get exactly the base identifier you want
        clean_base_name = filename_noext.split('-preproc_bold')[0]
        # Create the exact output folder path without the 'subj_xxx' prefix
        subj_output_folder = os.path.join(args.output_folder, clean_base_name)
        os.makedirs(subj_output_folder, exist_ok=True)
        tasks.append((idx, total_subjects, nii_file, mask_file, subj_output_folder, args))

    # --------------------------------------------------------------------------
    # Collect paths while preserving exact matching between c_files and masks
    # --------------------------------------------------------------------------
    c_file_paths_ordered = [None] * len(tasks)
    mask_files_ordered = [None] * len(tasks)

    if "train-pca-group" not in args.mode and "test-pca-project" not in args.mode:
        # Use ThreadPoolExecutor to act as a lightweight manager
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map each future back to its original task data
            future_to_task = {executor.submit(process_subject_subprocess, *task): task for task in tasks}

            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                orig_idx = task[0]  # The original subject index
                task_mask = task[3]  # The specific mask for this subject
                try:
                    c_path = future.result()
                    # Place the result in the exact original index spot
                    c_file_paths_ordered[orig_idx] = c_path
                    mask_files_ordered[orig_idx] = task_mask
                except Exception as exc:
                    logger.error(f"A subprocess generated an exception for subject {orig_idx}: {exc}")
    else: # "train-pca-group" in args.mode or "train-pca-group" in args.mode
        # Collect existing files using the correct index
        logger.info("Collecting existing files...")
        for task in tasks:
            orig_idx = task[0]
            task_mask = task[3]
            subj_output_folder = task[4]
            expected_c_path = os.path.join(subj_output_folder, "eigvecs_eigval_F.npz")
            if os.path.exists(expected_c_path):
                c_file_paths_ordered[orig_idx] = expected_c_path
                mask_files_ordered[orig_idx] = task_mask
            else:
                logger.warning(f"Missing expected file: {expected_c_path}")

        # If the global matrices file is missing due to moving folders,
        # regenerate it quickly by running ONLY the very first subject.
        group_mats_path = os.path.join(args.output_folder, "global_F_U_matrices.npz")
        if not os.path.exists(group_mats_path) and len(tasks) > 0:
            logger.info("Global matrices missing. Generating them by running the first subject (~1 min)...")
            try:
                # Save the returned path and original mask to our ordered lists so it isn't discarded
                rescued_c_path = process_subject_subprocess(*tasks[0])
                c_file_paths_ordered[tasks[0][0]] = rescued_c_path
                mask_files_ordered[tasks[0][0]] = tasks[0][3]
            except Exception as exc:
                logger.error(f"Rescue block failed for the first subject: {exc}")

        c_file_paths = [p for p in c_file_paths_ordered if p is not None]
        valid_nii_files = [nii_files[i] for i in range(len(nii_files)) if c_file_paths_ordered[i] is not None]
        valid_mask_files = [m for m in mask_files_ordered if m is not None]

        # --------------------------------------------------------------------------
        # Group fPCA Execution
        # --------------------------------------------------------------------------
        # Added explicit checks to ensure lists are not empty, silencing IDE warnings
        if c_file_paths and valid_nii_files and valid_mask_files:
            logger.info("==================================================")
            logger.info("All single-subject runs complete. Starting Group fPCA.")

            fmri_group_instance = FunctionalMRI(
                nii_file=valid_nii_files[0], mask_file=valid_mask_files[0], degree=args.degree,
                n_basis=args.n_basis, threshold=args.threshold, num_pca_comp=args.num_pca_comp,
                batch_size=args.batch_size, output_folder=args.output_folder, TR=args.TR,
                smooth_size=args.smooth_size, lambda_min=args.lambda_min, lambda_max=args.lambda_max,
                derivatives_num_p=args.derivatives_num_p, derivatives_num_u=args.derivatives_num_u,
                processed=args.processed, bad_margin_size=args.bad_margin_size,
                no_penalty=args.no_penalty,
                calc_penalty_bspline_accurately=args.calc_penalty_bspline_accurately,
                calc_penalty_skfda=args.calc_penalty_skfda,
                n_skip_vols_start=args.n_skip_vols_start, n_skip_vols_end=args.n_skip_vols_end,
                highpass=args.highpass, lowpass=args.lowpass, low_mem=args.low_mem,
                use_nilearn_filter=args.use_nilearn_filter, n_compcor_nilearn_filter=args.n_compcor_nilearn_filter,
                smoothing_fwhm_nilearn_filter=args.smoothing_fwhm_nilearn_filter
            )

            # Load the global matrices (F and U) created dynamically by the first subprocess
            group_mats_path = os.path.join(args.output_folder, "global_F_U_matrices.npz")
            if os.path.exists(group_mats_path):
                mats = np.load(group_mats_path)
                fmri_group_instance.F = mats['F']
                fmri_group_instance.U = mats['U']
            else:
                logger.error("Could not find the global F and U matrices file!")
            if "train-pca-group" in args.mode:
                fmri_group_instance.run_group_fpca(c_file_paths, valid_mask_files)
                logger.info("Group fPCA finished successfully.")
            else: # "test-pca-project" in args.mode
                fmri_group_instance.project_test_group_fpca(c_file_paths, valid_mask_files)
                logger.info("Test projection finished successfully.")

        else:
            logger.error("No valid subjects were processed successfully. Skipping Group fPCA.")

if __name__ == "__main__":
    main()
