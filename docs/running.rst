Running the fMRI Pipeline
==========================

This section describes how to run the `fmri-main` command-line tool to perform functional PCA on your fMRI data.

Synopsis
--------

.. code-block:: bash

   fmri-main --nii-file <PATH_TO_4D_NIFTI> \
             --mask-file <PATH_TO_3D_MASK_NIFTI> \
             --output-folder <OUTPUT_DIR> \
             [--degree <INT>] \
             [--n-basis <INT>] \
             [--threshold <FLOAT>] \
             [--num-pca-comp <INT>] \
             [--batch-size <INT>] \
             [--TR <FLOAT>] \
             [--smooth-size <INT>] \
             [--lambda-min <FLOAT>] \
             [--lambda-max <FLOAT>] \
             [--derivatives-num-p <INT>] \
             [--derivatives-num-u <INT>] \
             [--processed <BOOL>] \
             [--bad-margin-size <INT>] \
             [--no-penalty <BOOL>] \
             [--calc-penalty-bspline-accurately <BOOL>] \
             [--calc-penalty-skfda <BOOL>] \
             [--n-skip-vols-start <int>] \
             [--n-skip-vols-end <int>] \
             [--highpass <FLOAT>] \
             [--lowpass <FLOAT>]

Arguments
---------

\-\-nii-file `<PATH>`
  Path to the 4D fMRI NIfTI file (required).

\-\-mask-file `<PATH>`
  Path to the 3D binary mask NIfTI file (required).

\-\-output-folder `<DIR>`
  Directory where all output maps and plots will be saved (required).

\-\-degree `<INT>`
  Degree of the B-spline basis (default: 3).

\-\-n-basis `<INT>`
  Number of B-spline basis functions. Use 0 to determine it as number of timepoints or use several values for finding the best automatically based on the interpolation threshold (default: 0).

\-\-threshold `<FLOAT>`
  Interpolation error threshold for basis selection (default: 1e-6).

\-\-num-pca-comp `<INT>`
  Number of principal components to extract (default: 3).

\-\-batch-size `<INT>`
  Number of voxels processed per batch (default: 200).

\-\-TR `<FLOAT>`
  Repetition time (TR) in seconds. If not specified, the TR will be inferred from the NIfTI header (default: None).

\-\-smooth-size `<INT>`
  Box size of smoothing kernel. Relevant only if --processed is not set (default: 5).

\-\-lambda-min `<FLOAT>`
  Minimum value of lambda in log10 scale (i.e., 10^-4) (default: -4).

\-\-lambda-max `<FLOAT>`
  Maximum value of lambda in log10 scale (i.e., 10^3) (default: 3).

\-\-derivatives-num-p `<INT>`
  Number of derivatives in calculation of penalty matrix P (default: 2)

\-\-derivatives-num-u `<INT>`
  Number of derivatives in calculation of penalty matrix U (default: 0)

\-\-processed `<optional>`
  If specified, the input data is assumed to be post-processed (e.g., smoothing, filtering), and no additional post-processing will be applied. If not specified the pipeline will apply basic post-processing steps (default: not set).

\-\-bad-margin-size `<INT>`
  Size of the margin to ignore in calculating direction of eigvecs (default: 50).

\-\-no-penalty `<BOOL>`
  If specified, no penalty will be used (default: not set).

\-\-calc-penalty-bspline-accurately `<BOOL>`
  If set, the penalty matrix will be calculated using bspline package with an accurate method. If not set, an approximate method of bspline will be used (default: not set).

\-\-calc-penalty-skfda `<BOOL>`
  If set, the penalty matrix will be calculated using skfda package an accurate method. If not set, an approximate method of bsplie will be used (default: not set).

\-\-n-skip-vols-start `<INT>`
  Number of initial fMRI volumes to discard from the beginning of the signal (default: 0).

\-\-n-skip-vols-end `<ING>`
  Number of initial fMRI volumes to discard from the end of the signal (default: 0).

\-\-highpass `<FLOAT>`
  High-pass filter cutoff frequency in Hz. Filters out slow drifts below this frequency (default: 0.01).

\-\-lowpass `<FLOAT>`
  Low-pass filter cutoff frequency in Hz. Filters out high-frequency noise above this frequency (default: 0.08).

Examples
--------

Run a two-component analysis on a toy dataset:

.. code-block:: bash

   fmri-main \
     --nii-file tests/test_input/toy50-53_drc144images.nii \
     --mask-file tests/test_input/toy50-53_mask.nii \
     --output-folder output/toy_run \
     --num-pca-comp 2 \
     --n-basis 300

Run with all defaults (except output folder):

.. code-block:: bash

   fmri-main \
     --nii-file data/sub-01_task-rest_bold.nii.gz \
     --mask-file data/sub-01_mask.nii.gz \
     --output-folder results/sub-01


The argument *threshold*:
------------------------

    Maximum allowed mean absolute interpolation error when selecting the number of
    B-spline basis functions automatically (i.e. when ``--n-basis 0``).

    If you set ``n_basis=0`` (the default “auto” mode), the pipeline will:

    1. Try successive values of ``n_basis`` (from ``degree+1`` up to ``n_timepoints+20`` in steps of 10).
    2. For each candidate, fit the spline and compute the mean absolute error between the original
       voxel signals and their spline reconstructions.
    3. Stop at the first ``n_basis`` whose error ≤ ``threshold``, log that choice, and proceed.
    4. If none meets the threshold, choose the ``n_basis`` with the smallest observed error,
       log the achieved mean error, and continue.

    In practice, a smaller ``threshold`` forces more basis functions (and thus a finer interpolation),
    at the cost of higher computational time; a larger ``threshold`` results in fewer basis
    functions and a coarser fit.
Notes
-----

- Make sure the output folder exists before running; otherwise, the command will raise a `FileNotFoundError`.
- Output files include:

  - **voxel importance maps** (`eigenfunction_<k>_importance_map.nii.gz`)

  - **intensity plots** (`eigenfunction_<k>_signal_intensity.png`)

  - **best-voxel fit plots** (`eigenfunction_<k>_best_voxel.png`

