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
             [--processed <BOOL>] \
             [--calc-penalty-accurately <optional>]

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
  Number of B-spline basis functions (default: 0 → automatic selection).

\-\-threshold `<FLOAT>`
  Interpolation error threshold for basis selection (default: 1e-6).

\-\-num-pca-comp `<INT>`
  Number of principal components to extract (default: 3).

\-\-batch-size `<INT>`
  Number of voxels processed per batch (default: 200).

\-\-TR `<FLOAT>`
  Repetition time (TR) in seconds. If not specified, the TR will be inferred from the NIfTI header (default: None).

\-\-processed `<BOOL>`
  If `True`, the input data is assumed to be preprocessed (e.g., motion-corrected, normalized).
  If `False`, the pipeline will apply basic preprocessing steps (default: `False`).

\-\-calc-penalty-accurately `<optional>`
  If specified, the penalty matrix will be calculated with higher accuracy. (default: not set).

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

