Run fPCA
============

This section describes how to run the `fpca-main` command-line tool to perform functional PCA on each example of fMRI data.

Main command-line tool
----------------------

Synopsis
~~~~~~~~

.. code-block:: bash

   fpca-main --mode <STR> \
             --nii-file <PATH_TO_4D_NIFTI> \
             --mask-file <PATH_TO_3D_MASK_NIFTI> \
             --output-folder <OUTPUT_DIR> \
             [--degree <INT>] \
             [--n-basis <INT>] \
             [--threshold <FLOAT>] \
             [--num-pca-comp <INT>] \
             [--batch-size <INT>] \
             [--TR <FLOAT>] \
             [--lambda-min <FLOAT>] \
             [--lambda-max <FLOAT>] \
             [--derivatives-num-p <INT>] \
             [--derivatives-num-u <INT>] \
             [--no-penalty <BOOL>] \
             [--calc-penalty-bspline-accurately <BOOL>] \
             [--calc-penalty-skfda <BOOL>] \
             [--n-skip-vols-start <int>] \
             [--n-skip-vols-end <int>] \
             [--processed <BOOL>] \
             [--smooth-size <INT>] \
             [--highpass <FLOAT>] \
             [--lowpass <FLOAT>] \
             [--use-nilearn-filter <BOOL>] \
             [--n-compcor-nilearn-filter <INT>] \
             [--smoothing-fwhm-nilearn-filter <INT>] \
             [--low-mem <BOOL>] \
             [--n-jobs <INT>]


Arguments
~~~~~~~~~

\-\-mode `<STR>`
  The default is `pca-singles`. See the following documentation for additional options.

\-\-nii-file `<PATH>`
  Path to the 4D fMRI NIfTI file (required).

\-\-mask-file `<PATH>`
  Path to the 3D binary mask NIfTI file (required).

\-\-output-folder `<DIR>`
  Directory where all output maps and plots will be saved (required).

\-\-degree `<INT>`
  Degree of the B-spline basis (default: 3).

\-\-n-basis `<INT>`
  Number of B-spline basis functions. Use 0 to determine it as number of timepoints or use several values for finding the best automatically based on the interpolation threshold (default: 100).

\-\-threshold `<FLOAT>`
  Interpolation error threshold for basis selection (default: 1e-6).

\-\-num-pca-comp `<INT>`
  Number of principal components to extract (default: 7).

\-\-batch-size `<INT>`
  Number of voxels processed per batch (default: 200).

\-\-TR `<FLOAT>`
  Repetition time (TR) in seconds. If not specified, the TR will be inferred from the NIfTI header (default: 0.75).

\-\-lambda-min `<FLOAT>`
  Minimum value of lambda in log10 scale (i.e., 10^-6) (default: -6).

\-\-lambda-max `<FLOAT>`
  Maximum value of lambda in log10 scale (i.e., 10^12) (default: 12).

\-\-derivatives-num-p `<INT>`
  Number of derivatives in calculation of penalty matrix P (default: 2)

\-\-derivatives-num-u `<INT>`
  Number of derivatives in calculation of penalty matrix U (default: 0)

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

\-\-processed `<optional>`
  If specified, the input data is assumed to be post-processed (e.g., smoothing, filtering), and no additional post-processing will be applied. If not specified the pipeline will apply basic post-processing steps (default: not set).

\-\-smooth-size `<INT>`
  Box size of smoothing kernel. Relevant only if --processed is not set (default: 5).

\-\-highpass `<FLOAT>`
  High-pass filter cutoff frequency in Hz. Filters out slow drifts below this frequency (default: 0.01).

\-\-lowpass `<FLOAT>`
  Low-pass filter cutoff frequency in Hz. Filters out high-frequency noise above this frequency (default: 0.08).

\-\-use-nilearn-filter `<BOOL>`
  If set, nilearn's NiftiMasker using confound files will be used for filtering instead of the default method (default: not set).

\-\-n-compcor-nilearn-filter `<INT>`
  Number of CompCor components to regress out during nilearn filtering (default: 5).

\-\-smoothing-fwhm-nilearn-filter `<INT>`
  Full-width at half-maximum (FWHM) of the Gaussian smoothing kernel in mm applied during nilearn filtering (default: 6).

\-\-low-mem `<BOOL>`
  If set, only NPZ/TXT/NIfTI files are written (no PNG plots). (default: not set).

\-\-n-jobs `<INT>`
  Number of parallel processes to run. Use -1 for all cores (default: 1).


Input file naming
~~~~~~~~~~~~~~~~~

The pipeline expects preprocessed BOLD files and matching masks:

.. code-block:: text

   <sample1>-preproc_bold.nii.gz
   <sample1>-brain_mask.nii.gz

The subject's outputs are written to a sub-folder named after ``<base>`` (the part before ``-preproc_bold``).

Output directory layout
~~~~~~~~~~~~~~~~~~~~~~~

After a full training run, the output folder typically looks like:

.. code-block:: text

   output-folder/
   ├── global_F_U_matrices.npz
   ├── sample1/
   │   ├── eigvecs_eigval_F.npz
   │   ├── original_averaged_signal_intensity.png
   │   ├── eigenfunction_0_importance_map_group.nii.gz
   │   ├── eigenfunction_1_importance_map_group.nii.gz
   │   ├── ...
   │   ├── eigenfunction_0_best_voxel.txt
   │   ├── eigenfunction_1_best_voxel.txt
   │   ├── ...
   │   ├── temporal_profile_pc_0.txt
   │   ├── temporal_profile_pc_1.txt
   │   └── ...



Per-subject outputs
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Description
   * - ``eigvecs_eigval_F.npz``
     - Compressed archive with ``C`` (coefficient matrix), ``eigvecs_sorted``, ``eigvals_sorted``, ``F``, ``times``. Required input for group modes.

       For loading the arrays back, use:

         .. code-block:: text

            data = np.load("eigvecs_eigval_F.npz")
            eigvecs_sorted = data['eigvecs_sorted']
            eigvals_sorted = data['eigvals_sorted']
            F = data['F']
            C = data['C']
            times = data['times']

   * - ``eigenfunction_<k>_importance_map.nii.gz``
     - 3D brain map showing voxel-wise importance/loading for principal component ``k``. High values indicate strong contribution to that component's spatial pattern.
   * - ``temporal_profile_pc_<k>.txt``
     - Time series showing temporal dynamics of principal component ``k`` across the fMRI scan.
   * - ``temporal_profile_pc_<k>.png``
     - Plot of the temporal profile (skipped with ``--low-mem``).
   * - ``eigenfunction_<k>_importance_map.png``
     - Middle-slice plot of the importance map (skipped with ``--low-mem``).
   * - ``eigenfunction_<k>_best_voxel.png`` / ``.txt``
     - Best-fitting voxel signal for PC ``k`` (skipped with ``--low-mem`` for PNG).
   * - ``original_averaged_signal_intensity.png`` / ``.txt``
     - Mean masked signal intensity over time.

Examples
~~~~~~~~

Run a two-component analysis on a toy dataset:

.. code-block:: bash

   fpca-main \
     --nii-file tests/test_input/toy50-53_drc144images.nii \
     --mask-file tests/test_input/toy50-53_mask.nii \
     --output-folder output/toy_run \
     --num-pca-comp 2 \
     --n-basis 300

Run with all defaults (except output folder):

.. code-block:: bash

   fpca-main \
     --nii-file data/sub-01_task-rest_bold.nii.gz \
     --mask-file data/sub-01_mask.nii.gz \
     --output-folder results/sub-01


Optional: separate preprocessing
--------------------------------

By default, ``fpca-main`` applies temporal filtering and spatial smoothing during ``pca-singles``. To preprocess once and reuse the filtered data across multiple parameter sweeps, use ``preprocess-nii-file`` and then pass ``--processed`` to ``fpca-main``.

**Preprocess a single file**

.. code-block:: bash

   preprocess-nii-file \
       --nii-file /path/to/subject-preproc_bold.nii.gz \
       --mask-file /path/to/subject-brain_mask.nii.gz \
       --output-folder /path/to/preprocessed/ \
       --TR 0.75 \
       --smooth-size 5 \
       --highpass 0.01 \
       --lowpass 0.08

This writes ``<original_name>_filtered.nii.gz`` into the output folder.

**Then run fpca-main on the filtered file**

.. code-block:: bash

   fpca-main \
       --mode pca-singles \
       --nii-files /path/to/preprocessed/subject-preproc_bold_filtered.nii.gz \
       --mask-files /path/to/subject-brain_mask.nii.gz \
       --output-folder /path/to/analysis/outputs_mov1 \
       --processed \
       ... [other parameters as above]

.. note::

   When using ``--processed``, the input BOLD file must already be filtered/smoothed. The mask file path is still required but no additional smoothing is applied.




The argument *threshold*:
-------------------------

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

  - **best-voxel fit plots** (`eigenfunction_<k>_best_voxel.png`)

