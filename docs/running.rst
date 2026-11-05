Run fPCA
================================================================================

Use ``fpca-main`` for direct control of fPCA. Use ``run_fpca.sh`` to process
the two movement folders in the current workflow. The complete run order,
including classifiers, is in :doc:`group_pipeline`.

Activate the environment from :doc:`installation` before running commands
on this page.

Run one recording
--------------------------------------------------------------------------------

This example uses the same spline settings as the current batch runs.
Replace the input paths with your recording and its matching mask:

.. code-block:: bash

   fpca-main \
     --mode pca-singles \
     --nii-files /path/to/subject-preproc_bold.nii.gz \
     --mask-files /path/to/subject-brain_mask.nii.gz \
     --output-folder /path/to/new_fpca_output \
     --calc-penalty-skfda \
     --degree 3 \
     --n-basis 100 \
     --num-pca-comp 7 \
     --TR 0.75 \
     --low-mem

Use a new output folder. The command creates it. If a ``pca-singles`` output
folder already exists, the command requires ``--force`` to continue.
That flag allows existing outputs to be overwritten; it does not clean
out stale files from an earlier analysis.

The example uses the basic preprocessing branch. To use the Nilearn branch
with the settings used by the shell scripts, add:

.. code-block:: bash

   --use-nilearn-filter \
   --n-compcor-nilearn-filter 2 \
   --smoothing-fwhm-nilearn-filter 4

These three lines are options to append to the previous command, not a
separate command. This branch also needs matching fMRIPrep confound files.

Inputs and modes
--------------------------------------------------------------------------------

``--nii-files``
   One or more 4D NIfTI recordings. This argument is required. Its name is
   plural, even for one recording.

``--mask-files``
   Matching 3D brain masks, in the same order as the recordings. One mask
   may be supplied for all recordings only when it really applies to all
   of them. Otherwise, the two lists must have equal length.

``--output-folder``
   The parent folder for outputs. Supply it for every run. The program
   creates a separate child folder for each recording.

``--mode``
   One of the following modes. Use one mode at a time in the order below.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Purpose
   * - ``pca-singles``
     - Fit each recording and save its coefficients and individual fPCA
       results. Run this for both training and test participants.
   * - ``train-pca-group``
     - Read the saved training coefficients and fit the shared group basis.
       Pass training recordings and masks only.
   * - ``test-pca-project``
     - Read the saved test coefficients and project them using the training
       group mean and basis. Pass test recordings and masks only.

Group fitting and test projection still require the recording and mask
arguments. They use these paths to locate the matching saved outputs.
Use matching time grids and spline settings across participants within
each movement. Keep the two movements in separate fPCA output folders.

Parameters used in the current workflow
--------------------------------------------------------------------------------

The defaults below belong to ``fpca-main``. The wrapper changes some of
them, as explained in the next section.

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Option
     - Default
     - Meaning
   * - ``--degree``
     - ``3``
     - Polynomial degree of the spline pieces. Degree 3 gives cubic splines.
   * - ``--n-basis``
     - ``100``
     - Number of basis functions. Use one fixed value for the current workflow.
   * - ``--num-pca-comp``
     - ``7``
     - Number of retained components. It must be between 1 and the basis size.
   * - ``--batch-size``
     - ``200``
     - Number of voxels processed together when choosing their smoothing levels.
   * - ``--TR``
     - ``0.75``
     - Time between volumes, in seconds. The default is not automatic
       detection from the image header. Set the correct value for the data.
   * - ``--lambda-min`` / ``--lambda-max``
     - ``-6`` / ``12``
     - Base-10 exponents for 100 candidate smoothing levels, from
       :math:`10^{-6}` to :math:`10^{12}`.
   * - ``--derivatives-num-p``
     - ``2``
     - Derivative order used in the smoothing penalty.
   * - ``--derivatives-num-u``
     - ``0``
     - Order used for the functional inner product. Keep 0 for the
       mathematical method described here.
   * - ``--calc-penalty-skfda``
     - Off
     - Select the scikit-fda spline and Gram-matrix route used in these runs.
       Supply the flag without a value.
   * - ``--n-skip-vols-start`` / ``--n-skip-vols-end``
     - ``0`` / ``0``
     - Number of volumes removed from the start and end after preprocessing.
       The current runs remove none.
   * - ``--smooth-size``
     - ``5``
     - Spatial box width in voxels for basic preprocessing. This is not a
       Gaussian width in millimetres.
   * - ``--highpass`` / ``--lowpass``
     - ``0.01`` / ``0.08``
     - Band-pass limits in Hz for basic preprocessing. They do not define
       the Nilearn confound-removal branch.
   * - ``--processed``
     - Off
     - Skip the loader's extra preprocessing. The current batch scripts
       do not set this flag.
   * - ``--use-nilearn-filter``
     - Off
     - Use the Nilearn confound-removal and smoothing branch.
   * - ``--n-compcor-nilearn-filter``
     - ``5``
     - Requested number of CompCor confound components for that branch.
   * - ``--smoothing-fwhm-nilearn-filter``
     - ``6.0``
     - Gaussian smoothing full width at half maximum, in millimetres,
       for that branch.
   * - ``--n-jobs``
     - ``1``
     - Number of subject workers. ``-1`` requests all available CPU cores.
   * - ``--low-mem``
     - Off
     - Skip fPCA PNG plots while keeping numerical, text, and NIfTI outputs.
   * - ``--force``
     - Off
     - Allow use of an existing output folder.

The optional ``--threshold`` argument defaults to ``1e-6``. It is used
only when ``--n-basis`` contains several candidate values: the first value
whose mean absolute fitting error reaches the threshold is selected.
If none reaches it, the candidate with the smallest error is selected.
With a single value such as 100, this search does not run.
``--n-basis 0`` means one basis function per retained time point; it is
not a threshold search. Neither alternative is used in the current runs.

Run the two-movement wrapper
--------------------------------------------------------------------------------

The current filename is ``run_fpca.sh``. Its three required arguments are
the mode, the input folder, and the fPCA output parent folder:

.. code-block:: bash

   ./run_fpca.sh pca-singles \
     /path/to/raw_data_train \
     /path/to/output_pca \
     --n-jobs 4

There is no environment-directory argument. Activate ``fpca-env`` first.

The wrapper reads ``*.nii.gz`` files directly inside the input folder.
For a recording named ``<base>-preproc_bold.nii.gz``, it looks for
``<base>-brain_mask.nii.gz``. The base must contain ``-movement1`` or
``-movement2``. Files without a matching mask are not added to the run.
Check the reported file counts against your participant list. The current
wrapper stops if either movement has no matching files.

It writes movement outputs under ``outputs_mov1`` and ``outputs_mov2``.
It always adds ``--calc-penalty-skfda``, ``--low-mem``, and ``--force``.
It also limits numerical-library threads to one per worker. Its default
worker count is 20, which may be too high for a smaller machine.

The wrapper accepts only these extra options:

* ``--n-jobs``
* ``--use-nilearn-filter``
* ``--n-compcor-nilearn-filter``
* ``--smoothing-fwhm-nilearn-filter``

For the Nilearn branch, the wrapper supplies 2 CompCor components and
4 mm smoothing unless changed. These differ from the direct command's
defaults. It does not forward arbitrary ``fpca-main`` options. For example,
to change TR or the number of basis functions, use the direct command or
update the wrapper explicitly before the run.

Inspect the outputs
--------------------------------------------------------------------------------

Each recording folder contains ``eigvecs_eigval_F.npz``. It stores the
coefficient matrix, basis values, Gram matrix, eigenvectors, eigenvalues,
time grid, mask, affine transform, and mean function. It is needed by
later stages; keep it with the score maps.

Individual outputs include ``eigenfunction_0_importance_map.nii.gz`` and
``temporal_profile_pc_0.txt``, with corresponding files for other components.
These indices start at 0. The files called importance maps contain signed
fPCA scores, not classifier feature importance.

Group fitting also saves ``global_mean.npy``, ``global_eigvecs.npy``,
group temporal profiles, and participant group score maps. Test projection
writes maps with the suffix ``_group_test_proj.nii.gz``.

Review the logs after each stage. Numerical warnings about negative
eigenvalues, invalid smoothing scores, or absent variance require inspection.
A command finishing does not establish that every recording produced valid
results. After changing preprocessing, time grids, masks, or code, use a
new analysis output folder and rebuild the downstream features.

