Implementation of the current runs
================================================================================

This page explains how the current code carries out the method for the
recent commands: Sets 0, 1, and 4, with either fPCA or preprocessed BOLD,
and with either basic or Nilearn preprocessing. The fPCA wrapper selects
``--calc-penalty-skfda``. The main equations of the method are introduced
in :doc:`math`.

Other parameter combinations and feature sets are available. A description
of their implementation is left for a future extension of this guide.

Settings selected by the commands
--------------------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Part of the calculation
     - Current setting
   * - Sampling interval
     - TR = 0.75 seconds; no volumes removed at the start or end.
   * - Spline basis
     - 100 cubic B-splines, built with scikit-fda.
   * - Smoothing penalty
     - Integrated squared second derivative.
   * - Functional inner product
     - Integrated product of the basis functions, with derivative order 0.
   * - Smoothing candidates
     - 100 logarithmically spaced values from :math:`10^{-6}` to :math:`10^{12}`.
   * - Voxel batch size
     - 200 for the smoothing search.
   * - Retained fPCA components
     - Seven per participant or fitted group, separately for each movement.
   * - Sets 0 and 1 with fPCA
     - Separate classifier analyses for group component indices 0 and 1.
   * - Set 1 windows
     - Six windows per movement.
   * - Set 4 with fPCA
     - Sum the participant's own seven component contributions before regression.
   * - Classifier families
     - Logistic regression (LR), support vector machine (SVM), neural
       network (NN), decision tree (DTree), and random forest (RandForest).
   * - Training permutation mode
     - Five placeholder permutations; no valid training significance claim.

These are the settings selected by the current scripts. A saved cache can
bypass part of the calculation, as described below. To reproduce a run,
keep its source revision, environment, metadata, input paths, and logs.

Preprocessing
--------------------------------------------------------------------------------

Both branches start from the supplied BOLD images and brain masks. The
``raw`` comparison still passes through the data loader.

Without ``--filter``, the loader applies a spatial uniform box filter
with width 5 voxels and a third-order Butterworth band-pass filter from
0.01 to 0.08 Hz. The temporal filter is applied forward and backward with
``filtfilt``. The name ``output_no_filter`` therefore does not mean that
the data were left unfiltered.

With ``--filter``, the wrapper selects Nilearn processing with basic motion
confounds, high-pass cosine confounds, and two anatomical combined CompCor
components. CompCor components summarize nuisance variation in tissue
signals. A NiftiMasker applies 4 mm Gaussian smoothing, centers each voxel's
time series, and divides it by its sample standard deviation. Detrending is off.
This branch does not add the basic branch's 0.01–0.08 Hz band-pass filter.

Nilearn's returned ``sample_mask`` is used for diagnostics only. The code
keeps all time points; it does not remove the flagged initial volumes or
add spike regressors for them. Both branches then use the requested volume
cropping, which is zero at both ends in the current runs.

Efficient selection of the smoothing level
--------------------------------------------------------------------------------

Let :math:`F` be the sampled spline basis and :math:`P` its second-derivative
penalty matrix. The time grid is
:math:`t_j=j\,\mathrm{TR}`, for :math:`j=0,\ldots,n-1` in these runs.
The spline interval runs from the first to the last recorded time.

The program avoids forming and inverting a new matrix for every voxel and
every smoothing level. It first defines :math:`A=F^\top F` and solves

.. math::

   PV=AVD, \qquad V^\top AV=I,
   \qquad D=\operatorname{diag}(d_1,\ldots,d_K).

This is a generalized eigenvalue problem: each column of :math:`V` solves
:math:`Pv_j=d_jAv_j`. SciPy's ``eigh(P, A)`` solves it. It requires
:math:`A` to be positive definite, so the sampled basis must have full
column rank: no column can be a linear combination of the others.
The penalty matrix can be singular; the curvature penalty
is zero on constant and linear functions.

Define :math:`Q=FV`. Its columns satisfy :math:`Q^\top Q=I`.
For a batch of :math:`B` voxels, arrange their data in a matrix :math:`Y`
of size :math:`n\times B` and calculate

.. math::

   Z=Q^\top Y, \qquad
   r_{0,b}=\lVert Y_{:,b}-QZ_{:,b}\rVert^2.

The value :math:`r_{0,b}` is the part of a voxel's squared error that no
curve in this basis can remove. It is calculated once per batch.

For each candidate :math:`\lambda`, the diagonal shrinkage factors are

.. math::

   s_k(\lambda)=\frac{1}{1+\lambda d_k}.

The residual sum of squares and effective degrees of freedom give

.. math::

   \operatorname{RSS}_b(\lambda)
   &=r_{0,b}+\sum_{k=1}^{K}
       [1-s_k(\lambda)]^2 Z_{kb}^2,\\
   \operatorname{GCV}_b(\lambda)
   &=\frac{n\,\operatorname{RSS}_b(\lambda)}
           {[n-\sum_{k=1}^{K}s_k(\lambda)]^2}.

The code chooses the best candidate separately for every voxel in the
batch. Its coefficients are then

.. math::

   c_b=V\operatorname{diag}(s_k(\lambda_b))Z_{:,b}.

This is the same penalized fit and GCV expression as in :doc:`math`, under
the stated matrix conditions. Batching reduces temporary storage. It does
not force the voxels in a batch to share one smoothing level.

The implementation symmetrizes :math:`P` and :math:`A` before solving.
For the current second-derivative penalty, it checks the two expected
zero eigenvalues and sets those two values to exactly zero. It warns if
they are not close to zero, or if other penalty eigenvalues are not positive.
These warnings need investigation; forcing the expected zeros does not
validate an otherwise incorrect matrix.

Candidates whose remaining degrees of freedom are too small are skipped.
If any voxel has no finite valid GCV candidate, the calculation stops with
an error. Other numerical warnings may allow the run to continue. A warning
is therefore neither always fatal nor evidence that the result was repaired.

A large selected :math:`\lambda` can suppress curvature while retaining
constant or linear terms. Its numerical size also depends on the time
units and basis. It should not be interpreted alone as a measure of signal
quality.

The fPCA eigenvalue problem
--------------------------------------------------------------------------------

After smoothing, the coefficient matrix is centered across voxels.
For one participant,

.. math::

   S=\frac{1}{N}\widetilde C^\top\widetilde C,
   \qquad U_{kl}=\int f_k(t)f_l(t)\,dt.

The current code symmetrizes :math:`S` and :math:`U` separately and calls
``scipy.linalg.eigh(S, U, type=2)``. This solves

.. math::

   SU\Phi=\Phi\Gamma, \qquad \Phi^\top U\Phi=I.

The product :math:`SU` is not generally symmetric. The solver works with
the two symmetric input matrices and the positive definite Gram matrix.
Replacing :math:`SU` by :math:`(SU+US)/2` would change the problem.
Likewise, ``eigh(S, U)`` without ``type=2`` solves
:math:`S\Phi=U\Phi\Gamma`, which is a different equation.

This fPCA problem is separate from the earlier smoothing problem
:math:`PV=AVD`. Both use a generalized solver, but with different matrices
and equations.

The program sorts fPCA eigenvalues from largest to smallest and retains
seven components. It warns about negative eigenvalues, clips negative
values to zero, and warns if no positive variance remains. Large negative
values are not an expected feature of a valid covariance calculation.

Scores and sampled temporal profiles are

.. math::

   \text{scores}=\widetilde C U\Phi,
   \qquad \text{profiles}=F\Phi.

Scores keep their signs. The saved mean function is
:math:`F\mu_c`. The numerical archive also stores the mask and affine
transform so later stages can match a score map to the participant's brain.

Fit a shared basis from training participants
--------------------------------------------------------------------------------

Group fPCA is fitted separately for the two movements. Let :math:`C_s`
contain the coefficients for participant :math:`s`, with :math:`N_s`
included voxels. The group mean is

.. math::

   \mu_g=\frac{\sum_s\sum_{i=1}^{N_s}c_{si}}{\sum_s N_s}.

The code makes passes over the saved coefficient arrays. It first computes
this mean, then accumulates

.. math::

   S_g=\frac{\sum_s(C_s-\mathbf{1}\mu_g^\top)^\top
                       (C_s-\mathbf{1}\mu_g^\top)}
                    {\sum_s N_s}.

Thus, each included voxel has equal weight. Participants with more included
voxels contribute more to the group mean and covariance. This is not equal
weighting of participant covariance matrices.

The group covariance uses the same fPCA eigenvalue problem. The training
mean and eigenvectors are saved. Every participant is projected using

.. math::

   \text{scores}_s=(C_s-\mathbf{1}\mu_g^\top)U\Phi_g.

For a test participant, :math:`\mu_g` and :math:`\Phi_g` still come only
from training participants. The test data do not refit the group basis.
The saved coefficient archives must agree in basis values, Gram matrix,
and acquisition times; incompatible archives are rejected.

Regional signals and Sets 0 and 1
--------------------------------------------------------------------------------

The Schaefer atlas is resampled to each image grid with nearest-neighbour
interpolation. A region's score is averaged only over its intersection
with the participant's brain mask. A region with no included voxels gets
a zero value. Coverage can therefore affect the regional representation.

For group component :math:`j`, participant :math:`s`, and region
:math:`r`, the reconstructed contribution is

.. math::

   x_{srj}(t)=\overline a_{srj}\,\psi_{gj}(t).

The temporal profile is shared by participants in that movement; the
regional score varies between participants. The group mean function is
not added to these component contributions.

Set 0 joins the two movement matrices and flattens them into one feature
vector. The order is all regions at the first time point, then all regions
at the next time point, and so on. The basic unit used by classification
is a participant, not a voxel or a time point.

Set 1 computes a standard deviation, a mean absolute value, and six signed
window means for each region in each movement. There are therefore
:math:`2(2+6)R=16R` features for :math:`R` regional slots.
The window means describe consecutive parts of each movement, not musical
events identified from the workbook.

In the raw comparison, Sets 0 and 1 use regional means of the preprocessed
BOLD time series instead of a group-component reconstruction. The same
feature layouts are then applied.

The code currently derives the number of regional slots from the atlas
label list. If this includes a leading Background label, the mapping is
shifted and includes an extra empty slot. Do not assume that the feature
matrix has exactly 100 correctly labelled slots. See
:doc:`roi_explorer_guide` before interpreting regional names.

Set 4: relate individual signals to the music
--------------------------------------------------------------------------------

For fPCA Set 4, the program reads each participant's individual component
maps and temporal profiles and reconstructs

.. math::

   x_{sr}(t)=\sum_{j=1}^{7}\overline a_{srj}\psi_{sj}(t).

It does not use the shared group profiles here and does not add the
participant's saved mean function. The signal describes deviation from
that participant's spatial mean curve. Each participant's component
sign cancels in the product of score and profile.

Using individual profiles allows the reconstructed time course to differ
between participants. A group-profile signal could still be regressed on
the music, but its time course would be constrained by those shared profiles.

The individual reconstruction loader requires matching acquisition times
within each movement and the requested number of component files. It does
not shorten all recordings to the shortest one. This loader currently
reconstructs the signals again instead of reusing its old cache.

For raw Set 4, freshly built data are spatially centered at each time point:
the mean over included brain voxels is subtracted before regional averaging.
However, the raw cache is shared with Sets 0 and 1 unless a separate cache
parent is used. Loading their cache bypasses that centering. Consequently,
the original combined raw command alone does not establish that a result
used centered signals. Check its cache history or use the separate-cache
run in :doc:`group_pipeline`.

Build the music regressors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The workbook records the scan TR at which each musical bar starts.
These are one-based scan positions. The code converts them to zero-based
indices, rounds them to the nearest scan, and holds each annotation level
until the next bar. The final bar uses the median spacing between bar
starts as an estimated duration. Values outside the recording are cut;
the raw annotation signal is zero after the final bar ends.

In the current configuration, instrument rows are combined into wind,
guitars, keyboards, strings, and drums. Woodwind and brass contribute to
wind. A group's level is the largest level among its member rows:
0 for absent, 1 for playing, and 2 for leading or solo. This is a group
level, not a count of instruments.

The workbook's genre rows are included as additional regressors.
A density regressor counts instruments that are playing. An unmapped
instrument family does not get an instrument-group regressor but still
contributes to this density count. Inspect the reported unmapped families.

Each time series is convolved with a sampled, SPM-style double-gamma
hemodynamic response function. This models the delayed BOLD response.
The kernel is sampled at the scan TR, spans 32 seconds, and is normalized
to sum to one. The convolved series is then centered and divided by its
standard deviation. A constant annotation column is kept as zeros.
Both movements must produce the same regressor columns in the same order.

The generated raw and convolved regressors are saved under ``0_regressors``.
Check those files against the recording length and music timing. These
inputs describe the stimulus annotation; they are not participant labels
or measured audio features.

Fit the regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before regression, the code centers each regional signal over time and
divides it by its temporal standard deviation, with a small constant in
the denominator. It also standardizes the regressor columns again. This
normalization is separate from the earlier subtraction of a spatial mean.

For each participant and movement, the design matrix contains an intercept
and these regressor columns. Ordinary least squares fits the standardized
regional signals together:

.. math::

   \widehat B_s=\operatorname*{arg\,min}_{B}
     \lVert Y_s-X_{\mathrm{music}}B\rVert_F^2.

The code uses ``numpy.linalg.lstsq``. It drops the intercept coefficients
and lists the remaining coefficients for all regions of the first
regressor, then all regions of the next regressor, and so on.
Movement 1 comes before movement 2. With :math:`L` regressors and
:math:`R` regional slots, Set 4 has :math:`2LR` features.

These coefficients describe association between standardized regional
signals and the specified, often correlated regressors. They do not retain
the original regional signal amplitude. They are not causal effects or voxelwise
significance tests. A zero or dependent design column can make individual
coefficients uninformative even though a least-squares solution exists.

Train and evaluate the classifiers
--------------------------------------------------------------------------------

The metadata's ``is_train_set`` field restricts classifier fitting to
training participants and held-out testing to test participants.
``instrument_group`` supplies the labels. ``NM`` denotes non-musicians;
``mus`` combines the non-NM instrument groups for the NM-versus-mus comparison.

For each pair and model family, the code chooses the outer validation
procedure from the pair's training counts:

* With at least 20 participants and at least 12 in the smaller class,
  it uses five stratified outer folds and a balanced bagging classifier
  containing 30 fitted estimators. Each estimator uses a resampled training
  set with balanced classes, and their predictions are combined.
* Otherwise, it uses leave-one-out validation without bagging. Each
  participant is predicted by a model fitted without that participant.

Model settings are selected inside each outer training split. The inner
search uses three stratified folds when the pair's smaller training class
has at least three participants; otherwise it requests leave-one-out.
Very small classes can still make an inner fit fail. Check the logs and
finite scores; the fold choice alone does not ensure sufficient data.

Scaling and feature selection are inside the estimator pipeline and are
refitted during classifier validation. The score used for tuning is
balanced accuracy, the average of the two class recalls. After validation,
the selected settings are fitted again on the full training data and the
model is saved for held-out testing.

The shared group fPCA basis for Sets 0 and 1 was fitted before classifier
cross-validation, using all training participants. It is not refitted
inside each outer fold. Thus, the classifier tuning is nested, but the
entire fPCA-to-classifier process is not nested. The separate test set
provides the evaluation on participants excluded from group-basis fitting.

The summary also reports a majority vote of model predictions when they
can be matched across the same participants. This uses class predictions,
not an average of predicted probabilities. The models are not independent
statistical replications.

Statistical summaries for these runs
--------------------------------------------------------------------------------

Without ``--run-permutations``, the code produces five placeholder
training permutations using already selected model settings. These do
not repeat the full nested selection procedure. The corrected summary
does not turn them into a valid training p-value and does not supply an
analytic fallback.

On held-out data, the saved model produces one fixed set of predictions.
For a pair with :math:`n` participants, let :math:`n_1` be the number of
true class-1 labels, :math:`m_1` the number of predicted class-1 labels,
and :math:`h` the number of true positives. Under random assignment of
the labels with their class counts fixed,

.. math::

   H\sim\operatorname{Hypergeometric}(n,n_1,m_1),
   \qquad p=\Pr(H\geq h).

For fixed class and prediction counts, balanced accuracy increases with
:math:`h`, so this is a one-sided exact test for the fixed predictions.
It assumes participant labels are exchangeable under the null hypothesis;
it does not account for unmodelled site or other participant dependencies.

The summary selects the model family using training cross-validation,
takes that family's held-out p-value, and applies a Bonferroni correction
over the pairs in the summary. With :math:`M` pairs, the corrected value is
:math:`\min(1,Mp)`. It is written as ``test_sel_p_fwer``.
The correction does not include additional searches across analysis
configurations. Test-set model maxima and model averages remain descriptive.

The ``align-signals`` stage separately performs RSA on training participants
only. Its label permutations and multiple-test correction concern distances
between feature representations, not classifier accuracy or regional
importance. Inspect these outputs as a separate analysis.

Source locations
--------------------------------------------------------------------------------

The following files contain the calculations described above:

* ``src/fPCA/preprocess.py``: data loading and preprocessing.
* ``src/fPCA/fpca.py`` and ``src/fPCA/gcv_solver.py``: spline fitting, GCV,
  individual and group fPCA, and test projection.
* ``src/PCAGroupDiscrimination/build_signals_from_files.py``: regional
  signal extraction and raw/group-signal caches.
* ``src/PCAGroupDiscrimination/fpca_subject_signals.py``: individual
  component reconstruction for Set 4.
* ``src/PCAGroupDiscrimination/annotation_regressors.py`` and
  ``process_signals.py``: music regressors and participant feature vectors.
* ``src/PCAGroupDiscrimination/ml_analyzer.py``, ``ml_tester.py``, and
  ``stat.py``: classifier fitting, held-out predictions, and summary tests.

SciPy documents the different generalized eigenvalue equations in
`scipy.linalg.eigh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html>`_.

