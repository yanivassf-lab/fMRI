===============
Neuro-fPCA-fMRI
===============

.. image:: https://img.shields.io/pypi/v/neuro-fpca-fmri.svg
   :target: https://pypi.org/project/Neuro-fPCA-fMRI/
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/fmri/badge/?version=latest
   :target: https://fmri.readthedocs.io/en/latest/
   :alt: Documentation status


Project links
-------------

* Documentation: https://fmri.readthedocs.io
* GitHub repository: https://github.com/yanivassf-lab/fMRI
* PyPI package: https://pypi.org/project/Neuro-fPCA-fMRI/

Overview
--------

Neuro-fPCA-fMRI uses functional principal component analysis (fPCA)
to describe patterns of change in fMRI recordings.

The method fits a smooth curve to the signal at each voxel. It then
finds temporal patterns that explain variation across these curves.
Each temporal pattern has a signed score at each voxel, which can
be displayed as a brain map.

The package also provides a group analysis pipeline. It builds
features for each participant, trains classifiers to distinguish
participant groups, and evaluates the saved classifiers on separate
test participants. An interactive dashboard displays regional
feature importance from the fitted classifiers.

How fPCA works
--------------

1. **Load and prepare the data**

   Load a 4D fMRI recording and its brain mask. Extract the voxel
   signals and apply the selected preprocessing settings.

2. **Represent each signal with smooth functions**

   Represent each voxel signal as a weighted sum of basis functions.
   The package supports Fourier and B-spline bases. The current
   workflow uses a Fourier basis.

3. **Control the amount of smoothing**

   A roughness penalty limits rapid changes in the fitted curves.
   Generalized cross-validation (GCV) selects the smoothing strength
   separately for each voxel.

4. **Find the main temporal patterns**

   After centering the fitted curves across voxels, fPCA finds
   temporal components that explain their variation. The calculation
   accounts for the inner products between the basis functions.

5. **Save temporal profiles and spatial scores**

   Each component has a temporal profile and a score at each voxel.
   The temporal profile describes its shape over time. The voxel
   scores describe its spatial distribution.

Group analysis
--------------

The group workflow uses recordings from two musical movements.
Training and test participants are kept separate.

For analyses that use a shared component space, the pipeline fits
a group fPCA basis using the training participants and projects
the test participants onto that basis.

The current workflow includes three feature sets:

* **Set 0:** regional signals joined into one feature vector.
* **Set 1:** regional signal summaries, including standard deviation,
  mean absolute value, and means within time windows.
* **Set 4:** regression coefficients relating regional signals to
  the music annotation.

For fPCA Sets 0 and 1, each selected group component is analysed
separately. Set 4 instead uses a reconstruction from each participant's
own retained fPCA components before fitting the regression.

The pipeline also supports a comparison using preprocessed BOLD
signals directly. The scripts call this the ``raw`` branch.
Here, ``raw`` means that feature construction does not use fPCA;
it does not mean unprocessed scanner data.

Example outputs
---------------

The following images come from earlier example runs. They illustrate
the output formats, not the results or expected performance of a
new analysis.

PNG plots are skipped when ``--low-mem`` is enabled, as in the
current fPCA wrapper. See the documentation for saved numerical
outputs and plotting options.

Spatial component scores
~~~~~~~~~~~~~~~~~~~~~~~~

The spatial map shows the signed score of one component at each
voxel. The older image calls this value "Importance". It is an
fPCA score, not classifier feature importance or a statistical
test of a brain region.

.. figure:: https://fmri.readthedocs.io/en/latest/_images/eigenfunction_2_importance_map.png
   :align: center
   :width: 520px
   :alt: Brain slice showing signed voxel scores for fPCA component 2.

   Example spatial score map for component 2. Component indices
   start at 0.

Temporal profile
~~~~~~~~~~~~~~~~

The temporal profile shows the shape of a component over the
recording. The example below shows component 0, while the spatial
map above shows component 2.

.. figure:: https://fmri.readthedocs.io/en/latest/_images/eigenfunction_0_signal_intensity.png
   :align: center
   :width: 800px
   :alt: Temporal profile of fPCA component 0.

   Example temporal profile. This older image labels time in scans;
   the current plotting code uses seconds.

Fitted signal at one voxel
~~~~~~~~~~~~~~~~~~~~~~~~~~

The program also displays the voxel with the largest signed score
for a component. Red points show the samples used for fitting.
The blue line shows the fitted smooth curve.

The name "best voxel" refers to the component score used to select
the voxel. It does not mean that this voxel has the smallest
fitting error.

.. figure:: https://fmri.readthedocs.io/en/latest/_images/eigenfunction_0_best_voxel.png
   :align: center
   :width: 800px
   :alt: Voxel samples in red and the fitted smooth curve in blue.

   Example voxel selected using its score for component 0.
   The blue curve is the smooth fit, not a reconstruction
   from component 0 alone.

Installation and running
------------------------

Start with the installation guide, which describes the supported
setup using a mamba environment. Then follow the run instructions
for your analysis.

* `Installation guide <https://fmri.readthedocs.io/en/latest/installation.html>`_
* `fPCA commands and parameters <https://fmri.readthedocs.io/en/latest/running.html>`_
* `Group pipeline guide <https://fmri.readthedocs.io/en/latest/group_pipeline.html>`_
* `ROI Explorer dashboard <https://fmri.readthedocs.io/en/latest/roi_explorer_guide.html>`_

Reading the results
-------------------

fPCA voxel scores, classification performance, and regional feature
importance describe different things.

* **fPCA scores** describe how temporal components contribute
  to voxel signals.
* **Classification scores** describe prediction of participant labels.
* **Regional feature importance** describes how a fitted classifier
  uses its inputs.

A high prediction score or a highly ranked region does not by
itself establish a biological mechanism. Follow the documentation
when interpreting validation scores, statistical tests, and atlas
labels.

Mathematical background
-----------------------

The fPCA method follows the work of:

Roberto Viviani, Georg Grön and Manfred Spitzer, `Functional Principal Component Analysis of fMRI Data, Human Brain Mapping 24, 109–129 (2005) <https://onlinelibrary.wiley.com/doi/epdf/10.1002/hbm.20074>`_.

The documentation separates the mathematical explanation from
the implementation used in the current workflow:

* `Mathematical background <https://fmri.readthedocs.io/en/latest/math.html>`_
* `Implementation details <https://fmri.readthedocs.io/en/latest/implementation.html>`_

Credits
-------

The initial package structure was created using Cookiecutter_
and the `audreyr/cookiecutter-pypackage`_ template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _audreyr/cookiecutter-pypackage: https://github.com/audreyr/cookiecutter-pypackage

License
-------

Free software distributed under the GNU General Public License,
version 3 (GPLv3). See the LICENSE file in the repository.

Author
------

Developed by **Refael Kohen**.

**Contact:** `refael.kohen@gmail.com <mailto:refael.kohen@gmail.com>`_

Yaniv Assaf Lab, Tel Aviv University.
