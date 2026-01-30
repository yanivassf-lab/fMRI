===============
Neuro-fPCA-fMRI
===============

.. image:: https://img.shields.io/pypi/v/neuro-fpca-fmri.svg
   :target: https://pypi.org/project/neuro-fpca-fmri/
.. image:: https://readthedocs.org/projects/fmri/badge/?version=latest
   :target: https://fmri.readthedocs.io/en/latest/?version=latest
   :alt: Documentation Status
.. image:: https://pyup.io/repos/github/yanivassf-lab/fmri/shield.svg
   :target: https://pyup.io/repos/github/yanivassf-lab/fmri/
   :alt: Updates

Full Documentation
------------------

* Github Repository: https://github.com/yanivassf-lab/fMRI
* Documentation: https://fmri.readthedocs.io
* PyPI Package: https://pypi.org/project/Neuro-fPCA-fMRI

Overview
--------

This module implements a functional principal component analysis (fPCA) pipeline designed for fMRI data. The overall procedure involves the following steps:

1. **Data Loading and Preprocessing**

   The 4D fMRI image and an associated 3D binary mask are loaded. The mask extracts voxel-specific time series, and the time axis is defined based on the number of timepoints.

2. **B-Spline Basis Construction**

   A set of B-spline basis functions is generated to represent smooth temporal dynamics. Each voxel’s time series is approximated as a linear combination of these basis functions.

3. **Penalty Matrix Construction**

   A penalty matrix is computed to ensure smoothness of the estimated coefficients. This penalty is defined via an integral involving products of derivatives (typically the second derivative) of the basis functions.

4. **Regularized Spline Regression**

   Each voxel time series is approximated by a spline expansion. The coefficients are estimated via a regularized regression problem, and a generalized cross-validation (GCV) criterion selects a voxel-specific regularization parameter (:math:`\lambda`).

5. **Functional Principal Component Analysis**

   PCA is performed on the coefficient matrix. Multiplying eigenfunctions by coefficients yields voxel-specific importance scores mapped back to brain images, while the basis applied to each eigenfunction gives the temporal dynamics curve shared across voxels.


Output
------

For each eigenfunction, the following figures are generated:

- **Voxel importance maps**

  .. figure:: _static/eigenfunction_2_importance_map.png
     :align: center
     :figwidth: 80%
     :alt: Voxel importance map

- **Intensity plot**

  .. figure:: _static/eigenfunction_0_signal_intensity.png
     :align: center
     :figwidth: 80%
     :alt: Temporal intensity plot

- **Signal intensity of best-scoring voxel**

  .. figure:: _static/eigenfunction_0_best_voxel.png
     :align: center
     :figwidth: 80%
     :alt: Best voxel fit


Mathematical background
-----------------------

This program implements the methodology from:

*Roberto Viviani, Georg Grön and Manfred Spitzer.*
*Functional Principal Component Analysis of fMRI Data.*

Full mathematical derivation is available here:
`Detailed Math Docs <https://fmri.readthedocs.io/en/latest/math.html>`_


Credits
-------

This package was created using Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _audreyr/cookiecutter-pypackage: https://github.com/audreyr/cookiecutter-pypackage


License
-------

* Free software: GNU GPLv3


Author
------

*Refael Kohen* <refael.kohen@gmail.com>,
Yaniv Assaf Lab, Tel Aviv University.
