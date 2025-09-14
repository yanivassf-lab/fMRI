====
fMRI
====


.. image:: https://img.shields.io/pypi/v/fmri.svg
        :target: https://pypi.python.org/pypi/fmri

.. image:: https://img.shields.io/travis/yanivassf-lab/fmri.svg
        :target: https://travis-ci.com/yanivassf-lab/fmri

.. image:: https://readthedocs.org/projects/fmri/badge/?version=latest
        :target: https://fmri.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/yanivassf-lab/fmri/shield.svg
     :target: https://pyup.io/repos/github/yanivassf-lab/fmri/
     :alt: Updates




Overview
--------

This module implements a functional principal component analysis (fPCA) pipeline designed for fMRI data. The overall procedure involves the following steps:

1. Data Loading and Preprocessing:

   The 4D fMRI image and an associated 3D binary mask are loaded. The mask extracts voxel-specific time series, and the time axis is defined based on the number of timepoints.

2. B-Spline Basis Construction:

   A set of B-spline basis functions is generated to represent the smooth temporal dynamics. Each voxel’s time series is approximated as a linear combination of these basis functions.

3. Penalty Matrix Construction:

   A penalty matrix is computed to ensure smoothness of the estimated coefficients. This penalty is defined via an integral that involves the product of derivatives (typically the second derivative) of the basis functions.

4. Regularized Spline Regression:

   For every voxel, the time series is approximated by a linear expansion in the spline basis. The coefficients of this expansion are estimated by solving a regularized regression problem. A generalized cross-validation (GCV) criterion is used to select a voxel-specific regularization parameter (:math:`\lambda`).

5. Functional Principal Component Analysis:

   PCA is then performed on the coefficients matrix to identify the dominant patterns of temporal variation. Multiplying the eigenfunctions by the coefficients yields voxel-specific importance scores that are mapped back into brain images, while the product of the basis matrix with each eigenfunction generates an intensity plot showing the common (to all voxels) temporal dynamics.

Output
------

   Finally, for each eignfunction, voxel importance maps and intensity plot are generated and saved.



   - voxel importance maps:

     The computed score indicates the contribution or importance of that voxel for the corresponding principal component.
     Using the spatial information from the original brain mask, these voxel scores are then mapped back to a volume of the brain. This produces an importance map saved as a file with the brain’s dimensions, where each voxel’s value indicates its contribution to the fPCA component.

     .. figure:: _static/eigenfunction_2_importance_map.png
        :align: center
        :figwidth: 80%
        :alt: Signal and fitted function for best-scoring voxel
        :figclass: align-center

   - intensity plot:

     The PC temporal profile values are plotted as a graph (intensity plot) that illustrates the time-course of the fPCA component across the entire brain.

     .. figure:: _static/eigenfunction_0_signal_intensity.png
        :align: center
        :figwidth: 80%
        :alt: Signal and fitted function for best-scoring voxel
        :figclass: align-center

   - Signal Intensity for Best-Scoring Voxel

     This plot illustrates the signal intensity over time for the voxel with the highest score in a given eigenfunction :math:`i`. The red dots represent the original fMRI signal, while the blue curve shows the corresponding fitted function based on the selected eigenfunction. The voxel is identified by its index in the 3D brain volume, and the optimal regularization parameter :math:`\lambda`, along with the voxel’s score and spatial coordinates :math:`(x, y, z)`, are annotated in the figure. This visualization helps assess the quality of the functional fit at the most representative voxel for each eigenfunction.

     .. figure:: _static/eigenfunction_0_best_voxel.png
        :align: center
        :figwidth: 80%
        :alt: Signal and fitted function for best-scoring voxel
        :figclass: align-center



Mathematical background
-----------------------

This program implements the methodology from the paper:

Roberto Viviani, Georg Grön and Manfred Spitzer.
*Functional Principal Component Analysis of fMRI Data*.

See full mathematical background here: :doc:`Detailed Math Docs <math>`



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

License
-------

* Free software: GNU General Public License v3
* Documentation: https://fmri.readthedocs.io.

Author
------

The code was written by *Refael Kohen* <refael.kohen@gmail.com>, Yaniv Assaf Lab, Tel Aviv University.
