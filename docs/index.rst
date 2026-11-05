Neuro-fPCA-fMRI
================================================================================

|pypi-version| |documentation-status|

.. |pypi-version| image:: https://img.shields.io/pypi/v/neuro-fpca-fmri.svg
   :target: https://pypi.org/project/Neuro-fPCA-fMRI/
   :alt: PyPI version

.. |documentation-status| image:: https://readthedocs.org/projects/fmri/badge/?version=latest
   :target: https://fmri.readthedocs.io/en/latest/?version=latest
   :alt: Documentation Status


* Github Repository: https://github.com/yanivassf-lab/fMRI
* Documentation: https://fmri.readthedocs.io
* PyPI Package: https://pypi.org/project/Neuro-fPCA-fMRI

| Developed by **Refael Kohen** · `refael.kohen@gmail.com <mailto:refael.kohen@gmail.com>`_
| Yaniv Assaf Lab, Tel Aviv University

This guide explains how to prepare fMRI signals, run functional principal
component analysis (fPCA), compare participant groups, and inspect regional
feature importance.

Start with :doc:`readme`, then follow :doc:`installation` and
:doc:`group_pipeline`. Use :doc:`running` when you want to run the fPCA steps
separately.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   readme
   math
   installation
   running
   group_pipeline
   roi_explorer_guide
   implementation

The mathematical method is based on Roberto Viviani, Georg Grön, and
Manfred Spitzer, `Functional Principal Component Analysis of fMRI Data <https://onlinelibrary.wiley.com/doi/epdf/10.1002/hbm.20074>`_.,
Human Brain Mapping 24, 109–129 (2005).




