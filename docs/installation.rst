.. highlight:: shell

============
Installation
============

Stable release (pip)
--------------------

To install fPCA from PyPI with pip:

.. code-block:: console

    $ pip install Neuro-fPCA-fMRI

This installs the latest stable release and the CLI entry points (``fpca-main``, ``preprocess-nii-file``, ``nifti-viewer``, ``compare-pcs``).

Windows (recommended: conda/mamba)
----------------------------------

On Windows we recommend using a conda environment (via Miniconda or Mambaforge) for smooth installation of scientific dependencies:

1) Install Miniconda or Mambaforge

   - Miniconda: https://docs.conda.io/en/latest/miniconda.html
   - Mambaforge: https://github.com/conda-forge/miniforge

2) Create and activate an environment (Python 3.11 suggested)

.. code-block:: console

    # Install mamba in base (optional but faster)
    conda install -c conda-forge mamba

    # Create environment
    mamba create -n fpca-env -c conda-forge python=3.11

    # Activate environment
    mamba activate fpca-env

3) Install core scientific packages from conda-forge (ensures MKL/OpenBLAS are configured correctly)

.. code-block:: console

    mamba install -c conda-forge numpy scipy mkl mkl-service pandas matplotlib seaborn scikit-learn scikit-learn-extra imbalanced-learn plotly joblib nilearn nibabel scikit-fda dtaidistance pytorch shap

4) Install the Neuro-fPCA-fMRI package (PyPI) into the same environment

.. code-block:: console

    pip install Neuro-fPCA-fMRI

From sources
------------

The sources for Neuro-fPCA-fMRI can be downloaded from the `Github repo`_.

Clone the repository:

.. code-block:: console

    $ git clone https://github.com/yanivassf-lab/fmri
    $ cd fmri

Install in editable mode (development):

.. code-block:: console

    $ conda install -c conda-forge mamba
    $ mamba create -n fpca-env -c conda-forge python=3.11
    $ mamba activate fpca-env
    $ mamba install -c conda-forge numpy scipy mkl mkl-service pandas matplotlib seaborn scikit-learn scikit-learn-extra imbalanced-learn plotly joblib nilearn nibabel scikit-fda dtaidistance pytorch shap
    $ pip install -e .

Verify installation
-------------------

.. code-block:: console

    fpca-main --help
    preprocess-nii-file --help
    fmri-fpca-pipeline --help

.. _Github repo: https://github.com/yanivassf-lab/fmri
