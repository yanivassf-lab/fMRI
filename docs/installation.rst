.. highlight:: shell

============
Installation
============

Stable release (pip)
--------------------

To install fMRI from PyPI with pip:

.. code-block:: console

    $ pip install fmri

This installs the latest stable release and the CLI entry points (``fmri-main``, ``preprocess-nii-file``, ``nifti-viewer``, ``compare-pcs``).

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
    mamba create -n fmri-env -c conda-forge python=3.11

    # Activate environment
    mamba activate fmri-env

3) Install core scientific packages from conda-forge (ensures MKL/OpenBLAS are configured correctly)

.. code-block:: console

    mamba install -c conda-forge numpy scipy mkl mkl-service matplotlib nibabel scikit-fda dtaidistance

4) Install the fMRI package (PyPI) into the same environment

.. code-block:: console

    pip install fmri

From sources
------------

The sources for fMRI can be downloaded from the `Github repo`_.

Clone the repository:

.. code-block:: console

    $ git clone https://github.com/yanivassf-lab/fmri
    $ cd fmri

Install in editable mode (development):

.. code-block:: console

    # Option A: system Python
    $ pip install -e .

    # Option B: conda
    $ conda install -c conda-forge mamba
    $ mamba create -n fmri-dev -c conda-forge python=3.11
    $ mamba activate fmri-dev
    $ mamba install -c conda-forge numpy scipy mkl mkl-service matplotlib nibabel scikit-fda dtaidistance
    $ pip install -e .

Verify installation
-------------------

.. code-block:: console

    fmri-main --help
    preprocess-nii-file --help
    compare-pcs --help

.. _Github repo: https://github.com/yanivassf-lab/fmri
