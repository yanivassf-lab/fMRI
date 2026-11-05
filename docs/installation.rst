Installation
================================================================================

Use one mamba environment for the Python package, the shell scripts, and
the dashboard. The examples below use the environment name ``fpca-env``.
The full-run script activates this exact name.

The shell scripts require Bash. Use a Bash terminal on Linux or macOS.
On Windows, run the shell workflow inside WSL with its own Linux environment.
These instructions do not describe a separate Windows Python installation.

Install Miniforge
--------------------------------------------------------------------------------

Install `Miniforge3 <https://github.com/conda-forge/miniforge>`_ for your
operating system and processor. It provides both conda and mamba. Follow
its installer instructions, initialize the shell, and open a new terminal.

Check that mamba is available:

.. code-block:: bash

   mamba --version

Create the environment
--------------------------------------------------------------------------------

Create a Python 3.11 environment and install the scientific packages from
conda-forge:

.. code-block:: bash

   mamba create -n fpca-env -c conda-forge --strict-channel-priority \
     python=3.11 pip setuptools wheel git \
     numpy scipy pandas joblib matplotlib seaborn \
     scikit-learn imbalanced-learn pytorch umap-learn \
     nibabel nilearn scikit-fda dtaidistance \
     openpyxl plotly dash

   eval "$(mamba shell hook --shell bash)"
   mamba activate fpca-env

This is an environment setup, not a version lock for a published analysis.
The supplied code snapshot does not contain a complete, tested environment
lock. Keep the resolved package versions with each run, as shown below.
Do not treat a successful dependency installation as a full pipeline test.

Install the project into this environment
--------------------------------------------------------------------------------

Get a complete checkout of the project:

.. code-block:: bash

   git clone https://github.com/yanivassf-lab/fMRI.git
   cd fMRI

Use the source revision chosen for your analysis. For an existing checkout,
open that folder instead. Use the Python code and all three shell scripts
from the same revision.

The source folder must contain the package's installation metadata, such
as ``pyproject.toml`` or ``setup.py``. An archive containing only ``src`` and
``docs`` is not a complete installable checkout. In that case, obtain the
complete source for the same revision before continuing.

Register the local project and its command-line tools in the active
environment:

.. code-block:: bash

   python -m pip install --no-deps --no-build-isolation -e .

This command installs the project itself inside ``fpca-env``. Mamba has
already installed its dependencies. It is part of the same installation
route, not a second environment or a separate dependency setup.

Using the current source matters: an older packaged release may not contain
the commands or corrections described in these pages.

Check the installation
--------------------------------------------------------------------------------

.. code-block:: bash

   command -v python
   command -v fpca-main
   command -v fmri-fpca-pipeline

   fpca-main --help
   fmri-fpca-pipeline --help
   python -m PCAGroupDiscrimination.cli.explore_roi --help

   python -c "import fPCA, PCAGroupDiscrimination; print(list(fPCA.__path__)); print(list(PCAGroupDiscrimination.__path__))"
   python -c "import numpy, scipy, pandas, sklearn, imblearn, torch, umap, nibabel, nilearn, skfda, dtaidistance, openpyxl, dash, plotly; print('Imports passed')"

The printed package paths should point to the intended source checkout.
The help output should include the options described in :doc:`running`
and :doc:`group_pipeline`.

These checks confirm imports and command availability. Before a large run,
check a small input example with its matching mask and confirm that its
outputs and log look correct.

Record the environment
--------------------------------------------------------------------------------

Run these commands from the project checkout and keep the files with the
analysis record:

.. code-block:: bash

   conda list --explicit > conda-packages.txt
   python -m pip freeze > python-packages.txt
   git rev-parse HEAD > source-revision.txt
   git diff > source-local-changes.patch

The conda package list records a particular operating system and processor.
The source revision and local changes are needed because the project is
installed from a local checkout.

Next, prepare the input folders and run the steps in :doc:`group_pipeline`.

