Developers Guide
================

This page describes how to build and publish releases to PyPI and (optionally) conda-forge, and how to automate releases via GitHub Actions.

Local development setup
-----------------------

- Create and activate an environment (pip or conda). For conda, see the Installation page.
- Install in editable mode and dev tools:

.. code-block:: console

    pip install -e .
    pip install -r requirements.txt  # if present
    pip install -e .[dev]            # ruff, pytest, tox, etc. per pyproject.toml

- Sanity checks:

.. code-block:: console

    ruff check fmri/
    pytest -q

Versioning
----------

We use bump2version for semantic versioning:

.. code-block:: console

    bump2version patch   # or minor / major

This updates version metadata and tags a commit.

Build packages (PyPI)
---------------------

Use the standard Python build backend:

.. code-block:: console

    python -m build   # generates dist/*.whl and dist/*.tar.gz

Upload to PyPI (manual)
-----------------------

Set up PyPI API token in ~/.pypirc or use environment variable. Then:

.. code-block:: console

    twine upload dist/*

Conda package (conda-forge)
---------------------------

Recommended path is to publish to PyPI, then feed conda-forge via a recipe:

1) Fork `staged-recipes` and create a recipe for this project.
2) The recipe should reference the PyPI sdist/whl and declare runtime deps:
   - numpy, scipy, matplotlib, nibabel, scikit-fda, dtaidistance, etc.
3) Submit PR to conda-forge/staged-recipes. Once merged, a feedstock is created and builds are automated across platforms.

Useful links:
- conda-forge staged-recipes: https://github.com/conda-forge/staged-recipes
- Feedstock maintenance: rerender, version bumps via PRs to the feedstock

Automated releases (GitHub Actions)
-----------------------------------

We recommend a workflow that on a new tag:
- runs lint/tests
- builds wheels/sdist
- uploads to PyPI via a PyPI token secret

Example workflow skeleton (save as .github/workflows/release.yml):

.. code-block:: yaml

    name: Release
    on:
      push:
        tags:
          - 'v*'
    jobs:
      build:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: '3.11'
          - name: Install build deps
            run: |
              python -m pip install --upgrade pip
              pip install build twine
          - name: Build
            run: |
              python -m build
          - name: Publish to PyPI
            env:
              TWINE_USERNAME: __token__
              TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
            run: |
              python -m twine upload dist/*

CI for tests and lint
---------------------

Add a separate workflow to run tests on PRs and pushes, e.g. matrix on OS/Python versions, installing runtime deps from conda-forge or pip.

Release checklist
-----------------
- Update CHANGELOG
- bump2version patch|minor|major
- push branch and tag (bump2version can push tags automatically)
- verify CI green
- verify PyPI artifacts
- update/trigger conda-forge feedstock (if applicable)


Local conda build (advanced)
----------------------------

If you want to build a local conda package (for testing before conda-forge):

1) Install conda-build and conda-verify

.. code-block:: console

    conda install -c conda-forge conda-build conda-verify

2) Create a minimal recipe (meta.yaml) under a recipe directory (e.g., conda.recipe/)

.. code-block:: yaml

    package:
      name: fmri
      version: {{ GIT_DESCRIBE_TAG }}

    source:
      git_url: https://github.com/yanivassf-lab/fmri.git
      git_rev: {{ GIT_DESCRIBE_TAG }}

    build:
      noarch: python
      script: python -m pip install --no-deps --ignore-installed .

    requirements:
      host:
        - python >=3.10
        - pip
      run:
        - python >=3.10
        - numpy
        - scipy
        - matplotlib
        - nibabel
        - scikit-fda
        - dtaidistance

    about:
      home: https://github.com/yanivassf-lab/fmri
      license: GPL-3.0
      summary: Functional PCA on fMRI data to extract dominant temporal components.

3) Build the package

.. code-block:: console

    conda build conda.recipe

4) Install the local build for testing

.. code-block:: console

    conda install --use-local fmri
