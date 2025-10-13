Utility Scripts Documentation
============================

This page provides detailed documentation for the utility scripts included in the fMRI package. These scripts are designed to streamline common preprocessing and analysis tasks.

.. contents:: Table of Contents
   :depth: 3

Preprocessing Scripts
--------------------

preprocess-nii-file
~~~~~~~~~~~~~~~~~~~

The ``preprocess-nii-file`` script preprocesses a single 4D fMRI NIfTI file using a 3D mask file, applying filtering and smoothing to prepare the data for later analysis. Its main purpose is to make this preprocessing step reusable — so that when running ``fmri-main`` multiple times with different parameters, you don’t need to repeat the same costly preprocessing. Instead, once the data has been preprocessed, you can run ``fmri-main`` directly on the generated outputs using the ``--processed`` flag to indicate that the preprocessing stage has already been completed.

**Command Line Interface:**

.. code-block:: bash

   preprocess-nii-file --nii-file <path_to_nii> --mask-file <path_to_mask> --output-folder <output_path> [options]

**Required Arguments:**

* ``--output-folder`` (str): Path to an existing folder where output files will be saved.

**Optional Arguments:**

* ``--nii-file`` (str): Path to the 4D fMRI NIfTI file. If not provided, the script will prompt for input.
* ``--mask-file`` (str): Path to the 3D mask NIfTI file. If not provided, the script will prompt for input.
* ``--TR`` (float): Repetition time (TR) in seconds. If not provided, it will be extracted from the NIfTI header.
* ``--smooth_size`` (int): Box size of smoothing kernel (default: 5).

**What the Script Does:**

1. Loads the 4D fMRI data and 3D mask
2. Applies temporal filtering and spatial smoothing
3. Saves the filtered data as a new NIfTI file with "_filtered" suffix
4. Outputs the processed file to the specified output folder

**Example Usage:**

.. code-block:: bash

   # Basic usage
   preprocess-nii-file --nii-file /path/to/subject_bold.nii.gz \
                       --mask-file /path/to/brain_mask.nii.gz \
                       --output-folder /path/to/preprocessed_data/

   # With custom parameters
   preprocess-nii-file --nii-file /path/to/subject_bold.nii.gz \
                       --mask-file /path/to/brain_mask.nii.gz \
                       --output-folder /path/to/preprocessed_data/ \
                       --TR 0.75 \
                       --smooth_size 3

**Output:**

The script creates a filtered NIfTI file in the output folder with the naming convention: ``<original_filename>_filtered.nii``

**Notes:**

* The output folder must exist before running the script
* The script automatically handles both compressed (.nii.gz) and uncompressed (.nii) NIfTI files
* Processing time depends on the size of the input data and smoothing parameters

Batch Processing Scripts
------------------------

preprocess_multiple_files.sh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``preprocess_multiple_files.sh`` script is a Bash script designed to batch process multiple NIfTI files with their corresponding mask files. It automates the preprocessing multiple nii files.

**Purpose:**

This script automates the preprocessing of multiple fMRI files by:
- Finding all bold files in a specified input directory
- Matching each bold file with its corresponding brain mask
- Running the ``preprocess-nii-file`` command for each valid pair
- Organizing output files in a structured manner

**Script Structure:**

The script expects files to be named with a specific pattern:
- Bold files: ``<base_name>-preproc_bold.nii*``
- Mask files: ``<base_name>-brain_mask.nii.gz``

**Current Configuration:**

.. code-block:: bash

   INPUT_DIR="/folder/with/raw-files/"
   OUTPUT_DIR="preprocessed_data"

**How to Customize:**

1. **Update Input Directory:** Modify the ``INPUT_DIR`` variable to point to your data directory:

   .. code-block:: bash

      INPUT_DIR="/path/to/your/fmri/data/"

2. **Update Output Directory:** Change the ``OUTPUT_DIR`` variable for your preferred output location:

   .. code-block:: bash

      OUTPUT_DIR="/path/to/your/preprocessed/data/"

3. **Update Virtual Environment Path:** Modify the activation path if using a different virtual environment:

   .. code-block:: bash

      source /path/to/your/venv/bin/activate

**Usage:**

.. code-block:: bash

   # Make the script executable
   chmod +x preprocess_multiple_files.sh

   # Run the script
   ./preprocess_multiple_files.sh

**What the Script Does:**

1. Creates the output directory if it doesn't exist
2. Searches for all bold files matching the pattern ``*bold*.nii*``
3. For each bold file, attempts to find the corresponding brain mask
4. Runs ``preprocess-nii-file`` for each valid file pair
5. Logs the processing status for each file

**File Organization Expected:**

.. code-block:: text

   input_directory/
   ├── subject1-preproc_bold.nii.gz
   ├── subject1-brain_mask.nii.gz
   ├── subject2-preproc_bold.nii.gz
   ├── subject2-brain_mask.nii.gz
   └── ...

**Output Structure:**

.. code-block:: text

   preprocessed_data/
   ├── subject1-preproc_bold_filtered.nii
   ├── subject2-preproc_bold_filtered.nii
   └── ...

**Troubleshooting:**

* Ensure the virtual environment path is correct
* Verify that file naming conventions match the expected patterns
* Check that the input directory contains both bold and mask files
* Make sure the fMRI package is installed in the virtual environment

run_parameter_combinations.sh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``run_parameter_combinations.sh`` script performs comprehensive parameter sweeps for fMRI analysis, testing multiple combinations of parameters to find optimal settings for your data.

**Purpose:**

This script systematically tests different parameter combinations for fMRI analysis including:
- Different derivative orders (p and u parameters)
- Various threshold values
- Multiple lambda ranges for regularization
- Different numbers of basis functions
- Both penalized and non-penalized approaches

**Current Parameter Sets:**

.. code-block:: bash

   derivatives=(0 1 2)
   thresholds=(1e-6 1e-3)
   lambda_ranges=("0 3" "-4 0" "-4 3")
   n_basis_values=(100 200 300)

**Configuration Variables:**

* ``INPUT_DIR``: Directory containing original raw files (for mask files)
* ``PROCESSED_DIR``: Directory containing preprocessed data (for filtered NIfTI files)
* ``BASE_OUTPUT_DIR``: Base directory for all analysis results

**How to Customize:**

1. **Update Directory Paths:**

   .. code-block:: bash

      INPUT_DIR="/path/to/your/raw/files/"
      PROCESSED_DIR="/path/to/your/preprocessed/data/"
      BASE_OUTPUT_DIR="/path/to/your/analysis/results/"

2. **Modify Parameter Ranges:**

   .. code-block:: bash

      # Example: More comprehensive parameter sweep
      derivatives=(0 1 2 3)
      thresholds=(1e-8 1e-6 1e-4 1e-3 1e-2)
      lambda_ranges=("0 1" "0 2" "0 3" "-2 0" "-4 0" "-4 2" "-4 3")
      n_basis_values=(50 100 150 200 250 300 400)

3. **Adjust Analysis Parameters:**

   .. code-block:: bash

      # Modify these parameters in the fmri-main calls
      --num-pca-comp 7          # Number of PCA components
      --TR 0.75                 # Repetition time
      --calc-penalty-skfda      # Use scikit-fda for penalty calculation

    You can add '&' at the end of the fmri-main command to run analyses in parallel on the files (ensure sufficient CPU and RAM).

**Usage:**

.. code-block:: bash

   # Make the script executable
   chmod +x run_parameter_combinations.sh

   # Run the script
   ./run_parameter_combinations.sh

**Analysis Structure:**

The script runs two types of analyses:

1. **Non-penalized Analysis:** Tests different numbers of basis functions without regularization
2. **Penalized Analysis:** Tests combinations of all parameters with regularization

**Output Directory Structure:**

.. code-block:: text

   fmri_combinations_results/
   └── <base_filename>/
       ├── no_penalty_nb100/
       ├── no_penalty_nb200/
       ├── no_penalty_nb300/
       ├── p0_u0_t1e-6_l0_3_nb100/
       ├── p0_u0_t1e-6_l0_3_nb200/
       ├── p0_u1_t1e-6_l-4_0_nb100/
       └── ...

**Parameter Combinations Explained:**

* ``p0_u1_t1e-6_l-4_0_nb100`` means:
  - p=0 (0th derivative penalty)
  - u=1 (1st derivative penalty)
  - t=1e-6 (threshold value)
  - l=-4_0 (lambda range from -4 to 0)
  - nb=100 (100 basis functions)

**Computational Considerations:**

* **Time:** This script can run for many hours depending on data size and parameter ranges
* **Storage:** Each parameter combination creates a full output directory
* **Memory:** Ensure sufficient RAM for parallel processing
* **CPU:** The script can be modified to run in parallel for faster execution

**Monitoring Progress:**

The script outputs progress information including:
- Current file being processed
- Parameter combination being tested
- Output directory creation status

Analysis Scripts
----------------

compare-peaks
~~~~~~~~~~~~~

The ``compare-peaks`` script compares temporal peaks between different movements across subjects and parameter combinations. It generates similarity matrices and visualizations to identify optimal parameter settings.

**Command Line Interface:**

.. code-block:: bash

   compare-peaks --files-path <subjects_path> --output-folder <output_path> --pc_num <num_components> [options]

**Required Arguments:**

* ``--files-path`` (str): Path to the directory `fmri_combinations_results` containing the output of the previous script: `run_parameter_combinations.sh`.
* ``--output-folder`` (str): Path to the output folder (must not exist)
* ``--pc_num`` (int): Number of principal components/functions to analyze

**Optional Arguments:**

* ``--movements`` (list[int]): List of movements to compare (default: [1, 2])
* ``--alpha`` (float): Weighting parameter for combined score (0-1, default: 0.5)
* ``--num-scores`` (int): Number of top scores to keep (default: 5)

**Input File Structure:**

The script expects data organized as follows:

.. code-block:: text

   fmri_combinations_results/
   ├── sub-01_movement1/
   │   ├── no_penalty_nb100/
   │   │   ├── temporal_profile_pc_0.txt
   │   │   ├── temporal_profile_pc_1.txt
   │   │   └── ...
   │   ├── p0_u1_t1e-6_l-4_0_nb100/
   │   │   ├── temporal_profile_pc_0.txt
   │   │   └── ...
   │   └── ...
   ├── sub-01_movement2/
   │   ├── no_penalty_nb100/
   │   │   └── ...
   │   └── ...
   └── sub-02_movement1/
       └── ...

**What the Script Does:**

1. **Peak Detection:** Extracts temporal peaks from signal intensity profiles
2. **Similarity Calculation:** Computes similarity matrices between subjects and movements
3. **Scoring:** Calculates three types of scores:
   The scoring process computes three related similarity measures that evaluate how consistent and comparable the fMRI signals are within and across movements:

   - Within-movement similarity:
     Measures how similar the signal patterns of different examples (subjects) are within the same movement.
     A higher value indicates that subjects performing the same movement exhibit more consistent signal shapes.
     (Computed as the average pairwise similarity among all examples within each movement.)

   - Between-movement similarity:
     Measures how similar each example’s signal is across different movements.
     A higher value means that the same example shows a more similar signal when performing different movements.
     (Computed as the average similarity across movements for corresponding examples.)

   - Weighted combined score:
     Represents a weighted average of the two measures above:
     the between-movement score is weighted by ``alpha``, and the within-movement score by ``1 - alpha``.
     This allows balancing the importance between within-movement consistency and between-movement similarity.

4. **Visualization:** Creates comprehensive plots showing:

   Each parameter combination generates a figure containing:

   - Left panel: Signal intensity profiles for all subjects with detected peaks marked
   - Right panel: Similarity matrix heatmap showing relationships between all subjects
   - Title: Parameter combination and computed scores

5. **logging:**

   The script creates detailed logs including:

   - Command line arguments used
   - Processing progress for each parameter combination
   - Best parameter combinations for each score type
   - Error messages and warnings

**Example Usage:**

.. code-block:: bash

   # Basic usage
   compare-peaks --files-path /path/to/subjects/ \
                 --output-folder /path/to/peak_analysis/ \
                 --pc_num 7

   # Advanced usage with custom parameters
   compare-peaks --files-path /path/to/subjects/ \
                 --output-folder /path/to/peak_analysis/ \
                 --pc_num 7 \
                 --movements 1 2 3 \
                 --alpha 0.7 \
                 --num-scores 10

**Output Structure:**

.. code-block:: text

   output_folder/
   ├── compare_peaks_log.txt
   ├── pc_0/
   │   ├── peaks_no_penalty_nb100_pc_0.png
   │   ├── peaks_p0_u1_t1e-6_l-4_0_nb100_pc_0.png
   │   └── ...
   ├── pc_1/
   │   └── ...
   └── ...


**Best Practices:**

1. **Output Management:** The output folder must not exist; the script will create it
2. **Movement Validation:** Ensure movement numbers are between 1 and 9

**Troubleshooting:**

* **File not found errors:** Verify the directory structure matches the expected format
* **Missing temporal profile files:** Ensure all parameter combinations have been run
* **Alpha parameter:** Must be between 0 and 1
* **Movement numbers:** Must be between 1 and 9

