#!/bin/bash
# -----------------------------------------------------------------------------
# Script for parallel execution of fMRI comparison analyses
# -----------------------------------------------------------------------------

# 1. User Settings (please edit)
# List all your input directories here
INPUT_DIRS=(
    "/Users/user/Documents/pythonProject/fMRI-runs/outputs/run_params_comb/fmri_combinations_results_skfda_skip_100_volums_100_basis"
    "/Users/user/Documents/pythonProject/fMRI-runs/outputs/run_params_comb/fmri_combinations_results_skfda_skip_4_volums_100_basis"
    "/Users/user/Documents/pythonProject/fMRI-runs/outputs/run_params_comb/fmri_combinations_results_skfda_skip_0_volums_100_basis"
)

# Define the base output directory
BASE_OUTPUT_DIR="/Users/user/Documents/pythonProject/fMRI-runs/outputs/run_params_comb/compare_peaks_new"

# Define stimulus times for each movement
STIM_TIMES_MOV1="--combine-pcs-stimulus-times 30.0 150.75 324.75 360.75 427.5"
STIM_TIMES_MOV2="--combine-pcs-stimulus-times 85.5 228.0 234.75 337.5"

# Define BOLD lag times for each movement
#BOLD_LAG_MOV1="--combine-pcs-bold-lag-seconds 8.0 8.0 8.0 8.0"
#BOLD_LAG_MOV2="--combine-pcs-bold-lag-seconds 8.0 8.0 8.0 8.0"

# Correlation threshold for combining PCs (after normalization)
CORRELATION_THR=0.1

# 2. Main execution loop
# Loop over each input directory defined
for INPUT_PATH in "${INPUT_DIRS[@]}"; do

    # Create a unique output folder name based on input folder name
    INPUT_NAME=$(basename "$INPUT_PATH")
    echo "--- Starting batch for input: $INPUT_NAME ---"

    # Define the base command shared by all runs
    # Note that --max-workers=2 is for internal command execution
    BASE_CMD="compare-pcs --files-path $INPUT_PATH --movements 1 2 --num-scores 10 --max-workers 2 --skip-timepoints 0"

    # 3. Auto-PC runs (4 commands)
    echo "Starting Auto-PC runs (4)..."
    OUTPUT_FOLDER_1="$BASE_OUTPUT_DIR/${INPUT_NAME}"
    mkdir -p "$OUTPUT_FOLDER_1" && $BASE_CMD --output-folder "$OUTPUT_FOLDER_1/auto_best_fixori" --pc-sim-auto --pc-sim-auto-best-similar-pc --pc-sim-auto-weight-similar-pc 2 --fix-orientation &

    OUTPUT_FOLDER_2="$BASE_OUTPUT_DIR/${INPUT_NAME}"
    mkdir -p "$OUTPUT_FOLDER_2" && $BASE_CMD --output-folder "$OUTPUT_FOLDER_2/auto_avg_fixori" --pc-sim-auto --pc-sim-auto-weight-similar-pc 2 --fix-orientation &

    OUTPUT_FOLDER_3="$BASE_OUTPUT_DIR/${INPUT_NAME}"
    mkdir -p "$OUTPUT_FOLDER_3" && $BASE_CMD --output-folder "$OUTPUT_FOLDER_3/auto_best_peaksabs" --pc-sim-auto --pc-sim-auto-best-similar-pc --pc-sim-auto-weight-similar-pc 2 --peaks-abs &

    OUTPUT_FOLDER_4="$BASE_OUTPUT_DIR/${INPUT_NAME}"
    mkdir -p "$OUTPUT_FOLDER_4" && $BASE_CMD --output-folder "$OUTPUT_FOLDER_4/auto_avg_peaksabs" --pc-sim-auto --pc-sim-auto-weight-similar-pc 2 --peaks-abs &

    # 4. Fixed-PC runs (4 commands)
    echo "Starting Fixed-PC runs (4)..."
    OUTPUT_FOLDER_5="$BASE_OUTPUT_DIR/${INPUT_NAME}"
    mkdir -p "$OUTPUT_FOLDER_5" && $BASE_CMD --output-folder "$OUTPUT_FOLDER_5/fixed_pc0_fixori" --pc-num-comp 0 --fix-orientation &

    OUTPUT_FOLDER_6="$BASE_OUTPUT_DIR/${INPUT_NAME}"
    mkdir -p "$OUTPUT_FOLDER_6" && $BASE_CMD --output-folder "$OUTPUT_FOLDER_6/fixed_pc0_peaksabs" --pc-num-comp 0 --peaks-abs &

    OUTPUT_FOLDER_7="$BASE_OUTPUT_DIR/${INPUT_NAME}"
    mkdir -p "$OUTPUT_FOLDER_7" && $BASE_CMD --output-folder "$OUTPUT_FOLDER_7/fixed_pc1_fixori" --pc-num-comp 1 --fix-orientation &

    OUTPUT_FOLDER_8="$BASE_OUTPUT_DIR/${INPUT_NAME}"
    mkdir -p "$OUTPUT_FOLDER_8" && $BASE_CMD --output-folder "$OUTPUT_FOLDER_8/fixed_pc1_peaksabs" --pc-num-comp 1 --peaks-abs &

    # 5. Combine-PC runs (2 new commands)
    echo "Starting Combine-PC runs (2)..."

    # Define the new flags based on the times you set above
    # I'm using threshold=60 as we discussed (after normalization)
    COMBINE_FLAGS="--combine-pcs --combine-pcs-correlation-threshold $CORRELATION_THR $STIM_TIMES_MOV1 $STIM_TIMES_MOV2"

    OUTPUT_FOLDER_9="$BASE_OUTPUT_DIR/${INPUT_NAME}"
    mkdir -p "$OUTPUT_FOLDER_9" && $BASE_CMD --output-folder "$OUTPUT_FOLDER_9/combine_fixori" $COMBINE_FLAGS --fix-orientation &

    OUTPUT_FOLDER_10="$BASE_OUTPUT_DIR/${INPUT_NAME}"
    mkdir -p "$OUTPUT_FOLDER_10" && $BASE_CMD --output-folder "$OUTPUT_FOLDER_10/combine_peaksabs" $COMBINE_FLAGS --peaks-abs &

    # Wait for all 10 runs to complete
    echo "Waiting for all 10 runs for $INPUT_NAME to complete..."
    wait
    echo "--- Batch for $INPUT_NAME finished. ---"

done

echo "All input directories processed."
