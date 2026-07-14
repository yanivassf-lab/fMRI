#!/usr/bin/env bash

# Set default values
FEATURES_SETS="1"
PC_INDEX="0"
RAW_DATA_PATH=""
N_PERMUTATIONS=5
MAX_PC=6

# Parse arguments sequentially
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --venv) VENV="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --features-sets) FEATURES_SETS="$2"; shift 2 ;;
        --pc-index) PC_INDEX="$2"; shift 2 ;;
        --input-dir) INPUT_DIR="$2"; shift 2 ;;
        --output-dir) OUTDIR="$2"; shift 2 ;;
        --metadata-file) METADATA_FILE="$2"; shift 2 ;;
        --hyperparameters-file) HYPERPARAMETERS_FILE="$2"; shift 2 ;;
        --test-pairs-file) TEST_PAIRS_FILE="$2"; shift 2 ;;
        --raw-data-path) RAW_DATA_PATH="$2"; shift 2 ;;
        --n-permutations) N_PERMUTATIONS="$2"; shift 2 ;;
        --max-pc) MAX_PC="$2"; shift 2 ;;
        *) shift ;; # Ignore unknown parameters
    esac
done

# Evaluate conditions after parsing arguments
if [ "$RAW_DATA_PATH" != "" ]; then
    echo "Using raw data path: $RAW_DATA_PATH"
    MAX_PC=0
fi

# Activate virtual environment
source "$VENV/bin/activate"

#########################################################################################

mkdir -p "$OUTDIR"

# Iterate over feature sets
for SET_NUM in $FEATURES_SETS; do

    # Define common base arguments for this specific feature set
    BASE_CMD_ARGS=(
        "--input-dir" "$INPUT_DIR"
        "--output-dir" "$OUTDIR/full_pipeline_set${SET_NUM}"
        "--metadata-csv" "$METADATA_FILE"
        "--ml-hyperparameters-file" "$HYPERPARAMETERS_FILE"
        "--test-pairs-file" "$TEST_PAIRS_FILE"
        "--extra-features-set" "$SET_NUM"
        "--n-permutations" "$N_PERMUTATIONS"
        "--use-nilearn-filter"
    )

    # Conditionally append the raw data arguments if the path is provided
    if [ -n "$RAW_DATA_PATH" ]; then
        BASE_CMD_ARGS+=("--use-raw-data" "--raw-data-path" "$RAW_DATA_PATH" "--output-dir" "$OUTDIR/full_pipeline_set${SET_NUM}-raw")
    fi

    # Execute the pipeline based on the requested mode
    if [ "$MODE" == "align-signals" ]; then
        for ((i=0; i<=MAX_PC; i++)); do
            # Append the dynamic arguments on the fly and run in background
            fmri-fpca-pipeline "${BASE_CMD_ARGS[@]}" "--mode" "$MODE" "--target-pc-index" "$i" &
        done
        # Wait for all background processes to finish before moving to the next feature set
        wait

    elif [[ "$MODE" == "ml" || "$MODE" == "stat" || "$MODE" == "ml-test" ]]; then
        for i in $PC_INDEX; do
            # Append the dynamic arguments on the fly and run synchronously
            fmri-fpca-pipeline "${BASE_CMD_ARGS[@]}" "--mode" "$MODE" "--target-pc-index" "$i"
        done

    else
        echo "Warning: Unrecognized mode '$MODE'"
    fi

done

echo "All tasks completed successfully!"
