#!/usr/bin/env bash


# ./run_pipeline.sh \
#   --mode "ml-test" \
#   --features-sets "1" \
#   --pc-index "0 1" \
#   --input-dir "INPUT_DIR" \
#   --output-dir "OUTPUT_DIR" \
#   --metadata-file "status_fMRI_participants-example.csv" \
#   --hyperparameters-file "ml_hyperparameters.json" \
#   --test-pairs-file "test_pairs.json" \
#   --raw-data-path "RAW_DATA_PATH" #"/Volumes/BackUps_Amir/To Refael"


#
# Optional flags (all passed through to fmri-fpca-pipeline):
#   --ml-models "LR SVM"                 models to train (default: SVM LR NN DTree RandForest)
#   --groups "Keyboards NM"              restrict the pairs to the combinations of these groups
#   --jobs 32                            parallel workers of the grid search (default: 32)
#   --run-permutations --n-permutations 1000
#                                        nested permutation test (otherwise 5 placeholder permutations)
#   --annotation-xlsx visualize.xlsx     feature set 4: stimulus regressors from the orchestration workbook
#   --encoding-source fpca|raw           feature set 4: signal to encode (default fpca; raw needs --raw-data-path)
#   --acoustic-features-dir DIR          feature set 4: precomputed acoustic_mov1.csv / acoustic_mov2.csv
#   --use-nilearn-filter                 raw-data mode only: CompCor filtering of the BOLD series, with
#   --n-compcor-nilearn-filter 2 --smoothing-fwhm-nilearn-filter 4
#   --max-pc 0                           align-signals: run PCs 0..max-pc (use 0 for feature sets 3 and 4)

# Set default values
FEATURES_SETS="1"
PC_INDEX="0"
RAW_DATA_PATH=""
N_PERMUTATIONS=5
MAX_PC=6
ML_MODELS="SVM LR NN DTree RandForest"
JOBS=32
USE_NILEARN_FILTER=""
N_COMPCOR_NILEARN_FILTER=2
SMOOTHING_FWHM_NILEARN_FILTER=4

# Parse arguments sequentially
while [[ "$#" -gt 0 ]]; do
    case $1 in
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
        --run-permutations) RUN_PERMUTATIONS="1"; shift 1 ;;
        --max-pc) MAX_PC="$2"; shift 2 ;;
        --ml-models) ML_MODELS="$2"; shift 2 ;;
        --groups) GROUPS_LIST="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --annotation-xlsx) ANNOTATION_XLSX="$2"; shift 2 ;;
        --encoding-source) ENCODING_SOURCE="$2"; shift 2 ;;
        --acoustic-features-dir) ACOUSTIC_DIR="$2"; shift 2 ;;
        --use-nilearn-filter) USE_NILEARN_FILTER="1"; shift 1 ;;
        --n-compcor-nilearn-filter) N_COMPCOR_NILEARN_FILTER="$2"; shift 2 ;;
        --smoothing-fwhm-nilearn-filter) SMOOTHING_FWHM_NILEARN_FILTER="$2"; shift 2 ;;
        *) shift ;; # Ignore unknown parameters
    esac
done

# Evaluate conditions after parsing arguments
if [ "$RAW_DATA_PATH" != "" ]; then
    echo "Using raw data path: $RAW_DATA_PATH"
    MAX_PC=0
fi

# Force mathematical libraries to use exactly one thread per process
# This is crucial to prevent CPU thrashing when running parallel jobs
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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
        "--ml-models" $ML_MODELS
        "--jobs" "$JOBS"
    )

    # Conditionally append the raw data arguments if the path is provided
    if [ -n "$RAW_DATA_PATH" ]; then
        BASE_CMD_ARGS+=("--use-raw-data" "--raw-data-path" "$RAW_DATA_PATH" "--output-dir" "$OUTDIR/full_pipeline_set${SET_NUM}-raw")
    fi
    # CompCor filtering of the raw BOLD series (raw-data mode only; ignored for the PC-based feature sets)
    if [ -n "$USE_NILEARN_FILTER" ]; then
        BASE_CMD_ARGS+=("--use-nilearn-filter"
                        "--n-compcor-nilearn-filter" "$N_COMPCOR_NILEARN_FILTER"
                        "--smoothing-fwhm-nilearn-filter" "$SMOOTHING_FWHM_NILEARN_FILTER")
    fi
    # Optional restriction of the pairs (e.g. for a nested permutation run on one pair)
    if [ -n "$GROUPS_LIST" ]; then
        BASE_CMD_ARGS+=("--groups" $GROUPS_LIST)
    fi
    if [ -n "$RUN_PERMUTATIONS" ]; then
        BASE_CMD_ARGS+=("--run-permutations")
    fi
    # Feature set 4: stimulus regressors and the signal they are regressed on
    if [ -n "$ANNOTATION_XLSX" ]; then
        BASE_CMD_ARGS+=("--annotation-xlsx" "$ANNOTATION_XLSX")
    fi
    if [ -n "$ENCODING_SOURCE" ]; then
        BASE_CMD_ARGS+=("--encoding-source" "$ENCODING_SOURCE")
    fi
    if [ -n "$ACOUSTIC_DIR" ]; then
        BASE_CMD_ARGS+=("--acoustic-features-dir" "$ACOUSTIC_DIR")
    fi

    # Execute the pipeline based on the requested mode
    if [ "$MODE" == "align-signals" ]; then
        pids=()
        for ((i=0; i<=MAX_PC; i++)); do
            fmri-fpca-pipeline "${BASE_CMD_ARGS[@]}" \
                "--mode" "$MODE" \
                "--target-pc-index" "$i" &
            pids+=("$!")
        done

        failed=0
        for pid in "${pids[@]}"; do
            wait "$pid" || failed=1
        done

        if ((failed)); then
            echo "At least one alignment task failed" >&2
            exit 1
        fi
    elif [[ "$MODE" == "ml" || "$MODE" == "stat" || "$MODE" == "ml-test" || "$MODE" == "stat-test" ]]; then
        for i in $PC_INDEX; do
            # Append the dynamic arguments on the fly and run synchronously
            fmri-fpca-pipeline "${BASE_CMD_ARGS[@]}" "--mode" "$MODE" "--target-pc-index" "$i" || exit 1
        done

    else
        echo "Warning: Unrecognized mode '$MODE'"
    fi

done

echo "All tasks completed successfully!"

