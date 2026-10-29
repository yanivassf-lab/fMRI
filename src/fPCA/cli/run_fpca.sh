#!/usr/bin/env bash

# ./run_fpca.sh "pca-singles" "Input Directory" "Project Output Directory"
#
# Positional arguments: MODE INPUT_DIR PROJECT_OUTPUT_DIR
# Optional flags after them:
#   --use-nilearn-filter                 CompCor filtering of the BOLD series before the fPCA, with
#   --n-compcor-nilearn-filter 2 --smoothing-fwhm-nilearn-filter 4
#   --n-jobs 20                          parallel jobs of fpca-main (default: 20)
#
# Example of a filtered run:
# ./run_fpca.sh pca-singles /path/to/input_dir_train /path/to/output --use-nilearn-filter --n-compcor-nilearn-filter 2 --smoothing-fwhm-nilearn-filter 4

# Assign the positional arguments to variables
MODE="$1"
INPUT_DIR="$2"
PROJECT_OUTPUT_DIR="$3"
shift 3

# Optional flags
N_JOBS=20
FILTER_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --use-nilearn-filter) USE_NILEARN_FILTER="1"; shift 1 ;;
        --n-compcor-nilearn-filter) N_COMPCOR_NILEARN_FILTER="$2"; shift 2 ;;
        --smoothing-fwhm-nilearn-filter) SMOOTHING_FWHM_NILEARN_FILTER="$2"; shift 2 ;;
        --n-jobs) N_JOBS="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done
if [ -n "$USE_NILEARN_FILTER" ]; then
    FILTER_ARGS=("--use-nilearn-filter"
                 "--n-compcor-nilearn-filter" "${N_COMPCOR_NILEARN_FILTER:-2}"
                 "--smoothing-fwhm-nilearn-filter" "${SMOOTHING_FWHM_NILEARN_FILTER:-4}")
    echo "CompCor filtering enabled: ${FILTER_ARGS[*]}"
else
    echo "No CompCor filtering"
fi
#########################################################################################

mkdir -p "$PROJECT_OUTPUT_DIR"

# Force mathematical libraries to use exactly one thread per process
# This is crucial to prevent CPU thrashing when running parallel jobs
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Initialize separate arrays for movement 1 and movement 2
NII_FILES_MOV1=()
MASK_FILES_MOV1=()

NII_FILES_MOV2=()
MASK_FILES_MOV2=()

# Loop through all files and categorize them based on the task string
for bold_file in "$INPUT_DIR"/*.nii.gz; do
    if [ -f "$bold_file" ]; then
        filename=$(basename "$bold_file")
        base=${filename%%-preproc_bold*}
        mask_file="$INPUT_DIR/${base}-brain_mask.nii.gz"

        if [ -f "$mask_file" ]; then
            # Route to mov1 arrays if the filename contains '-movement1'
            if [[ "$filename" == *"-movement1"* ]]; then
                NII_FILES_MOV1+=("$bold_file")
                MASK_FILES_MOV1+=("$mask_file")
            # Route to mov2 arrays if the filename contains '-movement2'
            elif [[ "$filename" == *"-movement2"* ]]; then
                NII_FILES_MOV2+=("$bold_file")
                MASK_FILES_MOV2+=("$mask_file")
            fi
        fi
    fi
done


# ==========================================
# Execute Group fPCA for Movement 1
# ==========================================
echo "Found ${#NII_FILES_MOV1[@]} subjects for Movement 1. Starting Group fPCA pipeline..."

OUTPUT_DIR_MOV1="$PROJECT_OUTPUT_DIR"/outputs_mov1
if [ ${#NII_FILES_MOV1[@]} -gt 0 ]; then
    fpca-main \
        --mode "$MODE" \
        --nii-files "${NII_FILES_MOV1[@]}" \
        --mask-files "${MASK_FILES_MOV1[@]}" \
        --output-folder "$OUTPUT_DIR_MOV1" \
        --calc-penalty-skfda \
        "${FILTER_ARGS[@]}" \
        --n-jobs "$N_JOBS" \
        --low-mem \
        --force || exit 1
else
    echo "No files found for Movement 1" >&2
    exit 1
fi

echo "--------------------------------------------------"

# ==========================================
# Execute Group fPCA for Movement 2
# ==========================================
echo "Found ${#NII_FILES_MOV2[@]} subjects for Movement 2. Starting Group fPCA pipeline..."

OUTPUT_DIR_MOV2="$PROJECT_OUTPUT_DIR"/outputs_mov2
if [ ${#NII_FILES_MOV2[@]} -gt 0 ]; then
    fpca-main \
        --mode "$MODE" \
        --nii-files "${NII_FILES_MOV2[@]}" \
        --mask-files "${MASK_FILES_MOV2[@]}" \
        --output-folder "$OUTPUT_DIR_MOV2" \
        --calc-penalty-skfda \
        "${FILTER_ARGS[@]}" \
        --n-jobs "$N_JOBS" \
        --low-mem \
        --force || exit 1
else
    echo "Warning: No files found for Movement 2. Skipping..."
    exit 1
fi

echo "All tasks completed successfully!"

