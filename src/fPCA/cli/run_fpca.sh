#!/usr/bin/env bash

# Activate the virtual environment from the first argument
source "$1/bin/activate"

# Assign the remaining arguments to variables
MODE="$2"
INPUT_DIR="$3"
PROJECT_OUTPUT_DIR="$4"
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
        --n-jobs 5 \
        --use-nilearn-filter \
        --low-mem
else
    echo "Warning: No files found for Movement 1. Skipping..."
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
        --n-jobs 5 \
        --calc-penalty-skfda \
        --use-nilearn-filter \
        --low-mem
else
    echo "Warning: No files found for Movement 2. Skipping..."
fi

echo "All tasks completed successfully!"
