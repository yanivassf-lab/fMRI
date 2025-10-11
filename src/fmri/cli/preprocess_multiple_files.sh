#!/bin/bash

INPUT_DIR="/Users/user/My Drive (refaelkohen@mail.tau.ac.il)/TLV-U-drive/BrainWork/fMRI/files_from_amir_for_checking/raw-files/"
OUTPUT_DIR="preprocessed_data"

mkdir -p "$OUTPUT_DIR"

# Find and preprocess each NIfTI file
for bold_file in "$INPUT_DIR"/*bold*.nii*; do
    if [ -f "$bold_file" ]; then
        filename=$(basename "$bold_file")
        base=${filename%%-preproc_bold*}
        mask_file="$INPUT_DIR/${base}-brain_mask.nii.gz"
        if [ -f "$mask_file" ]; then
            echo $mask_file
            source /Users/user/Documents/pythonProject/fMRI-env/bin/activate
            echo "Preprocessing: $filename with mask: $(basename "$mask_file")"
            preprocess-nii-file --nii-file "$bold_file" --mask-file "$mask_file" --output-folder "$OUTPUT_DIR"
        fi
    fi
done
