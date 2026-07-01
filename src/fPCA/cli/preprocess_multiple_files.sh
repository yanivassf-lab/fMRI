#!/usr/bin/env bash

INPUT_DIR="/path/to/raw-files/"
OUTPUT_DIR="/path/to/preprocessed_data"
# ~12 GB RAM per process, adjust based on your system
MAX_PARALLEL=1  # Maximum number of parallel processes

source "/path/to/fMRI-env/bin/activate"

mkdir -p "$OUTPUT_DIR"

# Array to store PIDs of background processes
pids=()

for bold_file in "$INPUT_DIR"/*bold*.nii*; do
    if [ -f "$bold_file" ]; then
        # Start a background subshell to process the file
        (
            filename=$(basename "$bold_file")
            base=${filename%%-preproc_bold*}
            mask_file="$INPUT_DIR/${base}-brain_mask.nii.gz"

            if [ -f "$mask_file" ]; then
                echo "Preprocessing: $filename with mask: $(basename "$mask_file")"
                preprocess-nii-file --nii-file "$bold_file" --mask-file "$mask_file" --output-folder "$OUTPUT_DIR"
            fi
        ) &

        # Save the PID of the background process
        pids+=($!)

        # Limit the number of parallel processes
        while [ ${#pids[@]} -ge $MAX_PARALLEL ]; do
            # Wait for the first process in the array to finish
            wait "${pids[0]}"
            # Remove the finished PID from the array
            pids=("${pids[@]:1}")
        done
    fi
done

# Wait for any remaining background processes to finish
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All files processed."
