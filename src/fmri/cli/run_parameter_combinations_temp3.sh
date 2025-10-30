#!/usr/bin/env bash

INPUT_DIR="/Users/user/Documents/pythonProject/fMRI-runs/fMRI-files/files_from_amir_for_test/raw-files"
PROCESSED_DIR="/Users/user/Documents/pythonProject/fMRI-runs/outputs/preprocessed_data"
BASE_OUTPUT_DIR="/Users/user/Documents/pythonProject/fMRI-runs/outputs/run_params_comb/fmri_combinations_results_skfda"

# Activate Python environment
source "/Users/user/Documents/pythonProject/fMRI-env/bin/activate"

# Parameter sets
derivatives=(0 1 2)
thresholds=(1e-3 1e-6)
lambda_ranges=("0 6" "-6 0" "-6 6")
n_basis_values=(100 200 300 400)
MAX_PARALLEL=8  # Maximum number of parallel processes


# Array to store PIDs of background processes
pids=()

# Process each filtered NIfTI file
#for bold_file in "$PROCESSED_DIR"/*_filtered.nii; do
#    if [ -f "$bold_file" ]; then
#        (
#            filename=$(basename "$bold_file")
#            base=${filename%%-preproc_bold*}
#            mask_file="$INPUT_DIR/${base}-brain_mask.nii.gz"
#
#            if [ -f "$mask_file" ]; then
                # Run with --no-penalty for each n-basis value
#                for n_basis in "${n_basis_values[@]}"; do
#                    output_dir="$BASE_OUTPUT_DIR/${base}/no_penalty_nb${n_basis}"
#                    mkdir -p "$output_dir"
#
#                    fmri-main \
#                        --nii-file "$bold_file" \
#                        --mask-file "$mask_file" \
#                        --output-folder "$output_dir" \
#                        --processed \
#                        --calc-penalty-skfda \
#                        --no-penalty \
#                        --num-pca-comp 7 \
#                        --threshold 1e-6 \
#                        --n-basis "$n_basis"
#                done

# Run combinations with penalties
for p in "${derivatives[@]}"; do
    for u in "${derivatives[@]}"; do
        for thresh in "${thresholds[@]}"; do
            for lambda in "${lambda_ranges[@]}"; do
                for n_basis in "${n_basis_values[@]}"; do
                    read min max <<< "$lambda"
                    for bold_file in "$PROCESSED_DIR"/*_filtered.nii; do
                        if [ -f "$bold_file" ]; then
                            filename=$(basename "$bold_file")
                            base=${filename%%-preproc_bold*}
                            mask_file="$INPUT_DIR/${base}-brain_mask.nii.gz"

                            if [ -f "$mask_file" ]; then
                                (
                                output_dir="$BASE_OUTPUT_DIR/${base}/p${p}_u${u}_t${thresh}_l${min}_${max}_nb${n_basis}"
                                mkdir -p "$output_dir"

                                fmri-main \
                                    --TR 0.75 \
                                    --nii-file "$bold_file" \
                                    --mask-file "$mask_file" \
                                    --output-folder "$output_dir" \
                                    --processed \
                                    --calc-penalty-skfda \
                                    --derivatives-num-p "$p" \
                                    --derivatives-num-u "$u" \
                                    --lambda-min "$min" \
                                    --lambda-max "$max" \
                                    --threshold "$thresh" \
                                    --num-pca-comp 7 \
                                    --n-basis "$n_basis"
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
                        fi
                    done
                done
            done
        done
    done
done

# Wait for any remaining background processes to finish
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All tasks finished."
