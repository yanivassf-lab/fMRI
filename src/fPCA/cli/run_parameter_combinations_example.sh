#!/usr/bin/env bash

INPUT_DIR="/Users/user/Documents/pythonProject/fMRI-runs/fMRI-files/files_from_amir_for_test/mask-files"
#PROCESSED_DIR="/Users/user/Documents/pythonProject/fMRI-runs/outputs/preprocessed_data"
PROCESSED_DIR="/Users/user/Documents/pythonProject/fMRI-runs/fMRI-files/files_from_amir_for_test/raw-files"
BASE_OUTPUT_DIR="/Users/user/Documents/pythonProject/fMRI-runs/outputs/run_params_comb/fmri_combinations_results_skfda_sk0_hf0.15_lf0.2_nb200_lmi-6_lma0_sd40"

# Activate Python environment
source "/Users/user/Documents/pythonProject/fMRI-env/bin/activate"

# Parameter sets
derivatives_p=(2)
derivatives_u=(0)
lambda_ranges=("-6 0")
n_basis_values=(200)
skips_start=(0)
MAX_PARALLEL=5  # Maximum number of parallel processes


# Array to store PIDs of background processes
pids=()

# Run combinations with penalties

for skip_start in "${skips_start[@]}"; do
    for p in "${derivatives_p[@]}"; do
        for u in "${derivatives_u[@]}"; do
            for lambda in "${lambda_ranges[@]}"; do
                for n_basis in "${n_basis_values[@]}"; do
                    read min max <<< "$lambda"
#                    for bold_file in "$PROCESSED_DIR"/*_filtered.nii; do
                    for bold_file in "$PROCESSED_DIR"/*.nii.gz; do
                        if [ -f "$bold_file" ]; then
                            filename=$(basename "$bold_file")
                            base=${filename%%-preproc_bold*}
                            mask_file="$INPUT_DIR/${base}-brain_mask.nii.gz"

                            if [ -f "$mask_file" ]; then
                                (
                                output_dir="$BASE_OUTPUT_DIR/${base}/p${p}_u${u}_sk${skip_start}"

                                fpca-main \
                                    --TR 0.75 \
                                    --nii-file "$bold_file" \
                                    --mask-file "$mask_file" \
                                    --output-folder "$output_dir" \
                                    --calc-penalty-skfda \
                                    --derivatives-num-p "$p" \
                                    --derivatives-num-u "$u" \
                                    --lambda-min "$min" \
                                    --lambda-max "$max" \
                                    --num-pca-comp 7 \
                                    --n-basis "$n_basis" \
                                    --n-skip-vols-start "$skip_start" \
                                    --n-skip-vols-end 0 \
                                    --highpass 0.15 \
                                    --lowpass 0.2 \
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
