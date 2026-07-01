#!/usr/bin/env bash

INPUT_DIR="/path/to/raw-files/"
PROCESSED_DIR="/path/to/preprocessed_data"
BASE_OUTPUT_DIR="/path/to/fmri_combinations_results"

# Activate Python environment
source "/path/to/fMRI-env/bin/activate"

# Parameter sets
derivatives=(0 1 2)
lambda_ranges=("0 6" "-6 0" "-6 6")
n_basis_values=(100 200 300 400)
MAX_PARALLEL=8  # Maximum number of parallel processes


# Array to store PIDs of background processes
pids=()

#!/usr/bin/env bash

INPUT_DIR="/path/to/raw-files/"
PROCESSED_DIR="/path/to/preprocessed_data"
BASE_OUTPUT_DIR="/path/to/fmri_combinations_results"

# Activate Python environment
source "/path/to/fMRI-env/bin/activate"

# Parameter sets
derivatives_p=(2 1 0)
derivatives_u=(0 1 2)
lambda_ranges=("-6 12")
n_basis_values=(100)
skips_start=(0)
MAX_PARALLEL=2  # Maximum number of parallel processes


# Array to store PIDs of background processes
pids=()

# Run with --no-penalty for each n-basis value
for n_basis in "${n_basis_values[@]}"; do
    for bold_file in "$PROCESSED_DIR"/*_filtered.nii; do
        if [ -f "$bold_file" ]; then
            filename=$(basename "$bold_file")
            base=${filename%%-preproc_bold*}
            mask_file="$INPUT_DIR/${base}-brain_mask.nii.gz"

            if [ -f "$mask_file" ]; then
                (
                    output_dir="$BASE_OUTPUT_DIR/${base}/no_penalty_nb${n_basis}"

                    fmri-main \
                        --nii-file "$bold_file" \
                        --mask-file "$mask_file" \
                        --output-folder "$output_dir" \
                        --processed \
                        --calc-penalty-skfda \
                        --no-penalty \
                        --num-pca-comp 7 \
                        --n-basis "$n_basis" \
                        --n-skip-vols-start $skip_start \
                        --n-skip-vols-end 12
                ) &

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

for pid in "${pids[@]}"; do
    wait "$pid"
done



# Run combinations with penalties
for skip_start in "${skips_start[@]}"; do
    for p in "${derivatives_p[@]}"; do
        for u in "${derivatives_u[@]}"; do
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
                                output_dir="$BASE_OUTPUT_DIR/${base}/p${p}_u${u}_sk${skip_start}"

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
                                    --num-pca-comp 7 \
                                    --n-basis "$n_basis" \
                                    --n-skip-vols-start "$skip_start" \
                                    --n-skip-vols-end 12

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

# Run combinations with penalties
for p in "${derivatives[@]}"; do
    for u in "${derivatives[@]}"; do
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
                            output_dir="$BASE_OUTPUT_DIR/${base}/p${p}_u${u}_l${min}_${max}_nb${n_basis}"

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
                                --num-pca-comp 7 \
                                --n-basis "$n_basis" \
                                --n-skip-vols-start 4 \
                                --n-skip-vols-end 12
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

# Wait for any remaining background processes to finish
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All tasks finished."
