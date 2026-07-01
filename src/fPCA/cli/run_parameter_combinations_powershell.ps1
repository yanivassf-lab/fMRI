# ========================================================
# PowerShell script to run fMRI analysis with multiple parameter combinations
# Supports both no-penalty and penalty runs
# Limits parallel processes to number of physical cores
# ========================================================

# --- Core Detection and Environment Setup ---

# Detect number of physical cores
$physicalCores = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum
Write-Host "Detected physical cores: $physicalCores"
$MAX_PARALLEL = $physicalCores
Write-Host "Setting MAX_PARALLEL to $MAX_PARALLEL"

# Set thread environment variables for BLAS/MKL to limit CPU usage per job
$env:MAX_PARALLEL = $MAX_PARALLEL
$env:MKL_NUM_THREADS = 1 # We want each individual job to use only 1 thread
$env:OMP_NUM_THREADS = 1 # We manage parallelism via process throttling
$env:OPENBLAS_NUM_THREADS = 1

# --- Paths and Parameters ---
$INPUT_DIR = "C:\path\to\raw-files"
$PROCESSED_DIR = "C:\path\to\preprocessed_data"
$BASE_OUTPUT_DIR = "C:\path\to\fmri_combinations_results"

# Conda executable
$CONDA_BAT = "C:\path\to\Miniconda3\condabin\conda.bat"
$ENV_NAME = "fmri-env"

# Parameter sets
$derivatives = @(2, 1, 0)
$lambda_ranges = @("0 6", "-6 6", "-6 0")
$n_basis_values = @(400, 300, 200, 100)

# Array to store running processes (Using ArrayList for reliable Add/Remove operations)
[System.Collections.ArrayList]$processes = @()

# -------------------------------------------------------------------------
# Helper function to manage throttling (reused for both NO-PENALTY and PENALTY)
# -------------------------------------------------------------------------
function Wait-ForSlot {
    param(
        [ref]$processesRef,
        [int]$MAX_PARALLEL
    )

    # Loop until the number of active processes is below the maximum limit
    while ($processesRef.Value.Count -ge $MAX_PARALLEL) {

        Write-Host "--- MAX LIMIT REACHED: Waiting for a slot. Current running: $($processesRef.Value.Count) / $MAX_PARALLEL ---"

        # 1. Reliable cleanup: Iterate backwards and remove all finished processes
        # This prevents the array from resizing unpredictably and cleans multiple finishes.
        $list = $processesRef.Value
        for ($i = $list.Count - 1; $i -ge 0; $i--) {
            # Check if the process has exited
            if ($list[$i].HasExited) {
                # Remove the finished process from the ArrayList
                $list.RemoveAt($i)
            }
        }

        # 2. Check the count again. If still too high, pause briefly before re-checking
        if ($processesRef.Value.Count -ge $MAX_PARALLEL) {
            Write-Host "--- Still full. Sleeping for 2 seconds to allow jobs to finish... ---"
            Start-Sleep -Seconds 2
        }
    }

    # Final cleanup before exiting the function (just in case)
    $list = $processesRef.Value
    for ($i = $list.Count - 1; $i -ge 0; $i--) {
        if ($list[$i].HasExited) {
            $list.RemoveAt($i)
        }
    }
}


# -----------------------
# Run NO-PENALTY jobs
# -----------------------
Write-Host "Starting NO-PENALTY jobs..."
foreach ($n_basis in $n_basis_values) {
    foreach ($bold_file in Get-ChildItem "$PROCESSED_DIR\*_filtered.nii") {
        $filename = $bold_file.Name
        $base = $filename -replace "-preproc_bold.*",""
        $mask_file = Join-Path $INPUT_DIR "$base-brain_mask.nii.gz"

        if (Test-Path $mask_file) {
            $output_dir = Join-Path $BASE_OUTPUT_DIR "$base\no_penalty_nb$n_basis"

            # --- Wait for a slot before launching the next job ---
            Wait-ForSlot -processesRef ([ref]$processes) -MAX_PARALLEL $MAX_PARALLEL

            # Build arguments for conda run
            $args = @("run", "-n", $ENV_NAME, "fmri-main",
                     "--nii-file", $bold_file.FullName,
                     "--mask-file", $mask_file,
                     "--output-folder", $output_dir,
                     "--processed",
                     "--calc-penalty-skfda",
                     "--no-penalty",
                     "--num-pca-comp", "7",
                     "--n-basis", "$n_basis"),
                     "--n-skip-vols-start" "4",
                     "--n-skip-vols-end", "12"


            # Start process in background
            $proc = Start-Process -FilePath $CONDA_BAT -ArgumentList $args -PassThru
            # Use .Add() instead of += to avoid 'op_Addition' error
            $processes.Add($proc) | Out-Null
            Write-Host "Launched NO-PENALTY job $($proc.Id): $base (Basis: $n_basis). Total running: $($processes.Count)"
        }
    }
}

Write-Host "Waiting for last $($processes.Count) tasks to finish..."
# Use the same cleanup/wait logic to ensure all remaining jobs are handled
while ($processes.Count -gt 0) {
    Wait-ForSlot -processesRef ([ref]$processes) -MAX_PARALLEL 1 # Set MAX_PARALLEL to 1 to force a wait for a slot
}

# -----------------------
# Run PENALTY jobs
# -----------------------
Write-Host "Starting PENALTY jobs..."
foreach ($p in $derivatives) {
    foreach ($u in $derivatives) {
        foreach ($lambda in $lambda_ranges) {
            $parts = $lambda -split " "
            $min = $parts[0]
            $max = $parts[1]

            foreach ($n_basis in $n_basis_values) {
                foreach ($bold_file in Get-ChildItem "$PROCESSED_DIR\*_filtered.nii") {
                    $filename = $bold_file.Name
                    $base = $filename -replace "-preproc_bold.*",""
                    $mask_file = Join-Path $INPUT_DIR "$base-brain_mask.nii.gz"

                    if (Test-Path $mask_file) {
                        $output_dir = Join-Path $BASE_OUTPUT_DIR "$base\p${p}_u${u}_l${min}_${max}_nb${n_basis}"

                        # --- Wait for a slot before launching the next job ---
                        Wait-ForSlot -processesRef ([ref]$processes) -MAX_PARALLEL $MAX_PARALLEL

                        # Build arguments for conda run
                        $args = @("run", "-n", $ENV_NAME, "fmri-main",
                                 "--TR", "0.75",
                                 "--nii-file", $bold_file.FullName,
                                 "--mask-file", $mask_file,
                                 "--output-folder", $output_dir,
                                 "--processed",
                                 "--calc-penalty-skfda",
                                 "--derivatives-num-p", "$p",
                                 "--derivatives-num-u", "$u",
                                 "--lambda-min", "$min",
                                 "--lambda-max", "$max",
                                 "--num-pca-comp", "7",
                                 "--n-basis", "$n_basis"),
                                 "--n-skip-vols-start" "4",
                                 "--n-skip-vols-end", "12"


                        # Start process in background
                        $proc = Start-Process -FilePath $CONDA_BAT -ArgumentList $args -PassThru
                        # Use .Add() instead of += to avoid 'op_Addition' error
                        $processes.Add($proc) | Out-Null
                        Write-Host "Launched PENALTY job $($proc.Id): $base (P: $p, U: $u, Lambda: $lambda). Total running: $($processes.Count)"
                    }
                }
            }
        }
    }
}

# -----------------------
# Wait for remaining processes to finish
# -----------------------
Write-Host "Waiting for last $($processes.Count) tasks to finish..."
# Use the same cleanup/wait logic to ensure all remaining jobs are handled
while ($processes.Count -gt 0) {
    Wait-ForSlot -processesRef ([ref]$processes) -MAX_PARALLEL 1 # Set MAX_PARALLEL to 1 to force a wait for a slot
}

Write-Host "All tasks finished."
