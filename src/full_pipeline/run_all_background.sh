#!/bin/bash

# Full run (fPCA + all feature sets), with CompCor filtering:
#   mkdir -p /mnt/my_passport/R_output/output_filter1
#   nohup ./run_all_background.sh --filter > /mnt/my_passport/R_output/output_filter1/full_master_run.log 2>&1 &
# Without filtering:
#   mkdir -p /mnt/my_passport/R_output/output_no_filter
#   nohup ./run_all_background.sh > /mnt/my_passport/R_output/output_no_filter/full_master_run.log 2>&1 &
# Only feature set 4 on an existing fPCA output:
#   nohup ./run_all_background.sh --filter --sets "4" --skip-fpca > /mnt/my_passport/R_output/output_filter1/set4_run.log 2>&1 &
# Feature sets 0 and 1 on the raw BOLD series instead of the fPCA reconstruction (no fPCA needed):
#   nohup ./run_all_background.sh --sets "0 1" --raw --skip-fpca > /mnt/my_passport/R_output/output_no_filter/raw_run.log 2>&1 &
#
# Flags:
#   --sets "0 1 3 4"        feature sets to run (default: all four)
#   --filter                CompCor filtering (--n-compcor-nilearn-filter 2, --smoothing-fwhm-nilearn-filter 4);
#                           output goes to output_filter1 instead of output_no_filter
#   --raw                   run on the raw BOLD series (--use-raw-data): raw_data_train for
#                           align-signals/ml/stat, raw_data_test for ml-test/stat-test; outputs go to
#                           full_pipeline_set<N>-raw. Applies to sets 0, 1 and 4 (set 3 needs the fPCA output
#                           and is skipped); for set 4 it is the same as --encoding-source raw
#   --encoding-source X     feature set 4: fpca (default) or raw
#   --skip-fpca             do not rerun the three fPCA steps (use the existing output_pca)
#   --annotation-xlsx F     orchestration annotation workbook for feature set 4

# Initialize environment
eval "$(mamba shell hook --shell bash)"
mamba activate fpca-env

# =====================================================================
# Configuration (fixed paths) and flags
# =====================================================================
OUTPUT_DIR_RAW=/mnt/my_passport/R_output          # raw_data_train / raw_data_test live here
METADATA=status_fMRI_participants_05102026.csv
HYPERPARAMS=ml_hyperparameters.json
TEST_PAIRS=test_pairs.json

SETS="0 1 3 4"
FILTER_ARGS=""
ENCODING_SOURCE=fpca
SKIP_FPCA=""
RAW=""
ANNOTATION_XLSX=visualize_05102026.xlsx

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --sets) SETS="$2"; shift 2 ;;
        --filter) FILTER_ARGS="--use-nilearn-filter --n-compcor-nilearn-filter 2 --smoothing-fwhm-nilearn-filter 4"; shift 1 ;;
        --raw) RAW="1"; shift 1 ;;
        --encoding-source) ENCODING_SOURCE="$2"; shift 2 ;;
        --skip-fpca) SKIP_FPCA="1"; shift 1 ;;
        --annotation-xlsx) ANNOTATION_XLSX="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

if [ -n "$RAW" ]; then
    ENCODING_SOURCE=raw
fi

if [ -n "$FILTER_ARGS" ]; then
    OUTPUT_DIR=/mnt/my_passport/R_output/output_filter1
else
    OUTPUT_DIR=/mnt/my_passport/R_output/output_no_filter
fi
mkdir -p "$OUTPUT_DIR"
echo "Output: $OUTPUT_DIR | sets: $SETS | filter: '${FILTER_ARGS:-none}' | raw: ${RAW:-no} | set-4 source: $ENCODING_SOURCE | skip fPCA: ${SKIP_FPCA:-no}"

#./create_raw_data_links_train_test.sh

# =====================================================================
# fPCA
# =====================================================================
if [ -z "$SKIP_FPCA" ]; then
    # 1. Run individual PCA (training and test subjects, same output tree)
    ./run_fpca.sh pca-singles "$OUTPUT_DIR_RAW/raw_data_train" "$OUTPUT_DIR/output_pca" $FILTER_ARGS > "$OUTPUT_DIR/fpca_run-train.log" 2>&1 || exit 1
    ./run_fpca.sh pca-singles "$OUTPUT_DIR_RAW/raw_data_test"  "$OUTPUT_DIR/output_pca" $FILTER_ARGS > "$OUTPUT_DIR/fpca_run-test.log"  2>&1 || exit 1

    # 2. Train group PCA (training subjects only)
    ./run_fpca.sh train-pca-group "$OUTPUT_DIR_RAW/raw_data_train" "$OUTPUT_DIR/output_pca" $FILTER_ARGS > "$OUTPUT_DIR/fpca_run-group-train.log" 2>&1 || exit 1

    # 3. Project test data onto the group PCA
    ./run_fpca.sh test-pca-project "$OUTPUT_DIR_RAW/raw_data_test" "$OUTPUT_DIR/output_pca" $FILTER_ARGS > "$OUTPUT_DIR/fpca_run-group-test.log" 2>&1 || exit 1
fi

# =====================================================================
# Downstream pipeline
# =====================================================================
# run_set <feature sets> <pc indices> <log tag> [extra run_pipeline.sh flags...]
# Runs the five modes in order for the given feature sets; the extra flags are
# appended to every mode (e.g. --max-pc 0, --annotation-xlsx ...).
# With --raw, the raw training data are passed to align-signals/ml/stat and the
# raw test data to ml-test/stat-test (the pipeline loads the BOLD series itself).
run_set() {
    local SETS_="$1"; local PCS="$2"; local TAG="$3"; shift 3
    local COMMON=(--input-dir "$OUTPUT_DIR/output_pca"
                  --output-dir "$OUTPUT_DIR/output_pipeline"
                  --features-sets "$SETS_"
                  --pc-index "$PCS"
                  --metadata-file "$METADATA"
                  --hyperparameters-file "$HYPERPARAMS"
                  --test-pairs-file "$TEST_PAIRS"
                  "$@")
    local RAW_TRAIN=() RAW_TEST=()
    if [ -n "$RAW" ]; then
        RAW_TRAIN=(--raw-data-path "$OUTPUT_DIR_RAW/raw_data_train" $FILTER_ARGS)
        RAW_TEST=(--raw-data-path "$OUTPUT_DIR_RAW/raw_data_test" $FILTER_ARGS)
        TAG="${TAG}-raw"
    fi

    # 4. Align signals / RSA
    ./run_pipeline.sh --mode "align-signals" "${COMMON[@]}" "${RAW_TRAIN[@]}" > "$OUTPUT_DIR/pipeline_${TAG}_1_align.log"     2>&1 || exit 1
    # 5. Train Machine Learning models
    ./run_pipeline.sh --mode "ml"            "${COMMON[@]}" "${RAW_TRAIN[@]}" > "$OUTPUT_DIR/pipeline_${TAG}_2_ml.log"        2>&1 || exit 1
    # 6. Run statistics (training side)
    ./run_pipeline.sh --mode "stat"          "${COMMON[@]}" "${RAW_TRAIN[@]}" > "$OUTPUT_DIR/pipeline_${TAG}_3_stat.log"      2>&1 || exit 1
    # 7. Test Machine Learning models
    ./run_pipeline.sh --mode "ml-test"       "${COMMON[@]}" "${RAW_TEST[@]}"  > "$OUTPUT_DIR/pipeline_${TAG}_4_ml-test.log"   2>&1 || exit 1
    # 8. Run statistics (training + test)
    ./run_pipeline.sh --mode "stat-test"     "${COMMON[@]}" "${RAW_TEST[@]}"  > "$OUTPUT_DIR/pipeline_${TAG}_5_stat-test.log" 2>&1 || exit 1
}

# Feature sets 0 and 1: one run per PC (0 and 1)
SETS01=""
for s in $SETS; do [[ "$s" == "0" || "$s" == "1" ]] && SETS01="$SETS01 $s"; done
if [ -n "$SETS01" ]; then
    if [ -n "$RAW" ]; then
        # raw mode: no principal components, one run
        run_set "$SETS01" "0" "set01"
    else
        run_set "$SETS01" "0 1" "set01"
    fi
fi

# Feature set 3 (loadings of all PCs): the PC index does not apply -> one run, no per-PC alignment loop
if [[ " $SETS " == *" 3 "* ]]; then
    if [ -n "$RAW" ]; then
        echo "Feature set 3 needs the fPCA output and is skipped in --raw mode."
    else
        run_set "3" "0" "set3" --max-pc 0
    fi
fi

# Feature set 4 (encoding betas on the stimulus regressors): one run
if [[ " $SETS " == *" 4 "* ]]; then
    if [ "$ENCODING_SOURCE" == "raw" ]; then
        # raw signal: the pipeline loads the BOLD series itself; training data for align/ml/stat,
        # test data for ml-test/stat-test, so the modes are run separately
        COMMON4=(--input-dir "$OUTPUT_DIR/output_pca" --output-dir "$OUTPUT_DIR/output_pipeline"
                 --features-sets "4" --pc-index "0" --metadata-file "$METADATA"
                 --hyperparameters-file "$HYPERPARAMS" --test-pairs-file "$TEST_PAIRS"
                 --annotation-xlsx "$ANNOTATION_XLSX" --encoding-source raw $FILTER_ARGS)
        ./run_pipeline.sh --mode "align-signals" "${COMMON4[@]}" --raw-data-path "$OUTPUT_DIR_RAW/raw_data_train" > "$OUTPUT_DIR/pipeline_set4_1_align.log"     2>&1 || exit 1
        ./run_pipeline.sh --mode "ml"            "${COMMON4[@]}" --raw-data-path "$OUTPUT_DIR_RAW/raw_data_train" > "$OUTPUT_DIR/pipeline_set4_2_ml.log"        2>&1 || exit 1
        ./run_pipeline.sh --mode "stat"          "${COMMON4[@]}" --raw-data-path "$OUTPUT_DIR_RAW/raw_data_train" > "$OUTPUT_DIR/pipeline_set4_3_stat.log"      2>&1 || exit 1
        ./run_pipeline.sh --mode "ml-test"       "${COMMON4[@]}" --raw-data-path "$OUTPUT_DIR_RAW/raw_data_test"  > "$OUTPUT_DIR/pipeline_set4_4_ml-test.log"   2>&1 || exit 1
        ./run_pipeline.sh --mode "stat-test"     "${COMMON4[@]}" --raw-data-path "$OUTPUT_DIR_RAW/raw_data_test"  > "$OUTPUT_DIR/pipeline_set4_5_stat-test.log" 2>&1 || exit 1
    else
        # subject-level fPCA reconstruction: everything comes from output_pca
        run_set "4" "0" "set4" --max-pc 0 --annotation-xlsx "$ANNOTATION_XLSX" --encoding-source fpca
    fi
fi

# Finished
echo "Full pipeline executed successfully."
