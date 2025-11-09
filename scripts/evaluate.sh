PRED_DIR=$1
GROUND_TRUTH_FILE=$2
OUTPUT_FILE="${PRED_DIR}/evaluation.json"

uv run python -m src.evaluate -p $PRED_DIR/out.csv -g $GROUND_TRUTH_FILE -o $OUTPUT_FILE

# bash scripts/evaluate.sh out_gpt-oss-20b sample_gt.csv
