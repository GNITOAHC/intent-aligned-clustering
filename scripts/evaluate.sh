PRED_DIR=$1
GROUND_TRUTH_FILE=$2

uv run python -m src.evaluate -p $PRED_DIR/out.csv -g $GROUND_TRUTH_FILE

# bash scripts/evaluate.sh out_gpt-oss-20b sample_gt.csv
