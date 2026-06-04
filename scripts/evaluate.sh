#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: bash scripts/evaluate.sh <pred_dir> <ground_truth_file>" >&2
    exit 2
fi

PRED_DIR=$1
GROUND_TRUTH_FILE=$2
OUTPUT_FILE="${PRED_DIR}/evaluation.json"

uv run python -m src.evaluate -p "$PRED_DIR/out.csv" -g "$GROUND_TRUTH_FILE" -o "$OUTPUT_FILE"

# bash scripts/evaluate.sh out_gpt-oss-20b sample_gt.csv
