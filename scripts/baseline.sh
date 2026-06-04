#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: bash scripts/baseline.sh <doc_csv> <prompt_file> <output_dir>" >&2
    exit 2
fi

DOC_CSV=$1
PROMPT_FILE=$2
OUTPUT_DIR=$3

PROMPT_STR=$(cat "$PROMPT_FILE")

echo "Running baseline clustering..."
echo "Document CSV: $DOC_CSV"
echo "Output Directory: $OUTPUT_DIR"
echo "Prompt: $PROMPT_STR"


uv run python -m src.baseline -o "$OUTPUT_DIR" -d "$DOC_CSV" -p "$PROMPT_STR"

# bash scripts/baseline.sh sample.csv prompt.txt baseline_out
