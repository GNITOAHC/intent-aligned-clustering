DOC_CSV=$1
PROMPT_FILE=$2
OUTPUT_DIR=$3

PROMPT_STR=$(cat $PROMPT_FILE)

echo "Running baseline clustering..."
echo "Document CSV: $DOC_CSV"
echo "Output Directory: $OUTPUT_DIR"
echo "Prompt: $PROMPT_STR"


uv run python -m src2.main -o $OUTPUT_DIR -d $DOC_CSV -p "$PROMPT_STR"
# uv version is not tested yet