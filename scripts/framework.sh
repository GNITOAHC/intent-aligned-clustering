DOC_CSV=$1
PROMPT_FILE=$2
OUTPUT_DIR=$3
MODEL=$4

PROMPT_STR=$(cat $PROMPT_FILE)

echo "Running baseline clustering..."
echo "Document CSV: $DOC_CSV"
echo "Output Directory: $OUTPUT_DIR"
echo "Model: $MODEL"
echo "Prompt: $PROMPT_STR"


uv run python -m src.main -o $OUTPUT_DIR -d $DOC_CSV -p "$PROMPT_STR" -m $MODEL

# bash scripts/framework.sh sample.csv prompt.txt out_framework gpt-oss-20b
