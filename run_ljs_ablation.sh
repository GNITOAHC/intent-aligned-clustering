#!/usr/bin/env bash
# LJS (LLM-as-a-Judge) evaluation for ablation experiment outputs.
# Evaluates all ./out/ablation_* directories produced by run_ablation.sh.
#
# Run from: intent-aligned-clustering/
# Requires: OUTER_MEDUSA_ENDPOINT and OUTER_MEDUSA_API_KEY in .env
#
# Override judge model:
#   JUDGE_MODEL=gpt-4o bash run_ljs_ablation.sh
set -euo pipefail

[[ -f .env ]] && set -a && source .env && set +a

export JUDGE_MODEL="${JUDGE_MODEL:-gpt-oss-120b}"

echo "=== LJS Ablation Evaluation ==="
echo "Judge model : ${JUDGE_MODEL}"
echo "Log file    : ./LJS_ablation.log"
echo ""

uv run python run_ljs_ablation.py --verbose "$@"
