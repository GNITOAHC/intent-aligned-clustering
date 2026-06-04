#!/usr/bin/env bash
# LJS (LLM-as-a-Judge) evaluation for ablation experiment outputs.
# Evaluates all ./out/ablation_* directories produced by scripts/run_ablation.sh.
#
# Requires: OUTER_MEDUSA_ENDPOINT and OUTER_MEDUSA_API_KEY in .env
#
# Override judge model:
#   JUDGE_MODEL=gpt-4o bash scripts/run_ljs_ablation.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

[[ -f .env ]] && set -a && source .env && set +a

export JUDGE_MODEL="${JUDGE_MODEL:-gpt-5}"

echo "=== LJS Ablation Evaluation ==="
echo "Judge model : ${JUDGE_MODEL}"
echo "Log file    : ./LJS_ablation.log"
echo ""

uv run python llm-as-a-judge/run_ljs_ablation.py --verbose "$@"
