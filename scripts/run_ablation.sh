#!/usr/bin/env bash
# Refinement Loop Ablation
#   Condition 1: IAC w/o refinement  (--max_rounds 1, full post-processing)
#   Condition 2: IAC w/o post-proc.  (--max_rounds 5, --no_postproc)
# Requires: OUTER_MEDUSA_ENDPOINT and OUTER_MEDUSA_API_KEY in .env
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

[[ -f .env ]] && set -a && source .env && set +a

MODEL="gpt-oss-120b"
SEEDS=(42 43 44)
DATA_ROOT="./data"
DATASETS=(20news arxiv gomotion sarcasm yahoo)
MAX_JOBS=3

# ----------------------------------------------------------------
# Condition 1: IAC w/o refinement (max_rounds=1, full post-proc)
# ----------------------------------------------------------------
run_no_refinement() {
    local dataset=$1 seed=$2
    local doc_csv="${DATA_ROOT}/${dataset}/sample.csv"
    local prompt_file="${DATA_ROOT}/${dataset}/guided_proxy_intent.txt"
    local gt_file="${DATA_ROOT}/${dataset}/sample_gt.csv"
    local output_dir="./out/ablation_no_refinement_${dataset}_seed${seed}"

    [[ ! -f "$doc_csv" ]]     && { echo "SKIP $dataset/$seed: no sample.csv"; return; }
    [[ ! -f "$prompt_file" ]] && { echo "SKIP $dataset/$seed: no guided_proxy_intent.txt"; return; }

    echo "[no_refinement] $dataset seed${seed} -> $output_dir"
    uv run python -m src.main \
        -d "$doc_csv" \
        -p "$(cat "$prompt_file")" \
        -o "$output_dir" \
        -m "$MODEL" \
        --max_rounds 1 \
        --seed "$seed"

    [[ -f "$gt_file" ]] && \
        uv run python -m src.evaluate \
            --pred "$output_dir/out.csv" \
            --ground "$gt_file" \
            --output "$output_dir/eval.json"
}

# ----------------------------------------------------------------
# Condition 2: IAC w/o post-processing (max_rounds=5, --no_postproc)
# ----------------------------------------------------------------
run_no_postproc() {
    local dataset=$1 seed=$2
    local doc_csv="${DATA_ROOT}/${dataset}/sample.csv"
    local prompt_file="${DATA_ROOT}/${dataset}/guided_proxy_intent.txt"
    local gt_file="${DATA_ROOT}/${dataset}/sample_gt.csv"
    local output_dir="./out/ablation_no_postproc_${dataset}_seed${seed}"

    [[ ! -f "$doc_csv" ]]     && { echo "SKIP $dataset/$seed: no sample.csv"; return; }
    [[ ! -f "$prompt_file" ]] && { echo "SKIP $dataset/$seed: no guided_proxy_intent.txt"; return; }

    echo "[no_postproc] $dataset seed${seed} -> $output_dir"
    uv run python -m src.main \
        -d "$doc_csv" \
        -p "$(cat "$prompt_file")" \
        -o "$output_dir" \
        -m "$MODEL" \
        --max_rounds 5 \
        --no_postproc \
        --seed "$seed"

    [[ -f "$gt_file" ]] && \
        uv run python -m src.evaluate \
            --pred "$output_dir/out.csv" \
            --ground "$gt_file" \
            --output "$output_dir/eval.json"
}

# ----------------------------------------------------------------
# Run Condition 1
# ----------------------------------------------------------------
echo "=== Launching Condition 1: IAC w/o refinement (max_rounds=1) ==="
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_no_refinement "$dataset" "$seed" &
        while (( $(jobs -r | wc -l) >= MAX_JOBS )); do sleep 2; done
    done
done
wait
echo "=== Condition 1 done ==="

# ----------------------------------------------------------------
# Run Condition 2
# ----------------------------------------------------------------
echo ""
echo "=== Launching Condition 2: IAC w/o post-processing ==="
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_no_postproc "$dataset" "$seed" &
        while (( $(jobs -r | wc -l) >= MAX_JOBS )); do sleep 2; done
    done
done
wait
echo "=== Condition 2 done ==="

# ----------------------------------------------------------------
# Aggregate ACC/NMI/ARI for both conditions
# ----------------------------------------------------------------
echo ""
echo "=== Computing aggregate metrics ==="
uv run python3 - <<'PYEOF'
import json, os, statistics

RESULTS_DIR  = "./out"
all_datasets = ["20news", "arxiv", "yahoo", "gomotion", "sarcasm"]

conditions = {
    "no_refinement": "ablation_no_refinement",
    "no_postproc":   "ablation_no_postproc",
}

for cond_name, prefix in conditions.items():
    print(f"\n{'='*60}")
    print(f"IAC w/o {cond_name.replace('_', ' ')} — per-dataset mean ± std")
    print(f"{'='*60}")
    for ds in all_datasets:
        evals = []
        for seed in [42, 43, 44]:
            p = f"{RESULTS_DIR}/{prefix}_{ds}_seed{seed}/eval.json"
            if os.path.exists(p):
                with open(p) as f:
                    evals.append(json.load(f))
        if not evals:
            print(f"  {ds:12s}: no data")
            continue
        def ms(vals):
            return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0)
        am, as_ = ms([e.get("Hungarian", 0) for e in evals])
        nm, ns  = ms([e.get("NMI", 0)       for e in evals])
        rm, rs  = ms([e.get("ARI", 0)       for e in evals])
        print(f"  {ds:12s}: ACC={am:.2f}±{as_:.2f}  NMI={nm:.2f}±{ns:.2f}  ARI={rm:.2f}±{rs:.2f}")
PYEOF
