#!/usr/bin/env bash
# IAC ablation — Condition 2: w/o post-processing (max_rounds=5, --no_postproc)
# Run from: intent-aligned-clustering/
# Requires: OUTER_MEDUSA_ENDPOINT and OUTER_MEDUSA_API_KEY in .env
set -euo pipefail

[[ -f .env ]] && set -a && source .env && set +a

MODEL="gpt-oss-120b"
SEEDS=(42 43 44)
DATA_ROOT="./data"
DATASETS=(20news arxiv gomotion sarcasm yahoo)
MAX_JOBS=3

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

echo "=== IAC w/o post-processing ablation ==="
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_no_postproc "$dataset" "$seed" &
        while (( $(jobs -r | wc -l) >= MAX_JOBS )); do sleep 2; done
    done
done
wait
echo "=== Done ==="

echo ""
echo "=== Aggregate metrics ==="
uv run python3 - <<'PYEOF'
import json, os, statistics

RESULTS_DIR  = "./out"
all_datasets = ["20news", "arxiv", "yahoo", "gomotion", "sarcasm"]

print(f"{'dataset':12s}  ACC            NMI            ARI")
for ds in all_datasets:
    evals = []
    for seed in [42, 43, 44]:
        p = f"{RESULTS_DIR}/ablation_no_postproc_{ds}_seed{seed}/eval.json"
        if os.path.exists(p):
            with open(p) as f:
                evals.append(json.load(f))
    if not evals:
        print(f"{ds:12s}  no data")
        continue
    def ms(vals):
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0)
    am, as_ = ms([e.get("Hungarian", 0) for e in evals])
    nm, ns  = ms([e.get("NMI", 0)       for e in evals])
    rm, rs  = ms([e.get("ARI", 0)       for e in evals])
    print(f"{ds:12s}  {am:.2f}±{as_:.2f}  {nm:.2f}±{ns:.2f}  {rm:.2f}±{rs:.2f}")
PYEOF
