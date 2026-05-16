"""
LJS evaluation for ablation experiment outputs produced by run_ablation.sh.

Discovers all ./out/ablation_* directories, runs ClusterJudge.evaluate() on
each, and writes one JSON line per experiment to LJS_ablation.log.

Usage:
    uv run python run_ljs_ablation.py [--model gpt-oss-120b] [--verbose]
"""

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT  = REPO_ROOT / "out"
LOG_FILE  = REPO_ROOT / "LJS_ablation.log"

KNOWN_DATASETS = {"20news", "arxiv", "gomotion", "sarcasm", "yahoo"}


# ── load llm-as-a-judge via importlib (hyphen prevents normal import) ──────────
def _load_ljs_package():
    pkg_dir  = REPO_ROOT / "llm-as-a-judge"
    pkg_name = "llm_as_a_judge"

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    pkg_spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(pkg_dir / "__init__.py"),
        submodule_search_locations=[str(pkg_dir)],
    )
    pkg = importlib.util.module_from_spec(pkg_spec)
    pkg.__path__    = [str(pkg_dir)]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg

    for sub in ("rubric", "prompts", "sampler", "report", "evaluator"):
        full = f"{pkg_name}.{sub}"
        spec = importlib.util.spec_from_file_location(full, str(pkg_dir / f"{sub}.py"))
        mod  = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg_name
        sys.modules[full] = mod
        spec.loader.exec_module(mod)

    pkg_spec.loader.exec_module(pkg)
    return pkg


# ── helpers ────────────────────────────────────────────────────────────────────
def parse_ablation_dir(name: str):
    """
    Parse condition / dataset / seed from directory names such as:
      ablation_no_refinement_20news_seed42
      ablation_no_postproc_gomotion_seed44

    Returns (condition, dataset, seed_int) or (None, None, None) on failure.
    """
    parts = name.split("_")
    # strip leading "ablation"
    if parts[0] != "ablation":
        return None, None, None

    dataset = next((p for p in parts if p in KNOWN_DATASETS), None)
    if dataset is None:
        return None, None, None

    ds_idx    = parts.index(dataset)
    condition = "_".join(parts[1:ds_idx])          # e.g. "no_refinement"
    seed_part = next((p for p in parts[ds_idx+1:] if p.startswith("seed")), None)
    seed      = int(seed_part[4:]) if seed_part else None

    return condition, dataset, seed


def already_evaluated(exp_dir: Path) -> bool:
    return (exp_dir / "ljs.json").exists()


def print_summary(log_path: Path):
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if '"final_score"' in line:
                rows.append(json.loads(line))

    if not rows:
        print("(no completed evaluations in log)")
        return

    scores = defaultdict(list)
    for r in rows:
        key = (r.get("condition", "?"), r.get("dataset", "?"))
        scores[key].append(r["final_score"])

    import statistics
    print(f"\n{'='*65}")
    print("LJS Summary — mean ± std  (condition / dataset / n)")
    print(f"{'='*65}")
    for (cond, ds), vals in sorted(scores.items()):
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {cond:22s}  {ds:12s}  {m:.3f} ± {s:.3f}  (n={len(vals)})")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LJS evaluation for ablation runs")
    parser.add_argument(
        "--model", "-m",
        default=os.environ.get("JUDGE_MODEL", "gpt-oss-120b"),
        help="Judge model (default: env JUDGE_MODEL or gpt-oss-120b)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip dirs that already have ljs.json (default: True)",
    )
    args = parser.parse_args()

    pkg          = _load_ljs_package()
    ClusterJudge = pkg.ClusterJudge
    judge        = ClusterJudge(model=args.model, verbose=args.verbose)

    ablation_dirs = sorted(
        d for d in OUT_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("ablation_")
    )

    if not ablation_dirs:
        print(f"No ablation directories found under {OUT_ROOT}")
        sys.exit(0)

    print(f"Found {len(ablation_dirs)} ablation directories")
    print(f"Judge model : {args.model}")
    print(f"Log file    : {LOG_FILE}\n")

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        for i, exp_dir in enumerate(ablation_dirs, 1):
            prefix = f"[{i}/{len(ablation_dirs)}] {exp_dir.name}"

            out_csv = exp_dir / "out.csv"
            if not out_csv.exists():
                print(f"{prefix}  SKIP — no out.csv")
                continue

            if args.resume and already_evaluated(exp_dir):
                print(f"{prefix}  SKIP — already evaluated (ljs.json exists)")
                continue

            condition, dataset, seed = parse_ablation_dir(exp_dir.name)
            if dataset is None:
                print(f"{prefix}  SKIP — cannot parse dataset from dir name")
                continue

            intent_file = DATA_ROOT / dataset / "guided_proxy_intent.txt"
            if not intent_file.exists():
                print(f"{prefix}  SKIP — missing {intent_file}")
                continue

            print(f"{prefix}  condition={condition}  ds={dataset}  seed={seed}")

            try:
                result = judge.evaluate(
                    output_csv=str(out_csv),
                    intent_file=str(intent_file),
                )
                row = result.to_dict()
                print(f"  -> final_score={row['final_score']:.3f}  "
                      f"A={row['dimensions']['A']['score']}  "
                      f"B={row['dimensions']['B']['score']}  "
                      f"C={row['dimensions']['C']['score']}  "
                      f"D={row['dimensions']['D']['score']}  "
                      f"E={row['dimensions']['E']['score']}")

                # save per-experiment result alongside eval.json
                (exp_dir / "ljs.json").write_text(
                    json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as e:
                row = {"error": str(e)}
                print(f"  -> ERROR: {e}")

            row.update({
                "exp_dir":    str(exp_dir),
                "condition":  condition,
                "dataset":    dataset,
                "seed":       seed,
                "judge_model": args.model,
            })
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()

    print(f"\nDone. Results appended to {LOG_FILE}")
    print_summary(LOG_FILE)


if __name__ == "__main__":
    main()
