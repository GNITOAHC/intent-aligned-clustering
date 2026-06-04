# Intent-Aligned Clustering

This repository contains the intent-aligned clustering pipeline, dataset samples,
evaluation helpers, LLM-as-a-judge scoring, and Sankey visualization utilities.

## Framework Flow

![framework](assets/framework.png)

## Common Commands

Run framework:

```bash
bash scripts/framework.sh data/sample/sample.csv data/sample/prompt.txt out/sample gpt-oss-20b
```

Baseline(kmeans):

```bash
bash scripts/baseline.sh data/sample/sample.csv data/sample/prompt.txt out/baseline_sample
```

Hard metrics:

```bash
bash scripts/evaluate.sh out/sample data/sample/sample_gt.csv
```

Soft metrics:

```bash
bash scripts/run_ljs_ablation.sh
```

Ablation:

```bash
bash scripts/run_ablation.sh
```


Convert a sankey.json for [SankeyMATIC](https://sankeymatic.com/) visualization:

```bash
uv run python sankey-visualization/analyze_sankey.py out/sample/sankey.json > sankey-visualization/sankey-diagram
```
