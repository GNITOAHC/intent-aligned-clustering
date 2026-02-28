# llm-as-a-judge

A structured, reproducible LLM-based evaluation framework for intent-aligned clustering.

## Installation

This package is part of the `intent-aligned-clustering` project. Ensure you have the project dependencies installed:

```bash
uv sync
```

## Quick Start

```bash
# Basic evaluation
uv run python -m llm-as-a-judge \
    --output out/out_gpt-oss-20b/out.csv \
    --intent data/arxiv/prompt.txt

# With custom model and verbose output
uv run python -m llm-as-a-judge \
    --output out/out_gpt-oss-20b/out.csv \
    --intent data/arxiv/prompt.txt \
    --model gpt-4o \
    --verbose
```

## Rubric (0-5 per dimension)

The judge evaluates clustering quality across five structured dimensions:

| Dimension                           | Description                                                                |
| ----------------------------------- | -------------------------------------------------------------------------- |
| A. Intent Alignment                 | Do clusters reflect the provided intent?                                   |
| B. Intra-cluster Coherence          | Are items within a cluster semantically consistent?                        |
| C. Inter-cluster Distinctness       | Are clusters meaningfully different from each other?                       |
| D. Label Quality                    | Are cluster labels accurate, concise, and informative?                     |
| E. Coverage & Analytical Usefulness | Do clusters provide useful structure for downstream reasoning or analysis? |

### Scoring Scale

| Score | Description              |
| ----- | ------------------------ |
| 0     | Complete failure         |
| 1     | Severely lacking         |
| 2     | Below expectations       |
| 3     | Meets basic expectations |
| 4     | Good quality             |
| 5     | Excellent                |

## Final Integrated Score

The final score is computed as a weighted aggregate:

```
FinalScore = wA*A + wB*B + wC*C + wD*D + wE*E
```

Where:

- A-E correspond to rubric dimensions
- Default weights are uniform (0.2 each)
- Custom weights can be specified via CLI

This enables:

- Cross-method comparability
- Controlled ablations
- Task-dependent emphasis

## CLI Usage

```
usage: llm-as-a-judge [-h] --output OUTPUT --intent INTENT [--model {gpt-oss-20b,gpt-oss-120b,Google-Gemma-3-27B,Llama-3.1-70B,gpt-4o-mini,gpt-4o,gpt-3.5-turbo}]
                      [--weights WEIGHTS] [--sample-size SAMPLE_SIZE] [--max-clusters MAX_CLUSTERS] [--max-text-length MAX_TEXT_LENGTH] [--seed SEED] [--retries RETRIES]
                      [--format {json,markdown}] [--save SAVE] [--verbose]

Evaluate clustering quality using LLM-based judgment

options:
  -h, --help            show this help message and exit
  --output, -o OUTPUT   Path to clustering output CSV (columns: id, label, text)
  --intent, -i INTENT   Path to intent/prompt file describing the clustering goal
  --model, -m {gpt-oss-20b,gpt-oss-120b,Google-Gemma-3-27B,Llama-3.1-70B,gpt-4o-mini,gpt-4o,gpt-3.5-turbo}
                        LLM model to use for evaluation (default: gpt-4o-mini)
  --weights, -w WEIGHTS
                        Custom dimension weights: 'A=0.3,B=0.2,C=0.2,D=0.15,E=0.15' (must sum to 1.0)
  --sample-size, -s SAMPLE_SIZE
                        Max items to sample per cluster (default: 5)
  --max-clusters, -c MAX_CLUSTERS
                        Max clusters to sample for coherence evaluation (default: 5)
  --max-text-length MAX_TEXT_LENGTH
                        Max character length for each text item (default: 500)
  --seed SEED           Random seed for reproducibility (default: 42)
  --retries RETRIES     Max LLM retry attempts (default: 3)
  --format, -f {json,markdown}
                        Output format (default: json)
  --save SAVE           Save results to file (path)
  --verbose, -v         Print progress information
```

## Examples

### Basic Evaluation

```bash
python -m llm-as-a-judge \
    --output out/out_gpt-oss-20b/out.csv \
    --intent data/arxiv/prompt.txt
```

### Custom Weights (Emphasize Intent Alignment)

```bash
python -m llm-as-a-judge \
    --output out/out_gpt-oss-20b/out.csv \
    --intent data/arxiv/prompt.txt \
    --weights "A=0.3,B=0.2,C=0.2,D=0.15,E=0.15"
```

### Save Markdown Report

```bash
python -m llm-as-a-judge \
    --output out/out_gpt-oss-20b/out.csv \
    --intent data/arxiv/prompt.txt \
    --format markdown \
    --save evaluation_report.md
```

### Different Model with More Samples

```bash
python -m llm-as-a-judge \
    --output out/out_gpt-oss-20b/out.csv \
    --intent data/arxiv/prompt.txt \
    --model gpt-4o \
    --sample-size 10 \
    --max-clusters 8
```

## Output Format

### JSON Output (Default)

```json
{
  "dataset": "out_gpt-oss-20b",
  "model_used": "gpt-4o-mini",
  "timestamp": "2026-02-28T14:30:00",
  "dimensions": {
    "A": {
      "dimension": "A",
      "name": "Intent Alignment",
      "score": 4,
      "score_description": "Good quality",
      "reasoning": "The clustering largely reflects the intent..."
    },
    "B": { ... },
    "C": { ... },
    "D": { ... },
    "E": { ... }
  },
  "weights": {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2},
  "final_score": 4.2,
  "token_usage": {
    "prompt_tokens": 8500,
    "completion_tokens": 1200,
    "total_tokens": 9700
  },
  "metadata": {
    "output_csv": "out/out_gpt-oss-20b/out.csv",
    "intent_file": "data/arxiv/prompt.txt",
    "num_clusters": 11,
    "total_items": 504
  }
}
```

### Markdown Output

The markdown report includes:

- Summary with final score
- Dimension scores table
- Detailed reasoning for each dimension
- Weights configuration
- Token usage statistics

## Programmatic Usage

```python
import sys
sys.path.insert(0, '/path/to/intent-aligned-clustering')

# Import using importlib for hyphenated package name
import importlib
judge_pkg = importlib.import_module('llm-as-a-judge')

ClusterJudge = judge_pkg.ClusterJudge

# Create judge
judge = ClusterJudge(
    model="gpt-4o-mini",
    weights={"A": 0.3, "B": 0.2, "C": 0.2, "D": 0.15, "E": 0.15},
    max_clusters_to_sample=5,
    max_items_per_cluster=5,
    verbose=True,
)

# Run evaluation
result = judge.evaluate(
    output_csv="out/out_gpt-oss-20b/out.csv",
    intent_file="data/arxiv/prompt.txt",
)

# Access results
print(f"Final Score: {result.final_score:.2f}")
for key, score in result.dimensions.items():
    print(f"{score.name}: {score.score}/5")

# Export
print(result.to_json())
print(result.to_markdown())
```

## File Structure

```
llm-as-a-judge/
├── __init__.py       # Package exports
├── __main__.py       # Module entry point
├── judge.py          # CLI implementation
├── evaluator.py      # Core ClusterJudge class
├── prompts.py        # LLM prompt templates
├── rubric.py         # Scoring definitions
├── sampler.py        # Cluster sampling logic
├── report.py         # Result data structures
└── README.md         # This file
```

## Input Data Format

### Clustering Output CSV

The clustering output should be a CSV file with three columns:

```csv
id,label,text
0,Machine Learning,"Paper abstract about ML..."
1,Computer Vision,"Paper abstract about CV..."
```

### Intent File

A plain text file describing the clustering goal:

```
Represent the scientific articles by their primary research subfield
for clustering, for example, machine learning, Robotics, Cryptography.
Please cluster them into 9 groups.
```

## Environment Variables

The package uses the parent project's LLM configuration. Ensure these environment variables are set in your `.env` file:

- `OPENAI_API_KEY` - For OpenAI models (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
- `OUTER_MEDUSA_ENDPOINT` - For internal models (gpt-oss-20b, etc.)
- `OUTER_MEDUSA_API_KEY` - API key for internal models
