# llm-as-a-judge

A structured, reproducible LLM-based evaluation framework for intent-aligned clustering.

## Rubric (0–5 per dimension)

The judge evaluates clustering quality across five structured dimensions:

| Dimension                           | Description                                                                |
| ----------------------------------- | -------------------------------------------------------------------------- |
| A. Intent Alignment                 | Do clusters reflect the provided intent?                                   |
| B. Intra-cluster Coherence          | Are items within a cluster semantically consistent?                        |
| C. Inter-cluster Distinctness       | Are clusters meaningfully different from each other?                       |
| D. Label Quality                    | Are cluster labels accurate, concise, and informative?                     |
| E. Coverage & Analytical Usefulness | Do clusters provide useful structure for downstream reasoning or analysis? |

Each dimension is scored 0–5 using a fixed rubric.

## Final Integrated Score

The final score is computed as a weighted aggregate:

```
FinalScore = wA*A + wB*B + wC*C + wD*D + wE*E
```

Where:

- A–E correspond to rubric dimensions
- Default weights are uniform (0.2 each)
- Custom weights can be specified in config

This enables:

- Cross-method comparability
- Controlled ablations
- Task-dependent emphasis
