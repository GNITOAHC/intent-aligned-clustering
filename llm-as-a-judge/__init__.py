"""
LLM-as-a-Judge: A structured, reproducible LLM-based evaluation framework
for intent-aligned clustering.

This package provides tools to evaluate clustering quality across five dimensions:
- A. Intent Alignment
- B. Intra-cluster Coherence
- C. Inter-cluster Distinctness
- D. Label Quality
- E. Coverage & Analytical Usefulness
"""

from .rubric import Score, Dimension, DIMENSIONS, DEFAULT_WEIGHTS
from .report import DimensionScore, EvaluationResult
from .sampler import SamplerConfig, ClusterSampler
from .evaluator import ClusterJudge

__all__ = [
    # Rubric
    "Score",
    "Dimension",
    "DIMENSIONS",
    "DEFAULT_WEIGHTS",
    # Report
    "DimensionScore",
    "EvaluationResult",
    # Sampler
    "SamplerConfig",
    "ClusterSampler",
    # Evaluator
    "ClusterJudge",
]

__version__ = "0.1.0"
