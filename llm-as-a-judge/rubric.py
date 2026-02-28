"""
Rubric definitions for LLM-as-a-Judge evaluation.

Defines the scoring scale (0-5), the five evaluation dimensions,
and default weights for computing the final integrated score.
"""

from dataclasses import dataclass
from enum import IntEnum


class Score(IntEnum):
    """
    Scoring scale for each evaluation dimension.

    0 = Complete failure / Does not meet criterion at all
    1 = Severely lacking
    2 = Below expectations
    3 = Meets basic expectations
    4 = Good quality
    5 = Excellent / Fully meets criterion
    """

    FAIL = 0
    SEVERELY_LACKING = 1
    BELOW_EXPECTATIONS = 2
    MEETS_BASIC = 3
    GOOD = 4
    EXCELLENT = 5


@dataclass(frozen=True)
class Dimension:
    """
    Represents an evaluation dimension.

    Attributes:
        key: Single letter identifier (A-E)
        name: Human-readable dimension name
        description: Brief description of what this dimension measures
    """

    key: str
    name: str
    description: str


# The five evaluation dimensions as defined in the README
DIMENSIONS: list[Dimension] = [
    Dimension(
        key="A",
        name="Intent Alignment",
        description="Do clusters reflect the provided intent?",
    ),
    Dimension(
        key="B",
        name="Intra-cluster Coherence",
        description="Are items within a cluster semantically consistent?",
    ),
    Dimension(
        key="C",
        name="Inter-cluster Distinctness",
        description="Are clusters meaningfully different from each other?",
    ),
    Dimension(
        key="D",
        name="Label Quality",
        description="Are cluster labels accurate, concise, and informative?",
    ),
    Dimension(
        key="E",
        name="Coverage & Analytical Usefulness",
        description="Do clusters provide useful structure for downstream reasoning or analysis?",
    ),
]

# Mapping from dimension key to Dimension object
DIMENSION_MAP: dict[str, Dimension] = {d.key: d for d in DIMENSIONS}

# Default weights for computing the final integrated score
# All dimensions are weighted equally by default
DEFAULT_WEIGHTS: dict[str, float] = {
    "A": 0.2,
    "B": 0.2,
    "C": 0.2,
    "D": 0.2,
    "E": 0.2,
}


def validate_weights(weights: dict[str, float]) -> bool:
    """
    Validate that weights are valid:
    - All dimension keys (A-E) are present
    - All values are non-negative
    - Weights sum to 1.0 (with tolerance)

    Args:
        weights: Dictionary mapping dimension keys to weight values

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = set(DEFAULT_WEIGHTS.keys())
    provided_keys = set(weights.keys())

    if provided_keys != required_keys:
        missing = required_keys - provided_keys
        extra = provided_keys - required_keys
        raise ValueError(f"Invalid weight keys. Missing: {missing}, Extra: {extra}")

    for key, value in weights.items():
        if value < 0:
            raise ValueError(f"Weight for {key} cannot be negative: {value}")

    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0, got {total}")

    return True


def get_score_description(score: int) -> str:
    """
    Get a human-readable description for a score value.

    Args:
        score: Integer score from 0-5

    Returns:
        Description string
    """
    descriptions = {
        0: "Complete failure",
        1: "Severely lacking",
        2: "Below expectations",
        3: "Meets basic expectations",
        4: "Good quality",
        5: "Excellent",
    }
    return descriptions.get(score, "Unknown")
