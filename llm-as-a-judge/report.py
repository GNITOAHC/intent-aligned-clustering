"""
Report data structures for LLM-as-a-Judge evaluation results.

Provides dataclasses for storing evaluation scores and methods for
formatting output in JSON and Markdown formats.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

from .rubric import DIMENSION_MAP, get_score_description


@dataclass
class DimensionScore:
    """
    Score for a single evaluation dimension.

    Attributes:
        dimension: Single letter identifier (A-E)
        name: Human-readable dimension name
        score: Integer score from 0-5
        reasoning: LLM's explanation for the score
    """

    dimension: str
    name: str
    score: int
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dimension": self.dimension,
            "name": self.name,
            "score": self.score,
            "score_description": get_score_description(self.score),
            "reasoning": self.reasoning,
        }


@dataclass
class TokenUsage:
    """
    Token usage statistics for the evaluation.

    Attributes:
        prompt_tokens: Total prompt tokens used
        completion_tokens: Total completion tokens used
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for a clustering output.

    Attributes:
        dataset: Name/identifier of the evaluated dataset
        model_used: LLM model used for evaluation
        timestamp: When the evaluation was performed
        dimensions: Dictionary mapping dimension keys to DimensionScore objects
        weights: Weights used for computing final score
        final_score: Weighted aggregate score
        token_usage: Token usage statistics
        metadata: Additional metadata (clustering file path, intent, etc.)
    """

    dataset: str
    model_used: str
    timestamp: datetime
    dimensions: dict[str, DimensionScore]
    weights: dict[str, float]
    final_score: float
    token_usage: TokenUsage
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dataset": self.dataset,
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat(),
            "dimensions": {
                key: score.to_dict() for key, score in self.dimensions.items()
            },
            "weights": self.weights,
            "final_score": round(self.final_score, 2),
            "token_usage": self.token_usage.to_dict(),
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Convert to JSON string.

        Args:
            indent: Indentation level for pretty printing

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """
        Convert to Markdown report format.

        Returns:
            Markdown formatted report string
        """
        lines = [
            "# LLM-as-a-Judge Evaluation Report",
            "",
            "## Summary",
            "",
            f"- **Dataset**: {self.dataset}",
            f"- **Evaluation Model**: {self.model_used}",
            f"- **Timestamp**: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Final Score**: **{self.final_score:.2f}** / 5.00",
            "",
            "## Dimension Scores",
            "",
            "| Dimension | Score | Description |",
            "|-----------|-------|-------------|",
        ]

        # Add dimension rows
        for key in sorted(self.dimensions.keys()):
            score = self.dimensions[key]
            weight = self.weights.get(key, 0)
            lines.append(
                f"| {score.dimension}. {score.name} | "
                f"{score.score}/5 (weight: {weight:.0%}) | "
                f"{get_score_description(score.score)} |"
            )

        lines.extend(
            [
                "",
                "## Detailed Reasoning",
                "",
            ]
        )

        # Add detailed reasoning for each dimension
        for key in sorted(self.dimensions.keys()):
            score = self.dimensions[key]
            lines.extend(
                [
                    f"### {score.dimension}. {score.name}",
                    "",
                    f"**Score**: {score.score}/5 ({get_score_description(score.score)})",
                    "",
                    f"**Reasoning**: {score.reasoning}",
                    "",
                ]
            )

        lines.extend(
            [
                "## Weights Configuration",
                "",
                "```",
                f"FinalScore = "
                + " + ".join(
                    f"{self.weights[k]:.2f}*{k}" for k in sorted(self.weights.keys())
                ),
                f"           = "
                + " + ".join(
                    f"{self.weights[k]:.2f}*{self.dimensions[k].score}"
                    for k in sorted(self.weights.keys())
                ),
                f"           = {self.final_score:.2f}",
                "```",
                "",
                "## Token Usage",
                "",
                f"- Prompt tokens: {self.token_usage.prompt_tokens:,}",
                f"- Completion tokens: {self.token_usage.completion_tokens:,}",
                f"- Total tokens: {self.token_usage.total_tokens:,}",
                "",
            ]
        )

        # Add metadata if present
        if self.metadata:
            lines.extend(
                [
                    "## Metadata",
                    "",
                    "```json",
                    json.dumps(self.metadata, indent=2, ensure_ascii=False),
                    "```",
                    "",
                ]
            )

        return "\n".join(lines)


def create_evaluation_result(
    dataset: str,
    model_used: str,
    dimension_scores: dict[str, DimensionScore],
    weights: dict[str, float],
    token_usage: TokenUsage,
    metadata: dict[str, Any] | None = None,
) -> EvaluationResult:
    """
    Factory function to create an EvaluationResult with computed final score.

    Args:
        dataset: Dataset identifier
        model_used: LLM model used
        dimension_scores: Dictionary of dimension scores
        weights: Weight configuration
        token_usage: Token usage statistics
        metadata: Optional additional metadata

    Returns:
        Complete EvaluationResult object
    """
    # Compute weighted final score
    final_score = sum(
        weights.get(key, 0) * score.score for key, score in dimension_scores.items()
    )

    return EvaluationResult(
        dataset=dataset,
        model_used=model_used,
        timestamp=datetime.now(),
        dimensions=dimension_scores,
        weights=weights,
        final_score=final_score,
        token_usage=token_usage,
        metadata=metadata or {},
    )
