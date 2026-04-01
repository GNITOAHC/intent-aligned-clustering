"""
Core evaluator for LLM-as-a-Judge clustering evaluation.

Provides the ClusterJudge class that orchestrates the evaluation of
clustering results across all five dimensions using LLM-based scoring.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.llm import get_llm_instance, LLM

from .rubric import DIMENSIONS, DIMENSION_MAP, DEFAULT_WEIGHTS, validate_weights
from .prompts import get_prompts
from .sampler import ClusterSampler, ClusterData, SamplerConfig
from .report import (
    DimensionScore,
    EvaluationResult,
    TokenUsage,
    create_evaluation_result,
)


class LLMResponseError(Exception):
    """Raised when LLM response cannot be parsed."""

    pass


class ClusterJudge:
    """
    Evaluates clustering quality using LLM-based judgment across five dimensions.

    The judge loads clustering output, samples representative items, and uses
    an LLM to score each dimension (A-E) based on structured prompts.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        weights: dict[str, float] | None = None,
        max_clusters_to_sample: int = 5,
        max_items_per_cluster: int = 5,
        max_text_length: int = 500,
        seed: int = 42,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        verbose: bool = False,
    ):
        """
        Initialize the ClusterJudge.

        Args:
            model: LLM model to use for evaluation
            weights: Custom weights for dimensions (must sum to 1.0)
            max_clusters_to_sample: Max clusters to sample for coherence evaluation
            max_items_per_cluster: Max items to sample from each cluster
            max_text_length: Max character length for each text item
            seed: Random seed for reproducibility
            max_retries: Maximum retry attempts for LLM calls
            retry_delay: Base delay between retries (uses exponential backoff)
            verbose: Print progress information
        """
        self.model_name = model
        self.llm = get_llm_instance(model)

        self.weights = weights or DEFAULT_WEIGHTS.copy()
        validate_weights(self.weights)

        self.sampler = ClusterSampler(
            SamplerConfig(
                max_clusters_to_sample=max_clusters_to_sample,
                max_items_per_cluster=max_items_per_cluster,
                max_text_length=max_text_length,
                seed=seed,
            )
        )

        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verbose = verbose

        # Track token usage across all LLM calls
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[ClusterJudge] {message}")

    def _load_clusters(self, output_csv: str) -> dict[str, ClusterData]:
        """
        Load clustering output from CSV file.

        Expected CSV format: id,label,text

        Args:
            output_csv: Path to the clustering output CSV

        Returns:
            Dictionary mapping cluster labels to ClusterData objects
        """
        clusters: dict[str, ClusterData] = {}

        with open(output_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["label"]
                item_id = row["id"]
                text = row["text"]

                if label not in clusters:
                    clusters[label] = ClusterData(label=label, items=[])

                clusters[label].items.append((item_id, text))

        return clusters

    def _load_intent(self, intent_file: str) -> str:
        """
        Load clustering intent from file.

        Args:
            intent_file: Path to the intent/prompt file

        Returns:
            Intent string
        """
        with open(intent_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _call_llm_with_retry(self, prompt: str, sys_prompt: str) -> dict[str, Any]:
        """
        Call LLM and parse JSON response with retry logic.

        Args:
            prompt: User prompt
            sys_prompt: System prompt

        Returns:
            Parsed JSON response with 'score' and 'reasoning' keys

        Raises:
            LLMResponseError: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response, prompt_tokens, completion_tokens = self.llm.generate(
                    prompt, sys_prompt
                )

                # Track token usage
                self._prompt_tokens += prompt_tokens
                self._completion_tokens += completion_tokens

                # Parse JSON response
                # Try to extract JSON from response (handle markdown code blocks)
                response_text = response.strip()

                # Remove markdown code block if present
                if response_text.startswith("```"):
                    lines = response_text.split("\n")
                    # Remove first and last lines (```json and ```)
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    response_text = "\n".join(lines)

                parsed = json.loads(response_text)

                # Validate required fields
                if "score" not in parsed:
                    raise LLMResponseError("Missing 'score' field in response")
                if "reasoning" not in parsed:
                    raise LLMResponseError("Missing 'reasoning' field in response")

                # Validate score range
                score = int(parsed["score"])
                if score < 0 or score > 5:
                    raise LLMResponseError(f"Score {score} out of range [0-5]")

                parsed["score"] = score
                return parsed

            except json.JSONDecodeError as e:
                last_error = LLMResponseError(f"Failed to parse JSON: {e}")
            except (KeyError, ValueError, TypeError) as e:
                last_error = LLMResponseError(f"Invalid response format: {e}")
            except Exception as e:
                last_error = LLMResponseError(f"LLM call failed: {e}")

            # Exponential backoff before retry
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2**attempt)
                self._log(
                    f"Retry {attempt + 1}/{self.max_retries} after {delay:.1f}s..."
                )
                time.sleep(delay)

        raise last_error or LLMResponseError("Unknown error")

    def _score_dimension(
        self,
        dimension_key: str,
        prompt_kwargs: dict[str, Any],
    ) -> DimensionScore:
        """
        Score a single dimension using LLM.

        Args:
            dimension_key: Dimension identifier (A-E)
            prompt_kwargs: Keyword arguments for prompt template formatting

        Returns:
            DimensionScore object
        """
        dimension = DIMENSION_MAP[dimension_key]
        sys_prompt, user_template = get_prompts(dimension_key)

        user_prompt = user_template.format(**prompt_kwargs)

        self._log(f"Evaluating dimension {dimension_key}: {dimension.name}")

        result = self._call_llm_with_retry(user_prompt, sys_prompt)

        return DimensionScore(
            dimension=dimension_key,
            name=dimension.name,
            score=result["score"],
            reasoning=result["reasoning"],
        )

    def _score_intent_alignment(
        self,
        clusters: dict[str, ClusterData],
        intent: str,
    ) -> DimensionScore:
        """Score dimension A: Intent Alignment."""
        # Sample items from each cluster for the prompt
        sampled = self.sampler.sample_clusters(clusters)

        return self._score_dimension(
            "A",
            {
                "intent": intent,
                "num_clusters": len(clusters),
                "cluster_summary": self.sampler.format_cluster_summary(clusters),
                "samples": self.sampler.format_cluster_samples(sampled),
            },
        )

    def _score_intra_coherence(
        self,
        clusters: dict[str, ClusterData],
    ) -> DimensionScore:
        """Score dimension B: Intra-cluster Coherence."""
        sampled = self.sampler.sample_clusters(clusters)

        return self._score_dimension(
            "B",
            {
                "cluster_samples": self.sampler.format_cluster_samples(sampled),
            },
        )

    def _score_inter_distinctness(
        self,
        clusters: dict[str, ClusterData],
    ) -> DimensionScore:
        """Score dimension C: Inter-cluster Distinctness."""
        pairs = self.sampler.sample_cluster_pairs(clusters, max_pairs=3)

        return self._score_dimension(
            "C",
            {
                "cluster_pairs": self.sampler.format_cluster_pairs(pairs),
            },
        )

    def _score_label_quality(
        self,
        clusters: dict[str, ClusterData],
        intent: str,
    ) -> DimensionScore:
        """Score dimension D: Label Quality."""
        sampled = self.sampler.sample_clusters(clusters)

        # Include original cluster sizes in the sampled data for context
        for label in sampled:
            if label in clusters:
                # Update the sampled cluster to reflect original size
                original_size = clusters[label].size
                sampled[label] = ClusterData(
                    label=label,
                    items=sampled[label].items,
                )
                # We'll format this specially in format_label_samples

        return self._score_dimension(
            "D",
            {
                "intent": intent,
                "label_samples": self.sampler.format_label_samples(sampled),
            },
        )

    def _score_coverage_usefulness(
        self,
        clusters: dict[str, ClusterData],
        intent: str,
    ) -> DimensionScore:
        """Score dimension E: Coverage & Analytical Usefulness."""
        total_items = sum(c.size for c in clusters.values())
        sampled = self.sampler.sample_clusters(clusters)

        # Format cluster distribution with percentages
        distribution_lines = []
        for label, cluster in sorted(
            clusters.items(), key=lambda x: x[1].size, reverse=True
        ):
            pct = (cluster.size / total_items) * 100
            distribution_lines.append(f"- {label}: {cluster.size} items ({pct:.1f}%)")

        return self._score_dimension(
            "E",
            {
                "intent": intent,
                "total_items": total_items,
                "num_clusters": len(clusters),
                "cluster_distribution": "\n".join(distribution_lines),
                "samples": self.sampler.format_cluster_samples(sampled),
            },
        )

    def evaluate(
        self,
        output_csv: str,
        intent_file: str,
    ) -> EvaluationResult:
        """
        Evaluate clustering quality across all five dimensions.

        Args:
            output_csv: Path to clustering output CSV (columns: id, label, text)
            intent_file: Path to intent/prompt file

        Returns:
            Complete EvaluationResult with scores for all dimensions
        """
        # Reset token counters
        self._prompt_tokens = 0
        self._completion_tokens = 0

        # Load data
        self._log(f"Loading clustering output from {output_csv}")
        clusters = self._load_clusters(output_csv)
        self._log(
            f"Loaded {len(clusters)} clusters with {sum(c.size for c in clusters.values())} total items"
        )

        self._log(f"Loading intent from {intent_file}")
        intent = self._load_intent(intent_file)

        # Score each dimension
        dimension_scores: dict[str, DimensionScore] = {}

        self._log("Starting evaluation...")

        dimension_scores["A"] = self._score_intent_alignment(clusters, intent)
        dimension_scores["B"] = self._score_intra_coherence(clusters)
        dimension_scores["C"] = self._score_inter_distinctness(clusters)
        dimension_scores["D"] = self._score_label_quality(clusters, intent)
        dimension_scores["E"] = self._score_coverage_usefulness(clusters, intent)

        # Extract dataset name from path
        dataset_name = Path(output_csv).parent.name
        if dataset_name in (".", ""):
            dataset_name = Path(output_csv).stem

        # Create result
        result = create_evaluation_result(
            dataset=dataset_name,
            model_used=self.model_name,
            dimension_scores=dimension_scores,
            weights=self.weights,
            token_usage=TokenUsage(
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
            ),
            metadata={
                "output_csv": str(output_csv),
                "intent_file": str(intent_file),
                "num_clusters": len(clusters),
                "total_items": sum(c.size for c in clusters.values()),
            },
        )

        self._log(f"Evaluation complete. Final score: {result.final_score:.2f}/5.00")

        return result
