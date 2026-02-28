#!/usr/bin/env python3
"""
LLM-as-a-Judge: CLI entry point for evaluating clustering quality.

Usage:
    python -m llm-as-a-judge.judge --output <clustering.csv> --intent <prompt.txt>
    
Example:
    python -m llm-as-a-judge.judge \\
        --output out/out_gpt-oss-20b/out.csv \\
        --intent data/arxiv/prompt.txt \\
        --model gpt-4o-mini
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.llm import llm_choices

from .evaluator import ClusterJudge
from .rubric import DEFAULT_WEIGHTS


def parse_weights(weight_str: str) -> dict[str, float]:
    """
    Parse weight string in format 'A=0.3,B=0.2,C=0.2,D=0.15,E=0.15'.

    Args:
        weight_str: Comma-separated key=value pairs

    Returns:
        Dictionary of weights

    Raises:
        ValueError: If format is invalid
    """
    weights = {}

    try:
        for pair in weight_str.split(","):
            pair = pair.strip()
            if "=" not in pair:
                raise ValueError(
                    f"Invalid weight format: '{pair}'. Expected 'KEY=VALUE'"
                )

            key, value = pair.split("=", 1)
            key = key.strip().upper()
            value = float(value.strip())

            if key not in DEFAULT_WEIGHTS:
                raise ValueError(
                    f"Unknown dimension key: '{key}'. Valid keys: A, B, C, D, E"
                )

            weights[key] = value

        # Fill in missing keys with 0
        for key in DEFAULT_WEIGHTS:
            if key not in weights:
                weights[key] = 0.0

        # Validate sum
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")

        return weights

    except Exception as e:
        raise ValueError(f"Failed to parse weights '{weight_str}': {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="llm-as-a-judge",
        description="Evaluate clustering quality using LLM-based judgment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python -m llm-as-a-judge.judge -o out/out_gpt-oss-20b/out.csv -i data/arxiv/prompt.txt

  # With custom weights (emphasize intent alignment)
  python -m llm-as-a-judge.judge -o out.csv -i prompt.txt -w "A=0.3,B=0.2,C=0.2,D=0.15,E=0.15"

  # Save markdown report
  python -m llm-as-a-judge.judge -o out.csv -i prompt.txt -f markdown --save report.md

  # Use different model
  python -m llm-as-a-judge.judge -o out.csv -i prompt.txt -m gpt-4o
""",
    )

    # Required arguments
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to clustering output CSV (columns: id, label, text)",
    )
    parser.add_argument(
        "--intent",
        "-i",
        required=True,
        help="Path to intent/prompt file describing the clustering goal",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        "-m",
        default="gpt-4o-mini",
        choices=llm_choices(),
        help="LLM model to use for evaluation (default: gpt-4o-mini)",
    )

    # Weight configuration
    parser.add_argument(
        "--weights",
        "-w",
        default=None,
        help="Custom dimension weights: 'A=0.3,B=0.2,C=0.2,D=0.15,E=0.15' (must sum to 1.0)",
    )

    # Sampling configuration
    parser.add_argument(
        "--sample-size",
        "-s",
        type=int,
        default=5,
        help="Max items to sample per cluster (default: 5)",
    )
    parser.add_argument(
        "--max-clusters",
        "-c",
        type=int,
        default=5,
        help="Max clusters to sample for coherence evaluation (default: 5)",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=500,
        help="Max character length for each text item (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Retry configuration
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max LLM retry attempts (default: 3)",
    )

    # Output configuration
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save results to file (path)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress information",
    )

    return parser


def validate_paths(output_path: str, intent_path: str) -> None:
    """
    Validate that required files exist.

    Raises:
        FileNotFoundError: If files don't exist
    """
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Clustering output not found: {output_path}")

    if not os.path.exists(intent_path):
        raise FileNotFoundError(f"Intent file not found: {intent_path}")


def main(args: list[str] | None = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Command-line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    try:
        # Validate paths
        validate_paths(parsed.output, parsed.intent)

        # Parse weights if provided
        weights = None
        if parsed.weights:
            weights = parse_weights(parsed.weights)

        # Create judge
        judge = ClusterJudge(
            model=parsed.model,
            weights=weights,
            max_clusters_to_sample=parsed.max_clusters,
            max_items_per_cluster=parsed.sample_size,
            max_text_length=parsed.max_text_length,
            seed=parsed.seed,
            max_retries=parsed.retries,
            verbose=parsed.verbose,
        )

        # Run evaluation
        result = judge.evaluate(
            output_csv=parsed.output,
            intent_file=parsed.intent,
        )

        # Format output
        if parsed.format == "json":
            output = result.to_json()
        else:
            output = result.to_markdown()

        # Print to stdout
        print(output)

        # Save to file if requested
        if parsed.save:
            save_dir = os.path.dirname(parsed.save)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            with open(parsed.save, "w", encoding="utf-8") as f:
                f.write(output)

            if parsed.verbose:
                print(
                    f"\n[ClusterJudge] Results saved to {parsed.save}", file=sys.stderr
                )

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        if parsed.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
