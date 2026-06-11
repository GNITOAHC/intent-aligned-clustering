"""CLI entry points for the intent-aligned-clustering package."""

import argparse
import os

from iac.utils.shared import mark_method_evaluated, write_method


def run_baseline():
    """Entry point for TF-IDF + K-means++ baseline clustering.

    Parses CLI arguments and delegates to :func:`iac.baseline.baseline`.
    """
    from iac.baseline import baseline

    parser = argparse.ArgumentParser(
        description="TF-IDF + K-means++ baseline document clustering"
    )
    # fmt: off
    parser.add_argument("--prompt", "-p", type=str, help="Clustering intent prompt")
    parser.add_argument("--docs", "-d", type=str, required=True, help="Path to documents: a directory, a CSV file, or a HuggingFace dataset ID (e.g. 'owner/repo' or 'owner/repo:subset')")
    parser.add_argument("--output", "-o", type=str, default="./out", help="Output directory for results")
    # fmt: on
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.prompt is None:
        args.prompt = input("Enter your prompt: ")

    write_method(args.output, "baseline", False, "n/a", args.docs)
    baseline(args)


def run_bertopic():
    """Entry point for BERTopic-based document clustering.

    Parses CLI arguments and delegates to :func:`iac.bertopic.bertopic_baseline`.
    Requires the ``bertopic`` optional dependency (``pip install intent-aligned-clustering[bertopic]``).
    """
    from iac.bertopic import bertopic_baseline

    parser = argparse.ArgumentParser(
        description="BERTopic-based document clustering"
    )
    # fmt: off
    parser.add_argument("--prompt", "-p", type=str, help="Clustering intent prompt")
    parser.add_argument("--docs", "-d", type=str, required=True, help="Path to documents: a directory, a CSV file, or a HuggingFace dataset ID (e.g. 'owner/repo' or 'owner/repo:subset')")
    parser.add_argument("--output", "-o", type=str, default="./out", help="Output directory for results")
    # fmt: on
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.prompt is None:
        args.prompt = input("Enter your prompt: ")

    write_method(args.output, "bertopic", False, "n/a", args.docs)
    bertopic_baseline(args)


def run_iac():
    """Entry point for LLM-based intent-aligned clustering.

    Parses CLI arguments, selects the requested LLM backend, and delegates to
    :func:`iac.iac.main`.

    Args are forwarded to the pipeline via the ``args`` namespace:

    * ``--prompt`` / ``-p``: clustering intent prompt.
    * ``--docs`` / ``-d``: path to the document corpus (directory or CSV).
    * ``--output`` / ``-o``: output directory (default ``./out``).
    * ``--model`` / ``-m``: LLM backend choice.
    * ``--max_rounds`` / ``-r``: maximum clustering iterations (default 5).
    * ``--seed`` / ``-s``: random seed for reproducibility (default 42).
    * ``--no_postproc``: disable small-cluster merging and final cleanup.
    """
    import iac.iac as iac_module
    from iac.iac import DEFAULT_MODEL, main
    from iac.utils.llm import get_llm_instance, llm_choices

    parser = argparse.ArgumentParser(
        description="LLM-based intent-aligned document clustering"
    )
    # fmt: off
    parser.add_argument("--prompt", "-p", type=str, help="Clustering intent prompt")
    parser.add_argument("--docs", "-d", type=str, required=True, help="Path to documents: a directory, a CSV file, or a HuggingFace dataset ID (e.g. 'owner/repo' or 'owner/repo:subset')")
    parser.add_argument("--output", "-o", type=str, default="./out", help="Output directory for results")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, choices=llm_choices(), help="LLM model to use")
    parser.add_argument("--max_rounds", "-r", type=int, default=5, help="Maximum number of clustering rounds")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no_postproc", action="store_true", help="Disable post-processing (small-cluster merging and final cleanup)")
    # fmt: on
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.prompt is None:
        args.prompt = input("Enter your prompt: ")

    iac_module.llm = get_llm_instance(args.model)
    write_method(args.output, "iac", False, args.model, args.docs)
    main(args)


def run_evaluate():
    """Entry point for evaluating clustering results against ground truth.

    Parses CLI arguments and delegates to :func:`iac.evaluate.evaluate`.
    """
    import iac.evaluate as eval_module
    from iac.evaluate import evaluate

    parser = argparse.ArgumentParser(
        description="Evaluate clustering predictions against ground truth"
    )
    # fmt: off
    parser.add_argument("--pred", "-p", type=str, required=True, help="Prediction CSV with (id, label) columns")
    parser.add_argument("--ground", "-g", type=str, required=True, help="Ground truth CSV with (id, label) columns")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file for evaluation results")
    # fmt: on
    args = parser.parse_args()

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    eval_module.VERBOSE = args.verbose
    eval_module.OUTPUT_FILE = args.output

    evaluate(args.pred, args.ground)

    pred_dir = os.path.dirname(os.path.abspath(args.pred))
    mark_method_evaluated(pred_dir)
