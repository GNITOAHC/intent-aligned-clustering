"""Evaluation script for comparing clustering results against ground truth."""

import argparse
import csv
import json
import os

from cluster_scoring import ClusteringEvaluator

from iac.dataset import IACDataset

evaluator = ClusteringEvaluator()

VERBOSE = False
OUTPUT_FILE = None


def load_ground_truth(source: str) -> dict[str, str]:
    """Load ground truth labels as a mapping from id to label.

    Args:
        source: One of:
            * Path to a CSV with columns ``id,label`` (legacy format).
            * Path to a directory or a HuggingFace dataset ID (with optional
              ``owner/repo:subset`` syntax). The dataset must expose a
              ``label`` column; ``id`` is taken from metadata when present,
              otherwise falls back to the row index.
    """
    if source.endswith(".csv"):
        with open(source, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "id" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError(
                    f"Ground truth CSV must contain 'id' and 'label' columns: {source}"
                )
            return {row["id"]: row["label"] for row in reader}

    dataset = IACDataset.load(source)
    gt: dict[str, str] = {}
    for i, (meta, _) in enumerate(dataset):
        if "label" not in meta:
            raise ValueError(
                f"Dataset '{source}' has no 'label' column; cannot use as ground truth."
            )
        id_ = str(meta["id"]) if "id" in meta else str(i)
        gt[id_] = meta["label"]
    return gt


def evaluate(input_csv: str, ground_truth_source: str):
    """Evaluate clustering predictions against ground truth.

    Args:
        input_csv: Path to prediction CSV with columns ``id,label``.
        ground_truth_source: CSV path, directory, or HuggingFace dataset ID
            with labels. See :func:`load_ground_truth`.
    """
    gt_dict = load_ground_truth(ground_truth_source)

    label_true: list[str] = []
    label_pred: list[str] = []
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_label = gt_dict.get(row["id"])
            if true_label is not None:
                label_true.append(true_label)
                label_pred.append(row["label"])

    results = evaluator.evaluate(label_true, label_pred)

    if VERBOSE:
        print(json.dumps(results, indent=2))

    if OUTPUT_FILE:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="intent-aligned-clustering evaluation")
    parser.add_argument("--pred", "-p", type=str, required=True, help="Prediction CSV with (id, label) columns")
    parser.add_argument("--ground", "-g", type=str, required=True, help="Ground truth source: CSV path, directory, or HuggingFace dataset ID (e.g. 'owner/repo' or 'owner/repo:subset')")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file to save the evaluation results")
    args = parser.parse_args()
    # fmt: on

    VERBOSE = args.verbose
    OUTPUT_FILE = args.output

    if OUTPUT_FILE:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    evaluate(args.pred, args.ground)
