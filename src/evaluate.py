import argparse
import csv
from cluster_scoring import ClusteringEvaluator

evaluator = ClusteringEvaluator()


def evaluate(input_csv, ground_truth_csv):
    """
    Evaluate clustering results against ground truth.
    evaluate.csv: id, label
    ground_truth.csv: id, label
    """
    label_true = []
    label_pred = []

    with open(ground_truth_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        gt_dict = {row["id"]: row["label"] for row in reader}

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_ = row["id"]
            pred_label = row["label"]
            true_label = gt_dict.get(id_, None)
            if true_label is not None:
                label_true.append(true_label)
                label_pred.append(pred_label)

    print("Classification Report:")
    # from sklearn.metrics import classification_report

    print(label_true)
    print(label_pred)
    results = evaluator.evaluate(label_true, label_pred)
    print(results)

    # print(classification_report(label_true, label_pred))


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="intent-aligned-clustering evaluation")
    parser.add_argument("--pred", "-p", type=str, required=True, help="Path to the documents, either a directory or a CSV file")
    parser.add_argument("--ground", "-g", type=str, required=True, help="Ground truth file for evaluation")
    args = parser.parse_args()
    # fmt: on

    evaluate(args.pred, args.ground)
