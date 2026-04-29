"""
Please complete the file `process.py`. It should read from

1. `sample.csv` which contains two fields, "id" and "text".
2. `sample_with_labels.csv` which contains three fields, "text" and "label".

Based on these two files, create the third file "ground_truth.csv" which contains "id" and "label" columns.
"""

import csv
import sys


def main():
    sample_csv = sys.argv[1]
    sample_with_labels_csv = sys.argv[2]
    ground_truth_csv = sys.argv[3]

    # Read sample.csv and build id -> text mapping
    id_to_text = {}
    with open(sample_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_to_text[row["id"]] = row["text"]

    # Read sample_with_labels.csv and build text -> label mapping
    text_to_label = {}
    with open(sample_with_labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text_to_label[row["text"]] = row["label"]

    # Write ground_truth.csv with id and label columns
    with open(ground_truth_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        for id_, text in id_to_text.items():
            label = text_to_label.get(text, "")
            writer.writerow({"id": id_, "label": label})


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python process.py <sample.csv> <sample_with_labels.csv> <ground_truth.csv>"
        )
        sys.exit(1)

    main()