import argparse
import json
from collections import defaultdict
from pathlib import Path


def analyze_sankey(file_path: str) -> None:
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{file_path}'.")
        return
    except OSError as exc:
        print(f"Error reading file: {exc}")
        return

    if not isinstance(data, list):
        print("Error: JSON root must be a list of steps.")
        return

    for i in range(1, len(data)):
        curr_entry = data[i]
        prev_entry = data[i - 1]

        curr_round = curr_entry.get("round", "?")
        prev_round = prev_entry.get("round", "?")

        print(f"From round {curr_round}:")

        curr_assignments = curr_entry.get("assignments", [])
        prev_assignments = prev_entry.get("assignments", [])

        if len(prev_assignments) != len(curr_assignments):
            print("  Warning: Assignment length mismatch. Skipping.")
            print("\n")
            continue

        transitions = defaultdict(lambda: defaultdict(int))
        for prev, curr in zip(prev_assignments, curr_assignments):
            transitions[prev][curr] += 1

        for prev_label in sorted(transitions):
            destinations = transitions[prev_label]
            sorted_dests = sorted(
                destinations.items(), key=lambda item: item[1], reverse=True
            )

            for dest_label, count in sorted_dests:
                print(f"{prev_label}({prev_round}) [{count}] {dest_label}({curr_round})")

        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze clustering sankey JSON history."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="out/sankey.json",
        help="Path to the sankey.json file (default: out/sankey.json)",
    )

    args = parser.parse_args()
    analyze_sankey(args.file)


if __name__ == "__main__":
    main()
