import json
import argparse
from collections import defaultdict
import os


def analyze_sankey(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{file_path}'.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not isinstance(data, list):
        print("Error: JSON root must be a list of steps.")
        return

    if not isinstance(data, list):
        print("Error: JSON root must be a list of steps.")
        return

    # Skip the first entry (Round 1 initial state) and iterate through transitions
    for i in range(1, len(data)):
        curr_entry = data[i]
        prev_entry = data[i - 1]

        curr_round = curr_entry.get("round", "?")
        prev_round = prev_entry.get("round", "?")

        # Determine section header
        # Using current round as the identifier for the transition block
        header = f"From round {curr_round}:"
        # If it's a specific step type, maybe include it?
        # But user requested simple "From round X"
        print(header)

        curr_assignments = curr_entry.get("assignments", [])
        prev_assignments = prev_entry.get("assignments", [])

        if len(prev_assignments) != len(curr_assignments):
            print(f"  Warning: Assignment length mismatch. Skipping.")
            print("\n\n")
            continue

        # Map transitions: prev_label -> curr_label -> count
        transitions = defaultdict(lambda: defaultdict(int))
        for prev, curr in zip(prev_assignments, curr_assignments):
            transitions[prev][curr] += 1

        # Sort by previous label for consistent output
        sorted_prev_labels = sorted(transitions.keys())

        for prev_label in sorted_prev_labels:
            destinations = transitions[prev_label]
            # Sort destinations by count (descending)
            sorted_dests = sorted(
                destinations.items(), key=lambda x: x[1], reverse=True
            )

            for dest_label, count in sorted_dests:
                # Format: <topic a>(1) [<number>] <topic b>(2)
                print(
                    f"{prev_label}({prev_round}) [{count}] {dest_label}({curr_round})"
                )

        print("\n")  # Two linebreaks (one from print, one explicit)


if __name__ == "__main__":
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
