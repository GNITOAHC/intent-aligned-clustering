import argparse
import os
import sys
import csv
import json
from collections import defaultdict
from tqdm import tqdm

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import IACDataset
from src.utils.llm import OuterMedusaLLM
from src2.prompts import create_labeling_prompt

def main():
    parser = argparse.ArgumentParser(description="IAC Baseline (Zero-shot Labeling)")
    parser.add_argument("--docs", "-d", type=str, required=True, help="Path to documents (CSV or directory)")
    parser.add_argument("--intent", "-i", type=str, required=True, help="Clustering intent/instruction")
    parser.add_argument("--output", "-o", type=str, default="./out_src2", help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM for testing")
    args = parser.parse_args()

    # 1. Input Loading
    print(f"Loading documents from {args.docs}...")
    if args.docs.endswith(".csv"):
        dataset = IACDataset.from_csv(args.docs)
    else:
        dataset = IACDataset.from_dir(args.docs)
    
    print(f"Loaded {len(dataset)} documents.")

    # 2. Initialize LLM
    if args.mock:
        print("Using Mock LLM.")
        class MockLLM:
            def generate(self, prompt):
                import random
                return random.choice(["Cluster A", "Cluster B", "Cluster C"]), 0, 0
        llm = MockLLM()
    else:
        try:
            llm = OuterMedusaLLM()
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            return

    # 3. Pipeline: Prompt -> Inference -> Aggregation
    print("Starting Zero-shot Labeling...")
    
    # Store results: label -> list of doc_indices
    clusters = defaultdict(list)
    doc_labels = [] # List of (doc_id, label)

    os.makedirs(args.output, exist_ok=True)
    
    # We'll use a simple index as doc_id for now, or filename if available
    # IACDataset items are (metadata, text)
    
    for idx, (metadata, text) in tqdm(enumerate(dataset), total=len(dataset)):
        # 4. Prompt Construction
        prompt = create_labeling_prompt(args.intent, text)
        
        # 5. LLM Inference
        try:
            # Using a low temperature for determinism, though OuterMedusaLLM defaults might apply
            # The generate method signature is generate(prompt, sys_prompt)
            label, _, _ = llm.generate(prompt)
            label = label.strip()
        except Exception as e:
            print(f"Error processing document {idx}: {e}")
            label = "Error"
        
        # 6. Naive Aggregation (Exact Match)
        clusters[label].append(idx)
        doc_labels.append((idx, label))

    # 7. Output Generation
    out_csv = os.path.join(args.output, "clustering.csv")
    print(f"Saving results to {out_csv}...")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(doc_labels)

    summary_file = os.path.join(args.output, "summary.json")
    summary = {
        "total_documents": len(dataset),
        "num_clusters": len(clusters),
        "intent": args.intent,
        "cluster_sizes": {k: len(v) for k, v in clusters.items()}
    }
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Baseline completed.")
    print(f"Created {len(clusters)} unique labels.")

if __name__ == "__main__":
    main()
