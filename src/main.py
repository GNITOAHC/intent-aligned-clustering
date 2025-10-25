import argparse
import os
import json

from tqdm import tqdm, trange

from src.prompts import (
    create_cluster_prompt,
    create_cluster_prompt_sys,
    create_cluster_refinement_prompt,
)
from src.dataset import IACDataset
from src.utils.llm import OuterMedusaLLM
from src.utils.shared import target_cluster_count, get_output_files

DEFAULT_MODEL = "gpt-oss-20b"
llm = OuterMedusaLLM()


def normalize_label(label: str) -> str:
    """Normalize cluster labels to improve consistency"""
    return (
        label.strip().lower().replace('"', "").replace("'", "").replace(":", "").strip()
    )


def find_best_matching_cluster(label: str, existing_clusters: list[str]) -> str:
    """Find the best matching existing cluster using fuzzy matching"""
    normalized_label = normalize_label(label)

    # First try exact match (case-insensitive)
    for cluster in existing_clusters:
        if normalize_label(cluster) == normalized_label:
            return cluster

    # Then try partial matching for similar concepts
    for cluster in existing_clusters:
        normalized_cluster = normalize_label(cluster)
        if (
            normalized_label in normalized_cluster
            or normalized_cluster in normalized_label
            or
            # Check for common synonyms/variations
            any(word in normalized_cluster for word in normalized_label.split())
            or any(word in normalized_label for word in normalized_cluster.split())
        ):
            return cluster

    # If no match found, return the original label
    return label


def initialize_clusters_with_samples(
    dataset: IACDataset, instruction: str, sample_size: int
) -> dict[str, list]:
    """Initialize clusters by analyzing a sample of documents to get initial cluster ideas"""
    import random

    try:
        # Sample random documents from the dataset
        sample_indices = random.sample(
            range(len(dataset)), min(sample_size, len(dataset))
        )
        sample_texts = []

        for idx in sample_indices:
            _, text = dataset[idx]
            # Use first 200 chars to keep prompt manageable
            sample_texts.append(f"Doc{idx}: {text[:200]}...")

        # Create initialization prompt
        init_prompt = f"""CLUSTER INITIALIZATION TASK:

Objective: {instruction}

Sample documents:
{chr(10).join(sample_texts)}

Based on these sample documents and the clustering objective, suggest 3-5 initial cluster labels that would be useful for categorizing similar documents.

Rules:
1. Labels should be 1-3 words each
2. Labels should directly relate to the objective: {instruction}
3. Labels should capture distinct categories visible in the samples
4. Avoid overly specific or overly broad categories

Return only the cluster labels, one per line, no explanations:"""

        response, _, _ = llm.generate(
            init_prompt, "You are an expert at document clustering and categorization."
        )

        # Parse cluster labels from response
        initial_clusters = {}
        for line in response.strip().split("\n"):
            label = line.strip().strip('"-').strip()
            if label and len(label) > 0:
                initial_clusters[label] = []

        print(
            f"    Generated {len(initial_clusters)} initial clusters: {list(initial_clusters.keys())}"
        )
        return initial_clusters

    except Exception as e:
        print(f"    Error initializing clusters: {e}")
        return {}


def refine_cluster_labels(
    clusters: dict[str, list], instruction: str
) -> dict[str, list]:
    """Refine cluster labels for better consistency and clarity"""
    try:
        refinement_prompt = create_cluster_refinement_prompt(clusters, instruction)
        response, _, _ = llm.generate(refinement_prompt)

        # Parse the refinement suggestions
        refined_clusters = {}
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if "->" in line:
                parts = line.split("->")
                if len(parts) >= 2:
                    old_label = parts[0].strip().strip("\"'")
                    action = parts[1].strip()

                    if old_label in clusters:
                        if action.upper() == "KEEP":
                            refined_clusters[old_label] = clusters[old_label]
                        elif action.upper().startswith("RENAME_TO:"):
                            new_label = action[10:].strip()  # Remove "RENAME_TO:"
                            if new_label:
                                refined_clusters[new_label] = clusters[old_label]
                            else:
                                refined_clusters[old_label] = clusters[old_label]

        # If refinement failed or didn't cover all clusters, keep originals
        if len(refined_clusters) != len(clusters):
            print("    Refinement incomplete, keeping original labels")
            return clusters

        print(
            f"    Refined {len([k for k in refined_clusters.keys() if k not in clusters])} cluster labels"
        )
        return refined_clusters

    except Exception as e:
        print(f"    Error during refinement: {e}")
        return clusters


def cluster_to_k(
    dataset: IACDataset, clusters: dict[str, list], instruction: str, k: int
) -> dict[str, list]:
    """
    Cluster documents into k clusters. The original clusters are provided as prior knowledge.
    To get the text of each document, use for id, (_, text) in enumerate(dataset): ...
    ---
    Parameters:
        dataset: IACDataset
        clusters: existing clusters
        instruction: prompt
        k: target number of clusters
    Returns:
        dictionary of clusters: dict[str, list[int]] e.g. {'happy': ['id1', 'id2'], 'sad': ['id3', 'id4']},
    """
    # Initialize new clusters dictionary
    new_clusters = {}

    # Get existing cluster types (labels)
    cluster_types = list(clusters.keys())

    # Get system prompt for classification
    sys_prompt = create_cluster_prompt_sys()

    # Track statistics for debugging
    label_changes = 0
    errors = 0

    # Process each document individually to avoid context window issues
    for doc_id, (metadata, text) in enumerate(
        tqdm(dataset, desc="Clustering documents")
    ):
        # Skip very short or empty documents
        if not text or len(text.strip()) < 10:
            fallback_label = cluster_types[0] if cluster_types else "empty"
            if fallback_label not in new_clusters:
                new_clusters[fallback_label] = []
            new_clusters[fallback_label].append(doc_id)
            continue

        # Create prompt for this specific document
        prompt = create_cluster_prompt(cluster_types, instruction, text)

        # Get LLM response for cluster assignment
        try:
            raw_response, _, _ = llm.generate(prompt, sys_prompt)

            # Clean and normalize the response
            cluster_label = raw_response.strip()

            # Remove common prefixes/suffixes that LLM might add
            prefixes_to_remove = ["cluster:", "label:", "category:", "class:"]
            for prefix in prefixes_to_remove:
                if cluster_label.lower().startswith(prefix):
                    cluster_label = cluster_label[len(prefix) :].strip()

            # Remove quotes and clean up
            cluster_label = cluster_label.strip("\"'").strip()

            # If empty after cleaning, use fallback
            if not cluster_label:
                cluster_label = cluster_types[0] if cluster_types else "unknown"

            # Try to match with existing clusters for consistency
            if cluster_types:
                matched_label = find_best_matching_cluster(cluster_label, cluster_types)
                if matched_label != cluster_label:
                    cluster_label = matched_label
                    label_changes += 1

            # Add document to the assigned cluster
            if cluster_label not in new_clusters:
                new_clusters[cluster_label] = []
            new_clusters[cluster_label].append(doc_id)

        except Exception as e:
            errors += 1
            print(f"Error processing document {doc_id}: {e}")
            # Fallback: assign to first existing cluster or create 'error' cluster
            fallback_label = cluster_types[0] if cluster_types else "error"
            if fallback_label not in new_clusters:
                new_clusters[fallback_label] = []
            new_clusters[fallback_label].append(doc_id)

    # Print statistics
    print(f"  Label normalizations: {label_changes}, Errors: {errors}")

    # Post-processing: merge very small clusters if we have too many
    if (
        len(new_clusters) > k * 1.5
    ):  # If we have significantly more clusters than target
        # Sort clusters by size (smallest first)
        sorted_clusters = sorted(new_clusters.items(), key=lambda x: len(x[1]))

        # Merge smallest clusters into larger ones based on semantic similarity
        clusters_to_merge = []
        for small_cluster, docs in sorted_clusters:
            if len(docs) == 1 and len(clusters_to_merge) < len(new_clusters) - k:
                clusters_to_merge.append((small_cluster, docs))

        # Simple merging strategy: add small clusters to the largest cluster
        if clusters_to_merge and len(new_clusters) > k:
            largest_cluster = max(new_clusters.items(), key=lambda x: len(x[1]))[0]
            for small_cluster, docs in clusters_to_merge:
                if small_cluster != largest_cluster:
                    new_clusters[largest_cluster].extend(docs)
                    del new_clusters[small_cluster]
            print(
                f"  Merged {len(clusters_to_merge)} small clusters into '{largest_cluster}'"
            )

    # Ensure we don't have empty clusters and maintain consistency
    filtered_clusters = {label: docs for label, docs in new_clusters.items() if docs}

    return filtered_clusters


def main(args):
    # fmt: off
    prompt = args.prompt
    dataset = args.docs.endswith(".csv") and IACDataset.from_csv(args.docs) or IACDataset.from_dir(args.docs)
    clusters: dict[str, list] = {}
    cluster_counts = target_cluster_count(llm, prompt)
    log_file, out_file, summary_file = get_output_files(args.output)
    # fmt: on

    ##############
    # Print info #
    ##############
    print(f"Prompt: {prompt}")
    print(f"Documents: {len(dataset)}")
    print(f"Target Cluster Count: {cluster_counts}")
    print("log_file:", log_file)
    print("out_file:", out_file)
    print("summary_file:", summary_file)

    # Run clustering rounds
    for r in trange(args.max_rounds, desc="Clustering rounds"):
        print(f"\n=== Round {r + 1} ===")

        # For the first round, sample a few documents to get initial cluster ideas
        if r == 0 and len(clusters) == 0:
            print("  Initializing clusters with sample documents...")
            clusters = initialize_clusters_with_samples(
                dataset, prompt, min(10, len(dataset) // 10)
            )

        clusters = cluster_to_k(dataset, clusters, prompt, cluster_counts)

        print(f"Clusters after round {r + 1}: {len(clusters)} clusters")
        for cluster_name, doc_ids in clusters.items():
            print(f"  {cluster_name}: {len(doc_ids)} documents")

        # Refine cluster labels every 2 rounds to improve consistency
        if r > 0 and (r + 1) % 2 == 0 and len(clusters) > 1:
            print("  Refining cluster labels...")
            clusters = refine_cluster_labels(clusters, prompt)

        # Check if we've reached the target number of clusters
        if len(clusters) == cluster_counts:
            print(f"Reached target cluster count ({cluster_counts}) in round {r + 1}")
            break

    # Final cluster quality check and cleanup
    print("\n=== Final Processing ===")

    # Remove any clusters that are too small (less than 1% of total documents)
    min_cluster_size = max(1, len(dataset) // 100)
    large_clusters = {
        name: docs for name, docs in clusters.items() if len(docs) >= min_cluster_size
    }

    if len(large_clusters) < len(clusters):
        # Merge small clusters into the largest cluster
        small_clusters = {
            name: docs
            for name, docs in clusters.items()
            if len(docs) < min_cluster_size
        }
        if large_clusters:
            largest_cluster = max(large_clusters.items(), key=lambda x: len(x[1]))[0]
            for small_name, small_docs in small_clusters.items():
                large_clusters[largest_cluster].extend(small_docs)
            print(
                f"Merged {len(small_clusters)} small clusters into '{largest_cluster}'"
            )
            clusters = large_clusters

    # Save results
    print("\nFinal clustering results:")
    print(f"Total clusters: {len(clusters)}")
    print("Cluster distribution:")
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    for name, docs in sorted_clusters:
        percentage = (len(docs) / len(dataset)) * 100
        print(f"  {name}: {len(docs)} documents ({percentage:.1f}%)")

    # Save cluster results to output file
    try:
        import csv

        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["document_id", "cluster_label", "text"])

            for cluster_label, doc_ids in clusters.items():
                for doc_id in doc_ids:
                    _, text = dataset[doc_id]
                    writer.writerow([doc_id, cluster_label, text])

        print(f"Results saved to: {out_file}")

        # Save summary
        summary = {
            "prompt": prompt,
            "total_documents": len(dataset),
            "target_clusters": cluster_counts,
            "actual_clusters": len(clusters),
            "rounds_completed": min(r + 1, args.max_rounds),
            "clusters": {name: len(docs) for name, docs in clusters.items()},
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to: {summary_file}")

    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="intent-aligned-clustering tool")
    # fmt: off
    parser.add_argument("--prompt", "-p", type=str, help="Prompt to use for clustering")
    parser.add_argument("--docs", "-d", type=str, required=True, help="Path to the documents, either a directory or a CSV file")
    parser.add_argument("--output", "-o", type=str, default="./out", help="Output directory for experiments")
    # parser.add_argument("--ground_truth", "-g", type=str, default=None, help="Ground truth file for evaluation")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="LLM model to use")
    parser.add_argument("--max_rounds", "-r", type=int, default=5, help="Maximum number of clustering rounds")
    # fmt: on
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    if args.prompt is None:
        args.prompt = input("Enter your prompt: ")

    if args.model != DEFAULT_MODEL:
        llm.change_model(args.model)

    main(args)
