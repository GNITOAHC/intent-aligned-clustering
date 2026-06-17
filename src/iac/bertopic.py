"""BERTopic-based topic modeling comparator for document clustering."""

from iac.dataset import IACDataset
import argparse
import os
from iac.utils.llm import OuterMedusaLLM
from iac.utils.shared import target_cluster_count, get_output_files
from bertopic import BERTopic
import json
import csv

llm = OuterMedusaLLM()


def perform_bertopic_clustering(
    texts: list[str], n_topics: int
) -> tuple[list[int], BERTopic]:
    """Cluster documents into topics using BERTopic.

    Args:
        texts: Document texts to cluster.
        n_topics: Target number of topics to reduce to. Values below 1 let
            BERTopic pick the number of topics automatically.

    Returns:
        A tuple of (topic id assigned to each document, fitted BERTopic model).
    """
    topic_model = BERTopic(nr_topics=n_topics if n_topics > 0 else "auto")
    topics, _ = topic_model.fit_transform(texts)
    return topics, topic_model


def save_clustering_results(
    dataset: IACDataset,
    topics: list[int],
    out_file: str,
    summary_file: str = None,
    log_file: str = None,
    prompt: str = None,
) -> dict[str, list]:
    """Save BERTopic clustering results to output directory.

    Args:
        dataset: Dataset whose documents were clustered.
        topics: Topic id assigned to each document, indexed by document id.
        out_file: Path to write the per-document cluster assignment CSV to.
        summary_file: Path to write the clustering summary JSON to.
        log_file: Path to the run log file. Unused here, kept for parity with
            the baseline result saver.
        prompt: The clustering intent prompt, recorded in the summary JSON.

    Returns:
        Mapping of cluster names (e.g. ``"topic_0"``) to lists of document ids.
    """
    clusters: dict[str, list] = {}
    for doc_id, topic_id in enumerate(topics):
        cluster_name = f"topic_{topic_id}"
        if cluster_name not in clusters:
            clusters[cluster_name] = []
        clusters[cluster_name].append(doc_id)

    # Save results as CSV
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "text"])
        for cluster_name, doc_indices in clusters.items():
            for doc_idx in doc_indices:
                _, text = dataset[doc_idx]
                writer.writerow([doc_idx, cluster_name, text])

    # Save cluster summary
    summary = {
        "prompt": prompt,
        "total_documents": len(dataset),
        "num_clusters": len(clusters),
        "cluster_sizes": {name: len(docs) for name, docs in clusters.items()},
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return clusters


def bertopic_baseline(args):
    """Run the BERTopic comparator end-to-end and persist its results.

    Args:
        args: Parsed CLI arguments providing ``prompt``, ``docs`` and ``output``.

    Returns:
        A tuple of (clusters, topic assignments, fitted BERTopic model).
    """
    # fmt: off
    prompt = args.prompt
    dataset = IACDataset.load(args.docs)
    cluster_counts = target_cluster_count(llm, prompt)
    log_file, out_file, summary_file, _ = get_output_files(args.output)
    # fmt: on

    print(f"Processing {len(dataset)} documents...")
    print(f"Target cluster count: {cluster_counts}")

    # Extract texts from dataset
    texts = [text for _, text in dataset]

    # Perform BERTopic clustering
    print(f"Performing BERTopic clustering into {cluster_counts} topics...")
    topics, topic_model = perform_bertopic_clustering(texts, cluster_counts)

    # Save results
    print("Saving clustering results as CSV...")
    clusters = save_clustering_results(dataset, topics, out_file, summary_file, log_file, prompt)

    # Log the process
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("BERTopic clustering completed\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Dataset: {args.docs}\n")
        f.write(f"Total documents: {len(dataset)}\n")
        f.write(f"Target clusters: {cluster_counts}\n")
        f.write(f"Actual clusters: {len(clusters)}\n")
        f.write(f"Cluster distribution: {[len(docs) for docs in clusters.values()]}\n")

    print(f"Clustering complete! Results saved to {args.output}")
    print(f"Created {len(clusters)} clusters")

    return clusters, topics, topic_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="intent-aligned-clustering tool")
    # fmt: off
    parser.add_argument("--prompt", "-p", type=str, help="Prompt to use for clustering")
    parser.add_argument("--docs", "-d", type=str, required=True, help="Path to the documents, either a directory or a CSV file")
    parser.add_argument("--output", "-o", type=str, default="./out", help="Output directory for experiments")
    parser.add_argument("--ground_truth", "-g", type=str, default=None, help="Ground truth file for evaluation")
    # fmt: on
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    if args.prompt is None:
        args.prompt = input("Enter your prompt: ")

    bertopic_baseline(args)
