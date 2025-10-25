from src.dataset import IACDataset
import argparse
import os
from src.utils.llm import OuterMedusaLLM
from src.utils.shared import target_cluster_count, get_output_files
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import csv

llm = OuterMedusaLLM()


def embed_documents(texts: list[str]) -> np.ndarray:
    """
    Embed documents using TF-IDF vectorization.
    In a real implementation, this could use sentence transformers or other embedding models.
    """
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    embeddings = vectorizer.fit_transform(texts)
    return embeddings.toarray()


def perform_kmeans_clustering(
    embeddings: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, KMeans]:
    """
    Perform K-means++ clustering on the embeddings.
    """
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels, kmeans


def save_clustering_results(
    clusters: dict[str, list],
    dataset: IACDataset,
    cluster_labels: np.ndarray,
    out_file: str,
    summary_file: str = None,
    log_file: str = None,
):
    """
    Save clustering results to output directory.
    """
    # Create cluster assignments
    for i, label in enumerate(cluster_labels):
        cluster_name = f"cluster_{label}"
        if cluster_name not in clusters:
            clusters[cluster_name] = []
        clusters[cluster_name].append(i)

    # Save results as CSV
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for cluster_name, doc_indices in clusters.items():
            for doc_idx in doc_indices:
                writer.writerow([doc_idx, cluster_name])

    # Save cluster summary
    summary = {
        "total_documents": len(dataset),
        "num_clusters": len(clusters),
        "cluster_sizes": {name: len(docs) for name, docs in clusters.items()},
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def baseline(args):
    # fmt: off
    prompt = args.prompt
    dataset = args.docs.endswith(".csv") and IACDataset.from_csv(args.docs) or IACDataset.from_dir(args.docs)
    clusters: dict[str, list] = {}
    cluster_counts = target_cluster_count(llm, prompt)
    log_file, out_file, summary_file = get_output_files(args.output)
    # fmt: on

    print(f"Processing {len(dataset)} documents...")
    print(f"Target cluster count: {cluster_counts}")

    # Extract texts from dataset
    texts = [text for _, text in dataset]

    # Generate embeddings for all documents
    print("Generating embeddings...")
    embeddings = embed_documents(texts)

    # Perform K-means++ clustering
    print(f"Performing K-means++ clustering into {cluster_counts} clusters...")
    cluster_labels, kmeans_model = perform_kmeans_clustering(embeddings, cluster_counts)

    # Save results
    print("Saving clustering results as CSV...")
    save_clustering_results(
        clusters, dataset, cluster_labels, out_file, summary_file, log_file
    )

    # Log the process
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Baseline clustering completed\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Dataset: {args.docs}\n")
        f.write(f"Total documents: {len(dataset)}\n")
        f.write(f"Target clusters: {cluster_counts}\n")
        f.write(f"Actual clusters: {len(set(cluster_labels))}\n")
        f.write(f"Cluster distribution: {np.bincount(cluster_labels)}\n")

    print(f"Clustering complete! Results saved to {args.output}")
    print(f"Created {len(set(cluster_labels))} clusters")

    return clusters, cluster_labels, kmeans_model


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

    baseline(args)
