"""
Sampling utilities for LLM-as-a-Judge evaluation.

Provides strategies for sampling clusters and items to make evaluation
efficient while maintaining representativeness.
"""

import random
from dataclasses import dataclass, field


@dataclass
class SamplerConfig:
    """
    Configuration for the cluster sampler.

    Attributes:
        max_clusters_to_sample: Maximum number of clusters to sample for
                                coherence evaluation (dimension B)
        max_items_per_cluster: Maximum items to sample from each cluster
        max_text_length: Maximum character length for each text item
        seed: Random seed for reproducibility
    """

    max_clusters_to_sample: int = 5
    max_items_per_cluster: int = 5
    max_text_length: int = 500
    seed: int = 42


@dataclass
class ClusterData:
    """
    Represents a cluster with its label and items.

    Attributes:
        label: The cluster label/name
        items: List of (id, text) tuples belonging to this cluster
    """

    label: str
    items: list[tuple[str, str]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.items)


class ClusterSampler:
    """
    Handles sampling of clusters and items for evaluation.

    Provides methods to sample representative subsets of clustering data
    for efficient LLM-based evaluation while maintaining fairness and
    reproducibility.
    """

    def __init__(self, config: SamplerConfig | None = None):
        """
        Initialize the sampler with configuration.

        Args:
            config: Sampling configuration. Uses defaults if None.
        """
        self.config = config or SamplerConfig()
        random.seed(self.config.seed)

    def truncate_text(self, text: str) -> str:
        """
        Truncate text to maximum length, adding ellipsis if truncated.

        Args:
            text: Original text

        Returns:
            Truncated text
        """
        if len(text) <= self.config.max_text_length:
            return text
        return text[: self.config.max_text_length - 3] + "..."

    def sample_items_from_cluster(self, cluster: ClusterData) -> list[tuple[str, str]]:
        """
        Sample items from a single cluster.

        Args:
            cluster: ClusterData object

        Returns:
            List of (id, truncated_text) tuples
        """
        items = cluster.items
        n = min(len(items), self.config.max_items_per_cluster)

        sampled = random.sample(items, n) if len(items) > n else items
        return [(id_, self.truncate_text(text)) for id_, text in sampled]

    def sample_clusters(
        self, clusters: dict[str, ClusterData]
    ) -> dict[str, ClusterData]:
        """
        Sample a subset of clusters for evaluation.

        Args:
            clusters: Dictionary mapping label to ClusterData

        Returns:
            Sampled subset of clusters with sampled items
        """
        labels = list(clusters.keys())
        n = min(len(labels), self.config.max_clusters_to_sample)

        sampled_labels = random.sample(labels, n) if len(labels) > n else labels

        result = {}
        for label in sampled_labels:
            cluster = clusters[label]
            sampled_items = self.sample_items_from_cluster(cluster)
            result[label] = ClusterData(label=label, items=sampled_items)

        return result

    def sample_cluster_pairs(
        self, clusters: dict[str, ClusterData], max_pairs: int = 3
    ) -> list[tuple[ClusterData, ClusterData]]:
        """
        Sample pairs of clusters for inter-cluster distinctness evaluation.

        Tries to sample diverse pairs from different parts of the cluster set.

        Args:
            clusters: Dictionary mapping label to ClusterData
            max_pairs: Maximum number of pairs to sample

        Returns:
            List of (cluster1, cluster2) tuples with sampled items
        """
        labels = list(clusters.keys())

        if len(labels) < 2:
            return []

        # Generate all possible pairs and sample
        all_pairs = [
            (labels[i], labels[j])
            for i in range(len(labels))
            for j in range(i + 1, len(labels))
        ]

        n_pairs = min(len(all_pairs), max_pairs)
        sampled_pairs = random.sample(all_pairs, n_pairs)

        result = []
        for label1, label2 in sampled_pairs:
            cluster1 = clusters[label1]
            cluster2 = clusters[label2]

            # Sample items from each cluster in the pair
            items1 = self.sample_items_from_cluster(cluster1)
            items2 = self.sample_items_from_cluster(cluster2)

            result.append(
                (
                    ClusterData(label=label1, items=items1),
                    ClusterData(label=label2, items=items2),
                )
            )

        return result

    def format_cluster_summary(self, clusters: dict[str, ClusterData]) -> str:
        """
        Format cluster distribution as a summary string.

        Args:
            clusters: Dictionary mapping label to ClusterData

        Returns:
            Formatted summary string
        """
        lines = []
        # Sort by size descending
        sorted_clusters = sorted(
            clusters.items(), key=lambda x: x[1].size, reverse=True
        )
        for label, cluster in sorted_clusters:
            lines.append(f"- {label}: {cluster.size} items")
        return "\n".join(lines)

    def format_cluster_samples(self, clusters: dict[str, ClusterData]) -> str:
        """
        Format sampled cluster items for prompt inclusion.

        Args:
            clusters: Dictionary of sampled clusters with sampled items

        Returns:
            Formatted string with cluster labels and items
        """
        lines = []
        for label, cluster in clusters.items():
            lines.append(f"### Cluster: {label}")
            for i, (id_, text) in enumerate(cluster.items, 1):
                lines.append(f"{i}. [ID: {id_}] {text}")
            lines.append("")  # Empty line between clusters
        return "\n".join(lines)

    def format_cluster_pairs(self, pairs: list[tuple[ClusterData, ClusterData]]) -> str:
        """
        Format cluster pairs for inter-distinctness evaluation.

        Args:
            pairs: List of cluster pairs with sampled items

        Returns:
            Formatted string showing pairs for comparison
        """
        lines = []
        for i, (cluster1, cluster2) in enumerate(pairs, 1):
            lines.append(f"## Pair {i}: '{cluster1.label}' vs '{cluster2.label}'")
            lines.append("")
            lines.append(f"### Cluster: {cluster1.label}")
            for j, (id_, text) in enumerate(cluster1.items, 1):
                lines.append(f"{j}. [ID: {id_}] {text}")
            lines.append("")
            lines.append(f"### Cluster: {cluster2.label}")
            for j, (id_, text) in enumerate(cluster2.items, 1):
                lines.append(f"{j}. [ID: {id_}] {text}")
            lines.append("")
            lines.append("---")
            lines.append("")
        return "\n".join(lines)

    def format_label_samples(self, clusters: dict[str, ClusterData]) -> str:
        """
        Format clusters with labels and sample items for label quality evaluation.

        Args:
            clusters: Dictionary of clusters with sampled items

        Returns:
            Formatted string showing labels and representative items
        """
        lines = []
        for label, cluster in clusters.items():
            lines.append(f'### Label: "{label}" ({cluster.size} items in full cluster)')
            lines.append("Sample items:")
            for i, (id_, text) in enumerate(cluster.items, 1):
                lines.append(f"  {i}. {text}")
            lines.append("")
        return "\n".join(lines)
