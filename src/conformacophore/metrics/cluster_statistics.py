import numpy as np
from src.conformacophore.metrics.cluster_metrics import ClusterMetrics
from typing import List

class ClusterStatistics(ClusterMetrics):
    """Class for calculating cluster statistics."""

    def calculate_metrics(self, rmsd_matrix: np.ndarray, clusters: List[int]) -> dict:
        unique_clusters = np.unique(clusters)
        cluster_sizes = {cluster: np.sum(clusters == cluster) for cluster in unique_clusters}
        return {'cluster_sizes': cluster_sizes}
