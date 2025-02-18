# handlers/clustering_handler.py
import numpy as np
from scipy.cluster import hierarchy
from typing import List
from src.conformacophore.contexts.clustering_context import ClusteringContext
from src.conformacophore.clustering.rmsd_clustering import RMSDClustering
from src.conformacophore.filters.representatives_finder import RepresentativeFinder
from src.conformacophore.metrics.optimal_cluster_count import OptimalClusterCount

class ClusteringHandler:
    def __init__(self):
        self.context = ClusteringContext(RMSDClustering())
        self.representative_finder = RepresentativeFinder()
        self.optimal_cluster_count = OptimalClusterCount()

    def calculate_rmsd_matrix(self, trajs) -> np.ndarray:
        """Calculate the RMSD matrix for the given trajectories."""
        return self.context.calculate_rmsd_matrix(trajs)

    def get_optimal_clusters(self, rmsd_matrix: np.ndarray, max_clusters: int, output_dir: str, compound_id: str):
        """Determine the optimal number of clusters."""
        return self.optimal_cluster_count.get_optimal_clusters(rmsd_matrix, max_clusters, output_dir, compound_id)

    def get_clusters(self, linkage_matrix: np.ndarray, n_clusters: int) -> List[int]:
        """Get the cluster assignments from the linkage matrix."""
        return hierarchy.fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

    def find_lowest_rmsd_structure(self, pdb_files: List[str], rmsd_matrix: np.ndarray, clusters: List[int]) -> tuple:
        """Find the structure with the lowest RMSD in each cluster."""
        return self.representative_finder.find_lowest_rmsd_structure(pdb_files, rmsd_matrix, clusters)
