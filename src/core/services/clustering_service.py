# src/core/services/clustering_service.py
"""Service for clustering molecular conformations."""

import numpy as np
from typing import List, Set, Optional, Tuple, Dict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from ..domain.models.alignment_result import AlignmentResult
import matplotlib.pyplot as plt
import os
from pathlib import Path


class ClusteringService:
    """Service for clustering molecular conformations."""

    def cluster_frames(
        self,
        results: List[AlignmentResult],
        rmsd_cutoff: float = 2.0,
        method: str = "complete",
    ) -> List[Set[int]]:
        """
        Cluster frames based on RMSD values.

        Args:
            results: List of alignment results
            rmsd_cutoff: RMSD cutoff for clustering
            method: Linkage method for hierarchical clustering

        Returns:
            List of sets containing frame indices for each cluster
        """
        # Extract RMSD matrix from results
        n_frames = len(results)
        rmsd_matrix = np.zeros((n_frames, n_frames))

        # Fill RMSD matrix
        for i in range(n_frames):
            for j in range(i + 1, n_frames):
                rmsd = results[i].rmsd
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd

        # Perform hierarchical clustering
        Z = linkage(rmsd_matrix, method=method)
        cluster_labels = fcluster(Z, rmsd_cutoff, criterion="distance")

        # Convert labels to sets of indices
        unique_labels = set(cluster_labels)
        clusters = []
        for label in unique_labels:
            cluster_indices = {
                i for i, c_label in enumerate(cluster_labels) if c_label == label
            }
            clusters.append(cluster_indices)

        # Sort clusters by size (largest first)
        clusters.sort(key=len, reverse=True)

        return clusters

    def get_cluster_representatives(
        self, clusters: List[Set[int]], results: List[AlignmentResult]
    ) -> List[int]:
        """
        Get representative frame for each cluster (lowest RMSD).

        Args:
            clusters: List of sets containing frame indices
            results: List of alignment results

        Returns:
            List of representative frame indices
        """
        representatives = []
        for cluster in clusters:
            # Get frame with lowest RMSD in cluster
            cluster_rmsds = [(i, results[i].rmsd) for i in cluster]
            best_frame = min(cluster_rmsds, key=lambda x: x[1])[0]
            representatives.append(best_frame)
        return representatives

    def get_optimal_clusters(
        self,
        results: List[AlignmentResult],
        max_clusters: int = 10,
        output_dir: Optional[Path] = None,
        compound_id: Optional[str] = None,
    ) -> Tuple[Dict[str, List[float]], Dict[str, int], np.ndarray]:
        """
        Calculate and visualize optimal number of clusters.

        Args:
            results: List of alignment results
            max_clusters: Maximum number of clusters to consider
            output_dir: Directory to save cluster metrics plot
            compound_id: Identifier for the compound

        Returns:
            Tuple containing:
            - Dictionary of clustering metrics for different numbers of clusters
            - Dictionary of suggested optimal number of clusters based on various metrics
            - Linkage matrix from hierarchical clustering
        """
        # Calculate metrics
        metrics, linkage_matrix = self.calculate_cluster_metrics(results, max_clusters)

        # Plot metrics if output directory is provided
        if output_dir and compound_id:
            self.plot_cluster_metrics(metrics, output_dir, compound_id)

        # Get suggestions
        suggestions = self.suggest_optimal_clusters(metrics)

        return metrics, suggestions, linkage_matrix

    def calculate_cluster_metrics(
        self, results: List[AlignmentResult], max_clusters: int
    ) -> Tuple[Dict[str, List[float]], np.ndarray]:
        """
        Calculate various clustering metrics for different numbers of clusters.

        Args:
            results: List of alignment results
            max_clusters: Maximum number of clusters to consider

        Returns:
            Tuple containing:
            - Dictionary of clustering metrics for different numbers of clusters
            - Linkage matrix from hierarchical clustering
        """
        # Convert RMSD matrix to condensed form for hierarchical clustering
        rmsd_matrix = np.array([[r.rmsd for r in results]])
        rmsd_matrix = (rmsd_matrix + rmsd_matrix.T) / 2  # Ensure symmetry
        rmsd_matrix = np.maximum(rmsd_matrix, 0)  # Ensure non-negativity
        np.fill_diagonal(rmsd_matrix, 0)  # Ensure zero diagonal
        condensed_matrix = squareform(rmsd_matrix)

        # Perform hierarchical clustering
        Z = linkage(condensed_matrix, method="ward")

        # Initialize metrics dictionaries
        metrics = {"elbow": [], "silhouette": [], "calinski": [], "davies": []}

        # Calculate metrics for different numbers of clusters
        for n_clusters in range(2, max_clusters + 1):
            # Get cluster labels
            labels = fcluster(Z, t=n_clusters, criterion="maxclust")

            # Calculate elbow metric (within-cluster sum of squares)
            within_ss = 0
            for i in range(1, n_clusters + 1):
                cluster_points = np.where(labels == i)[0]
                if len(cluster_points) > 1:
                    cluster_rmsd = rmsd_matrix[np.ix_(cluster_points, cluster_points)]
                    within_ss += np.sum(cluster_rmsd**2) / (2 * len(cluster_points))
            metrics["elbow"].append(within_ss)

            # Calculate silhouette score
            try:
                sil = silhouette_score(rmsd_matrix, labels, metric="precomputed")
                metrics["silhouette"].append(sil)
            except:
                metrics["silhouette"].append(np.nan)

            # Calculate Calinski-Harabasz score
            try:
                cal = calinski_harabasz_score(rmsd_matrix, labels)
                metrics["calinski"].append(cal)
            except:
                metrics["calinski"].append(np.nan)

            # Calculate Davies-Bouldin score
            try:
                dav = davies_bouldin_score(rmsd_matrix, labels)
                metrics["davies"].append(dav)
            except:
                metrics["davies"].append(np.nan)

        return metrics, Z

    def plot_cluster_metrics(
        self,
        metrics: Dict[str, List[float]],
        output_dir: Path,
        compound_id: str,
        dpi: int = 300,
    ) -> None:
        """Plot clustering metrics to help determine optimal number of clusters."""
        n_clusters = range(2, len(metrics["elbow"]) + 2)

        # Create a 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Elbow plot
        ax1.plot(n_clusters, metrics["elbow"], "bo-")
        ax1.set_title("Elbow Method")
        ax1.set_xlabel("Number of Clusters")
        ax1.set_ylabel("Within-cluster Sum of Squares")

        # Silhouette plot
        ax2.plot(n_clusters, metrics["silhouette"], "ro-")
        ax2.set_title("Silhouette Score\n(Higher is better)")
        ax2.set_xlabel("Number of Clusters")
        ax2.set_ylabel("Silhouette Score")

        # Calinski-Harabasz plot
        ax3.plot(n_clusters, metrics["calinski"], "go-")
        ax3.set_title("Calinski-Harabasz Score\n(Higher is better)")
        ax3.set_xlabel("Number of Clusters")
        ax3.set_ylabel("Calinski-Harabasz Score")

        # Davies-Bouldin plot
        ax4.plot(n_clusters, metrics["davies"], "mo-")
        ax4.set_title("Davies-Bouldin Score\n(Lower is better)")
        ax4.set_xlabel("Number of Clusters")
        ax4.set_ylabel("Davies-Bouldin Score")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"compound_{compound_id}_cluster_metrics.png"),
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()

    def suggest_optimal_clusters(
        self, metrics: Dict[str, List[float]]
    ) -> Dict[str, int]:
        """Suggest optimal number of clusters based on various metrics."""
        suggestions = {}

        # Elbow method - find point of maximum curvature
        elbow = np.array(metrics["elbow"])
        diffs = np.diff(elbow, 2)  # Second derivative
        suggestions["elbow"] = np.argmax(np.abs(diffs)) + 2

        # Silhouette - maximum score
        silhouette = np.array(metrics["silhouette"])
        suggestions["silhouette"] = np.nanargmax(silhouette) + 2

        # Calinski-Harabasz - maximum score
        calinski = np.array(metrics["calinski"])
        suggestions["calinski"] = np.nanargmax(calinski) + 2

        # Davies-Bouldin - minimum score
        davies = np.array(metrics["davies"])
        suggestions["davies"] = np.nanargmin(davies) + 2

        return suggestions
