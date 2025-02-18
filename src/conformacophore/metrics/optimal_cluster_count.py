import os
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import List

class OptimalClusterCount:
    """Class for calculating optimal cluster count."""

    def calculate_metrics(self, rmsd_matrix: np.ndarray, Z: np.ndarray) -> dict:
        metrics = {'elbow': [], 'silhouette': [], 'calinski': [], 'davies': []}

        for n_clusters in range(2, 11):
            labels = hierarchy.fcluster(Z, t=n_clusters, criterion='maxclust')

            within_ss = 0
            for i in range(1, n_clusters + 1):
                cluster_points = np.where(labels == i)[0]
                if len(cluster_points) > 1:
                    cluster_rmsd = rmsd_matrix[np.ix_(cluster_points, cluster_points)]
                    within_ss += np.sum(cluster_rmsd ** 2) / (2 * len(cluster_points))
            metrics['elbow'].append(within_ss)

            try:
                sil = silhouette_score(rmsd_matrix, labels, metric='precomputed')
                metrics['silhouette'].append(sil)
            except:
                metrics['silhouette'].append(np.nan)

            try:
                cal = calinski_harabasz_score(rmsd_matrix, labels)
                metrics['calinski'].append(cal)
            except:
                metrics['calinski'].append(np.nan)

            try:
                dav = davies_bouldin_score(rmsd_matrix, labels)
                metrics['davies'].append(dav)
            except:
                metrics['davies'].append(np.nan)

        return metrics

    def get_optimal_clusters(self, rmsd_matrix: np.ndarray, max_clusters: int, output_dir: str, compound_id: str):
        """Determine the optimal number of clusters."""
        condensed_matrix = squareform(rmsd_matrix)
        Z = hierarchy.linkage(condensed_matrix, method='ward')
        metrics = self.calculate_metrics(rmsd_matrix, Z)
        suggestions = {}

        # Determine optimal clusters based on the elbow method
        suggestions['elbow'] = np.argmin(metrics['elbow']) + 2

        # Determine optimal clusters based on silhouette score
        suggestions['silhouette'] = np.nanargmax(metrics['silhouette']) + 2

        # Determine optimal clusters based on calinski-harabasz score
        suggestions['calinski'] = np.nanargmax(metrics['calinski']) + 2

        # Determine optimal clusters based on davies-bouldin score
        suggestions['davies'] = np.nanargmin(metrics['davies']) + 2

        # Voting mechanism to determine final optimal cluster count
        all_suggestions = [suggestions['elbow'], suggestions['silhouette'], suggestions['calinski'], suggestions['davies']]
        optimal_clusters = max(set(all_suggestions), key=all_suggestions.count)
        suggestions['final'] = optimal_clusters

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return metrics, suggestions, Z
