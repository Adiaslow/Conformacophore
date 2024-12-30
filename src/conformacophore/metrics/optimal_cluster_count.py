import numpy as np
from metrics.cluster_metrics import ClusterMetrics
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import List

class OptimalClusterCount(ClusterMetrics):
    """Class for calculating optimal cluster count."""

    def calculate_metrics(self, rmsd_matrix: np.ndarray, clusters: List[int]) -> dict:
        condensed_matrix = squareform(rmsd_matrix)
        Z = hierarchy.linkage(condensed_matrix, method='ward')

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
