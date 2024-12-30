from sklearn.cluster import KMeans
import numpy as np
from typing import List
from ..strategies import ClusteringStrategy

class KMeansClustering(ClusteringStrategy):
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def cluster(self, data: np.ndarray) -> List[int]:
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(data)
        return kmeans.labels_.tolist()
