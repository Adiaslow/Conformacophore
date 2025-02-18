# clustering/rmsd_clustering.py
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from typing import List
from src.conformacophore.strategies.clustering_strategy import ClusteringStrategy

class RMSDClustering(ClusteringStrategy):
    def calculate_rmsd_matrix(self, structures: List[np.ndarray]) -> np.ndarray:
        """Calculate the RMSD matrix for a list of structures."""
        n_structs = len(structures)
        rmsd_matrix = np.zeros((n_structs, n_structs))

        for i in range(n_structs):
            for j in range(i + 1, n_structs):
                rmsd = self._calculate_rmsd(structures[i], structures[j])
                rmsd_matrix[i, j] = rmsd_matrix[j, i] = rmsd

        return rmsd_matrix

    def cluster(self, data: np.ndarray) -> List[int]:
        """Cluster the data based on the RMSD matrix."""
        rmsd_matrix = self.calculate_rmsd_matrix(data)
        condensed_matrix = squareform(rmsd_matrix)
        Z = hierarchy.linkage(condensed_matrix, method='ward')
        return hierarchy.fcluster(Z, t=1, criterion='maxclust')

    def _calculate_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate the RMSD between two sets of coordinates."""
        return np.sqrt(np.sum((coords1 - coords2) ** 2) / len(coords1))
