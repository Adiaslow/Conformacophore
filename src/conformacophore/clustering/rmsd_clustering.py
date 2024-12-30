import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from typing import List
from ..strategies import ClusteringStrategy

class RMSDClustering(ClusteringStrategy):
    def calculate_rmsd_matrix(self, structures: List[np.ndarray]) -> np.ndarray:
        n_structs = len(structures)
        rmsd_matrix = np.zeros((n_structs, n_structs))

        for i in range(n_structs):
            for j in range(i + 1, n_structs):
                rmsd = self._calculate_rmsd(structures[i], structures[j])
                rmsd_matrix[i, j] = rmsd_matrix[j, i] = rmsd

        return rmsd_matrix

    def cluster(self, data: np.ndarray) -> List[int]:
        rmsd_matrix = self.calculate_rmsd_matrix(data)
        condensed_matrix = squareform(rmsd_matrix)
        Z = hierarchy.linkage(condensed_matrix, method='ward')
        return hierarchy.fcluster(Z, t=1, criterion='maxclust')

    def _calculate_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        return np.sqrt(np.sum((coords1 - coords2) ** 2) / len(coords1))
