import numpy as np
from typing import List, Any

class RepresentativeFinder:
    """Class for finding representative structures."""

    @staticmethod
    def find_representative_structure(rmsd_matrix: np.ndarray, clusters: List[int], structures: List[Any]) -> Any:
        unique_clusters = set(clusters)
        representative_structures = []

        for cluster in unique_clusters:
            indices = [i for i, c in enumerate(clusters) if c == cluster]
            submatrix = rmsd_matrix[np.ix_(indices, indices)]
            avg_rmsds = np.mean(submatrix, axis=1)
            representative_idx = indices[np.argmin(avg_rmsds)]
            representative_structures.append(structures[representative_idx])

        return representative_structures
