import numpy as np
from typing import List, Any

class RepresentativeFinder:
    """Class for finding representative structures."""

    @staticmethod
    def find_representative_structure(rmsd_matrix: np.ndarray, clusters: List[int], structures: List[Any]) -> List[Any]:
        unique_clusters = set(clusters)
        representative_structures = []

        for cluster in unique_clusters:
            indices = [i for i, c in enumerate(clusters) if c == cluster]
            submatrix = rmsd_matrix[np.ix_(indices, indices)]
            avg_rmsds = np.mean(submatrix, axis=1)
            representative_idx = indices[np.argmin(avg_rmsds)]
            representative_structures.append(structures[representative_idx])

        return representative_structures

    def find_lowest_rmsd_structure(self, pdb_files: List[str], rmsd_matrix: np.ndarray, clusters: List[int]) -> tuple:
        """Find the structure with the lowest RMSD in each cluster."""
        unique_clusters = set(clusters)
        lowest_rmsd_structures = {}
        representative = None
        lowest_rmsd = float('inf')

        for cluster in unique_clusters:
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster]
            cluster_rmsd_matrix = rmsd_matrix[np.ix_(cluster_indices, cluster_indices)]
            avg_rmsds = np.mean(cluster_rmsd_matrix, axis=1)
            min_rmsd_idx = cluster_indices[np.argmin(avg_rmsds)]

            if avg_rmsds[np.argmin(avg_rmsds)] < lowest_rmsd:
                lowest_rmsd = avg_rmsds[np.argmin(avg_rmsds)]
                representative = pdb_files[min_rmsd_idx]

            lowest_rmsd_structures[cluster] = pdb_files[min_rmsd_idx]

        return representative, lowest_rmsd_structures
