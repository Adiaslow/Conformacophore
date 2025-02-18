# contexts/clustering_context.py
from typing import List
import numpy as np
from src.conformacophore.strategies.clustering_strategy import ClusteringStrategy

class ClusteringContext:
    def __init__(self, strategy: ClusteringStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ClusteringStrategy):
        self._strategy = strategy

    def cluster(self, data: np.ndarray) -> List[int]:
        return self._strategy.cluster(data)

    def calculate_rmsd_matrix(self, structures: List[np.ndarray]) -> np.ndarray:
        return self._strategy.calculate_rmsd_matrix(structures)
