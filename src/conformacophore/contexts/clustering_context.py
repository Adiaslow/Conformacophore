from ..strategies import ClusteringStrategy
from typing import List
import numpy as np

class ClusteringContext:
    def __init__(self, strategy: ClusteringStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ClusteringStrategy):
        self._strategy = strategy

    def cluster(self, data: np.ndarray) -> List[int]:
        return self._strategy.cluster(data)
