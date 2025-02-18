# strategies/clustering_strategy.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class ClusteringStrategy(ABC):
    @abstractmethod
    def cluster(self, data: np.ndarray) -> List[int]:
        pass
