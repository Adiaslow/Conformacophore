from abc import ABC, abstractmethod
from typing import List
import numpy as np

class ClusteringStrategy(ABC):
    @abstractmethod
    def cluster(self, data: np.ndarray) -> List[int]:
        """Cluster the data and return cluster labels."""
        pass
