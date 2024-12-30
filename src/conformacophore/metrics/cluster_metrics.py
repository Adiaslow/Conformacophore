from abc import ABC, abstractmethod
import numpy as np
from typing import List

class ClusterMetrics(ABC):
    """Abstract base class for cluster metrics."""

    @abstractmethod
    def calculate_metrics(self, rmsd_matrix: np.ndarray, clusters: List[int]) -> dict:
        pass
