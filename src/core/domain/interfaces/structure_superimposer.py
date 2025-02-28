"""Interface for structure superimposition strategies."""

from abc import ABC, abstractmethod
from ..models.molecular_graph import MolecularGraph
from ..models.alignment_result import AlignmentResult


class StructureSuperimposer(ABC):
    """Abstract base class for structure superimposition strategies."""

    @abstractmethod
    def align(self, mol1: MolecularGraph, mol2: MolecularGraph) -> AlignmentResult:
        """
        Align two molecular structures.

        Args:
            mol1: First molecular structure
            mol2: Second molecular structure

        Returns:
            AlignmentResult containing alignment metrics and transformation
        """
        pass
