from abc import ABC, abstractmethod
from src.conformacophore.entities.molecular_graph import MolecularGraph
from src.conformacophore.results.alignment_result import AlignmentResult

class AlignmentStrategy(ABC):
    """Abstract base class for structure alignment strategies."""

    @abstractmethod
    def align(self, mol1: 'MolecularGraph', mol2: 'MolecularGraph') -> AlignmentResult:
        """
        Align two molecular structures.

        Args:
            mol1: First molecular structure to align
            mol2: Second molecular structure to align

        Returns:
            AlignmentResult containing RMSD and transformation matrix
        """
        pass
