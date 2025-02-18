from typing import Optional
from src.conformacophore.results.alignment_result import AlignmentResult
from src.conformacophore.strategies.alignment_strategy import AlignmentStrategy
from src.conformacophore.entities.molecular_graph import MolecularGraph

class AlignmentContext:
    """Context for handling different alignment strategies."""

    def __init__(self, strategy: Optional[AlignmentStrategy] = None):
        """
        Initialize alignment context.

        Args:
            strategy: Initial alignment strategy to use
        """
        self._strategy = strategy

    def set_strategy(self, strategy: AlignmentStrategy) -> None:
        """
        Set the alignment strategy.

        Args:
            strategy: New alignment strategy to use
        """
        self._strategy = strategy

    def align(self, mol1: 'MolecularGraph', mol2: 'MolecularGraph') -> AlignmentResult:
        """
        Align structures using current strategy.

        Args:
            mol1: First molecular structure
            mol2: Second molecular structure

        Returns:
            AlignmentResult
        """
        if self._strategy is None:
            raise ValueError("No alignment strategy set")
        return self._strategy.align(mol1, mol2)
