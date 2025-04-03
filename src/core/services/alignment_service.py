"""Service for molecular structure alignment."""

from typing import Optional
import numpy as np
from ..domain.models.molecular_graph import MolecularGraph
from ..domain.models.alignment_result import AlignmentResult
from ..domain.models.clash_result import ClashResult
from ..domain.interfaces.structure_superimposer import StructureSuperimposer
import logging


class AlignmentService:
    """Service for performing structure alignments."""

    def __init__(self, superimposer: Optional[StructureSuperimposer] = None):
        """Initialize service with superimposition strategy."""
        from ..domain.implementations.rmsd_superimposer import RMSDSuperimposer

        self._superimposer = superimposer or RMSDSuperimposer()

    def align_structures(
        self,
        reference: MolecularGraph,
        target: MolecularGraph,
        clash_cutoff: float = 2.0,
    ) -> AlignmentResult:
        """
        Align two molecular structures.

        Args:
            reference: Reference structure
            target: Structure to align
            clash_cutoff: Distance cutoff for clash detection

        Returns:
            AlignmentResult containing alignment metrics and transformation
        """
        # Use the configured superimposer to align structures
        result = self._superimposer.align(reference, target)

        # If alignment failed, return empty result
        if not result.matched_pairs:
            return AlignmentResult(
                rmsd=float("inf"),
                matched_atoms=0,
                transformation_matrix=None,
                matched_pairs=[],
            )

        return result

    def _calculate_transformation(
        self, coords1: np.ndarray, coords2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate optimal rotation and translation."""
        # Center coordinates
        center1 = np.mean(coords1, axis=0)
        center2 = np.mean(coords2, axis=0)
        coords1_centered = coords1 - center1
        coords2_centered = coords2 - center2

        # Calculate correlation matrix
        correlation_matrix = np.dot(coords2_centered.T, coords1_centered)

        # SVD
        U, _, Vt = np.linalg.svd(correlation_matrix)

        # Calculate rotation matrix
        rotation = np.dot(U, Vt)

        # Ensure right-handed coordinate system
        if np.linalg.det(rotation) < 0:
            Vt[-1] *= -1
            rotation = np.dot(U, Vt)

        # Calculate translation
        translation = center1 - np.dot(center2, rotation.T)

        return rotation, translation

    def _detect_clashes(
        self,
        reference: MolecularGraph,
        target: MolecularGraph,
        alignment: AlignmentResult,
        cutoff: float,
    ) -> ClashResult:
        """Check for steric clashes between aligned structures."""
        rotation, translation = alignment.transformation_matrix

        # Get coordinates
        ref_coords = reference.get_coordinates()
        target_coords = target.get_coordinates()

        # Apply transformation to target
        aligned_coords = np.dot(target_coords, rotation.T) + translation

        # Find close contacts
        clash_pairs = []

        for i, coord1 in enumerate(ref_coords):
            for j, coord2 in enumerate(aligned_coords):
                dist = np.linalg.norm(coord1 - coord2)
                if dist < cutoff:
                    clash_pairs.append((i, j))

        return ClashResult(
            has_clashes=len(clash_pairs) > 0,
            num_clashes=len(clash_pairs),
            clash_pairs=clash_pairs,
        )
