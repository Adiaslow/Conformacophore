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
        # Get coordinates
        ref_coords = reference.get_coordinates()
        target_coords = target.get_coordinates()

        # Match atoms based on name only (ignore residue name for now)
        matched_pairs = []
        used_target_indices = set()

        logger = logging.getLogger(__name__)
        logger.debug("Starting atom matching")
        logger.debug(
            f"Reference atoms: {len(reference.atoms)}, Target atoms: {len(target.atoms)}"
        )

        for i, ref_atom in enumerate(reference.atoms):
            ref_name = ref_atom["atom_name"]
            logger.debug(
                f"Looking for match for {ref_name}({ref_atom['residue_name']})"
            )

            for j, target_atom in enumerate(target.atoms):
                if j in used_target_indices:
                    continue

                target_name = target_atom["atom_name"]
                if ref_name == target_name:
                    matched_pairs.append((i, j))
                    used_target_indices.add(j)
                    logger.debug(
                        f"Matched {ref_name}({ref_atom['residue_name']}) -> "
                        f"{target_name}({target_atom['residue_name']})"
                    )
                    break

        logger.debug(f"Found {len(matched_pairs)} atom matches")

        if not matched_pairs:
            return AlignmentResult(
                rmsd=float("inf"),
                matched_atoms=0,
                transformation_matrix=None,
                matched_pairs=[],
            )

        # Extract matched coordinates
        ref_matched = np.array([ref_coords[i] for i, _ in matched_pairs])
        target_matched = np.array([target_coords[j] for _, j in matched_pairs])

        # Calculate optimal transformation
        rotation, translation = self._calculate_transformation(
            ref_matched, target_matched
        )

        # Apply transformation and calculate RMSD
        aligned_coords = np.dot(target_matched, rotation.T) + translation
        rmsd = np.sqrt(np.mean(np.sum((ref_matched - aligned_coords) ** 2, axis=1)))

        return AlignmentResult(
            rmsd=rmsd,
            matched_atoms=len(matched_pairs),
            transformation_matrix=(rotation, translation),
            matched_pairs=matched_pairs,
        )

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
        min_distance = float("inf")

        for i, coord1 in enumerate(ref_coords):
            for j, coord2 in enumerate(aligned_coords):
                dist = np.linalg.norm(coord1 - coord2)
                if dist < cutoff:
                    clash_pairs.append((i, j))
                    min_distance = min(min_distance, dist)

        return ClashResult(
            has_clashes=len(clash_pairs) > 0,
            num_clashes=len(clash_pairs),
            clash_pairs=clash_pairs,
            min_distance=min_distance if clash_pairs else float("inf"),
        )
