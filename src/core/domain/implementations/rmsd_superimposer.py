"""Implementation of structure superimposition using RMSD."""

import numpy as np
from typing import List, Tuple, Optional

from ..interfaces.structure_superimposer import StructureSuperimposer
from ..models.molecular_graph import MolecularGraph
from ..models.alignment_result import AlignmentResult
from ..models.clash_result import ClashResult


class RMSDSuperimposer(StructureSuperimposer):
    """Superimpose structures by minimizing RMSD."""

    def align(self, mol1: MolecularGraph, mol2: MolecularGraph) -> AlignmentResult:
        """
        Align two molecular structures.

        Args:
            mol1: First molecular structure
            mol2: Second molecular structure

        Returns:
            AlignmentResult containing alignment metrics and transformation
        """
        # Get coordinates
        coords1 = mol1.get_coordinates()
        coords2 = mol2.get_coordinates()

        # Find matching atoms
        matched_pairs = self._match_atoms(mol1, mol2)

        if not matched_pairs:
            return AlignmentResult(
                rmsd=float("inf"),
                matched_atoms=0,
                transformation_matrix=None,
                matched_pairs=[],
            )

        # Extract matched coordinates
        matched_coords1 = np.array([coords1[i] for i, _ in matched_pairs])
        matched_coords2 = np.array([coords2[j] for _, j in matched_pairs])

        # Calculate optimal transformation
        rotation, translation = self._calculate_transformation(
            matched_coords1, matched_coords2
        )

        # Apply transformation and calculate RMSD
        aligned_coords = np.dot(matched_coords2, rotation.T) + translation
        rmsd = np.sqrt(np.mean(np.sum((matched_coords1 - aligned_coords) ** 2, axis=1)))

        return AlignmentResult(
            rmsd=rmsd,
            matched_atoms=len(matched_pairs),
            transformation_matrix=(rotation, translation),
            matched_pairs=matched_pairs,
        )

    def _match_atoms(
        self, mol1: MolecularGraph, mol2: MolecularGraph
    ) -> List[Tuple[int, int]]:
        """Match atoms between structures based on chemical identity."""
        matched_pairs = []

        for i, atom1 in enumerate(mol1.atoms):
            for j, atom2 in enumerate(mol2.atoms):
                if self._atoms_match(atom1, atom2):
                    matched_pairs.append((i, j))
                    break

        return matched_pairs

    def _atoms_match(self, atom1: dict, atom2: dict) -> bool:
        """Check if two atoms match based on identity."""
        return (
            atom1["atom_name"] == atom2["atom_name"]
            and atom1["residue_name"] == atom2["residue_name"]
            and atom1["residue_num"] == atom2["residue_num"]
        )

    def _calculate_transformation(
        self, coords1: np.ndarray, coords2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
