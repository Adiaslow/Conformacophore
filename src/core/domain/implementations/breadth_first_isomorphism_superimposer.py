"""Implementation of isomorphic graph matching for molecular structure alignment."""

import numpy as np
import networkx as nx
from typing import List, Tuple, Optional, Dict, Set
from ..interfaces.structure_superimposer import StructureSuperimposer
from ..models.molecular_graph import MolecularGraph
from ..models.alignment_result import AlignmentResult


class BreadthFirstIsomorphismSuperimposer(StructureSuperimposer):
    """Superimposer that uses breadth-first isomorphic graph matching."""

    def __init__(self, max_iterations: int = 1000):
        """Initialize superimposer."""
        self.max_iterations = max_iterations

    def _create_networkx_graph(self, graph: MolecularGraph) -> nx.Graph:
        """Convert MolecularGraph to NetworkX graph."""
        G = nx.Graph()

        # Add nodes with atom attributes
        for atom in graph.atoms:
            G.add_node(
                atom.atom_id,
                element=atom.element,
                name=atom.atom_name,
                residue=atom.residue_name,
            )

        # Add edges from bonds
        for bond in graph.bonds:
            G.add_edge(bond.atom1_id, bond.atom2_id)

        return G

    def _find_isomorphic_mapping(
        self, ref_graph: nx.Graph, target_graph: nx.Graph
    ) -> Optional[Dict[int, int]]:
        """Find isomorphic mapping between two graphs."""

        # Get node match function that requires matching elements
        def node_match(n1, n2):
            return n1["element"] == n2["element"]

        # Try to find isomorphic mapping
        try:
            matcher = nx.isomorphism.GraphMatcher(
                ref_graph, target_graph, node_match=node_match
            )
            if matcher.is_isomorphic():
                return matcher.mapping
        except Exception:
            pass

        return None

    def _calculate_transformation(
        self, ref_coords: np.ndarray, target_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate optimal rotation and translation."""
        # Center coordinates
        ref_center = np.mean(ref_coords, axis=0)
        target_center = np.mean(target_coords, axis=0)

        ref_centered = ref_coords - ref_center
        target_centered = target_coords - target_center

        # Calculate correlation matrix
        correlation_matrix = np.dot(target_centered.T, ref_centered)

        # SVD
        U, _, Vt = np.linalg.svd(correlation_matrix)

        # Calculate rotation matrix
        rotation = np.dot(U, Vt)

        # Ensure right-handed coordinate system
        if np.linalg.det(rotation) < 0:
            Vt[-1] *= -1
            rotation = np.dot(U, Vt)

        # Calculate translation
        translation = ref_center - np.dot(target_center, rotation.T)

        return rotation, translation

    def align(
        self, reference: MolecularGraph, target: MolecularGraph
    ) -> AlignmentResult:
        """
        Align target structure to reference using isomorphic graph matching.

        Args:
            reference: Reference structure
            target: Structure to align

        Returns:
            AlignmentResult containing alignment metrics and transformation
        """
        # Convert to NetworkX graphs
        ref_graph = self._create_networkx_graph(reference)
        target_graph = self._create_networkx_graph(target)

        # Find isomorphic mapping
        mapping = self._find_isomorphic_mapping(ref_graph, target_graph)

        if mapping is None:
            # No isomorphic mapping found
            return AlignmentResult(
                rmsd=float("inf"),
                matched_atoms=0,
                transformation_matrix=None,
                matched_pairs=[],
                isomorphic_match=False,
            )

        # Create matched pairs from mapping
        matched_pairs = [
            (ref_id - 1, target_id - 1) for ref_id, target_id in mapping.items()
        ]

        # Get coordinates for matched atoms
        ref_coords = np.array(
            [reference.atoms[i].coordinates for i, _ in matched_pairs]
        )
        target_coords = np.array(
            [target.atoms[j].coordinates for _, j in matched_pairs]
        )

        # Calculate optimal transformation
        rotation, translation = self._calculate_transformation(
            ref_coords, target_coords
        )

        # Apply transformation and calculate RMSD
        aligned_coords = np.dot(target_coords, rotation.T) + translation
        rmsd = np.sqrt(np.mean(np.sum((ref_coords - aligned_coords) ** 2, axis=1)))

        return AlignmentResult(
            rmsd=rmsd,
            matched_atoms=len(matched_pairs),
            transformation_matrix=(rotation, translation),
            matched_pairs=matched_pairs,
            isomorphic_match=True,
        )
