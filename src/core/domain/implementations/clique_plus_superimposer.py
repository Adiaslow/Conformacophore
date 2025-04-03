"""Implementation of Clique+ algorithm for molecular structure alignment."""

import numpy as np
import networkx as nx
import time
from typing import List, Tuple, Optional, Dict, Set, Any, FrozenSet, cast
from ..interfaces.structure_superimposer import StructureSuperimposer
from ..models.molecular_graph import MolecularGraph
from ..models.alignment_result import AlignmentResult
import logging


def node_match(node1: Dict[str, Any], node2: Dict[str, Any]) -> bool:
    """Check if two nodes match based on chemical properties.

    Args:
        node1: First node's attributes
        node2: Second node's attributes

    Returns:
        True if nodes match chemically, False otherwise
    """
    # Must have element attribute
    if "element" not in node1 or "element" not in node2:
        return False

    # Get cleaned element values
    element1 = str(node1.get("element", "")).strip().upper()
    element2 = str(node2.get("element", "")).strip().upper()

    # Skip hydrogens for now as they can be more flexible
    if element1 == "H" or element2 == "H":
        return False

    # First try exact element match
    if element1 == element2:
        return True

    # Define groups of chemically similar elements
    similar_elements = {
        frozenset(["C", "X", "DU"]): "Carbon and dummy atoms",
        frozenset(["N", "NX"]): "Nitrogen atoms",
        frozenset(["O", "OX"]): "Oxygen atoms",
        frozenset(["S", "SX"]): "Sulfur atoms",
        frozenset(["P", "PX"]): "Phosphorus atoms",
        frozenset(["F", "CL", "BR", "I"]): "Halogens",
    }

    # Check if elements belong to the same group
    for group in similar_elements:
        if element1 in group and element2 in group:
            return True

    return False


def edge_match(edge1: Dict[str, Any], edge2: Dict[str, Any]) -> bool:
    """Check if two edges match based on chemical properties.

    Args:
        edge1: First edge's attributes
        edge2: Second edge's attributes

    Returns:
        True if edges match chemically, False otherwise
    """
    if edge1 is None or edge2 is None:
        return False

    # Get bond types if they exist
    bond_type1 = str(edge1.get("bond_type", "SINGLE")).strip().upper()
    bond_type2 = str(edge2.get("bond_type", "SINGLE")).strip().upper()

    # Get bond orders if they exist
    bond_order1 = float(edge1.get("bond_order", 1.0))
    bond_order2 = float(edge2.get("bond_order", 1.0))

    # If bond types are the same, it's a match
    if bond_type1 == bond_type2:
        return True

    # If bond orders are exactly equal, it's a match
    if bond_order1 == bond_order2:
        return True

    # Define strict bond type equivalences
    bond_equivalences = {
        ("AROMATIC", "SINGLE"): lambda o1, o2: o1 == 1.5 and o2 == 1.0,
        ("AROMATIC", "DOUBLE"): lambda o1, o2: o1 == 1.5 and o2 == 2.0,
        ("SINGLE", "AROMATIC"): lambda o1, o2: o1 == 1.0 and o2 == 1.5,
        ("DOUBLE", "AROMATIC"): lambda o1, o2: o1 == 2.0 and o2 == 1.5,
    }

    # Check strict bond type equivalences
    key = (bond_type1, bond_type2)
    if key in bond_equivalences:
        return bond_equivalences[key](bond_order1, bond_order2)

    return False


class CliquePlusSuperimposer(StructureSuperimposer):
    """Superimposer that uses the Clique+ algorithm for molecular graph isomorphism."""

    def __init__(self, timeout: float = 60.0):
        """Initialize superimposer.

        Args:
            timeout: Maximum time in seconds to spend finding cliques
        """
        self.timeout = timeout
        self._reference_graph: Optional[nx.Graph] = None
        self._reference_molecular_graph: Optional[MolecularGraph] = None
        self.logger = logging.getLogger(__name__)

    def _create_product_graph(
        self, ref_graph: nx.Graph, target_graph: nx.Graph
    ) -> nx.Graph:
        """Create product graph for Clique+ algorithm.

        The product graph is created with nodes representing compatible atom pairs
        and edges representing compatible bonds.

        Args:
            ref_graph: Reference molecular graph
            target_graph: Target molecular graph to align

        Returns:
            Product graph where nodes are pairs of compatible atoms and edges represent
            compatible bonding patterns
        """
        product = nx.Graph()

        # Create nodes for compatible atom pairs
        node_count = 0
        compatible_pairs: List[Tuple[int, int]] = []

        # Match atoms with compatible elements
        for ref_id in ref_graph.nodes():
            ref_attrs = ref_graph.nodes[ref_id]
            ref_element = str(ref_attrs.get("element", "")).strip().upper()

            if ref_element == "H":  # Skip hydrogens
                continue

            for target_id in target_graph.nodes():
                target_attrs = target_graph.nodes[target_id]

                if node_match(ref_attrs, target_attrs):
                    compatible_pairs.append((ref_id, target_id))
                    product.add_node(
                        (ref_id, target_id),
                        ref_element=ref_element,
                        target_element=str(target_attrs.get("element", ""))
                        .strip()
                        .upper(),
                    )
                    node_count += 1

        if self.logger:
            self.logger.info(f"Created {node_count} compatible atom pairs")

        # Add edges between compatible pairs based on explicit bonds
        edge_count = 0

        # Create adjacency lookup for quick bond checking
        ref_adj: Dict[Tuple[int, int], Dict[str, Any]] = {}
        target_adj: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for e in ref_graph.edges():
            ref_adj[(e[0], e[1])] = ref_graph.edges[e]
            ref_adj[(e[1], e[0])] = ref_graph.edges[e]

        for e in target_graph.edges():
            target_adj[(e[0], e[1])] = target_graph.edges[e]
            target_adj[(e[1], e[0])] = target_graph.edges[e]

        # Process pairs of compatible pairs
        for i, (ref1, target1) in enumerate(compatible_pairs):
            for ref2, target2 in compatible_pairs[i + 1 :]:
                if ref1 == ref2 or target1 == target2:
                    continue

                # Check bonds in both graphs
                ref_bond = ref_adj.get((ref1, ref2))
                target_bond = target_adj.get((target1, target2))

                # Add edge if bond patterns match
                if (ref_bond is not None) == (target_bond is not None):
                    if ref_bond is None or edge_match(ref_bond, target_bond):
                        product.add_edge((ref1, target1), (ref2, target2))
                        edge_count += 1

        if self.logger:
            self.logger.info(f"Created {edge_count} compatible bond pairs")
            self.logger.info(
                f"Product graph has {len(product.nodes)} nodes and {len(product.edges)} edges"
            )

        return product

    def _find_maximum_clique(self, product_graph: nx.Graph) -> List[Tuple[Any, Any]]:
        """Find maximum clique in the product graph using NetworkX's implementation.

        Args:
            product_graph: The product graph to find maximum clique in

        Returns:
            List of node pairs representing the maximum clique found
        """
        try:
            # Use NetworkX's find_clique function since max_weight_clique is not available in all versions
            clique = list(nx.find_cliques(product_graph))
            if not clique:
                return []

            # Get the largest clique
            max_clique = max(clique, key=len)

            if self.logger:
                self.logger.info(f"Found maximum clique of size {len(max_clique)}")

            return cast(List[Tuple[Any, Any]], max_clique)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to find maximum clique: {str(e)}")
            return []

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

        # Add edges from explicit bonds
        for bond in graph.bonds:
            G.add_edge(bond.atom1_id, bond.atom2_id)

        return G

    def set_reference(self, reference: MolecularGraph) -> None:
        """Set a reference structure that will be cached for future alignments."""
        self._reference_molecular_graph = reference
        self._reference_graph = self._create_networkx_graph(reference)

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

    def align(self, mol1: MolecularGraph, mol2: MolecularGraph) -> AlignmentResult:
        """Align target structure to reference using Clique+ algorithm."""
        # Use cached reference if it's the same object
        if (
            mol1 is self._reference_molecular_graph
            and self._reference_graph is not None
        ):
            ref_graph = self._reference_graph
        else:
            # If a new reference is provided, update the cache
            self.set_reference(mol1)
            ref_graph = self._reference_graph

        if ref_graph is None:
            self.logger.error("Failed to create reference graph")
            return AlignmentResult(
                rmsd=float("inf"),
                matched_atoms=0,
                transformation_matrix=None,
                matched_pairs=[],
                isomorphic_match=False,
            )

        # Convert target to NetworkX graph
        target_graph = self._create_networkx_graph(mol2)

        # Create product graph
        self.logger.info("Creating product graph for Clique+ algorithm")
        product_graph = self._create_product_graph(ref_graph, target_graph)

        # Find maximum clique
        self.logger.info("Finding maximum clique")
        clique = self._find_maximum_clique(product_graph)

        if not clique:
            self.logger.warning("No isomorphic mapping found")
            return AlignmentResult(
                rmsd=float("inf"),
                matched_atoms=0,
                transformation_matrix=None,
                matched_pairs=[],
                isomorphic_match=False,
            )

        # Create matched pairs from mapping
        matched_pairs = [(ref_id, target_id) for ref_id, target_id in clique]

        # Get coordinates for matched atoms
        ref_coords = np.array([mol1.atoms[i].coordinates for i, _ in matched_pairs])
        target_coords = np.array([mol2.atoms[j].coordinates for _, j in matched_pairs])

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
