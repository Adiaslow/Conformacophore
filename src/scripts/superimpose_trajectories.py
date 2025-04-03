#!/usr/bin/env python3
# src/scripts/superimpose_trajectories.py

"""
Script to superimpose test compound conformations onto a reference ligand structure.

The script handles two main types of structures:
1. Reference Structure:
   - Contains protein chains (A,B,C) and a reference ligand (chain D)
   - Used as the template for superimposition
   - Chain D contains the reference ligand conformation

2. Test Compound Models:
   - Contains conformations of test compounds from MD trajectories
   - Each model represents a different conformation to be aligned
   - Compounds are matched to the reference ligand using graph isomorphism
"""

import os
import argparse
import logging
import tempfile
import json
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    Set,
    Literal,
    Union,
    Generator,
    TypeVar,
    cast,
)
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import networkx as nx
import functools
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO, Select
from Bio.PDB.Superimposer import Superimposer
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFMCS import FindMCS
from numpy.typing import NDArray

# Global cache for reference data
_REFERENCE_DATA: Dict[str, Any] = {
    "target_protein_coords": None,  # Nx3 numpy array of protein atom coordinates
    "ref_ligand_graph": None,  # MolecularGraph object for reference ligand
    "ref_ligand_coords": None,  # Mx3 numpy array of ligand atom coordinates
    "ref_conect": None,  # CONECT records for the reference ligand
}

# Add a cache for isomorphic mappings
_ISOMORPHIC_CACHE: Dict[str, Dict[str, Any]] = {}

# File cache to avoid re-reading the same files
_FILE_CACHE = {}
_PDB_STRUCTURE_CACHE: Dict[str, Structure] = {}
_CONECT_CACHE = {}

# Initialize PDB parser once
PDB_PARSER = PDBParser(QUIET=True)

# Type variables for graph attributes
NodeData = TypeVar("NodeData", bound=Dict[str, Any])


class ChainSelect(Select):
    """Select specific chains when saving a structure."""

    def __init__(self, chain_ids: List[str]):
        """Initialize chain selector.

        Args:
            chain_ids: List of chain IDs to select
        """
        super().__init__()
        self.chain_ids = chain_ids

    def accept_chain(self, chain: Chain) -> Literal[1]:
        """Determine if a chain should be included.

        Args:
            chain: BioPython Chain object

        Returns:
            1 if chain should be included, 0 if not

        Note:
            BioPython's Select class expects integer return values:
            1 to accept the chain, 0 to reject it
        """
        if chain.id in self.chain_ids:
            return 1
        return super().accept_chain(chain)


class AlignmentResult:
    """Result of an alignment operation."""

    def __init__(
        self,
        rmsd: float,
        matched_atoms: int,
        transformation_matrix: Optional[
            Tuple[NDArray[np.float64], NDArray[np.float64]]
        ],
        matched_pairs: List[Tuple[Any, Any]],
    ) -> None:
        """Initialize an AlignmentResult.

        Args:
            rmsd: Root mean square deviation of the alignment
            matched_atoms: Number of matched atoms
            transformation_matrix: Optional tuple of (rotation matrix, translation vector)
            matched_pairs: List of matched atom pairs
        """
        self.rmsd = rmsd
        self.matched_atoms = matched_atoms
        self.transformation_matrix = transformation_matrix
        self.matched_pairs = matched_pairs
        self.isomorphic_match = transformation_matrix is not None


class MolecularGraph:
    """Represents a molecular structure as a graph for matching."""

    def __init__(
        self,
        atoms: List[Any],
        logger: Optional[logging.Logger] = None,
        name: str = "",
        conect_dict: Optional[Dict[int, List[int]]] = None,
        structure: Optional[Structure] = None,
        chain_id: Optional[str] = None,
        is_reference: bool = False,
    ):
        """Initialize molecular graph from list of atoms.

        Args:
            atoms: List of BioPython atoms
            logger: Optional logger instance
            name: Name identifier for the molecule
            conect_dict: Dictionary of CONECT records from PDB
            structure: Original structure for reference
            chain_id: Chain ID to filter CONECT records
            is_reference: Whether this is the reference ligand (True) or test compound (False)
        """
        self.atoms = atoms
        self.name = name
        self.logger = logger
        self.conect_dict = conect_dict
        self.structure = structure
        self.chain_id = chain_id
        self.is_reference = is_reference
        self.serial_to_atom = {}
        self.graph = self._create_graph()

    def get_coordinates(self) -> np.ndarray:
        """Get coordinates of all atoms in the graph.

        Returns:
            numpy array of shape (n_atoms, 3) containing xyz coordinates
        """
        return np.array([atom.coord for atom in self.atoms])

    def _create_graph(self) -> nx.Graph:
        """Create NetworkX graph from atoms using PDB connectivity."""
        G = nx.Graph()

        # Create a map from atom serial number to atom object
        serial_to_atom: Dict[int, Atom] = {}
        name_to_atom: Dict[str, Atom] = {}  # Also map by atom name
        serial_numbers: List[int] = []  # Track all serials for debugging

        for atom in self.atoms:
            serial_to_atom[atom.serial_number] = atom
            serial_numbers.append(atom.serial_number)

            # Map by name (format: "N1", "C1", etc.)
            name_to_atom[atom.name.strip()] = atom

            # The atom may have a property with its original serial number
            if hasattr(atom, "original_serial"):
                original_serial = getattr(atom, "original_serial")
                serial_to_atom[original_serial] = atom
                serial_numbers.append(original_serial)
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f"Atom {atom.name} has original serial: {original_serial}"
                    )

        # Store for use in other methods
        self.serial_to_atom = serial_to_atom

        # Log the serial numbers we have
        if self.logger:
            self.logger.debug(
                f"{self.name} Atom serial numbers: {sorted(serial_numbers)}"
            )

        # Add nodes with attributes
        for i, atom in enumerate(self.atoms):
            residue = atom.get_parent()
            residue_name, residue_id = self._get_residue_info(residue)
            element = self._get_atom_element(atom)

            node_attrs: Dict[str, Any] = {
                "name": atom.name.strip(),  # Strip whitespace
                "element": element,
                "coord": tuple(atom.coord),
                "residue_name": residue_name,
                "residue_id": residue_id,
                "idx": i,  # Keep track of original index
                "serial": atom.serial_number,  # Keep track of PDB serial number
            }

            # Store original serial number if available
            if hasattr(atom, "original_serial"):
                node_attrs["original_serial"] = getattr(atom, "original_serial")

            G.add_node(atom, **node_attrs)

            if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                serial_info = (
                    f", original: {getattr(atom, 'original_serial')}"
                    if hasattr(atom, "original_serial")
                    else ""
                )
                self.logger.debug(
                    f"{self.name} Node {i}: {atom.name} {element} {residue_name} {residue_id} (serial: {atom.serial_number}{serial_info})"
                )

        # Check if we have CONECT records
        if not self.conect_dict:
            error_msg = f"{self.name} ERROR: No CONECT records found for this structure. CONECT records are required."
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Add edges using CONECT records - this is the only supported connectivity method
        edge_count = self._add_conect_edges(G)

        # Check if we have enough edges for connectivity
        if edge_count < 1:
            error_msg = (
                f"{self.name} ERROR: Failed to create any edges from CONECT records."
            )
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Additional validation - ensure the graph is reasonably connected
        # For test compounds, be more lenient with connectivity requirements
        min_edges = len(self.atoms) - 1 if self.is_reference else len(self.atoms) - 3
        if len(G.edges) < min_edges:
            # Try to infer missing edges based on distance
            self._infer_missing_edges(G)

            # Check connectivity again
            if len(G.edges) < min_edges:
                error_msg = f"{self.name} ERROR: Graph is not fully connected after using CONECT records and distance-based inference. Got {len(G.edges)} edges but need at least {min_edges}."
                if self.logger:
                    self.logger.error(error_msg)
                    # Log detailed connectivity information
                    self._log_connectivity_info(G)
                raise ValueError(error_msg)

                                if self.logger:
            self.logger.info(
                f"{self.name} Graph: {len(G.nodes)} nodes, {len(G.edges)} edges"
            )

        # Print degree distribution
        degrees = {}
        for node in G.nodes():
            degrees[node] = len(
                list(G.neighbors(node))
            )  # Use neighbors instead of degree
        terminal_nodes = [n for n in G.nodes if degrees[n] == 1]
                if self.logger:
                    self.logger.info(
                f"{self.name} Degree distribution: {sorted(degrees.values())}"
            )
            self.logger.info(f"{self.name} Terminal nodes: {len(terminal_nodes)}")

            # Print element distribution
            elements = [cast(Dict[str, str], G.nodes[n])["element"] for n in G.nodes]
            element_counts: Dict[str, int] = {}
            for e in elements:
                if e not in element_counts:
                    element_counts[e] = 0
                element_counts[e] += 1
            self.logger.info(f"{self.name} Element distribution: {element_counts}")

            # Visualize the graph connectivity for debugging
            components = list(nx.connected_components(G))
            self.logger.info(f"{self.name} Connected components: {len(components)}")
            for i, comp in enumerate(components):
                self.logger.info(f"{self.name} Component {i+1}: {len(comp)} nodes")
                # Print a few atoms in each component
                if len(comp) > 0:
                    sample = list(comp)[:3]  # First 3 atoms
                        self.logger.info(
                        f"  Sample atoms: {[cast(Dict[str, str], G.nodes[n])['name'] for n in sample]}"
                    )

        return G

    def _infer_missing_edges(self, G: nx.Graph) -> None:
        """Infer missing edges based on atomic distances.

        Args:
            G: NetworkX graph to add edges to
        """
        if self.logger:
            self.logger.info(f"{self.name} Inferring missing edges based on distances")

        # Define typical bond lengths (in Angstroms)
        max_bond_lengths = {
            ("C", "C"): 1.6,  # Single C-C bond
            ("C", "N"): 1.5,  # C-N bond
            ("C", "O"): 1.5,  # C-O bond
            ("N", "N"): 1.5,  # N-N bond
            ("N", "O"): 1.5,  # N-O bond
            ("O", "O"): 1.5,  # O-O bond
        }

        edges_added = 0
        for i, atom1 in enumerate(self.atoms):
            for j, atom2 in enumerate(self.atoms[i + 1 :], i + 1):
                # Skip if edge already exists
                if G.has_edge(atom1, atom2):
                    continue

                # Get elements
                elem1 = G.nodes[atom1].get("element", "")
                elem2 = G.nodes[atom2].get("element", "")

                # Skip if either atom is hydrogen
                if elem1 == "H" or elem2 == "H":
                    continue

                # Get bond length threshold - try both orderings
                bond_pair = tuple(sorted([elem1, elem2]))
                max_length = max_bond_lengths.get(bond_pair, 1.6)  # Default to 1.6Å

                # Calculate distance
                dist = np.linalg.norm(atom1.coord - atom2.coord)

                # Add edge if atoms are close enough
                if dist <= max_length:
                                        G.add_edge(atom1, atom2)
                    edges_added += 1
                                        if self.logger:
                                            self.logger.debug(
                            f"{self.name} Added inferred edge: {atom1.name}-{atom2.name} ({dist:.2f}Å)"
                                            )

            if self.logger:
            self.logger.info(f"{self.name} Added {edges_added} inferred edges")

    def _log_connectivity_info(self, G: nx.Graph) -> None:
        """Log detailed connectivity information for debugging.

        Args:
            G: NetworkX graph to analyze
        """
        if not self.logger:
            return

        # Log all edges
        self.logger.debug(f"\n{self.name} Edge list:")
        for edge in G.edges():
            atom1, atom2 = edge
            self.logger.debug(
                f"  {atom1.name}({atom1.serial_number}) - {atom2.name}({atom2.serial_number})"
            )

        # Find isolated atoms
        isolated = list(nx.isolates(G))
        if isolated:
            self.logger.debug(f"\n{self.name} Isolated atoms:")
            for atom in isolated:
                self.logger.debug(
                    f"  {atom.name}({atom.serial_number}) - Element: {atom.element if hasattr(atom, 'element') else 'Unknown'}"
                )

        # Analyze connected components
        components = list(nx.connected_components(G))
        self.logger.debug(f"\n{self.name} Connected components analysis:")
        for i, comp in enumerate(components):
            self.logger.debug(f"\nComponent {i+1} ({len(comp)} atoms):")
            for atom in comp:
                neighbors = list(G.neighbors(atom))
                self.logger.debug(
                    f"  {atom.name}({atom.serial_number}) - {len(neighbors)} connections: "
                    f"{[f'{n.name}({n.serial_number})' for n in neighbors]}"
                )

    def _add_conect_edges(self, G: nx.Graph) -> int:
        """Add edges based on CONECT records from PDB file.

        Returns:
            Number of edges added
        """
        if not self.conect_dict:
            return 0

        if self.logger:
            self.logger.info(
                f"{self.name} Using CONECT records from PDB for connectivity"
            )

            edge_count = 0
        missing_serials = []
        missing_connections = []

        # Direct mapping using CONECT records based on serial numbers
            for serial, connected_serials in self.conect_dict.items():
            # Serial number needs to be in our atom list
            if serial in self.serial_to_atom:
                atom1 = self.serial_to_atom[serial]
                    for connected_serial in connected_serials:
                    if connected_serial in self.serial_to_atom:
                        atom2 = self.serial_to_atom[connected_serial]
                            # Only add each edge once (avoid duplicates)
                            if atom1 != atom2 and not G.has_edge(atom1, atom2):
                                G.add_edge(atom1, atom2)
                                edge_count += 1
                        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                                    self.logger.debug(
                                        f"{self.name} Edge from CONECT: {atom1.name}({serial})-{atom2.name}({connected_serial})"
                                    )
        else:
                        missing_connections.append(connected_serial)
                        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(
                                f"Connected serial {connected_serial} not found in atom list"
                            )
            else:
                missing_serials.append(serial)
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Serial {serial} not found in atom list")

        # Log any issues with missing serials
        if missing_serials and self.logger:
            self.logger.warning(
                f"{self.name} Could not find {len(missing_serials)} atom serials in the structure: {missing_serials[:10]}..."
            )

        if missing_connections and self.logger:
            self.logger.warning(
                f"{self.name} Could not find {len(missing_connections)} connected atom serials: {missing_connections[:10]}..."
                                )

        if self.logger:
            self.logger.info(
                f"{self.name} Added {edge_count} edges from CONECT records"
            )

        return edge_count

    def _get_residue_info(self, residue: Optional[Residue]) -> Tuple[str, str]:
        """Get residue name and ID safely.

        Args:
            residue: Bio.PDB.Residue object or None

        Returns:
            Tuple of (residue_name, residue_id)
        """
        if residue is None:
            return "", ""
        return getattr(residue, "resname", ""), str(getattr(residue, "id", ""))

    def _get_atom_element(self, atom: Atom) -> str:
        """Get the element of an atom safely.

        Args:
            atom: Bio.PDB.Atom object

        Returns:
            Element of the atom
        """
        return getattr(atom, "element", "C")  # Default to carbon if element is not set


class IsomorphicSuperimposer:
    """Graph isomorphism-based structure alignment."""

    def __init__(self, logger=None):
        """Initialize with optional logger."""
        self.logger = logger

    def align(self, mol1: MolecularGraph, mol2: MolecularGraph) -> AlignmentResult:
        """Implement graph matching algorithm."""
        if self.logger:
            self.logger.info(
                f"Starting isomorphic matching between graphs with {len(mol1.graph.nodes)} and {len(mol2.graph.nodes)} nodes"
            )

        matches = self._custom_graph_match(mol1.graph, mol2.graph)

        if not matches:
            if self.logger:
                self.logger.warning("No isomorphic matches found")
            return self._empty_result()

        matched_pairs = list(matches.items())
        if self.logger:
            self.logger.info(f"Found {len(matched_pairs)} matched atom pairs")
            for i, (a1, a2) in enumerate(
                matched_pairs[:10]
            ):  # Show first 10 for brevity
                self.logger.info(
                    f"  Match {i+1}: {a1.name} ({mol1.graph.nodes[a1]['element']}) -> {a2.name} ({mol2.graph.nodes[a2]['element']})"
                )

        matched_mol1_atoms = [pair[0] for pair in matched_pairs]
        matched_mol2_atoms = [pair[1] for pair in matched_pairs]

        try:
            superimposer = Superimposer()
            superimposer.set_atoms(matched_mol1_atoms, matched_mol2_atoms)
            rotran = superimposer.rotran  # Store this to avoid None issues

            if self.logger and rotran:
                self.logger.info(f"Superimposition RMSD: {superimposer.rms:.4f}")
                if rotran[0] is not None:
                    self.logger.info(f"Rotation matrix:\n{rotran[0]}")
                if rotran[1] is not None:
                    self.logger.info(f"Translation vector: {rotran[1]}")

            # Make sure we have a valid RMSD
            rmsd = float(
                superimposer.rms if superimposer.rms is not None else float("inf")
            )

            return AlignmentResult(
                rmsd=rmsd,
                matched_atoms=len(matched_pairs),
                transformation_matrix=(
                    rotran
                    if rotran and rotran[0] is not None and rotran[1] is not None
                    else None
                ),
                matched_pairs=matched_pairs,
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during superimposition: {str(e)}")
            return self._empty_result()

    def _custom_graph_match(
        self, reference_graph: nx.Graph, model_graph: nx.Graph
    ) -> Dict[Any, Any]:
        """Custom graph matching implementation."""
        if self.logger:
            self.logger.info("Starting custom graph matching...")

        def node_matches(node1, node2):
            """Match atoms based on element type with fallback to name matching."""
            # Get element and name information
            element1 = reference_graph.nodes[node1].get("element", "").strip()
            element2 = model_graph.nodes[node2].get("element", "").strip()
            name1 = reference_graph.nodes[node1].get("name", "").strip()
            name2 = model_graph.nodes[node2].get("name", "").strip()

            # Skip hydrogen atoms
            if element1 == "H" or element2 == "H":
                return False

            # First, check if elements match
            if element1 == element2:
                return True

            # If elements don't match but names suggest they should
            # (Sometimes the same atom might be labeled with different elements in different files)
            if name1 == name2 and len(name1) > 0:
                if self.logger:
                    self.logger.debug(
                        f"Element mismatch but names match: {name1}({element1}) and {name2}({element2})"
                    )
                return True

            # Special case for common element mismatches
            # For example, sometimes carbons are labeled as X or dummy atoms
            if (element1 == "C" and element2 in ["X", "Du"]) or (
                element2 == "C" and element1 in ["X", "Du"]
            ):
                if self.logger:
                    self.logger.debug(
                        f"Special case element match: {element1} and {element2}"
                    )
                return True

            return False

        def explore_neighborhood(
            ref_node, model_node, visited_ref=None, visited_model=None, depth=0
        ):
            """Recursively explore and match the neighborhoods of two nodes."""
            if visited_ref is None:
                visited_ref = set()
            if visited_model is None:
                visited_model = set()

            indent = "  " * depth
            if self.logger:
                self.logger.debug(
                    f"{indent}Exploring: {ref_node.name} -> {model_node.name}"
                )

            current_match = {ref_node: model_node}
            visited_ref.add(ref_node)
            visited_model.add(model_node)

            ref_neighbors = list(reference_graph.neighbors(ref_node))
            model_neighbors = list(model_graph.neighbors(model_node))

            if self.logger:
                self.logger.debug(
                    f"{indent}Ref neighbors: {[n.name for n in ref_neighbors]}"
                )
                self.logger.debug(
                    f"{indent}Model neighbors: {[n.name for n in model_neighbors]}"
                )

            for ref_neighbor in ref_neighbors:
                if ref_neighbor in visited_ref:
                    if self.logger:
                        self.logger.debug(
                            f"{indent}Skipping visited: {ref_neighbor.name}"
                        )
                    continue

                potential_matches = [
                    model_neighbor
                    for model_neighbor in model_neighbors
                    if node_matches(ref_neighbor, model_neighbor)
                    and model_neighbor not in visited_model
                ]

                if self.logger:
                    self.logger.debug(
                        f"{indent}For {ref_neighbor.name}, potential matches: {[n.name for n in potential_matches]}"
                    )

                if not potential_matches:
                    if self.logger:
                        self.logger.debug(
                            f"{indent}No potential matches for {ref_neighbor.name}"
                        )
                    return None

                for model_match in potential_matches:
                    if self.logger:
                        self.logger.debug(
                            f"{indent}Trying: {ref_neighbor.name} -> {model_match.name}"
                        )

                    neighbor_match = explore_neighborhood(
                        ref_neighbor,
                        model_match,
                        visited_ref.copy(),
                        visited_model.copy(),
                        depth + 1,
                    )

                    if neighbor_match:
                        current_match.update(neighbor_match)
                        break
                else:
                    if self.logger:
                        self.logger.debug(
                            f"{indent}Failed to match any neighbor for {ref_neighbor.name}"
                        )
                    return None

            if self.logger:
                self.logger.debug(
                    f"{indent}Returning match set with {len(current_match)} pairs"
                )
            return current_match

        # Get all non-hydrogen atoms from both graphs
        ref_atoms = [
            n
            for n in reference_graph.nodes()
            if reference_graph.nodes[n].get("element", "").strip() != "H"
        ]
        model_atoms = [
            n
            for n in model_graph.nodes()
            if model_graph.nodes[n].get("element", "").strip() != "H"
        ]

        if self.logger:
            self.logger.info(
                f"Found {len(ref_atoms)} non-H reference atoms and {len(model_atoms)} non-H model atoms"
            )

        best_match = {}

        # Try all possible starting points
        for ref_atom in ref_atoms:
            ref_element = reference_graph.nodes[ref_atom].get("element", "").strip()

            # Only try atoms with matching elements
            matching_model_atoms = [
                m
                for m in model_atoms
                if model_graph.nodes[m].get("element", "").strip() == ref_element
            ]

            if self.logger:
                self.logger.debug(
                    f"Trying reference atom: {ref_atom.name} ({ref_element}) - {len(matching_model_atoms)} potential matches"
                )

            # Progress tracking for large molecules
            if len(matching_model_atoms) > 10 and self.logger:
                self.logger.info(
                    f"Testing {ref_atom.name} against {len(matching_model_atoms)} model atoms..."
                )

            # Try each potential match
            for i, model_atom in enumerate(matching_model_atoms):
                # Print progress occasionally
                if i % 10 == 0 and len(matching_model_atoms) > 20 and self.logger:
                    self.logger.info(
                        f"  Tried {i}/{len(matching_model_atoms)} model atoms..."
                    )

                match = explore_neighborhood(ref_atom, model_atom)
                if match and len(match) > len(best_match):
                    best_match = match
                    if self.logger:
                        self.logger.info(
                            f"Found improved match with {len(match)} atoms"
                        )

                    # If we've found a complete match, stop searching
                    if len(match) == len(ref_atoms) and self.logger:
                        self.logger.info("Found complete match, stopping search")
                        break

            # If we've found a good enough match, stop searching
            # "Good enough" means most of the reference atoms are matched
            if len(best_match) >= 0.8 * len(ref_atoms) and self.logger:
                self.logger.info(
                    f"Found good match with {len(best_match)}/{len(ref_atoms)} atoms, stopping search"
                )
                break

        if self.logger:
            self.logger.info(f"Final match contains {len(best_match)} atom pairs")

        return best_match

    @staticmethod
    def _empty_result() -> AlignmentResult:
        """Return an empty alignment result."""
        return AlignmentResult(
            rmsd=float("inf"),
            matched_atoms=0,
            transformation_matrix=None,
            matched_pairs=[],
        )


def setup_logging(verbose_level) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(__name__)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set up file handler
    log_file = Path("superimposition.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Set up console handler if verbose
    if verbose_level:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

        if verbose_level == "debug":
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    return logger


@functools.lru_cache(maxsize=16)
def get_pdb_lines(pdb_file: str) -> List[str]:
    """Get lines from a PDB file with caching.

    Args:
        pdb_file: Path to the PDB file

    Returns:
        List of lines from the PDB file
    """
    # Check file cache first
    if pdb_file in _FILE_CACHE:
        return _FILE_CACHE[pdb_file]

    # Read the file and cache it
    with open(pdb_file, "r") as f:
        lines = f.readlines()

    # Cache the result for future use
    _FILE_CACHE[pdb_file] = lines
    return lines


def get_structure(pdb_file: str, structure_id: Optional[str] = None) -> Structure:
    """Get a BioPython structure from a PDB file with caching.

    Args:
        pdb_file: Path to the PDB file
        structure_id: Identifier for the structure

    Returns:
        BioPython Structure object
    """
    # Create a unique cache key from file path and structure ID
    cache_key = f"{pdb_file}_{structure_id if structure_id else 'default'}"

    # Check structure cache
    if cache_key in _PDB_STRUCTURE_CACHE:
        return _PDB_STRUCTURE_CACHE[cache_key]

    # Parse the structure
    if structure_id is None:
        structure_id = Path(pdb_file).stem

    structure = PDB_PARSER.get_structure(structure_id, pdb_file)
    if not isinstance(structure, Structure):
        raise ValueError(f"Failed to parse structure from {pdb_file}")

    # Cache the result
    _PDB_STRUCTURE_CACHE[cache_key] = structure
    return structure


def extract_chain_from_reference(
    reference_pdb: str, chain_ids: List[str], output_file: Path, logger: logging.Logger
) -> Path:
    """Extract specific chains from a PDB file.

    This optimized version uses caching to avoid re-reading files.

    Args:
        reference_pdb: Path to reference PDB file
        chain_ids: List of chain IDs to extract
        output_file: Output file path
        logger: Logger instance

    Returns:
        Path to extracted chain file
    """
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if output file already exists
    if output_file.exists():
        logger.info(f"Using existing extracted chain file: {output_file}")
        return output_file

    # Get the structure from cache or disk
    structure = get_structure(reference_pdb, f"ref_{'-'.join(chain_ids)}")

    # Create a new structure with only the desired chains
    logger.info(f"Extracting chains {chain_ids} from {reference_pdb}")

    # Write the structure with only the selected chains
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_file), ChainSelect(chain_ids))

    logger.info(f"Saved extracted chains to {output_file}")
    return output_file


def parse_conect_records(
    pdb_file: str, model_num: Optional[int] = None
) -> Dict[int, List[int]]:
    """Parse CONECT records from a PDB file.

    Args:
        pdb_file: Path to PDB file
        model_num: Model number to extract (if None, get all)

    Returns:
        Dictionary mapping atom serial numbers to lists of connected atom serial numbers
    """
    # Create a cache key that includes the model number
    cache_key = f"{pdb_file}_{model_num if model_num is not None else 'all'}"

    # Check cache first
    if cache_key in _CONECT_CACHE:
        return _CONECT_CACHE[cache_key]

    # Read the PDB file lines (using the cached version if available)
    lines = get_pdb_lines(pdb_file)

    # Extract CONECT records
        conect_dict = {}
    model_found = model_num is None  # If model_num is None, process all lines

    for line in lines:
        if model_num is not None:
            if line.startswith("MODEL"):
                # Check if this is the model we want
                try:
                    current_model = int(line.split()[1])
                    model_found = current_model == model_num
                except (IndexError, ValueError):
                    model_found = False
            elif line.startswith("ENDMDL"):
                model_found = False

            if not model_found:
                continue

        if line.startswith("CONECT"):
            parts = line.split()
            if len(parts) > 2:  # At least one connection
                atom_serial = int(parts[1])
                connected_serials = [int(parts[i]) for i in range(2, len(parts))]
                if atom_serial not in conect_dict:
                    conect_dict[atom_serial] = []
                conect_dict[atom_serial].extend(connected_serials)

    # Cache the result
    _CONECT_CACHE[cache_key] = conect_dict
    return conect_dict


def count_models_in_pdb(pdb_path: str) -> int:
    """Count the number of models in a PDB file.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Number of models
    """
    # This is a quick counting function that doesn't need to parse the full file
    lines = get_pdb_lines(pdb_path)

    model_count = 0
    for line in lines:
            if line.startswith("MODEL"):
                model_count += 1

    # If no MODEL records found, assume it's a single model
    return max(1, model_count)


def parse_pdb_for_ligand(
    pdb_file: str, chain_id: str, logger: logging.Logger
) -> Tuple[List[Atom], Dict[int, List[int]]]:
    """Parse a PDB file directly to extract atoms and CONECT records.

    This approach preserves original serial numbers and extracts exact connectivity information.

    Args:
        pdb_file: Path to PDB file
        chain_id: Chain ID to extract (e.g., "D" for ligand)
        logger: Logger instance

    Returns:
        Tuple of (list of atoms, CONECT dictionary)
    """
    # Get all lines from the PDB
    lines = get_pdb_lines(pdb_file)

    # Extract atoms and CONECT records
    atoms = []
    conect_dict = {}
    atom_serial_to_idx = {}

    # First extract all atoms for the specified chain
    for line in lines:
        if line.startswith(("ATOM ", "HETATM")):
            # Check chain ID (column 21-22)
            if line[21:22].strip() == chain_id:
                try:
                    serial = int(line[6:11].strip())
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()

                    # Add to our list and track serial number
                    atoms.append((serial, atom_name, residue_name, line))
                    atom_serial_to_idx[serial] = len(atoms) - 1
                except ValueError:
                    logger.warning(f"Skipping line with invalid serial number: {line}")

    logger.info(f"Extracted {len(atoms)} atoms from chain {chain_id}")

    # Now extract CONECT records and filter only those that connect our chain's atoms
    for line in lines:
        if line.startswith("CONECT"):
                    parts = line.split()
                    if len(parts) > 2:  # At least one connection
                        try:
                            atom_serial = int(parts[1])
                    # Only include if this atom is in our chain
                    if atom_serial in atom_serial_to_idx:
                        connected_serials = []
                        for i in range(2, len(parts)):
                            connected_serial = int(parts[i])
                            # Only include connections to atoms in our chain
                            if connected_serial in atom_serial_to_idx:
                                connected_serials.append(connected_serial)

                        if (
                            connected_serials
                        ):  # Only add if there are connections to our chain
                            if atom_serial not in conect_dict:
                                conect_dict[atom_serial] = []
                            conect_dict[atom_serial].extend(connected_serials)
                        except ValueError:
                    logger.warning(
                        f"Skipping CONECT with invalid serial number: {line}"
                    )

    # Log connectivity info
    total_connections = sum(len(v) for v in conect_dict.values())
    logger.info(
        f"Found {len(conect_dict)} atoms with connections in chain {chain_id} (total {total_connections} connections)"
    )

    # Load the structure with BioPython but preserve original serial numbers
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        for _, _, _, line in atoms:
            f.write(line)
        temp_file = f.name

    try:
        # Parse the structure
        structure = get_structure("chain_" + chain_id, temp_file)
        if not isinstance(structure, Structure):
            raise ValueError(f"Failed to parse structure from {temp_file}")

        # Get the atoms and tag them with original serials
        biopython_atoms = list(structure.get_atoms())

        # Verify we have the same number of atoms
        if len(biopython_atoms) != len(atoms):
            logger.warning(
                f"Warning: BioPython parsed {len(biopython_atoms)} atoms but we extracted {len(atoms)} atoms"
            )

        # Assign original serial numbers to BioPython atoms
        for i, atom in enumerate(biopython_atoms):
            if i < len(atoms):
                original_serial = atoms[i][0]
                setattr(atom, "original_serial", original_serial)
                # Log for debugging
                logger.debug(
                    f"Atom {i}: BioPython serial={atom.serial_number}, Original={original_serial}, Name={atom.name}"
                )

        return biopython_atoms, conect_dict
    finally:
        # Clean up temp file
        os.unlink(temp_file)


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Calculate RMSD between two sets of coordinates.

    Args:
        coords1: First set of coordinates (N x 3)
        coords2: Second set of coordinates (N x 3)

    Returns:
        RMSD value
    """
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff * diff, axis=1)))


def superimpose_coordinates(
    ref_coords: np.ndarray, mov_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Superimpose one set of coordinates onto another using SVD.

    Args:
        ref_coords: Reference coordinates (N x 3)
        mov_coords: Moving coordinates (N x 3)

    Returns:
        Tuple of (rotation matrix, translation vector, RMSD)
    """
    # Center both coordinate sets
    ref_center = np.mean(ref_coords, axis=0)
    mov_center = np.mean(mov_coords, axis=0)

    ref_centered = ref_coords - ref_center
    mov_centered = mov_coords - mov_center

    # Calculate correlation matrix
    corr = np.dot(mov_centered.T, ref_centered)

    # SVD
    V, S, Wt = np.linalg.svd(corr)

    # Ensure right-handed coordinate system
    d = np.sign(np.linalg.det(np.dot(V, Wt)))
    V[:, -1] *= d

    # Calculate rotation matrix
    R = np.dot(V, Wt)

    # Calculate translation
    t = ref_center - np.dot(mov_center, R)

    # Calculate RMSD
    aligned_coords = np.dot(mov_coords, R) + t
    rmsd = calculate_rmsd(ref_coords, aligned_coords)

    return R, t, rmsd


def check_clashes(
    coords: np.ndarray, protein_coords: np.ndarray, cutoff: float = 2.0
) -> Tuple[bool, int]:
    """Check for steric clashes between a molecule and protein.

    Args:
        coords: Coordinates to check (N x 3)
        protein_coords: Protein coordinates (M x 3)
        cutoff: Distance cutoff for clashes in Angstroms

    Returns:
        Tuple of (has_clashes, num_clashing_atoms)
    """
    num_clashing = 0
    for coord in coords:
        dists = np.sqrt(np.sum((protein_coords - coord) ** 2, axis=1))
        if np.any(dists < cutoff):
            num_clashing += 1
    return num_clashing > 0, num_clashing


def assign_chain_ids(
    structure: Structure, logger: Optional[logging.Logger] = None
) -> None:
    """Assign chain IDs based on atom connectivity and chemical properties.

    Instead of relying on residue types, this function identifies the ligand
    based on its connectivity pattern and chemical composition.

    Args:
        structure: BioPython Structure object
        logger: Optional logger instance
    """
    for model in structure:
        # First, collect all atoms that aren't part of standard amino acids
        residue_atoms = {}
        serial_map = {}  # Map old serial numbers to new ones

        for atom in model.get_atoms():
            residue = atom.get_parent()
            # Store original serial number
            serial_map[atom.serial_number] = atom.serial_number

            if residue.resname not in [
                "ALA",
                "CYS",
                "ASP",
                "GLU",
                "PHE",
                "GLY",
                "HIS",
                "ILE",
                "LYS",
                "LEU",
                "MET",
                "ASN",
                "PRO",
                "GLN",
                "ARG",
                "SER",
                "THR",
                "VAL",
                "TRP",
                "TYR",
            ]:
                if residue.id not in residue_atoms:
                    residue_atoms[residue.id] = {"non_protein": [], "residue": residue}
                residue_atoms[residue.id]["non_protein"].append(atom)
        else:
                if residue.id not in residue_atoms:
                    residue_atoms[residue.id] = {"protein": [], "residue": residue}
                residue_atoms[residue.id]["protein"].append(atom)

        # Create chain D for non-protein atoms (potential ligand)
        if any("non_protein" in atoms for atoms in residue_atoms.values()):
            if "D" not in model:
                chain_d = Chain("D")
                model.add(chain_d)
            else:
                chain_d = model["D"]

            # Move non-protein residues to chain D
            for res_id, res_data in residue_atoms.items():
                if "non_protein" in res_data:
                    # Create a new residue in chain D
                    new_residue = res_data["residue"].copy()
                    new_residue.detach_parent()  # Remove old parent reference
                    chain_d.add(new_residue)

                    # Clear existing atoms from the new residue
                    for atom in list(new_residue.get_atoms()):
                        new_residue.detach_child(atom.name)

                    # Add atoms to the new residue and update serial numbers
                    for atom in res_data["non_protein"]:
                        if atom.get_parent():
                            atom.get_parent().detach_child(atom.name)
                        new_atom = atom.copy()
                        new_atom.serial_number = (
                            atom.serial_number
                        )  # Preserve serial number
                        serial_map[atom.serial_number] = new_atom.serial_number
                        new_residue.add(new_atom)

            if logger:
                logger.info(f"Moved non-protein atoms to chain D")

        # Create chain A for protein atoms if needed
        if any("protein" in atoms for atoms in residue_atoms.values()):
            if "A" not in model:
                chain_a = Chain("A")
                model.add(chain_a)
            else:
                chain_a = model["A"]

            # Move protein residues to chain A
            for res_id, res_data in residue_atoms.items():
                if "protein" in res_data:
                    # Create a new residue in chain A
                    new_residue = res_data["residue"].copy()
                    new_residue.detach_parent()  # Remove old parent reference
                    chain_a.add(new_residue)

                    # Clear existing atoms from the new residue
                    for atom in list(new_residue.get_atoms()):
                        new_residue.detach_child(atom.name)

                    # Add atoms to the new residue and update serial numbers
                    for atom in res_data["protein"]:
                        if atom.get_parent():
                            atom.get_parent().detach_child(atom.name)
                        new_atom = atom.copy()
                        new_atom.serial_number = (
                            atom.serial_number
                        )  # Preserve serial number
                        serial_map[atom.serial_number] = new_atom.serial_number
                        new_residue.add(new_atom)

            if logger:
                logger.info(f"Moved protein atoms to chain A")

        # Update CONECT records based on serial number mapping
        if hasattr(structure, "original_conect"):
            new_conect = {}
            for old_serial, connected in structure.original_conect.items():
                if old_serial in serial_map:
                    new_serial = serial_map[old_serial]
                    new_conect[new_serial] = [
                        serial_map[s] for s in connected if s in serial_map
                    ]
            structure.original_conect = new_conect


def parse_reference_data(reference_pdb: str, logger: logging.Logger) -> bool:
    """Parse and store reference data in an optimized format.

    Args:
        reference_pdb: Path to reference PDB file
        logger: Logger instance

    Returns:
        True if successful, False otherwise
    """
    try:
        # Parse reference PDB file
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("reference", reference_pdb)
        model = structure[0]

        # Get CONECT records first
        conect_dict = parse_conect_records(reference_pdb)
        if not conect_dict:
            logger.error("No CONECT records found in reference structure")
            return False

        # Extract ligand chain (D) and create molecular graph
        if "D" not in model:
            logger.error("No chain D found in reference structure")
            return False

        ligand_chain = model["D"]
        ligand_atoms = list(ligand_chain.get_atoms())
        ligand_coords = np.array([atom.get_coord() for atom in ligand_atoms])

        # Create molecular graph for ligand
        ref_ligand_graph = MolecularGraph(
            ligand_atoms,
            logger=logger,
            name="reference",
            conect_dict=conect_dict,
            is_reference=True,
        )

        # Extract protein coordinates (chains A,B,C)
        protein_coords = []
        protein_atoms = []
        for chain_id in ["A", "B", "C"]:
            if chain_id in model:
                chain = model[chain_id]
                for atom in chain.get_atoms():
                    protein_coords.append(atom.get_coord())
                    protein_atoms.append(atom)

        if not protein_coords:
            logger.warning("No protein atoms found in chains A,B,C")
            protein_coords = np.empty((0, 3))
        else:
            protein_coords = np.array(protein_coords)

        # Cache reference data
        _REFERENCE_DATA["target_protein_coords"] = protein_coords
        _REFERENCE_DATA["ref_ligand_graph"] = ref_ligand_graph
        _REFERENCE_DATA["ref_ligand_coords"] = ligand_coords
        _REFERENCE_DATA["ref_conect"] = conect_dict

        # Log success with details
        logger.info(f"Successfully cached reference data:")
        logger.info(f"- Protein atoms: {len(protein_coords)}")
        logger.info(f"- Ligand atoms: {len(ligand_atoms)}")
        logger.info(f"- CONECT records: {len(conect_dict)}")
        return True

    except Exception as e:
        logger.error(f"Failed to parse reference data: {str(e)}")
        return False


def process_model(
    test_compound: Structure, model_idx: int, logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Process a single test compound model.

    This function reuses isomorphic mappings for subsequent models of the same compound.

    Args:
        test_compound: BioPython Structure object for the test compound
        model_idx: Model index in trajectory
        logger: Logger instance

    Returns:
        Dictionary with alignment results or None if failed
    """
    try:
        # Get compound ID from structure
        compound_id = test_compound._id

        # Check if we have cached isomorphic mapping
        if model_idx == 0 or compound_id not in _ISOMORPHIC_CACHE:
            # First model or new compound - need to find isomorphic mapping
            logger.info(f"Finding isomorphic mapping for compound {compound_id}")

            # Parse CONECT records and create molecular graph
            test_conect = parse_conect_records(compound_id)
            if not test_conect:
                logger.error(f"No CONECT records found for test compound {compound_id}")
                return None

            # Assign atoms to chains
            assign_chain_ids(test_compound, logger)

            # Get chain D (non-protein atoms)
            model = test_compound[0]
            if "D" not in model:
                logger.warning(
                    f"No non-protein atoms found to match in model {model_idx}"
                )
                return None

            test_chain = model["D"]
            test_atoms = list(test_chain.get_atoms())
            if not test_atoms:
                logger.warning(
                    f"No atoms found in non-protein chain in model {model_idx}"
                )
                return None

            # Create molecular graph for test compound
            test_graph = MolecularGraph(
                test_atoms, logger=logger, name="test_compound", conect_dict=test_conect
            )

            # Find isomorphic mapping
            superimposer = IsomorphicSuperimposer(logger)
            result = superimposer.align(_REFERENCE_DATA["ref_ligand_graph"], test_graph)

            if not result.isomorphic_match:
                logger.warning(f"No matching substructure found in model {model_idx}")
                return None

            # Cache the mapping and atom indices for future models
            _ISOMORPHIC_CACHE[compound_id] = {
                "matched_pairs": result.matched_pairs,
                "test_atoms": test_atoms,
                "test_atom_indices": [atom.serial_number for atom in test_atoms],
            }

        # Get cached mapping
        cached = _ISOMORPHIC_CACHE[compound_id]
        matched_pairs = cached["matched_pairs"]
        test_atoms = cached["test_atoms"]

        # Create mapping from reference atom index to matched test atom
        ref_to_test = {}
        for ref_atom, test_atom in matched_pairs:
            ref_idx = _REFERENCE_DATA["ref_ligand_graph"].atoms.index(ref_atom)
            ref_to_test[ref_idx] = test_atom

        # Get coordinates for matched atoms
        ref_coords = []
        test_coords = []
        for i, ref_atom in enumerate(_REFERENCE_DATA["ref_ligand_graph"].atoms):
            if i in ref_to_test:
                ref_coords.append(ref_atom.coord)
                test_coords.append(ref_to_test[i].coord)

        ref_coords = np.array(ref_coords)
        test_coords = np.array(test_coords)

        # Calculate optimal transformation
        rotation, translation = calculate_optimal_transform(
            ref_coords, test_coords, logger
        )

        # Apply transformation to matched atoms to get RMSD
        test_coords_transformed = np.dot(test_coords, rotation.T) + translation
        matched_rmsd = calculate_rmsd(ref_coords, test_coords_transformed)

        if logger:
            logger.info(f"RMSD for matched atoms: {matched_rmsd:.4f} Å")

        # Apply transformation to all test atoms for clash detection
        all_test_coords = np.array([atom.get_coord() for atom in test_atoms])
        transformed_coords = np.dot(all_test_coords, rotation.T) + translation

        # Check for clashes with protein
        has_clashes, num_clashing = False, 0
        if len(_REFERENCE_DATA["target_protein_coords"]) > 0:
            has_clashes, num_clashing = check_clashes(
                transformed_coords, _REFERENCE_DATA["target_protein_coords"]
            )
            if logger and has_clashes:
                logger.warning(f"Detected {num_clashing} atoms with protein clashes")

        return {
            "rmsd": float(matched_rmsd),  # RMSD of matched atoms only
            "matched_atoms": len(matched_pairs),
            "has_clashes": has_clashes,
            "num_clashing": num_clashing,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        }

    except Exception as e:
        logger.error(f"Failed to process test compound model {model_idx}: {str(e)}")
        return None


def calculate_optimal_transform(
    ref_coords: np.ndarray,
    mov_coords: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate optimal rotation and translation using Kabsch algorithm.

    Args:
        ref_coords: Reference coordinates (N x 3)
        mov_coords: Moving coordinates (N x 3)
        logger: Optional logger instance

    Returns:
        Tuple of (rotation matrix, translation vector)
    """
    if logger:
        logger.info("\nStarting optimal transform calculation:")
        logger.info(f"Reference coordinates shape: {ref_coords.shape}")
        logger.info(f"Moving coordinates shape: {mov_coords.shape}")
        logger.info("\nInitial coordinates:")
        for i, (ref, mov) in enumerate(zip(ref_coords, mov_coords)):
            logger.info(f"Atom {i}:")
            logger.info(f"  Reference: {ref}")
            logger.info(f"  Moving:    {mov}")
            logger.info(f"  Distance:  {np.linalg.norm(ref - mov):.4f} Å")

    # Center both coordinate sets
    ref_center = np.mean(ref_coords, axis=0)
    mov_center = np.mean(mov_coords, axis=0)

    if logger:
        logger.info(f"\nReference center: {ref_center}")
        logger.info(f"Moving center: {mov_center}")
        logger.info(f"Center distance: {np.linalg.norm(ref_center - mov_center):.4f} Å")

    ref_centered = ref_coords - ref_center
    mov_centered = mov_coords - mov_center

    if logger:
        logger.info("\nAfter centering:")
        for i, (ref, mov) in enumerate(zip(ref_centered, mov_centered)):
            logger.info(f"Atom {i}:")
            logger.info(f"  Reference: {ref}")
            logger.info(f"  Moving:    {mov}")
            logger.info(f"  Distance:  {np.linalg.norm(ref - mov):.4f} Å")

    # Calculate correlation matrix
    corr = np.dot(mov_centered.T, ref_centered)

    if logger:
        logger.info(f"\nCorrelation matrix:\n{corr}")

    # SVD
    V, S, Wt = np.linalg.svd(corr)

    if logger:
        logger.info(f"\nSVD components:")
        logger.info(f"V:\n{V}")
        logger.info(f"S: {S}")
        logger.info(f"Wt:\n{Wt}")

    # Ensure right-handed coordinate system
    d = np.sign(np.linalg.det(np.dot(V, Wt)))
    V[:, -1] *= d

    # Calculate rotation matrix
    rotation = np.dot(V, Wt)

    # Calculate translation
    translation = ref_center - np.dot(mov_center, rotation)

    if logger:
        logger.info(f"\nFinal transformation:")
        logger.info(f"Rotation matrix:\n{rotation}")
        logger.info(f"Translation vector: {translation}")

        # Test transformation
        mov_transformed = np.dot(mov_coords, rotation.T) + translation
        rmsd = np.sqrt(np.mean(np.sum((ref_coords - mov_transformed) ** 2, axis=1)))
        logger.info(f"\nRMSD after transformation: {rmsd:.4f} Å")

        logger.info("\nFinal coordinates:")
        for i, (ref, mov) in enumerate(zip(ref_coords, mov_transformed)):
            logger.info(f"Atom {i}:")
            logger.info(f"  Reference: {ref}")
            logger.info(f"  Transformed: {mov}")
            logger.info(f"  Distance: {np.linalg.norm(ref - mov):.4f} Å")

    return rotation, translation


def process_trajectory(args: Tuple[str, str, bool, Dict[str, Any]]) -> Optional[Dict]:
    """Process a single trajectory file.

    Args:
        args: Tuple containing (pdb_file, base_dir, verbose, reference_data)

    Returns:
        Dictionary of model metrics or None if failed
    """
    pdb_file, base_dir, verbose, reference_data = args

    # Set up logging for this process
    logger = setup_logging(verbose)

    try:
        # Initialize reference data for this process
        global _REFERENCE_DATA
        _REFERENCE_DATA = reference_data

        # Get number of models
        model_count = count_models_in_pdb(pdb_file)
        logger.info(f"Processing {model_count} models from {pdb_file}")

        # Create metrics file path
        metrics_file = Path(pdb_file).parent / "superimposition_metrics.json"
        metrics = {}

        # Load existing metrics if file exists
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not read existing metrics file: {metrics_file}")

        # Parse structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_file, pdb_file)  # Use pdb_file as ID
        model_count = len(structure)
        logger.info(f"Processing {model_count} models from {pdb_file}")

        # Process each model
        with tqdm(total=model_count, desc="Models", unit="model") as pbar:
            for model_idx, model in enumerate(structure):
                # Skip if already processed
                if str(model_idx + 1) in metrics:
                    pbar.update(1)
                    continue

                # Create a new structure for this model
                model_structure = Structure(pdb_file)  # Use pdb_file as ID
                model_structure.add(model)
                model_structure._id = pdb_file  # Set _id for CONECT record parsing

                # Superimpose model
                result = process_model(model_structure, model_idx, logger)
                if result:
                    metrics[str(model_idx + 1)] = result

                    # Update progress bar
                    pbar.set_postfix(
                        {
                            "RMSD": f"{result['rmsd']:.4f}",
                            "Clashes": result["has_clashes"],
                        }
                    )

                    # Save metrics after each model
                    with open(metrics_file, "w") as f:
                        json.dump(metrics, f, indent=2)

                pbar.update(1)

        return metrics

    except Exception as e:
        logger.error(f"Failed to process {pdb_file}: {str(e)}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Superimpose ligands onto a reference structure using RDKit MCS."
    )
    parser.add_argument("base_dir", help="Base directory containing md_Ref.pdb files")
    parser.add_argument(
        "reference_pdb",
        help="Reference PDB file with protein (chains A,B,C) and ligand (chain D)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: number of CPU cores)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already processed files",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("superimpose")

    # Initialize reference data
    print("\nInitializing reference structures...")
    if not parse_reference_data(args.reference_pdb, logger):
        return

    # Find trajectory files
    print(f"\nSearching for md_Ref.pdb files in {args.base_dir}...")
    base_path = Path(args.base_dir)
    pdb_files = list(base_path.rglob("md_Ref.pdb"))
    pdb_files = [str(path) for path in pdb_files]

    if not pdb_files:
        print(f"No md_Ref.pdb files found in {args.base_dir}")
        return

    print(f"Found {len(pdb_files)} files to process")

    # Set number of processes
    num_processes = args.num_processes or min(os.cpu_count() or 1, len(pdb_files))
    print(f"Using {num_processes} processes")

    # Process files in parallel
    with tqdm(total=len(pdb_files), desc="Files", unit="file") as pbar:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []

            for pdb_file in pdb_files:
                if not args.force:
                    metrics_file = (
                        Path(pdb_file).parent / "superimposition_metrics.json"
                    )
                    if metrics_file.exists():
                    pbar.update(1)
                    continue

                futures.append(
                    executor.submit(
                        process_trajectory,
                        (pdb_file, args.base_dir, args.verbose, _REFERENCE_DATA),
                    )
                )

            # Process results as they complete
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)

                if result:
                    avg_rmsd = np.mean([m["rmsd"] for m in result.values()])
                    clashes = sum(1 for m in result.values() if m["has_clashes"])
                    pbar.set_postfix(
                        {
                            "Avg RMSD": f"{avg_rmsd:.4f}",
                            "Clashes": f"{clashes}/{len(result)}",
                        }
                    )

    print("\nSuperimposition complete!")


def create_rdkit_mol_from_chain(
    chain: Chain, conect_dict: Dict[int, List[int]], logger: logging.Logger
) -> Optional[rdkit.Chem.Mol]:
    """Create an RDKit molecule from a BioPython chain and CONECT records.

    Args:
        chain: BioPython Chain object
        conect_dict: Dictionary of CONECT records
        logger: Logger instance

    Returns:
        RDKit molecule or None if failed
    """
    try:
        # Create empty RDKit molecule
        mol = Chem.RWMol()

        # Map from atom serial numbers to RDKit atom indices
        serial_to_idx = {}

        # Add atoms
        for residue in chain:
            for atom in residue:
                # Get element from atom name (strip numbers)
                element = "".join(c for c in atom.name if not c.isdigit()).strip()
                if element not in ["C", "N", "O", "S", "P", "H"]:
                    element = "C"  # Default to carbon for unknown elements

                # Add atom to molecule
                rdkit_atom = Chem.Atom(element)
                idx = mol.AddAtom(rdkit_atom)
                serial_to_idx[atom.serial_number] = idx

        # Add bonds from CONECT records
        bonds_added = set()
        for serial, connected in conect_dict.items():
            if serial in serial_to_idx:
                atom1_idx = serial_to_idx[serial]
                for connected_serial in connected:
                    if connected_serial in serial_to_idx:
                        atom2_idx = serial_to_idx[connected_serial]
                        # Avoid adding the same bond twice
                        bond_key = tuple(sorted([atom1_idx, atom2_idx]))
                        if bond_key not in bonds_added:
                            mol.AddBond(atom1_idx, atom2_idx, Chem.BondType.SINGLE)
                            bonds_added.add(bond_key)

        # Convert to non-editable molecule
        mol = mol.GetMol()

        # Calculate valence and add hydrogens
        for atom in mol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(mol)

        return mol

    except Exception as e:
        logger.error(f"Failed to create RDKit molecule: {str(e)}")
        return None


if __name__ == "__main__":
    main()
