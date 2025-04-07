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
    defaultdict,
)
import numpy as np
from numpy.typing import NDArray
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import sys
import functools
import traceback
import shutil

# Required import
import networkx as nx

try:
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB.PDBIO import PDBIO, Select
    from Bio.PDB.Superimposer import Superimposer
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import DisorderedAtom

    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print(
        "WARNING: BioPython not available. Please install it with: pip install biopython"
    )

    print(
        "WARNING: BioPython not available. Please install it with: pip install biopython"
    )

# Add RDKit import with fallback
try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available. Some advanced features will be disabled.")

    # Define placeholder for type annotations
    class rdkit:
        class Chem:
            class Mol:
                pass

    class Chem:
        pass


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
        require_conect: bool = False,  # Make CONECT records optional by default
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
            require_conect: Whether to require CONECT records (default: False)
        """
        self.atoms = atoms
        self.name = name
        self.logger = logger
        self.conect_dict = conect_dict
        self.structure = structure
        self.chain_id = chain_id
        self.is_reference = is_reference
        self.serial_to_atom = {}
        self.require_conect = require_conect
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
            # Get atom serial number
            serial = getattr(atom, "serial_number", None)

            # Some parsers use different attributes
            if serial is None:
                serial = getattr(atom, "serial", None)

            # Fallback to using an index
            if serial is None:
                if hasattr(atom, "get_serial_number"):
                    serial = atom.get_serial_number()
                else:
                    serial = len(serial_numbers) + 1

            # Track serial numbers for debugging
            serial_numbers.append(serial)

            # Store atom in maps
            serial_to_atom[serial] = atom
            name_to_atom[atom.name] = atom

            # Store atom data in graph node
            element = self._get_atom_element(atom)
            residue = getattr(atom, "parent", None)
            residue_name, residue_id = self._get_residue_info(residue)

            # Add node to graph
            G.add_node(
                atom,
                serial=serial,
                element=element,
                name=atom.name,
                residue_name=residue_name,
                residue_id=residue_id,
            )

        # Save serial to atom map for later use
        self.serial_to_atom = serial_to_atom

        # Add edges based on CONECT records
        if self.conect_dict:
            edges_from_conect = self._add_edges_from_conect(G)
            if edges_from_conect == 0 and self.require_conect:
                error_msg = f"{self.name} ERROR: No valid CONECT records found for this structure. CONECT records are required."
                if self.logger:
                    self.logger.error(error_msg)
                raise ValueError(error_msg)
            elif self.require_conect:
                error_msg = f"{self.name} ERROR: No CONECT records found for this structure. CONECT records are required."
                if self.logger:
                    self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                # If no CONECT records and not required, infer edges based on distances
                if self.logger:
                    self.logger.info(
                        f"{self.name} No CONECT records found, inferring edges based on distances"
                    )
                self._infer_missing_edges(G)

        # Debug connectivity issues
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            self._log_connectivity_info(G)

        if len(G.edges()) == 0:
            if self.logger:
                self.logger.warning(
                    f"{self.name} No edges found from CONECT records, inferring edges based on distances"
                )
            self._infer_missing_edges(G)

        # Visualize the graph connectivity for debugging
        components = list(nx.connected_components(G))
        if self.logger:
            self.logger.info(f"{self.name} Connected components: {len(components)}")
            for i, comp in enumerate(components):
                self.logger.info(f"{self.name} Component {i+1}: {len(comp)} nodes")

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

    def _add_edges_from_conect(self, G: nx.Graph) -> int:
        """Add edges from CONECT records.

        Args:
            G: NetworkX graph to add edges to

        Returns:
            Number of edges added
        """
        if not self.conect_dict:
            if self.require_conect:
                error_msg = f"{self.name} ERROR: No CONECT dictionary provided. CONECT records are required."
                if self.logger:
                    self.logger.error(error_msg)
                raise ValueError(error_msg)
            return 0

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
                f"{self.name} {len(missing_serials)} serials from CONECT records not found in atom list"
            )

        if missing_connections and self.logger:
            self.logger.warning(
                f"{self.name} {len(missing_connections)} connected serials from CONECT records not found in atom list"
            )

        if edge_count == 0 and self.require_conect:
            error_msg = f"{self.name} ERROR: No valid CONECT records found for this structure. CONECT records are required."
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)

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


def get_structure(structure_id: str, pdb_file: str) -> Structure:
    """Parse a PDB file and return the structure.

    Args:
        structure_id: ID for the structure
        pdb_file: Path to the PDB file

    Returns:
        BioPython Structure object
    """
    parser = PDBParser(QUIET=True)
    return parser.get_structure(structure_id, pdb_file)


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
    pdb_file: str, model_num: Optional[int] = None, chain_id: Optional[str] = None
) -> Dict[int, List[int]]:
    """Parse CONECT records from a PDB file.

    Args:
        pdb_file: Path to PDB file
        model_num: Model number to extract (if None, get all)
        chain_id: Chain ID to filter records for (if None, get all)

    Returns:
        Dictionary mapping atom serial numbers to lists of connected atom serial numbers
    """
    # Create a cache key that includes the model number and chain ID
    cache_key = f"{pdb_file}_{model_num if model_num is not None else 'all'}_{chain_id if chain_id is not None else 'all'}"

    # Check cache first
    if cache_key in _CONECT_CACHE:
        return _CONECT_CACHE[cache_key]

    # Read the PDB file lines (using the cached version if available)
    lines = get_pdb_lines(pdb_file)

    # First, if we're filtering by chain, get the atom serial numbers for that chain
    chain_serials = set()
    if chain_id is not None:
        for line in lines:
            if line.startswith(("ATOM ", "HETATM")):
                if line[21:22].strip() == chain_id:  # Chain ID is in column 22
                    try:
                        serial = int(line[6:11].strip())
                        chain_serials.add(serial)
                    except ValueError:
                        continue

    # Extract CONECT records
    conect_dict = {}
    model_found = model_num is None  # If model_num is None, process all lines

    for line in lines:
        if model_num is not None:
            if line.startswith("MODEL"):
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
            try:
                parts = line.split()
                if len(parts) > 2:  # At least one connection
                    atom_serial = int(parts[1])
                    # Only include if this atom is in our chain
                    if atom_serial in chain_serials:
                        connected_serials = []
                        for i in range(2, len(parts)):
                            connected_serial = int(parts[i])
                            # Only include connections to atoms in our chain
                            if connected_serial in chain_serials:
                                connected_serials.append(connected_serial)

                        if (
                            connected_serials
                        ):  # Only add if there are connections to our chain
                            if atom_serial not in conect_dict:
                                conect_dict[atom_serial] = []
                            conect_dict[atom_serial].extend(connected_serials)
            except ValueError as e:
                logger.warning(f"Skipping CONECT with invalid serial number: {line}")

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
            try:
                parts = line.split()
                if len(parts) > 2:  # At least one connection
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
            except ValueError as e:
                logger.warning(f"Skipping CONECT with invalid serial number: {line}")

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
    # Vectorized computation of all pairwise distances
    num_clashing = 0

    # For each atom in the test molecule
    for i, coord in enumerate(coords):
        # Calculate distances to all protein atoms
        distances = np.sqrt(np.sum((protein_coords - coord) ** 2, axis=1))

        # Check if this atom clashes with any protein atom
        if np.any(distances < cutoff):
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


def parse_reference_data(
    ref_pdb_path: str, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Parse reference PDB file.

    Args:
        ref_pdb_path: Path to reference PDB file
        logger: Logger instance

    Returns:
        Dictionary of reference data
    """
    try:
        if logger:
            logger.info(f"Parsing reference PDB: {ref_pdb_path}")

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("ref", ref_pdb_path)
        model = structure[0]

        # Get reference ligand atoms (chain D)
        ref_ligand_atoms = []
        if "D" in model:
            ref_ligand_chain = model["D"]
            if logger:
                logger.info(
                    f"Found chain D with {len(list(ref_ligand_chain.get_residues()))} residues"
                )

            for residue in ref_ligand_chain:
                if logger:
                    logger.info(f"Processing residue {residue.resname} {residue.id[1]}")
                for atom in residue:
                    ref_ligand_atoms.append(atom)

        if not ref_ligand_atoms:
            if logger:
                logger.error("No ligand atoms found in chain D of reference structure")
            raise ValueError("No ligand atoms found in chain D of reference structure")

        if logger:
            logger.info(f"Found {len(ref_ligand_atoms)} reference ligand atoms")
            logger.info(
                f"First few ligand atoms: {[atom.name for atom in ref_ligand_atoms[:5]]}"
            )

        # Create molecular graph for reference ligand
        ref_ligand_graph = MolecularGraph(
            ref_ligand_atoms,
            logger=logger,
            name="reference",
            conect_dict=parse_conect_records(ref_pdb_path, chain_id="D"),
            is_reference=True,
        )

        # Extract protein coordinates (chains A,B,C)
        protein_coords = []
        protein_atoms = []
        for chain_id in ["A", "B", "C"]:
            if chain_id in model:
                chain = model[chain_id]
                if logger:
                    logger.info(
                        f"Found chain {chain_id} with {len(list(chain.get_residues()))} residues"
                    )
                for atom in chain.get_atoms():
                    protein_coords.append(atom.get_coord())
                    protein_atoms.append(atom)

        if logger:
            logger.info(f"Found {len(protein_coords)} reference protein atoms")

        return {
            "ref_pdb_path": ref_pdb_path,
            "ref_ligand_atoms": ref_ligand_atoms,
            "ref_ligand_graph": ref_ligand_graph,
            "target_protein_coords": (
                np.array(protein_coords) if protein_coords else np.array([])
            ),
            "target_protein_atoms": protein_atoms,
            "pdb_parser": parser,
        }
    except Exception as e:
        if logger:
            logger.error(f"Error parsing reference data: {str(e)}")
            logger.error(traceback.format_exc())
        raise


def process_model(
    test_compound: Structure, model_idx: int, logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Process a single test compound model.

    Args:
        test_compound: Test compound structure
        model_idx: Model index
        logger: Logger instance

    Returns:
        Dictionary with results or None if processing failed
    """
    try:
        # Get the compound ID - use the path as fallback
        try:
            compound_id = test_compound.get_parent().id
        except (AttributeError, TypeError):
            # Fallback to using model index as the compound ID
            compound_id = f"model_{model_idx}"
            logger.info(f"Using fallback compound ID: {compound_id}")

        # Log basic info about test compound
        logger.info(f"Processing model {model_idx} of compound {compound_id}")

        try:
            atoms_count = len(list(test_compound.get_atoms()))
            logger.info(f"Structure has {atoms_count} atoms")
        except Exception as e:
            logger.warning(f"Could not count atoms: {str(e)}")
            atoms_count = 0

        # Get test atoms more robustly
        test_atoms = []
        try:
            # Log chains and residues for debugging
            chains = list(test_compound.get_chains())
            chain_ids = [chain.id for chain in chains]
            logger.info(f"Chains in test structure: {chain_ids}")

            for chain in chains:
                try:
                    residues = list(chain.get_residues())
                    residue_ids = [f"{res.resname}_{res.id[1]}" for res in residues]
                    logger.info(
                        f"Chain {chain.id} has {len(residues)} residues: {residue_ids[:5]}"
                    )

                    # Get atoms from residues
                    for residue in residues:
                        for atom in residue:
                            test_atoms.append(atom)
                except Exception as e:
                    logger.warning(
                        f"Error getting residues for chain {chain.id}: {str(e)}"
                    )
        except Exception as e:
            logger.warning(f"Error getting chains: {str(e)}, trying direct atom access")
            logger.info(
                "No atoms found through chain/residue hierarchy, trying direct atom access"
            )
            try:
                test_atoms = list(test_compound.get_atoms())
            except Exception as e2:
                logger.warning(
                    f"Error getting test atoms through structure hierarchy: {str(e2)}"
                )

                # Try a direct approach as last resort
                try:
                    test_atoms = list(test_compound.get_atoms())
                except Exception as e3:
                    logger.error(f"Could not get atoms from test compound: {str(e3)}")
                    return None

        if not test_atoms:
            logger.error("No atoms found in test compound")
            return None

        logger.info(f"Found {len(test_atoms)} atoms in test structure")
        logger.info(f"First few atom names: {[atom.name for atom in test_atoms[:5]]}")

        # Extract CONECT records from the test compound
        conect_dict = None
        if hasattr(test_compound, "_id") and test_compound._id:
            try:
                # Extract CONECT records directly from the PDB file
                pdb_file = test_compound._id
                conect_dict = parse_conect_records(pdb_file)
                logger.info(
                    f"Found {len(conect_dict)} CONECT records in test structure"
                )
            except Exception as e:
                logger.warning(f"Error extracting CONECT records: {str(e)}")

        # Create molecular graph with the correct parameter order
        test_graph = MolecularGraph(
            atoms=test_atoms,
            logger=logger,  # Pass logger correctly
            name="test_compound",
            conect_dict=conect_dict,
            structure=test_compound,
            chain_id=None,
            is_reference=False,
            require_conect=False,  # Don't require CONECT records
        )

        # Check if we have a cached mapping
        if compound_id not in _ISOMORPHIC_CACHE:
            logger.info(
                f"No cached mapping found for {compound_id}, calculating isomorphic match"
            )

            # Find isomorphic mapping
            superimposer = IsomorphicSuperimposer(logger)
            result = superimposer.align(_REFERENCE_DATA["ref_ligand_graph"], test_graph)

            if not result.isomorphic_match:
                logger.warning(f"No matching substructure found in model {model_idx}")
                logger.info(
                    f"Reference ligand has {len(_REFERENCE_DATA['ref_ligand_graph'].atoms)} atoms"
                )
                logger.info(f"Test compound has {len(test_graph.atoms)} atoms")
                logger.info(
                    f"Reference atom names: {[atom.name for atom in _REFERENCE_DATA['ref_ligand_graph'].atoms]}"
                )
                logger.info(
                    f"Test atom names: {[atom.name for atom in test_graph.atoms]}"
                )
                return None

            logger.info(
                f"Found isomorphic match with {len(result.matched_pairs)} atom pairs"
            )
            logger.info(
                f"Matched pairs: {[(p[0].name, p[1].name) for p in result.matched_pairs[:5]]}"
            )

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

        logger.info(
            f"Using {len(matched_pairs)} matched atom pairs for superimposition"
        )

        # Get coordinates for matched atoms
        ref_coords = []
        test_coords = []
        for ref_atom, test_atom in matched_pairs:
            ref_coords.append(ref_atom.coord)
            test_coords.append(test_atom.coord)

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
            logger.info(
                f"Checking clashes with {len(_REFERENCE_DATA['target_protein_coords'])} protein atoms"
            )
            has_clashes, num_clashing = check_clashes(
                transformed_coords, _REFERENCE_DATA["target_protein_coords"]
            )
            logger.info(
                f"Has clashes: {has_clashes}, Number of clashing atoms: {num_clashing}"
            )

        # Calculate metrics
        metrics = {
            "rmsd": float(matched_rmsd),
            "matched_atoms": len(matched_pairs),
            "has_clashes": has_clashes,
            "num_clashes": num_clashing,
            "success": True,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        }

        logger.info(f"Successfully processed model {model_idx}, metrics: {metrics}")
        return metrics

    except Exception as e:
        if logger:
            logger.error(f"Error processing model {model_idx}: {str(e)}")
            logger.error(traceback.format_exc())
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
    """Process a single frame PDB file.

    Args:
        args: Tuple containing (frame_file, frame_dir, verbose, reference_data)

    Returns:
        Dictionary of frame metrics or None if failed
    """
    frame_file, frame_dir, verbose, reference_data = args

    # Set up logging for this process
    logger = setup_logging(verbose)

    try:
        # Initialize reference data for this process
        global _REFERENCE_DATA
        _REFERENCE_DATA = reference_data

        # Parse structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("frame", frame_file)

        if not structure:
            logger.error(f"Failed to parse structure from {frame_file}")
            return None

        # Create a new structure for this frame
        frame_structure = Structure("frame")
        frame_structure.add(structure[0])  # Add the first (and only) model
        frame_structure._id = frame_file  # Set _id for CONECT record parsing

        # Get frame number from filename
        frame_num = int(Path(frame_file).stem.split("_")[1])

        # Superimpose frame
        result = process_model(frame_structure, frame_num, logger)
        if result:
            # Add frame file information to result
            result["frame_file"] = frame_file
            return result

        return None

    except Exception as e:
        logger.error(f"Failed to process {frame_file}: {str(e)}")
        return None


def save_simple_superimposed_structure(
    frame_file: str,
    metrics: Dict[str, Any],
    ref_pdb_path: str,
    output_dir: Optional[Path],
    logger: logging.Logger,
) -> Optional[Path]:
    """Save a superimposed structure as a PDB file using a simple approach.

    This function creates a PDB file by:
    1. Creating a copy of the reference PDB file
    2. Extracting just the protein chains (A, B, C) and ligand chain (D)
    3. Appending the transformed test ligand coordinates as chain E
    4. Adding CONECT records for test structure and cyclic bond if needed

    Args:
        frame_file: Path to the test compound frame file
        metrics: Dictionary with metrics including rotation and translation
        ref_pdb_path: Path to the reference PDB file
        output_dir: Directory to save output PDB files
        logger: Logger instance

    Returns:
        Path to the saved PDB file or None if failed
    """
    try:
        # Ensure output directory exists
        if output_dir is None:
            # Create default output directory in the same directory as frame_file
            frame_path = Path(frame_file)
            output_dir = frame_path.parent.parent / "superimposed"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get frame number from filename
        frame_name = Path(frame_file).stem

        # Create output file path
        output_file = output_dir / f"superimposed_{frame_name}.pdb"

        logger.info(f"Saving superimposed structure for {frame_name} to {output_file}")

        # Parse reference and test structures
        with open(frame_file, "r") as f:
            test_lines = f.readlines()

        # Create a new PDB file
        with open(output_file, "w") as f:
            # Write a header
            f.write(f"REMARK   4 Superimposed structure\n")
            f.write(f"REMARK   4 Reference: {ref_pdb_path}\n")
            f.write(f"REMARK   4 Test: {frame_file}\n")
            f.write(f"REMARK   4 RMSD: {metrics.get('rmsd', 0.0):.3f} Angstroms\n")
            f.write(f"REMARK   4 Matched atoms: {metrics.get('matched_atoms', 0)}\n")
            f.write(f"REMARK   4 Has clashes: {metrics.get('has_clashes', False)}\n")
            f.write(f"REMARK   4 Number of clashes: {metrics.get('num_clashes', 0)}\n")

            # Copy reference PDB content (chains A, B, C, D)
            with open(ref_pdb_path, "r") as ref_file:
                for line in ref_file:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        chain_id = line[21:22]
                        if chain_id in ["A", "B", "C", "D"]:
                            f.write(line)
                    elif line.startswith("TER"):
                        f.write(line)
                    elif line.startswith("CONECT"):
                        # Keep CONECT records from reference
                        f.write(line)

            # Get transformation data
            if "rotation" not in metrics or "translation" not in metrics:
                logger.error(f"Missing transformation data for {frame_name}")
                return None

            rotation = np.array(metrics["rotation"])
            translation = np.array(metrics["translation"])

            # Find original serial numbers and map to new ones
            atom_id_map = {}
            atom_num = 10001  # Start with high serial numbers to avoid conflicts

            # Process test frame atoms
            test_coords = []
            test_atom_info = []

            for line in test_lines:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # Extract original serial number
                    orig_serial = int(line[6:11].strip())
                    atom_id_map[orig_serial] = atom_num

                    # Extract coordinates
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords = np.array([x, y, z])
                    test_coords.append(coords)

                    # Store atom info for later
                    test_atom_info.append(
                        {
                            "line": line,
                            "serial": orig_serial,
                            "name": line[12:16].strip(),
                            "resname": line[17:20].strip(),
                            "resnum": int(line[22:26].strip()),
                            "element": (
                                line[76:78].strip()
                                if len(line) >= 78
                                else line[12:16].strip()[0]
                            ),
                        }
                    )

                    atom_num += 1

            # Extract CONECT records from test structure
            test_connections = defaultdict(list)
            for line in test_lines:
                if line.startswith("CONECT"):
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    try:
                        atom_serial = int(parts[1])
                        for i in range(2, len(parts)):
                            try:
                                connected_serial = int(parts[i])
                                test_connections[atom_serial].append(connected_serial)
                            except ValueError:
                                continue
                    except ValueError:
                        continue

            # Apply transformation to all atoms
            transformed_coords = []
            for coord in test_coords:
                new_coord = np.dot(coord, rotation) + translation
                transformed_coords.append(new_coord)

            # Write transformed coordinates as chain E
            for i, (coord, atom_info) in enumerate(
                zip(transformed_coords, test_atom_info)
            ):
                serial = atom_id_map[atom_info["serial"]]
                atom_name = atom_info["name"]
                residue_name = atom_info["resname"]
                residue_num = atom_info["resnum"]
                element = atom_info["element"]

                line = f"ATOM  {serial:5d} {atom_name:4s} {residue_name:3s} E{residue_num:4d}    "
                line += f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                line += f"  1.00  0.00          {element:>2s}  \n"

                f.write(line)

            # Add TER record
            f.write("TER\n")

            # Add CONECT records for test structure
            for orig_serial, connections in test_connections.items():
                if orig_serial in atom_id_map:
                    new_id = atom_id_map[orig_serial]
                    conect_line = f"CONECT{new_id:5d}"

                    for connected_serial in connections:
                        if connected_serial in atom_id_map:
                            conect_line += f"{atom_id_map[connected_serial]:5d}"

                    conect_line += "\n"
                    f.write(conect_line)

            # Add cyclic peptide bond if needed
            # Find first and last residue numbers
            if test_atom_info:
                residue_nums = [info["resnum"] for info in test_atom_info]
                min_res = min(residue_nums)
                max_res = max(residue_nums)

                # Find N atom in first residue and C atom in last residue
                first_N = None
                last_C = None

                for info in test_atom_info:
                    if info["resnum"] == min_res and info["name"] == "N":
                        first_N = info["serial"]
                    if info["resnum"] == max_res and info["name"] == "C":
                        last_C = info["serial"]

                # Add cyclic connection if both atoms found and not already connected
                if first_N is not None and last_C is not None:
                    if first_N in atom_id_map and last_C in atom_id_map:
                        # Check if connection already exists
                        if last_C not in test_connections.get(
                            first_N, []
                        ) and first_N not in test_connections.get(last_C, []):
                            logger.info(
                                f"Adding cyclic peptide connection between residues {min_res} and {max_res}"
                            )

                            # Add CONECT record
                            new_first_N = atom_id_map[first_N]
                            new_last_C = atom_id_map[last_C]
                            f.write(f"CONECT{new_first_N:5d}{new_last_C:5d}\n")
                            f.write(f"CONECT{new_last_C:5d}{new_first_N:5d}\n")

            # End the file
            f.write("END\n")

        logger.info(f"Successfully saved superimposed structure to {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Error saving superimposed structure: {str(e)}")
        traceback.print_exc()
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Superimpose ligands onto a reference structure using RDKit MCS."
    )
    parser.add_argument(
        "base_dir", help="Base directory containing pdb_frames directories"
    )
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
    parser.add_argument(
        "--save-structures",
        action="store_true",
        help="Save the first 5 superimposed structures as PDB files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save superimposed structures (default: base_dir/superimposed)",
    )

    args = parser.parse_args()

    # Check for required modules
    missing_modules = []
    if not BIOPYTHON_AVAILABLE:
        missing_modules.append("biopython")

    if missing_modules:
        print("ERROR: Required modules are missing:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nPlease install them with:")
        print(f"  pip install {' '.join(missing_modules)}")
        return 1

    # Set up logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("superimpose")

    # Initialize reference data
    print("\nInitializing reference structures...")
    try:
        global _REFERENCE_DATA
        _REFERENCE_DATA = parse_reference_data(args.reference_pdb, logger)
    except Exception as e:
        print(f"Error parsing reference data: {str(e)}")
        return 1

    # Set output directory for saved structures
    output_dir = None
    if args.save_structures:
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(args.base_dir) / "superimposed"

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Will save the first 5 superimposed structures to {output_dir}")
        except Exception as e:
            print(f"Warning: Could not create output directory {output_dir}: {str(e)}")
            print("Will use default directory next to frame files")
            output_dir = None

    # Find pdb_frames directories and their frame files
    print(f"\nSearching for frame PDB files in {args.base_dir}...")
    base_path = Path(args.base_dir)
    frame_dirs = list(base_path.rglob("pdb_frames"))

    if not frame_dirs:
        print(f"No pdb_frames directories found in {args.base_dir}")
        return 1

    # Collect all frame files
    frame_files = []
    skipped_dirs = []
    for frame_dir in frame_dirs:
        # Get all frame_*.pdb files in this directory
        frames = sorted(frame_dir.glob("frame_*.pdb"))
        if not frames:
            continue

        # Create metrics file path for this directory
        metrics_file = frame_dir / "superimposition_metrics.json"

        # Check if directory is already fully processed
        if not args.force and metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                    # Check if we have metrics for all frames
                    if len(metrics) == len(frames):
                        # Verify metrics are valid
                        all_valid = all(
                            isinstance(m, dict) and "rmsd" in m and "has_clashes" in m
                            for m in metrics.values()
                        )
                        if all_valid:
                            print(f"Skipping already processed directory: {frame_dir}")
                            skipped_dirs.append(frame_dir)
                            continue
            except (json.JSONDecodeError, KeyError):
                # If metrics file is invalid, we'll reprocess
                pass

        frame_files.extend(frames)

    if not frame_files:
        if skipped_dirs:
            print("\nAll directories have been processed. Use --force to reprocess.")
            print("\nProcessed directories:")
            for dir_path in skipped_dirs:
                print(f"  {dir_path}")
        else:
            print(f"No frame PDB files found in any pdb_frames directory")
        return 1

    print(
        f"\nFound {len(frame_files)} frame files to process in {len(frame_dirs) - len(skipped_dirs)} directories"
    )
    if skipped_dirs:
        print(f"Skipped {len(skipped_dirs)} already processed directories")

    # Limit to first 1000 frames for testing if there are too many
    if len(frame_files) > 1000:
        print(f"Limiting to first 1000 frames for processing")
        frame_files = frame_files[:1000]

    # Set number of processes
    num_processes = args.num_processes or min(os.cpu_count() or 1, len(frame_files))
    print(f"Using {num_processes} processes")

    # Simple progress reporting
    def print_progress(current, total, start_time):
        elapsed_time = time.time() - start_time
        if elapsed_time > 0 and current > 0:
            frames_per_second = current / elapsed_time
            eta = (total - current) / frames_per_second if frames_per_second > 0 else 0
            eta_str = f"{eta:.1f}s" if eta < 60 else f"{eta/60:.1f}m"

            # Create a simple progress bar
            bar_len = 30
            filled_len = int(round(bar_len * current / float(total)))
            bar = "=" * filled_len + "-" * (bar_len - filled_len)

            sys.stdout.write(
                f"\r[{bar}] {current}/{total} ({current/total*100:.1f}%) - ETA: {eta_str}"
            )
            sys.stdout.flush()

    # Process files in parallel
    start_time = time.time()
    total_frames = len(frame_files)
    processed_frames = 0
    print_progress(0, total_frames, start_time)

    # Store frames with successful results for saving later
    successful_frames = []
    saved_structures_count = 0

    # If save_structures is enabled, process a few frames first to get some saved structures
    if args.save_structures:
        # Take the first 20 frames to process first, one at a time, until we get 5 successful ones
        frames_to_try = frame_files[: min(20, len(frame_files))]
        print(
            f"\nProcessing first {len(frames_to_try)} frames to find successful superimpositions..."
        )

        for i, frame_file in enumerate(frames_to_try):
            frame_path = str(frame_file)
            frame_dir = str(frame_file.parent)
            result = process_trajectory(
                (frame_path, frame_dir, args.verbose, _REFERENCE_DATA)
            )

            if result and result.get("success", False):
                print(f"Found successful superimposition: {frame_file.name}")
                successful_frames.append((frame_path, result))

                # Save this structure
                if len(successful_frames) <= 5:
                    saved_file = save_simple_superimposed_structure(
                        frame_path, result, args.reference_pdb, output_dir, logger
                    )
                    if saved_file:
                        saved_structures_count += 1

                # Once we have 5, we can stop
                if len(successful_frames) >= 5:
                    print(
                        f"Found 5 successful superimpositions, continuing with bulk processing..."
                    )
                    break

            # Update progress
            processed_frames += 1
            if i % 5 == 0:  # Only update occasionally
                print(
                    f"Processed {i+1}/{len(frames_to_try)} frames, found {len(successful_frames)} successful superimpositions"
                )

        # Remove the frames we've already processed from the list
        frame_files = [
            f for f in frame_files if str(f) not in [x[0] for x in successful_frames]
        ]

    # Now process the rest of the frames
    if frame_files:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []

            # Group frame files by their parent directory
            frame_groups = {}
            for frame_file in frame_files:
                parent_dir = frame_file.parent
                if parent_dir not in frame_groups:
                    frame_groups[parent_dir] = []
                frame_groups[parent_dir].append(frame_file)

            # Process each directory's frames
            for frame_dir, dir_frames in frame_groups.items():
                metrics_file = frame_dir / "superimposition_metrics.json"
                metrics = {}

                # Load existing metrics if they exist
                if metrics_file.exists():
                    try:
                        with open(metrics_file, "r") as f:
                            metrics = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Could not read existing metrics file: {metrics_file}"
                        )

                # Submit each frame for processing
                for frame_file in dir_frames:
                    frame_num = frame_file.stem.split("_")[1]
                    # Only skip if we have valid metrics for this frame
                    if not args.force and str(frame_num) in metrics:
                        try:
                            frame_metrics = metrics[str(frame_num)]
                            if (
                                isinstance(frame_metrics, dict)
                                and "rmsd" in frame_metrics
                                and "has_clashes" in frame_metrics
                            ):
                                processed_frames += 1
                                print_progress(
                                    processed_frames, total_frames, start_time
                                )
                                continue
                        except (KeyError, TypeError):
                            pass

                    futures.append(
                        executor.submit(
                            process_trajectory,
                            (
                                str(frame_file),
                                str(frame_dir),
                                args.verbose,
                                _REFERENCE_DATA,
                            ),
                        )
                    )

            # Process results as they complete
            for future in as_completed(futures):
                result = future.result()
                processed_frames += 1
                print_progress(processed_frames, total_frames, start_time)

                if result:
                    # Get the directory and update its metrics file
                    frame_file = result.get("frame_file")
                    if frame_file:
                        frame_dir = Path(frame_file).parent
                        metrics_file = frame_dir / "superimposition_metrics.json"

                        # Load current metrics
                        current_metrics = {}
                        if metrics_file.exists():
                            try:
                                with open(metrics_file, "r") as f:
                                    current_metrics = json.load(f)
                            except json.JSONDecodeError:
                                pass

                        # Update with new result
                        frame_num = Path(frame_file).stem.split("_")[1]
                        current_metrics[str(frame_num)] = result

                        # Save updated metrics
                        with open(metrics_file, "w") as f:
                            json.dump(current_metrics, f, indent=2)

                        # Track successful results for saving structures later
                        if result.get("success", False) and len(successful_frames) < 5:
                            successful_frames.append((frame_file, result))

    print("\n")  # New line after progress bar

    # After processing is complete, save any remaining structures needed to reach 5
    if args.save_structures and successful_frames:
        structures_to_save = 5 - saved_structures_count
        if structures_to_save > 0 and len(successful_frames) > saved_structures_count:
            print(
                f"\nSaving {min(structures_to_save, len(successful_frames) - saved_structures_count)} more superimposed structures..."
            )
            for frame_file, result in successful_frames[saved_structures_count:]:
                if saved_structures_count >= 5:
                    break

                saved_file = save_simple_superimposed_structure(
                    frame_file,
                    result,
                    args.reference_pdb,
                    output_dir,
                    logger,
                )
                if saved_file:
                    saved_structures_count += 1
                    print(f"Saved superimposed structure for {Path(frame_file).name}")

    # After all processing is done, generate CSV files for each directory
    print("\nGenerating CSV result files...")
    for frame_dir in frame_dirs:
        metrics_file = frame_dir / "superimposition_metrics.json"
        if metrics_file.exists():
            try:
                # Create CSV file in the parent directory (e.g., x_177/superimposition_results.csv)
                csv_file = frame_dir.parent / "superimposition_results.csv"

                # Load metrics
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

                if not metrics:
                    print(f"No metrics found for {frame_dir}, skipping CSV generation")
                    continue

                # Convert JSON to CSV
                with open(csv_file, "w") as f:
                    # Write header
                    header = [
                        "frame",
                        "rmsd",
                        "matched_atoms",
                        "has_clashes",
                        "num_clashes",
                        "success",
                    ]
                    f.write(",".join(header) + "\n")

                    # Write rows
                    for frame_num, frame_metrics in sorted(
                        metrics.items(), key=lambda x: int(x[0])
                    ):
                        if isinstance(frame_metrics, dict):
                            row = [
                                f"frame_{frame_num}",
                                str(frame_metrics.get("rmsd", 0)),
                                str(frame_metrics.get("matched_atoms", 0)),
                                str(frame_metrics.get("has_clashes", False)),
                                str(frame_metrics.get("num_clashes", 0)),
                                str(frame_metrics.get("success", False)),
                            ]
                            f.write(",".join(row) + "\n")

                print(f"Created CSV results file: {csv_file}")
            except Exception as e:
                print(f"Error generating CSV for {frame_dir}: {str(e)}")

    if args.save_structures:
        print(
            f"\nSaved {saved_structures_count} superimposed structures to {output_dir}"
        )

    print("\nSuperimposition complete!")


def create_rdkit_mol_from_chain(
    chain: Chain, conect_dict: Dict[int, List[int]], logger: logging.Logger
) -> Optional[Any]:
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


def get_molblock_from_pdb(
    pdb_file: str, chain_id: str = "D", remove_hs: bool = False
) -> Optional[str]:
    """Convert a PDB file to an MDL molblock string.

    Args:
        pdb_file: PDB file path
        chain_id: Chain ID to extract
        remove_hs: Whether to remove hydrogens

    Returns:
        MDL molblock string or None if conversion failed
    """
    if not RDKIT_AVAILABLE:
        return None

    try:
        # Read PDB file and convert to RDKit molecule
        mol = Chem.MolFromPDBFile(
            pdb_file,
            sanitize=True,
            removeHs=remove_hs,
            flavor=Chem.PDBFlavor.AllChem,
        )
        if mol is None:
            return None

        # Filter by chain ID if specified
        if chain_id:
            atoms_to_keep = []
            for atom in mol.GetAtoms():
                info = atom.GetPDBResidueInfo()
                if info and info.GetChainId() == chain_id:
                    atoms_to_keep.append(atom.GetIdx())

            if atoms_to_keep:
                mol = Chem.PathToSubmol(mol, atoms_to_keep)

        # Generate molblock
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None


def molecule_from_pdb_file(
    pdb_file: str, chain_id: Optional[str] = None, remove_hs: bool = False
) -> Optional[Any]:
    """Create an RDKit molecule from a PDB file.

    Args:
        pdb_file: Path to PDB file
        chain_id: Optional chain ID to extract (default: all chains)
        remove_hs: Whether to remove hydrogens

    Returns:
        RDKit molecule or None if conversion failed
    """
    if not RDKIT_AVAILABLE:
        return None

    try:
        # Read PDB file and convert to RDKit molecule
        mol = Chem.MolFromPDBFile(
            pdb_file,
            sanitize=True,
            removeHs=remove_hs,
            flavor=Chem.PDBFlavor.AllChem,
        )
        if mol is None:
            return None

        # Filter by chain ID if specified
        if chain_id:
            atoms_to_keep = []
            for atom in mol.GetAtoms():
                info = atom.GetPDBResidueInfo()
                if info and info.GetChainId() == chain_id:
                    atoms_to_keep.append(atom.GetIdx())

            if atoms_to_keep:
                mol = Chem.PathToSubmol(mol, atoms_to_keep)

        # Generate molblock
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None


if __name__ == "__main__":
    main()
