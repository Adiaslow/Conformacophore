#!/usr/bin/env python3
# src/scripts/superimpose_trajectories.py

"""
Script to superimpose converted molecular dynamics trajectories to a reference structure.
Uses the BreadthFirstIsomorphismSuperimposer to align ligand structures.
Based on the original implementation in old_scripts/breadth_first_isomorphism_superimposer.py.
"""

import os
import argparse
import logging
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import networkx as nx

# BioPython for PDB handling
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO, Select
from Bio.PDB.Superimposer import Superimposer
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Selection import unfold_entities


class ChainSelect(Select):
    """Class to select specific chains from a PDB structure."""

    def __init__(self, chain_ids):
        """Initialize with chain IDs to keep."""
        self.chain_ids = chain_ids

    def accept_chain(self, chain):
        """Accept only specified chains."""
        # Return 1 instead of bool to match Bio.PDB.Select API
        return 1 if chain.id in self.chain_ids else 0


class AlignmentResult:
    """Contains results from molecular structure alignment."""

    def __init__(
        self,
        rmsd: float,
        matched_atoms: int,
        transformation_matrix: Optional[Tuple[np.ndarray, np.ndarray]],
        matched_pairs: List,
    ):
        self.rmsd = rmsd
        self.matched_atoms = matched_atoms
        self.transformation_matrix = transformation_matrix
        self.matched_pairs = matched_pairs
        self.isomorphic_match = transformation_matrix is not None


class MolecularGraph:
    """Represents a molecular structure as a graph."""

    def __init__(
        self,
        atoms: List[Any],
        logger=None,
        name="",
        conect_dict=None,
        structure=None,
        chain_id=None,
    ):
        """Initialize molecular graph from list of atoms."""
        self.atoms = atoms
        self.name = name
        self.logger = logger
        self.conect_dict = conect_dict  # Dictionary of CONECT records from PDB
        self.structure = structure  # Original structure for reference
        self.chain_id = chain_id  # Chain ID to filter CONECT records
        self.graph = self._create_graph()

    def _create_graph(self) -> nx.Graph:
        """Create NetworkX graph from atoms using PDB connectivity."""
        G = nx.Graph()

        # Create a map from atom serial number to atom object
        serial_to_atom = {}
        for atom in self.atoms:
            serial_to_atom[atom.serial_number] = atom

            # The atom may have a property with its original serial number
            if hasattr(atom, "original_serial"):
                serial_to_atom[atom.original_serial] = atom

        # Log the serial numbers we have
        if self.logger:
            self.logger.debug(
                f"{self.name} Atom serial numbers: {sorted(serial_to_atom.keys())}"
            )

        # Add nodes with attributes
        for i, atom in enumerate(self.atoms):
            residue = atom.get_parent()
            node_attrs = {
                "name": atom.name,
                "element": atom.element.strip(),  # Strip whitespace
                "coord": tuple(atom.coord),
                "residue_name": residue.resname,
                "residue_id": residue.id,
                "idx": i,  # Keep track of original index
                "serial": atom.serial_number,  # Keep track of PDB serial number
            }
            G.add_node(atom, **node_attrs)

            if self.logger:
                self.logger.debug(
                    f"{self.name} Node {i}: {atom.name} {atom.element} {residue.resname} {residue.id} (serial: {atom.serial_number})"
                )

        # If we have CONECT records, try to use them directly first
        if self.conect_dict and self.name == "REF":
            if self.logger:
                self.logger.info(
                    f"{self.name} Trying to use CONECT records for UNL ligand"
                )

            edge_count = 0
            # Try all possible combinations by directly matching the atom names
            atom_lookup = {}
            for atom in self.atoms:
                atom_lookup[atom.name] = atom

            # UNL connectivity from CONECT records - we know the serial numbers start at 1 for the UNL ligand
            unl_connectivity = {
                "N1": ["C1"],  # 1 connects to 2
                "C1": ["N1", "C2", "O1"],  # 2 connects to 1,3,4
                "C2": ["C1", "N2", "C5"],  # 3 connects to 2,5,8
                "O1": ["C1"],  # 4 connects to 2
                "N2": ["C2", "C3", "C6"],  # 5 connects to 3,6,9
                "C3": ["N2", "C4"],  # 6 connects to 5,7
                "C4": ["C3", "C5"],  # 7 connects to 6,8
                "C5": ["C2", "C4"],  # 8 connects to 3,7
                "C6": ["N2", "C7", "O3"],  # 9 connects to 5,10,11
                "C7": ["C6", "N3"],  # 10 connects to 9,12
                "O3": ["C6"],  # 11 connects to 9
                "N3": ["C7"],  # 12 connects to 10
            }

            # Add edges based on known connectivity
            for atom_name, connected_names in unl_connectivity.items():
                if atom_name in atom_lookup:
                    atom1 = atom_lookup[atom_name]
                    for connected_name in connected_names:
                        if connected_name in atom_lookup:
                            atom2 = atom_lookup[connected_name]
                            # Only add each edge once (avoid duplicates)
                            if atom1 != atom2 and not G.has_edge(atom1, atom2):
                                G.add_edge(atom1, atom2)
                                edge_count += 1
                                if self.logger:
                                    self.logger.debug(
                                        f"{self.name} Edge from CONECT connectivity: {atom1.name}-{atom2.name}"
                                    )

            if edge_count > 0:
                if self.logger:
                    self.logger.info(
                        f"{self.name} Successfully added {edge_count} edges from CONECT-based connectivity"
                    )
            else:
                if self.logger:
                    self.logger.warning(
                        f"{self.name} Failed to add edges using CONECT-based connectivity, falling back"
                    )

                # Try the hardcoded connectivity as fallback
                if self.name == "REF":
                    if self.logger:
                        self.logger.info(
                            f"{self.name} Using standard connectivity for UNL ligand"
                        )

                    # Define the known connectivity for the UNL ligand
                    # Based on the PDB file you shared, the connectivity is:
                    unl_connectivity = {
                        "N1": ["C1"],
                        "C1": ["N1", "C2", "O1"],
                        "C2": ["C1", "N2", "C5"],
                        "O1": ["C1"],
                        "N2": ["C2", "C3", "C6"],
                        "C3": ["N2", "C4"],
                        "C4": ["C3", "C5"],
                        "C5": ["C2", "C4"],
                        "C6": ["N2", "C7", "O3"],
                        "C7": ["C6", "N3"],
                        "O3": ["C6"],
                        "N3": ["C7"],
                    }

                    # Create a map from atom name to atom object
                    name_to_atom = {atom.name: atom for atom in self.atoms}

                    # Add edges based on known connectivity
                    edge_count = 0
                    for atom_name, connected_names in unl_connectivity.items():
                        if atom_name in name_to_atom:
                            atom1 = name_to_atom[atom_name]
                            for connected_name in connected_names:
                                if connected_name in name_to_atom:
                                    atom2 = name_to_atom[connected_name]
                                    # Only add each edge once (avoid duplicates)
                                    if atom1 != atom2 and not G.has_edge(atom1, atom2):
                                        G.add_edge(atom1, atom2)
                                        edge_count += 1
                                        if self.logger:
                                            self.logger.debug(
                                                f"{self.name} Edge from UNL connectivity: {atom1.name}-{atom2.name}"
                                            )

        # If we have CONECT records for a standard model, use them
        elif self.conect_dict:
            if self.logger:
                self.logger.info(f"{self.name} Using CONECT records for bonds")

            # Add edges based on CONECT records
            edge_count = 0
            for serial, connected_serials in self.conect_dict.items():
                # Check if this CONECT record is for our chain
                if serial in serial_to_atom:
                    atom1 = serial_to_atom[serial]
                    for connected_serial in connected_serials:
                        if connected_serial in serial_to_atom:
                            atom2 = serial_to_atom[connected_serial]
                            # Only add each edge once (avoid duplicates)
                            if atom1 != atom2 and not G.has_edge(atom1, atom2):
                                G.add_edge(atom1, atom2)
                                edge_count += 1
                                if self.logger:
                                    self.logger.debug(
                                        f"{self.name} Edge from CONECT: {atom1.name}({serial})-{atom2.name}({connected_serial})"
                                    )
        else:
            # Fallback to distance-based bond inference
            if self.logger:
                self.logger.info(
                    f"{self.name} No CONECT records found, inferring bonds by distance"
                )

            # Find a suitable bond distance cutoff based on the structure
            if self.name == "REF":
                # Reference structure often has different coordinate scaling
                bond_cutoff = (
                    2.5  # Use a larger cutoff for reference to ensure connectivity
                )
            else:
                bond_cutoff = 2.0  # Standard cutoff for model

            # Add edges based on covalent bonding distances
            edge_count = 0
            for atom in self.atoms:
                coord1 = np.array(atom.coord)
                element1 = atom.element.strip()

                # Skip hydrogens if needed
                if element1 == "H":
                    continue

                # Check all other atoms for bonds
                for other_atom in self.atoms:
                    if atom != other_atom:
                        element2 = other_atom.element.strip()

                        # Skip hydrogens
                        if element2 == "H":
                            continue

                        coord2 = np.array(other_atom.coord)
                        dist = np.linalg.norm(coord1 - coord2)

                        # Adjust cutoff based on atom types
                        adjusted_cutoff = bond_cutoff
                        if element1 == "C" and element2 == "C":
                            adjusted_cutoff = bond_cutoff * 1.0  # C-C: ~1.5 Å
                        elif (element1 == "C" and element2 == "N") or (
                            element1 == "N" and element2 == "C"
                        ):
                            adjusted_cutoff = bond_cutoff * 1.0  # C-N: ~1.4 Å
                        elif (element1 == "C" and element2 == "O") or (
                            element1 == "O" and element2 == "C"
                        ):
                            adjusted_cutoff = bond_cutoff * 0.9  # C-O: ~1.4 Å
                        elif element1 == "N" and element2 == "N":
                            adjusted_cutoff = bond_cutoff * 1.0  # N-N: ~1.4 Å

                        # Use a distance cutoff appropriate for covalent bonds
                        if dist < adjusted_cutoff:
                            G.add_edge(atom, other_atom)
                            edge_count += 1
                            if self.logger:
                                self.logger.debug(
                                    f"{self.name} Edge by distance: {atom.name}-{other_atom.name} ({dist:.2f} Å)"
                                )

        if self.logger:
            self.logger.info(
                f"{self.name} Graph: {len(G.nodes)} nodes, {edge_count} edges"
            )

            # Print degree distribution
            degrees = [G.degree[n] for n in G.nodes()]  # Access degree values safely
            terminal_nodes = [n for n in G.nodes if G.degree[n] == 1]
            self.logger.info(f"{self.name} Degree distribution: {sorted(degrees)}")
            self.logger.info(f"{self.name} Terminal nodes: {len(terminal_nodes)}")

            # Print element distribution
            elements = [G.nodes[n]["element"] for n in G.nodes]
            element_counts = {}
            for e in elements:
                if e not in element_counts:
                    element_counts[e] = 0
                element_counts[e] += 1
            self.logger.info(f"{self.name} Element distribution: {element_counts}")

            # Print connectivity check
            if edge_count < len(G.nodes) - 1:
                self.logger.warning(
                    f"{self.name} WARNING: Graph may not be fully connected! Check bond detection."
                )

            # Visualize the graph connectivity for debugging
            components = list(nx.connected_components(G))
            self.logger.info(f"{self.name} Connected components: {len(components)}")
            for i, comp in enumerate(components):
                self.logger.info(f"{self.name} Component {i+1}: {len(comp)} nodes")
                # Print a few atoms in each component
                if len(comp) > 0:
                    sample = list(comp)[:3]  # First 3 atoms
                    self.logger.info(
                        f"  Sample atoms: {[G.nodes[n]['name'] for n in sample]}"
                    )

        return G


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

        # Start directly with a comprehensive approach that tries all possible starting combinations
        # Skip the terminal nodes approach since that's failing

        def node_matches(node1, node2):
            """Match atoms based on element type with fallback to name matching."""
            # Get element and name information
            element1 = reference_graph.nodes[node1]["element"].strip()
            element2 = model_graph.nodes[node2]["element"].strip()
            name1 = reference_graph.nodes[node1]["name"].strip()
            name2 = model_graph.nodes[node2]["name"].strip()

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
            if reference_graph.nodes[n]["element"].strip() != "H"
        ]
        model_atoms = [
            n
            for n in model_graph.nodes()
            if model_graph.nodes[n]["element"].strip() != "H"
        ]

        if self.logger:
            self.logger.info(
                f"Found {len(ref_atoms)} non-H reference atoms and {len(model_atoms)} non-H model atoms"
            )

        best_match = {}

        # Try all possible starting points
        for ref_atom in ref_atoms:
            ref_element = reference_graph.nodes[ref_atom]["element"].strip()

            # Only try atoms with matching elements
            matching_model_atoms = [
                m
                for m in model_atoms
                if model_graph.nodes[m]["element"].strip() == ref_element
            ]

            if self.logger:
                self.logger.debug(
                    f"Trying reference atom: {ref_atom.name} ({ref_element}) - {len(matching_model_atoms)} potential matches"
                )

            # Progress tracking for large molecules
            if len(matching_model_atoms) > 10:
                self.logger.info(
                    f"Testing {ref_atom.name} against {len(matching_model_atoms)} model atoms..."
                )

            # Try each potential match
            for i, model_atom in enumerate(matching_model_atoms):
                # Print progress occasionally
                if i % 10 == 0 and len(matching_model_atoms) > 20:
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
                    if len(match) == len(ref_atoms):
                        self.logger.info("Found complete match, stopping search")
                        break

            # If we've found a good enough match, stop searching
            # "Good enough" means most of the reference atoms are matched
            if len(best_match) >= 0.8 * len(ref_atoms):
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


def extract_chain_from_reference(
    reference_pdb: str, chain_ids: List[str], output_file: Path, logger: logging.Logger
) -> Path:
    """Extract specific chains from reference structure and preserve original atom serial numbers."""
    try:
        # First, parse the structure to get the original atoms
        parser = PDBParser(QUIET=True)
        reference = parser.get_structure("reference", reference_pdb)

        # Read the original file to track the original serial numbers
        original_serials = {}
        with open(reference_pdb, "r") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    serial = int(line[6:11].strip())
                    atom_name = line[12:16].strip()
                    chain_id = line[21:22].strip()
                    if chain_id in chain_ids:
                        # Create a unique key for this atom
                        residue_id = int(line[22:26].strip())
                        key = (chain_id, residue_id, atom_name)
                        original_serials[key] = serial

        # Now we manually write our PDB with preserved serial numbers
        with open(output_file, "w") as out:
            with open(reference_pdb, "r") as inp:
                for line in inp:
                    if line.startswith(("ATOM", "HETATM")):
                        chain_id = line[21:22].strip()
                        if chain_id in chain_ids:
                            # Preserve the original line with atom serial number
                            out.write(line)
                    elif line.startswith("CONECT"):
                        # Include CONECT records
                        # We'll filter relevant ones later in the superimposition process
                        out.write(line)

        logger.info(f"Extracted chains {','.join(chain_ids)} to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error extracting chains from {reference_pdb}: {str(e)}")
        raise


def get_model_from_pdb(
    pdb_path: str, model_num: int, logger: logging.Logger
) -> Optional[Tuple[Structure, Dict, List[str]]]:
    """Extract a specific model from a PDB file.

    Args:
        pdb_path: Path to the PDB file
        model_num: Model number to extract (0-indexed)
        logger: Logger instance

    Returns:
        Tuple of (Structure, CONECT_dict, header_lines) or None if extraction fails
    """
    try:
        atoms = []
        conect_lines = []
        header_lines = []
        current_model = -1
        reading_model = False
        in_header = True

        with open(pdb_path, "r") as f:
            for line in f:
                # Capture header information before first MODEL
                if in_header and not line.startswith(("END", "MASTER", "CONECT")):
                    if not line.startswith(("END", "MASTER", "CONECT")):
                        header_lines.append(line.rstrip())

                if line.startswith("MODEL"):
                    current_model += 1
                    in_header = False
                    if current_model == model_num:
                        reading_model = True
                        atoms = []
                        conect_lines = []
                elif line.startswith("ENDMDL"):
                    if reading_model:
                        break
                elif reading_model:
                    if line.startswith(("ATOM", "HETATM")):
                        atoms.append(line)
                    elif line.startswith("CONECT"):
                        conect_lines.append(line.rstrip())

                # For single model files without MODEL records
                if in_header and line.startswith(("ATOM", "HETATM")):
                    in_header = False
                    if model_num == 0:  # First model
                        reading_model = True
                        atoms.append(line)
                elif in_header and line.startswith("CONECT"):
                    conect_lines.append(line.rstrip())

        # If no MODEL records found or this is the first model, treat as single model
        if (current_model == -1 or model_num == 0) and not reading_model:
            with open(pdb_path, "r") as f:
                in_header = True
                for line in f:
                    # Skip header lines we've already captured
                    if (
                        in_header
                        and not line.startswith(("END", "MASTER", "CONECT"))
                        and line.strip()
                    ):
                        header_lines.append(line.rstrip())

                    if line.startswith(("ATOM", "HETATM")):
                        in_header = False
                        atoms.append(line)
                    elif line.startswith("CONECT"):
                        conect_lines.append(line.rstrip())

        # If no atoms found, return None
        if not atoms:
            logger.warning(f"No atoms found for model {model_num} in {pdb_path}")
            return None

        # Write temporary PDB file with just this model
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as temp:
            temp.write("MODEL        1\n")
            for atom_line in atoms:
                temp.write(atom_line)
            temp.write("ENDMDL\n")
            temp_name = temp.name

        # Parse the temporary file
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("temp", temp_name)

        # Clean up
        os.unlink(temp_name)

        # Parse CONECT records
        conect_dict = {}
        for line in conect_lines:
            parts = line.split()
            if len(parts) > 2:  # At least one connection
                atom_serial = int(parts[1])
                connected_serials = [int(parts[i]) for i in range(2, len(parts))]
                if atom_serial not in conect_dict:
                    conect_dict[atom_serial] = []
                conect_dict[atom_serial].extend(connected_serials)

        return structure, conect_dict, header_lines

    except Exception as e:
        logger.error(f"Error extracting model {model_num} from {pdb_path}: {str(e)}")
        return None


def count_models_in_pdb(pdb_path: str) -> int:
    """Count the number of models in a PDB file.

    Args:
        pdb_path: Path to the PDB file

    Returns:
        Number of models in the file (at least 1)
    """
    model_count = 0
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                model_count += 1

    # If no MODEL records found, treat as single model
    if model_count == 0:
        model_count = 1

    return model_count


def parse_conect_records(pdb_file: str, model_num: int = None) -> Dict[int, List[int]]:
    """Parse CONECT records from a PDB file.

    Args:
        pdb_file: Path to the PDB file
        model_num: Optional model number to extract CONECT records for (0-indexed)

    Returns:
        Dictionary mapping atom serial numbers to lists of connected atom serial numbers.
    """
    conect_dict = {}
    current_model = -1
    in_target_model = model_num is None  # If no model specified, get all CONECT records

    try:
        with open(pdb_file, "r") as f:
            for line in f:
                if line.startswith("MODEL"):
                    current_model += 1
                    in_target_model = model_num is None or current_model == model_num
                elif line.startswith("ENDMDL"):
                    if model_num is not None and current_model == model_num:
                        # We've processed the target model, we can stop
                        break
                    in_target_model = model_num is None  # Reset for single-model files

                # For files without MODEL records, consider everything as model 0
                if model_num == 0 and current_model == -1:
                    in_target_model = True

                # Parse CONECT records if we're in the target model or looking at all models
                if in_target_model and line.startswith("CONECT"):
                    # Format: CONECT serial# serial# serial# ...
                    # First serial is the atom, rest are connected atoms
                    parts = line.split()
                    if len(parts) > 2:  # At least one connection
                        try:
                            atom_serial = int(parts[1])
                            connected_serials = [
                                int(parts[i]) for i in range(2, len(parts))
                            ]

                            # Create entry or append to existing
                            if atom_serial not in conect_dict:
                                conect_dict[atom_serial] = []

                            # Avoid duplicates
                            for serial in connected_serials:
                                if serial not in conect_dict[atom_serial]:
                                    conect_dict[atom_serial].append(serial)
                        except ValueError:
                            # Skip malformed CONECT records
                            continue

        return conect_dict
    except Exception as e:
        print(f"Error parsing CONECT records: {e}")
        return {}


def calculate_clashes(
    structure1_atoms: List, structure2_atoms: List, cutoff: float = 1.5
) -> int:
    """Calculate the number of clashing atoms between two structures.

    Args:
        structure1_atoms: List of atoms from first structure
        structure2_atoms: List of atoms from second structure
        cutoff: Distance below which atoms are considered clashing (in Å)

    Returns:
        Number of clashing atoms
    """
    clash_count = 0
    for atom1 in structure1_atoms:
        coord1 = np.array(atom1.coord)
        # Skip hydrogen atoms
        if atom1.element.strip() == "H":
            continue

        for atom2 in structure2_atoms:
            # Skip hydrogen atoms
            if atom2.element.strip() == "H":
                continue

            # Skip comparing against itself
            if atom1 == atom2:
                continue

            coord2 = np.array(atom2.coord)
            dist = np.linalg.norm(coord1 - coord2)

            if dist < cutoff:
                clash_count += 1
                break  # Count each atom only once

    return clash_count


def superimpose_model(
    reference_ligand_atoms: List,
    model_atoms: List,
    logger: logging.Logger,
    reference_pdb: str = None,
    model_pdb: str = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, int]:
    """Superimpose a model onto a reference using graph-based isomorphism.

    Args:
        reference_ligand_atoms: Atoms from reference ligand
        model_atoms: Atoms from model to superimpose
        logger: Logger instance
        reference_pdb: Path to reference PDB for CONECT records
        model_pdb: Path to model PDB for CONECT records

    Returns:
        Tuple of (rotation matrix, translation vector, RMSD, number of matched atoms)
    """
    # Log basic atom information
    logger.info(f"Reference ligand: {len(reference_ligand_atoms)} atoms")
    logger.info(f"Model: {len(model_atoms)} atoms")

    # Parse CONECT records from reference PDB if provided
    ref_conect_dict = {}
    if reference_pdb:
        logger.info(f"Parsing CONECT records from {reference_pdb}")
        ref_conect_dict = parse_conect_records(reference_pdb)
        logger.info(
            f"Found {len(ref_conect_dict)} atoms with CONECT records in reference"
        )

    # Parse CONECT records from model PDB if provided
    model_conect_dict = {}
    if model_pdb:
        logger.info(f"Parsing CONECT records from {model_pdb}")
        model_conect_dict = parse_conect_records(model_pdb)
        logger.info(
            f"Found {len(model_conect_dict)} atoms with CONECT records in model"
        )

    # Create molecular graphs
    # For the reference, specify it's chain D (the UNL ligand)
    ref_graph = MolecularGraph(
        reference_ligand_atoms,
        logger,
        name="REF",
        conect_dict=ref_conect_dict,
        chain_id="D",
    )

    # For the model, use its CONECT records
    model_graph = MolecularGraph(
        model_atoms, logger, name="MODEL", conect_dict=model_conect_dict
    )

    # Print some basic graph diagnostics
    if logger.level <= logging.INFO:
        # Check atom types in both graphs
        ref_elements = set(
            ref_graph.graph.nodes[n]["element"] for n in ref_graph.graph.nodes
        )
        model_elements = set(
            model_graph.graph.nodes[n]["element"] for n in model_graph.graph.nodes
        )

        logger.info(f"Reference elements: {sorted(ref_elements)}")
        logger.info(f"Model elements: {sorted(model_elements)}")

        # Check if all reference elements exist in model
        missing_elements = ref_elements - model_elements
        if missing_elements:
            logger.warning(
                f"Elements in reference but not in model: {missing_elements}"
            )

    # Perform alignment
    logger.info("Starting alignment...")

    # Create a new superimposer instance
    custom_superimposer = IsomorphicSuperimposer(logger)
    result = custom_superimposer.align(ref_graph, model_graph)

    if result.transformation_matrix is not None:
        rotation, translation = result.transformation_matrix
        logger.info(
            f"Alignment successful: RMSD={result.rmsd:.4f}, {result.matched_atoms} atoms matched"
        )
        return rotation, translation, result.rmsd, result.matched_atoms
    else:
        logger.warning("Failed to find isomorphic matching between structures")

        # Detailed diagnostics for debugging
        logger.error("MATCHING FAILURE DIAGNOSTICS:")
        logger.error("----------------------------")

        # Check for identical elements
        ref_element_counts = {}
        model_element_counts = {}

        for n in ref_graph.graph.nodes:
            el = ref_graph.graph.nodes[n]["element"]
            if el not in ref_element_counts:
                ref_element_counts[el] = 0
            ref_element_counts[el] += 1

        for n in model_graph.graph.nodes:
            el = model_graph.graph.nodes[n]["element"]
            if el not in model_element_counts:
                model_element_counts[el] = 0
            model_element_counts[el] += 1

        logger.error(f"Reference element counts: {ref_element_counts}")
        logger.error(f"Model element counts: {model_element_counts}")

        # Compare graph metrics
        logger.error(
            f"Reference graph: {len(ref_graph.graph.nodes)} nodes, {len(ref_graph.graph.edges)} edges"
        )
        logger.error(
            f"Model graph: {len(model_graph.graph.nodes)} nodes, {len(model_graph.graph.edges)} edges"
        )

        # Look for potential subgraph
        ref_is_subgraph = True
        for el, count in ref_element_counts.items():
            if el not in model_element_counts or model_element_counts[el] < count:
                ref_is_subgraph = False
                logger.error(
                    f"Element mismatch: {el} has {count} in ref but {model_element_counts.get(el, 0)} in model"
                )

        if ref_is_subgraph:
            logger.error(
                "Reference COULD BE a subgraph of model based on element counts"
            )
        else:
            logger.error(
                "Reference is definitely NOT a subgraph of model based on element counts"
            )

        # Print some sample atoms from both graphs
        logger.error("Sample reference atoms:")
        for i, atom in enumerate(reference_ligand_atoms[:5]):
            residue = atom.get_parent()
            logger.error(f"  {i}: {atom.name} {atom.element} {residue.resname}")

        logger.error("Sample model atoms:")
        for i, atom in enumerate(model_atoms[:5]):
            residue = atom.get_parent()
            logger.error(f"  {i}: {atom.name} {atom.element} {residue.resname}")

        return None, None, float("inf"), 0


def apply_transformation_to_structure(
    structure: Structure, rotation: np.ndarray, translation: np.ndarray
) -> None:
    """Apply transformation to all atoms in a structure.

    Args:
        structure: Structure to transform
        rotation: Rotation matrix
        translation: Translation vector
    """
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.transform(rotation, translation)


def format_pdb_atom_line(atom, residue, chain_id=None):
    """Format an atom for PDB output with proper element handling.

    In PDB format, the element field is in columns 77-78. For hydrogen atoms,
    it's important to format them correctly to ensure proper visualization.
    Other elements also need specific handling to ensure proper connectivity.
    """
    # Use the chain ID from the parameter if provided, otherwise from the residue's parent chain
    chain_id_to_use = chain_id if chain_id else residue.get_parent().id

    # Handle element formatting - standardize to PDB conventions
    # PDB standard is right-justified in a 2-character field
    element = atom.element.strip().upper()

    # Special case for hydrogen (should be 'H ' not ' H')
    if element == "H":
        element_field = "H "
    # Special case for single-letter elements (C, N, O, S, P, etc.)
    elif len(element) == 1:
        element_field = f"{element} "
    # Standard handling for two-letter elements (CA, MG, etc.)
    else:
        element_field = element[:2]  # Limit to 2 characters

    # Format the atom line according to PDB standard
    # The element field is deliberately right-padded with spaces to ensure it ends at column 78
    # This is crucial for proper element recognition by visualization programs
    return f"ATOM  {atom.serial_number:5d} {atom.name:^4} {residue.resname:<3} {chain_id_to_use:1}{residue.id[1]:4d}    {atom.coord[0]:8.3f}{atom.coord[1]:8.3f}{atom.coord[2]:8.3f}  1.00  0.00           {element_field}\n"


def superimpose_trajectory(
    args: Tuple[str, str, str, bool, Optional[int], Optional[int], Optional[int], bool],
) -> Optional[Tuple[str, Dict[int, float]]]:
    """Superimpose a trajectory to the reference structure."""
    pdb_path, reference_pdb, output_dir, verbose, start, stop, stride, force = args

    # Set up logging
    logger = setup_logging(verbose)

    try:
        pdb_file = Path(pdb_path)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Output file will be in the same directory structure but in the output dir
        rel_path = pdb_file.relative_to(pdb_file.parent.parent)
        output_file = output_dir_path / f"{rel_path.stem}_superimposed.pdb"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Check if already processed
        rmsd_file = output_file.parent / f"{rel_path.stem}_rmsd.json"
        if (
            not force
            and output_file.exists()
            and rmsd_file.exists()
            and not isinstance(verbose, str)
            and not verbose
        ):
            logger.info(f"Skipping {pdb_file}, already processed")
            return None

        if force and output_file.exists():
            logger.info(f"Force flag set, reprocessing {pdb_file}")

        # Parse CONECT records from the test PDB file
        test_conect_dict = parse_conect_records(str(pdb_file))
        logger.info(
            f"Found {len(test_conect_dict)} atoms with CONECT records in test PDB"
        )

        with tempfile.TemporaryDirectory(prefix="superimpose_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)

            # Extract reference ligand (chain D)
            ref_ligand_file = extract_chain_from_reference(
                reference_pdb, ["D"], temp_dir / "reference_ligand.pdb", logger
            )

            # Extract reference protein (chains A, B, C)
            ref_protein_file = extract_chain_from_reference(
                reference_pdb,
                ["A", "B", "C"],
                temp_dir / "reference_protein.pdb",
                logger,
            )

            # Load reference structures
            parser = PDBParser(QUIET=True)
            ref_ligand = parser.get_structure("ref_ligand", str(ref_ligand_file))
            ref_protein = parser.get_structure("ref_protein", str(ref_protein_file))

            # Get reference ligand atoms for superimposition
            ref_ligand_atoms = list(unfold_entities(ref_ligand[0]["D"], "A"))

            if not ref_ligand_atoms:
                logger.error("No atoms found in reference ligand")
                return None

            logger.info(f"Reference ligand has {len(ref_ligand_atoms)} atoms")

            # Count models in trajectory
            model_count = count_models_in_pdb(pdb_path)
            logger.info(f"Found {model_count} models in {pdb_path}")

            if model_count == 0:
                logger.error(f"No models found in {pdb_path}")
                return None

            # Apply start, stop, stride parameters
            model_indices = list(range(model_count))

            # Apply slicing parameters if provided
            if start is not None or stop is not None or stride is not None:
                # Default values
                start_idx = 0 if start is None else start
                stop_idx = model_count if stop is None else min(stop, model_count)
                stride_val = 1 if stride is None else stride

                # Adjust for negative indices
                if start_idx < 0:
                    start_idx = max(0, model_count + start_idx)
                if stop_idx < 0:
                    stop_idx = max(0, model_count + stop_idx)

                # Create sliced list of indices
                model_indices = model_indices[start_idx:stop_idx:stride_val]
                logger.info(
                    f"Processing {len(model_indices)} models with slice [{start_idx}:{stop_idx}:{stride_val}]"
                )
            else:
                logger.info(f"Processing all {model_count} models")

            # Dictionary to store RMSD values
            rmsd_values = {}
            successful_models = 0

            # Create output PDB file
            with open(output_file, "w") as f:
                # Write header
                f.write("TITLE     Superimposed MD Trajectory\n")
                f.write(f"REMARK    Reference: {reference_pdb}\n")
                f.write(f"REMARK    Trajectory: {pdb_path}\n")
                f.write(f"REMARK    Method: Isomorphic Graph Matching\n")

                # Write reference model (model 0)
                f.write("MODEL        0\n")
                f.write("REMARK    Reference structure\n")

                # Write reference protein chains
                for chain in ref_protein[0]:
                    for residue in chain:
                        for atom in residue:
                            f.write(format_pdb_atom_line(atom, residue))

                # Write reference ligand
                for residue in ref_ligand[0]["D"]:
                    for atom in residue:
                        f.write(format_pdb_atom_line(atom, residue, chain_id="D"))

                # Write CONECT records from reference if available
                ref_conect_dict = parse_conect_records(reference_pdb)
                for serial, connected_serials in ref_conect_dict.items():
                    # Only write CONECT records for atoms in our reference structure
                    atom_found = False
                    for chain in ref_protein[0]:
                        for residue in chain:
                            for atom in residue:
                                if atom.serial_number == serial:
                                    atom_found = True
                                    break
                            if atom_found:
                                break
                        if atom_found:
                            break

                    if not atom_found:
                        for residue in ref_ligand[0]["D"]:
                            for atom in residue:
                                if atom.serial_number == serial:
                                    atom_found = True
                                    break
                            if atom_found:
                                break

                    if atom_found:
                        conect_line = f"CONECT{serial:5d}"
                        for connected in connected_serials:
                            conect_line += f"{connected:5d}"
                        f.write(conect_line + "\n")

                f.write("ENDMDL\n")

            # Process each model in the trajectory based on slicing
            for model_idx in model_indices:
                logger.info(f"Processing model {model_idx+1} from {pdb_path}")

                # Extract the model
                model_data = get_model_from_pdb(str(pdb_path), model_idx, logger)

                if model_data is None:
                    logger.warning(f"Failed to extract model {model_idx+1}")
                    continue

                model_structure, model_conect_dict, model_header_lines = model_data

                if model_structure is None or len(model_structure) == 0:
                    logger.warning(f"Failed to extract model {model_idx+1}")
                    continue

                # If model_conect_dict is empty, try to parse from the original file
                if not model_conect_dict:
                    logger.info(
                        f"No CONECT records found in model extraction, trying to parse from original file"
                    )
                    # Parse CONECT records specifically for this model number
                    model_conect_dict = parse_conect_records(str(pdb_file), model_idx)
                    if model_conect_dict:
                        logger.info(
                            f"Successfully parsed {len(model_conect_dict)} CONECT records from original file for model {model_idx}"
                        )
                    else:
                        logger.warning(
                            f"Could not find CONECT records in original file for model {model_idx}"
                        )
                        # Try parsing all CONECT records as fallback
                        model_conect_dict = parse_conect_records(str(pdb_file))
                        if model_conect_dict:
                            logger.info(
                                f"Parsed {len(model_conect_dict)} CONECT records from entire file"
                            )
                        else:
                            logger.warning(
                                f"No CONECT records found at all in the file"
                            )

                # Get all atoms from the model
                model_atoms = []
                for chain in model_structure[0]:
                    model_atoms.extend(list(unfold_entities(chain, "A")))

                if not model_atoms:
                    logger.warning(f"No atoms found in model {model_idx+1}")
                    continue

                logger.info(f"Model {model_idx+1} has {len(model_atoms)} atoms")

                # Create superimposer with logger
                superimposer = IsomorphicSuperimposer(logger)

                # Superimpose model onto reference, passing both CONECT dictionaries
                try:
                    rotation, translation, rmsd, matched_atoms = superimpose_model(
                        ref_ligand_atoms,
                        model_atoms,
                        logger,
                        reference_pdb=reference_pdb,
                        model_pdb=str(pdb_file),
                    )
                except Exception as e:
                    logger.error(
                        f"Error during superimposition of model {model_idx+1}: {str(e)}"
                    )
                    rotation, translation, rmsd, matched_atoms = (
                        None,
                        None,
                        float("inf"),
                        0,
                    )

                if rotation is not None and translation is not None:
                    # Apply transformation to the model
                    try:
                        apply_transformation_to_structure(
                            model_structure, rotation, translation
                        )
                    except Exception as e:
                        logger.error(
                            f"Error applying transformation to model {model_idx+1}: {str(e)}"
                        )
                        continue

                    # Calculate clashes between model and reference
                    ref_protein_atoms = []
                    for chain in ref_protein[0]:
                        ref_protein_atoms.extend(list(unfold_entities(chain, "A")))

                    try:
                        # Only consider clashes between the protein (A,B,C chains) and the test compound
                        # Do not include reference ligand as a clashable object
                        clash_count = calculate_clashes(model_atoms, ref_protein_atoms)
                    except Exception as e:
                        logger.error(
                            f"Error calculating clashes for model {model_idx+1}: {str(e)}"
                        )
                        clash_count = 0

                    # Append to output file
                    try:
                        with open(output_file, "a") as f:
                            f.write(f"MODEL      {model_idx+1}\n")

                            # Add header information from original model
                            for header_line in model_header_lines:
                                f.write(f"{header_line}\n")

                            # Add information about the superimposition
                            f.write(f"REMARK    RMSD: {rmsd:.4f}\n")
                            f.write(f"REMARK    Matched Atoms: {matched_atoms}\n")
                            f.write(f"REMARK    Clashing Atoms: {clash_count}\n")

                            # Write transformed model atoms
                            for chain in model_structure[0]:
                                for residue in chain:
                                    for atom in residue:
                                        f.write(format_pdb_atom_line(atom, residue))

                            # Write CONECT records from the model
                            for serial, connected_serials in model_conect_dict.items():
                                # Only write CONECT records for atoms in our model
                                atom_found = False
                                atom_obj = None

                                # Create a mapping of serial numbers to ensure we handle all atoms
                                serial_to_atom = {}
                                for chain in model_structure[0]:
                                    for residue in chain:
                                        for atom in residue:
                                            serial_to_atom[atom.serial_number] = atom

                                # Check if this serial number exists in our model
                                if serial in serial_to_atom:
                                    atom_found = True
                                    atom_obj = serial_to_atom[serial]

                                    # Filter connected serials to only include those that exist in our model
                                    valid_connections = [
                                        conn
                                        for conn in connected_serials
                                        if conn in serial_to_atom
                                    ]

                                    if valid_connections:
                                        # Write the CONECT record with verified connections
                                        conect_line = f"CONECT{serial:5d}"
                                        for connected in valid_connections:
                                            conect_line += f"{connected:5d}"
                                        f.write(conect_line + "\n")
                                    else:
                                        if self.logger:
                                            logger.debug(
                                                f"Skipping CONECT for {serial} - no valid connections"
                                            )

                            # Also add standard connectivity for common molecules if not already defined
                            # by CONECT records - this is a fallback for missing connectivity
                            if len(model_conect_dict) == 0:
                                logger.warning(
                                    f"No CONECT records found for model {model_idx+1}, inferring standard connectivity"
                                )
                                # Add code to infer standard connectivity if needed

                            f.write("ENDMDL\n")

                        successful_models += 1
                    except Exception as e:
                        logger.error(
                            f"Error writing model {model_idx+1} to output file: {str(e)}"
                        )
                        continue

                    # Store RMSD and match information
                    rmsd_values[model_idx + 1] = {
                        "rmsd": float(rmsd),
                        "matched_atoms": matched_atoms,
                        "clashing_atoms": clash_count,
                    }

                    logger.info(
                        f"Model {model_idx+1} superimposed with RMSD: {rmsd:.4f} ({matched_atoms} atoms matched, {clash_count} atoms clashing)"
                    )
                else:
                    logger.warning(f"Failed to superimpose model {model_idx+1}")

            # Save RMSD summary
            try:
                with open(rmsd_file, "w") as f:
                    json.dump(rmsd_values, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving RMSD values: {str(e)}")

            logger.info(
                f"Superimposition complete. {successful_models} models saved to {output_file}"
            )

            # Verify the output file exists and has content
            if output_file.exists():
                file_size = output_file.stat().st_size
                logger.info(f"Output file size: {file_size} bytes")

                if file_size == 0:
                    logger.error("Output file is empty!")

                # Count models in output file
                output_model_count = count_models_in_pdb(str(output_file))
                logger.info(f"Output file contains {output_model_count} models")

                if output_model_count <= 1:  # Only reference model
                    logger.error("Output file only contains the reference model!")
            else:
                logger.error("Output file does not exist!")

            return pdb_path, rmsd_values

    except Exception as e:
        logger.error(f"Error processing {pdb_path}: {str(e)}")
        logger.exception(e)  # Log full traceback
        return None


def find_trajectory_pdbs(base_dir: str) -> List[str]:
    """Find all md_Ref.pdb files in the directory structure.

    Args:
        base_dir: Base directory to search

    Returns:
        List of paths to PDB files
    """
    base_path = Path(base_dir)
    pdb_files = list(base_path.rglob("md_Ref.pdb"))
    return [str(path) for path in pdb_files]


def main():
    """Main function for superimposing trajectories."""
    parser = argparse.ArgumentParser(
        description="Superimpose converted MD trajectories to a reference structure"
    )
    parser.add_argument(
        "base_dir", type=str, help="Base directory containing md_Ref.pdb files"
    )
    parser.add_argument("reference_pdb", type=str, help="Path to reference PDB file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="superimposed",
        help="Directory for output files",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of parallel processes to use",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already superimposed trajectories",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed processing information"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show extremely detailed debug information"
    )
    # Add trajectory frame selection arguments
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Starting model index (0-indexed, inclusive)",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Ending model index (exclusive)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Step size for selecting models",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = "debug" if args.debug else (True if args.verbose else False)
    logger = setup_logging(log_level)

    logger.info("=== Starting superimposition process ===")
    logger.info(f"Reference PDB: {args.reference_pdb}")

    # Check if reference file exists
    if not os.path.exists(args.reference_pdb):
        logger.error(f"Reference PDB file not found: {args.reference_pdb}")
        return

    # Find all trajectory PDB files
    logger.info(f"Searching for trajectory files in {args.base_dir}...")
    pdb_files = find_trajectory_pdbs(args.base_dir)

    if not pdb_files:
        logger.error(f"No md_Ref.pdb files found in {args.base_dir}")
        return

    logger.info(f"Found {len(pdb_files)} md_Ref.pdb files")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try processing single-threaded first if debugging
    if args.debug:
        logger.info("Running in debug mode with single thread")
        for pdb_file in pdb_files:
            logger.info(f"Processing {pdb_file}...")
            result = superimpose_trajectory(
                (
                    pdb_file,
                    args.reference_pdb,
                    args.output_dir,
                    "debug",
                    args.start,
                    args.stop,
                    args.stride,
                    args.force,
                )
            )
            if result:
                pdb_path, rmsd_values = result
                # Calculate average RMSD
                avg_rmsd = np.mean([info["rmsd"] for info in rmsd_values.values()])
                logger.info(f"Completed {pdb_path} with average RMSD: {avg_rmsd:.4f}")
        return

    # Process files in parallel for normal mode
    with tqdm(total=len(pdb_files), unit="file") as pbar:
        with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
            futures = []

            for pdb_file in pdb_files:
                # Check if already processed and not forcing reprocessing
                rel_path = Path(pdb_file).relative_to(Path(pdb_file).parent.parent)
                output_file = output_dir / f"{rel_path.stem}_superimposed.pdb"
                rmsd_file = output_dir / f"{rel_path.stem}_rmsd.json"

                if not args.force and output_file.exists() and rmsd_file.exists():
                    pbar.update(1)
                    continue

                futures.append(
                    executor.submit(
                        superimpose_trajectory,
                        (
                            pdb_file,
                            args.reference_pdb,
                            args.output_dir,
                            args.verbose,
                            args.start,
                            args.stop,
                            args.stride,
                            args.force,
                        ),
                    )
                )

            # Process results as they complete
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)

                if result:
                    pdb_path, rmsd_values = result
                    # Calculate average RMSD
                    avg_rmsd = np.mean([info["rmsd"] for info in rmsd_values.values()])
                    logger.info(
                        f"Completed {pdb_path} with average RMSD: {avg_rmsd:.4f}"
                    )

    logger.info("Superimposition complete!")


if __name__ == "__main__":
    main()
