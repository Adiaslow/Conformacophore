from abc import ABC, abstractmethod
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Superimposer import Superimposer
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.PDBIO import Select
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import multiprocessing
import numpy as np
import pandas as pd
import networkx as nx
import tempfile
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm


@dataclass
class ClashResult:
    """Contains results from clash detection."""

    has_clashes: bool
    num_clashes: int
    clash_pairs: List[tuple]


@dataclass
class AlignmentResult:
    """Contains results from molecular structure alignment."""

    rmsd: float
    matched_atoms: int
    transformation_matrix: Optional[Tuple[np.ndarray, np.ndarray]]
    matched_pairs: List[Tuple[Any, Any]]
    clash_results: Optional[ClashResult] = None


class ModelSelector(Select):
    """A selector for a specific model in a multi-model PDB file."""

    def __init__(self, model_number=0):
        self.model_number = model_number

    def accept_model(self, model):
        """Only accept the specified model."""
        return model.id == self.model_number


def get_structure_from_model(self, filepath: str, model_num: int = 0) -> Structure:
    """
    Read a specific model from a PDB file.

    Args:
        filepath (str): Path to the PDB file
        model_num (int, optional): Model number to extract. Defaults to 0 (first model).

    Returns:
        Structure: Biopython Structure object for the specified model
    """
    parser = PDBParser(QUIET=True)

    # Parse the entire structure
    full_structure = parser.get_structure("model", filepath)

    # Handle single model files
    if len(full_structure) <= model_num:
        return full_structure[0]  # Return first model if model_num is out of range

    # Create a new structure with only the desired model
    selected_structure = Structure("selected")
    selected_model = Model(0)
    selected_structure.add(selected_model)

    # Copy chains from the specified model
    for chain in full_structure[model_num]:
        new_chain = chain.copy()
        selected_model.add(new_chain)

    return selected_structure


# Create a default selector that accepts everything
class DefaultSelect(Select):
    def accept_model(self, model):
        return 1

    def accept_chain(self, chain):
        return 1

    def accept_residue(self, residue):
        return 1

    def accept_atom(self, atom):
        return 1


# Create a default selector instance
_select = DefaultSelect()


class PDBHeaderHandler:
    """Handles reading and writing of comprehensive PDB information."""

    def __init__(self):
        # Comprehensive dictionary to store all PDB file information
        self.full_pdb_info = {
            "global_headers": [],  # Global header lines
            "model_headers": {},  # Model-specific headers
            "connectivity": [],  # CONECT records
            "remarks": {},  # Detailed remarks
            "other_records": {  # Other specific records
                "SEQRES": [],
                "COMPND": [],
                "SOURCE": [],
                "KEYWDS": [],
                "EXPDTA": [],
                "TITLE": [],
                "AUTHOR": [],
                "REVDAT": [],
                "DBREF": [],
                "SEQADV": [],
                "HET": [],
                "HETNAM": [],
                "HETSYN": [],
                "FORMUL": [],
                "HELIX": [],
                "SHEET": [],
                "TURN": [],
                "SSBOND": [],
                "LINK": [],
                "MODRES": [],
            },
        }
        # Add compatibility with old code
        self.headers = self.full_pdb_info["global_headers"]
        self.model_headers = self.full_pdb_info["model_headers"]
        self.current_model = -1

    # Rest of the implementation remains the same as in the previous script
    def read_headers(self, pdb_path: str):
        """
        Comprehensively read all information from PDB file.

        Captures global and model-specific information, including
        connectivity, remarks, and other specialized records.
        """
        # Reset existing information
        self.__init__()

        # Specific record types to capture
        record_types = list(self.full_pdb_info["other_records"].keys()) + [
            "REMARK",
            "CONECT",
            "MODEL",
            "ENDMDL",
        ]

        with open(pdb_path, "r") as f:
            lines = f.readlines()

        current_model = -1
        reading_model = False
        model_specific_lines = {}

        for line in lines:
            record_type = line[:6].strip()

            # Global headers (before any MODEL)
            if not reading_model:
                # Capture global headers and other global records
                if record_type in record_types:
                    if record_type == "MODEL":
                        current_model += 1
                        reading_model = True
                        model_specific_lines[current_model] = []
                    else:
                        self.full_pdb_info["global_headers"].append(line)

            # Model-specific information
            if reading_model:
                if record_type == "ENDMDL":
                    reading_model = False
                    current_model = -1
                elif current_model != -1:
                    # Capture model-specific information
                    model_specific_lines[current_model].append(line)

            # Specialized record handling
            if record_type in self.full_pdb_info["other_records"]:
                self.full_pdb_info["other_records"][record_type].append(line)

            # Connectivity records
            if record_type == "CONECT":
                self.full_pdb_info["connectivity"].append(line)

            # Detailed REMARK handling
            if record_type == "REMARK":
                try:
                    remark_num = int(line[6:10].strip())
                    if remark_num not in self.full_pdb_info["remarks"]:
                        self.full_pdb_info["remarks"][remark_num] = []
                    self.full_pdb_info["remarks"][remark_num].append(line)
                except (ValueError, IndexError):
                    pass

        # Store model-specific headers
        self.full_pdb_info["model_headers"] = model_specific_lines

    def write_connectivity(self, fhandle):
        """
        Write connectivity information to a file handle.

        Args:
            fhandle: File handle to write the connectivity information to.
        """
        for record in self.full_pdb_info["connectivity"]:
            fhandle.write(record)


class CustomPDBIO(PDBIO):
    """
    Extended PDBIO class that preserves comprehensive PDB header and connection information.
    """

    def __init__(self):
        super().__init__()
        self._header_handler: Optional[PDBHeaderHandler] = None

    @property
    def header_handler(self) -> Optional[PDBHeaderHandler]:
        """Getter for header_handler"""
        return self._header_handler

    @header_handler.setter
    def header_handler(self, value: Optional[PDBHeaderHandler]):
        """Setter for header_handler"""
        self._header_handler = value

    def save(self, file, select=_select, write_end=True, preserve_atom_numbering=False):
        """
        Save structure to a file with comprehensive header handling.
        """
        # Prepare file handle
        if isinstance(file, str):
            fhandle = open(file, "w")
            close_file = True
        else:
            fhandle = file
            close_file = False

        # Write headers if available and file is a string (new file)
        if isinstance(file, str) and self._header_handler:
            # Write comprehensive headers
            self._header_handler.write_headers(fhandle)

        try:
            # Invoke the parent save method
            super().save(
                fhandle,
                select=select,
                write_end=write_end,
                preserve_atom_numbering=preserve_atom_numbering,
            )

            # Write connectivity records if available
            if self._header_handler:
                self._header_handler.write_connectivity(fhandle)

        finally:
            # Close file if we opened it
            if close_file:
                fhandle.close()


class MolecularGraph:
    """Represents a molecular structure as a graph."""

    def __init__(self, atoms: List[Any]):
        """Initialize molecular graph from list of atoms."""
        self.atoms = atoms
        self.graph = self._create_graph()

    def _create_graph(self) -> nx.Graph:
        """Create NetworkX graph from atoms using PDB connectivity."""
        G = nx.Graph()

        # Add nodes with attributes
        for atom in self.atoms:
            residue = atom.get_parent()
            node_attrs = {
                "name": atom.name,
                "element": atom.element,
                "coord": tuple(atom.coord),
                "residue_name": residue.resname,
                "residue_id": residue.id,
            }
            G.add_node(atom, **node_attrs)

        # Create a map of residues to their atoms for faster lookup
        residue_atoms = {}
        for atom in self.atoms:
            residue = atom.get_parent()
            if residue not in residue_atoms:
                residue_atoms[residue] = []
            residue_atoms[residue].append(atom)

        # Add edges based on covalent bonding distances
        for atom in self.atoms:
            coord1 = np.array(atom.coord)
            residue = atom.get_parent()

            # Check connections within the same residue
            for other_atom in residue_atoms[residue]:
                if atom != other_atom:
                    coord2 = np.array(other_atom.coord)
                    dist = np.linalg.norm(coord1 - coord2)
                    # Use a distance cutoff appropriate for covalent bonds
                    if dist < 2.0:  # Typical covalent bond length plus some tolerance
                        G.add_edge(atom, other_atom)

        return G


class StructureSuperimposer(ABC):
    """Abstract base class for structure superimposition strategies."""

    @abstractmethod
    def align(self, mol1: MolecularGraph, mol2: MolecularGraph) -> AlignmentResult:
        """Align two molecular structures."""
        pass


class IsomorphicSuperimposer(StructureSuperimposer):
    """Graph isomorphism-based structure alignment."""

    def align(self, mol1: MolecularGraph, mol2: MolecularGraph) -> AlignmentResult:
        """Implement graph matching algorithm."""
        matches = self._custom_graph_match(mol1.graph, mol2.graph)

        if not matches:
            return self._empty_result()

        matched_pairs = list(matches.items())
        matched_mol1_atoms = [pair[0] for pair in matched_pairs]
        matched_mol2_atoms = [pair[1] for pair in matched_pairs]

        superimposer = Superimposer()
        superimposer.set_atoms(matched_mol1_atoms, matched_mol2_atoms)

        return AlignmentResult(
            rmsd=superimposer.rms,
            matched_atoms=len(matched_pairs),
            transformation_matrix=(superimposer.rotran[0], superimposer.rotran[1]),
            matched_pairs=matched_pairs,
        )

    def _custom_graph_match(
        self, ligand_graph: nx.Graph, cyclic_peptide_graph: nx.Graph
    ) -> Dict[Any, Any]:
        """Custom graph matching implementation."""

        def find_terminal_nodes(graph):
            return [node for node in graph.nodes() if graph.degree(node) == 1]

        def node_matches(node1, node2):
            return (
                ligand_graph.nodes[node1]["element"]
                == cyclic_peptide_graph.nodes[node2]["element"]
            )

        def explore_neighborhood(
            ligand_node, cyclic_node, visited_ligand=None, visited_cyclic=None
        ):
            if visited_ligand is None:
                visited_ligand = set()
            if visited_cyclic is None:
                visited_cyclic = set()

            current_match = {ligand_node: cyclic_node}
            visited_ligand.add(ligand_node)
            visited_cyclic.add(cyclic_node)

            ligand_neighbors = list(ligand_graph.neighbors(ligand_node))
            cyclic_neighbors = list(cyclic_peptide_graph.neighbors(cyclic_node))

            for lig_neighbor in ligand_neighbors:
                if lig_neighbor in visited_ligand:
                    continue

                potential_matches = [
                    cyc_neighbor
                    for cyc_neighbor in cyclic_neighbors
                    if node_matches(lig_neighbor, cyc_neighbor)
                    and cyc_neighbor not in visited_cyclic
                ]

                if not potential_matches:
                    return None

                for cyc_match in potential_matches:
                    neighbor_match = explore_neighborhood(
                        lig_neighbor,
                        cyc_match,
                        visited_ligand.copy(),
                        visited_cyclic.copy(),
                    )

                    if neighbor_match:
                        current_match.update(neighbor_match)
                        break
                else:
                    return None

            return current_match

        matches = {}
        ligand_terminals = find_terminal_nodes(ligand_graph)
        cyclic_terminals = find_terminal_nodes(cyclic_peptide_graph)

        for lig_start in ligand_terminals:
            for cyc_start in cyclic_terminals:
                if node_matches(lig_start, cyc_start):
                    match = explore_neighborhood(lig_start, cyc_start)
                    if match and len(match) > len(matches):
                        matches = match

        return matches

    @staticmethod
    def _empty_result() -> AlignmentResult:
        """Return an empty alignment result."""
        return AlignmentResult(
            rmsd=float("inf"),
            matched_atoms=0,
            transformation_matrix=None,
            matched_pairs=[],
        )


class StructureAligner:
    """Main class for structure alignment using different strategies."""

    def __init__(self, strategy: StructureSuperimposer):
        self.strategy = strategy

    def get_structure_from_model(self, filepath: str, model_num: int) -> Structure:
        """Read a specific model from a PDB file."""
        atoms = []
        current_model = -1
        reading_model = False

        with open(filepath, "r") as f:
            for line in f:
                if line.startswith("MODEL"):
                    current_model += 1
                    if current_model == model_num:
                        reading_model = True
                        atoms = []
                elif line.startswith("ENDMDL"):
                    if reading_model:
                        break
                elif reading_model and (
                    line.startswith("ATOM") or line.startswith("HETATM")
                ):
                    atoms.append(line)

        # If no MODEL records found, treat as single model
        if current_model == -1:
            with open(filepath, "r") as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        atoms.append(line)

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

        return structure

    def detect_clashes(
        self,
        structure: Structure,
        chain_to_check: Chain,
        target_chains: List[str],
        distance_cutoff: float = 2.0,
    ) -> ClashResult:
        """
        Detect clashes between a chain and specified target chains.

        Args:
            structure: Complete structure containing all chains
            chain_to_check: Chain to check for clashes
            target_chains: List of chain IDs to check against
            distance_cutoff: Distance threshold for clash detection (Angstroms)

        Returns:
            ClashResult object containing clash information
        """
        # Get atoms from the chain we're checking
        chain_atoms = list(unfold_entities(chain_to_check, "A"))

        # Get atoms from target chains
        target_atoms = []
        for chain_id in target_chains:
            if chain_id in structure[0]:
                target_atoms.extend(list(unfold_entities(structure[0][chain_id], "A")))

        if not target_atoms:
            return ClashResult(
                has_clashes=False,
                num_clashes=0,
                clash_pairs=[],
            )

        # Create neighbor search for target atoms
        ns = NeighborSearch(target_atoms)

        # Check for clashes
        clash_pairs = []

        for atom in chain_atoms:
            # Skip hydrogen atoms
            if atom.element == "H":
                continue

            # Find all close atoms
            close_atoms = ns.search(atom.coord, distance_cutoff)

            for close_atom in close_atoms:
                # Skip hydrogen atoms
                if close_atom.element == "H":
                    continue

                # Calculate exact distance
                distance = np.linalg.norm(atom.coord - close_atom.coord)

                if distance < distance_cutoff:
                    clash_pairs.append((atom, close_atom))

        return ClashResult(
            has_clashes=len(clash_pairs) > 0,
            num_clashes=len(clash_pairs),
            clash_pairs=clash_pairs,
        )

    def align_structures(
        self,
        ligand_path: str,
        cyclic_path: str,
        output_dir: Optional[str] = None,
        ligand_chain: str = "A",
        cyclic_chain: str = "A",
        new_chain: str = "B",
        target_chains: Optional[List[str]] = None,
        clash_cutoff: float = 2.0,
        write_pdbs: bool = False,
        model_number: Optional[int] = None,
    ) -> Tuple[List[AlignmentResult], dict]:
        """
        Align structures and check for clashes with target chains.
        """
        parser = PDBParser(QUIET=True)

        # Default to first model if not specified
        if model_number is None:
            model_number = 0

        try:
            # Read ligand structure
            ligand_struct = parser.get_structure("ligand", ligand_path)

            # Read cyclic structure with specific model
            try:
                cyclic_struct = self.get_structure_from_model(cyclic_path, model_number)
            except Exception as model_err:
                print(f"Error extracting model {model_number}: {model_err}")
                return [self._empty_result()], {}

            # Initialize header handler and read headers
            header_handler = PDBHeaderHandler()
            header_handler.read_headers(cyclic_path)

            # Get model information
            model_info = get_model_info(cyclic_path, model_number)

            # Validate chain existence
            if cyclic_chain not in cyclic_struct[0]:
                print(f"Cyclic chain {cyclic_chain} not found in structure")
                return [self._empty_result()], model_info

            if ligand_chain not in ligand_struct[0]:
                print(f"Ligand chain {ligand_chain} not found in structure")
                return [self._empty_result()], model_info

            # Extract atoms
            cyclic_atoms = list(unfold_entities(cyclic_struct[0][cyclic_chain], "A"))
            ligand_atoms = list(unfold_entities(ligand_struct[0][ligand_chain], "A"))

            # Create molecular graphs
            ligand_graph = MolecularGraph(ligand_atoms)
            cyclic_graph = MolecularGraph(cyclic_atoms)

            # Align structures
            result = self.strategy.align(ligand_graph, cyclic_graph)

            # Create combined structure for clash detection and output
            combined = Structure("combined")
            out_model = Model(0)
            combined.add(out_model)

            if result.transformation_matrix is not None:
                # Add target chains for clash detection
                if target_chains:
                    for chain_id in target_chains:
                        if chain_id in ligand_struct[0]:
                            out_model.add(ligand_struct[0][chain_id].copy())

                # Add ligand chain
                out_model.add(ligand_struct[0][ligand_chain].copy())

                # Add transformed cyclic peptide with new chain ID
                new_chain_obj = Chain(new_chain)
                old_chain = cyclic_struct[0][cyclic_chain]

                for residue in old_chain:
                    new_chain_obj.add(residue.copy())

                # Apply transformation
                rotation, translation = result.transformation_matrix
                for atom in unfold_entities(new_chain_obj, "A"):
                    atom.transform(rotation, translation)

                out_model.add(new_chain_obj)

                # Perform clash detection if target chains are specified
                if target_chains:
                    clash_result = self.detect_clashes(
                        combined, new_chain_obj, target_chains, clash_cutoff
                    )
                    result.clash_results = clash_result

                # Write PDB if requested
                if write_pdbs and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(
                        output_dir, f"aligned_conf_{model_number+1}.pdb"
                    )

                    # Create a custom PDBIO with model-specific headers
                    io = CustomPDBIO()

                    # Set the header handler to preserve model-specific headers
                    io.header_handler = header_handler

                    # Set the structure
                    io.set_structure(combined)

                    # Write headers manually if they exist
                    with open(output_path, "w") as f:
                        # Write global headers
                        for header in header_handler.headers:
                            f.write(header)

                        # Write model-specific headers if they exist
                        if model_number in header_handler.model_headers:
                            for header in header_handler.model_headers[model_number]:
                                f.write(header)

                        # Save the structure
                        io.save(f)

            return [result], model_info

        except Exception as e:
            print(f"Unexpected error in alignment: {str(e)}")
            import traceback

            traceback.print_exc()
            return [self._empty_result()], {}


def get_model_info(pdb_path: str, model_idx: int) -> dict:
    """Extract model information from PDB file."""
    info = {"compound": None, "frame": None, "sequence": None, "headers": []}

    with open(pdb_path, "r") as f:
        lines = f.readlines()

    # Always try to capture global headers
    info["headers"] = [
        line
        for line in lines
        if any(
            line.startswith(keyword)
            for keyword in [
                "HEADER",
                "TITLE",
                "COMPND",
                "SOURCE",
                "KEYWDS",
                "EXPDTA",
                "AUTHOR",
                "REVDAT",
                "REMARK",
                "SEQRES",
            ]
        )
    ]

    # Track model-specific information
    current_model = -1
    reading_model = False

    for line in lines:
        if line.startswith("MODEL"):
            current_model += 1
            if current_model == model_idx:
                reading_model = True

        elif line.startswith("ENDMDL"):
            if reading_model:
                break

        elif reading_model:
            if line.startswith("COMPND"):
                # Extract just the compound number
                try:
                    info["compound"] = line.split()[1].strip()
                except IndexError:
                    print(f"Warning: Could not parse COMPND line: {line}")

            elif line.startswith("REMARK Frame"):
                try:
                    info["frame"] = line.split()[-1].strip()
                except IndexError:
                    print(f"Warning: Could not parse Frame line: {line}")

            elif line.startswith("SEQRES"):
                # Get everything after the residue count
                try:
                    parts = line.split()
                    info["sequence"] = " ".join(parts[4:])
                except IndexError:
                    print(f"Warning: Could not parse SEQRES line: {line}")

    # Print headers for debugging
    """
    print(f"\nHeaders for file {pdb_path} (Model {model_idx}):")
    for header in info['headers']:
        print(header.strip())

    print(f"\nParsed Model Info:")
    print(f"Compound: {info['compound']}")
    print(f"Frame: {info['frame']}")
    print(f"Sequence: {info['sequence']}")
    """
    return info


import os
import multiprocessing
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from tqdm import tqdm  # Added tqdm import

# Rest of the imports remain the same as in the previous script


def process_single_model(args):
    """
    Process a single model from a PDB file with detailed error handling.

    Args:
        args (tuple): Contains processing arguments

    Returns:
        Dict or None: Processing results for the specific model
    """
    # Unpack arguments (keeping implementation the same as previous script)
    (
        pdb_path,
        model_idx,
        reference_pdb,
        ligand_chain,
        target_chains,
        clash_cutoff,
        write_pdbs,
        output_dir,
    ) = args

    try:
        # Initialize aligner
        aligner = StructureAligner(IsomorphicSuperimposer())

        # Get model information
        model_info = get_model_info(pdb_path, model_idx)

        # Determine output directory for this model if writing PDbs
        conformer_output_dir = (
            os.path.join(output_dir, os.path.splitext(os.path.basename(pdb_path))[0])
            if write_pdbs
            else None
        )
        if write_pdbs and not os.path.exists(conformer_output_dir):
            os.makedirs(conformer_output_dir, exist_ok=True)

        # Align structures for this model
        results = aligner.align_structures(
            reference_pdb,
            pdb_path,
            conformer_output_dir,
            ligand_chain=ligand_chain,
            cyclic_chain="A",
            new_chain="X",
            target_chains=target_chains,
            clash_cutoff=clash_cutoff,
            write_pdbs=write_pdbs,
            model_number=model_idx,
        )[0]
        result = results[0]

        # Prepare clash information
        clash_info = {
            "Has_Clashes": False,
            "Num_Clashes": 0,
        }

        if result.clash_results:
            clash_info.update(
                {
                    "Has_Clashes": result.clash_results.has_clashes,
                    "Num_Clashes": result.clash_results.num_clashes,
                }
            )

        # Combine all information
        model_result = {
            "Filename": os.path.basename(pdb_path),
            "Model": model_idx + 1,
            "Compound": model_info["compound"],
            "Frame": model_info["frame"],
            "Sequence": model_info["sequence"],
            "RMSD": result.rmsd,
            "Matched_Atoms": result.matched_atoms,
            **clash_info,
        }

        return model_result

    except Exception as e:
        print(f"Error processing file {pdb_path}, model {model_idx + 1}: {str(e)}")
        return None

    def _empty_result(self):
        """Return an empty alignment result."""
        return AlignmentResult(
            rmsd=float("inf"),
            matched_atoms=0,
            transformation_matrix=None,
            matched_pairs=[],
        )


def process_directory(
    input_dir: str,
    output_dir: str,
    reference_pdb: str,
    ligand_chain: str,
    target_chains: Optional[List[str]] = None,
    clash_cutoff: float = 2.0,
    write_pdbs: bool = False,
    num_processes: Optional[int] = None,
):
    """
    Process PDB files using multiprocessing at the model level with tqdm progress tracking.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine number of processes
    if num_processes is None:
        # Use all available CPU cores
        num_processes = max(1, multiprocessing.cpu_count() - 1)

    print(f"Using {num_processes} parallel processes")

    # Collect all models from all PDB files to process
    processing_tasks = []
    for pdb_file in os.listdir(input_dir):
        if not pdb_file.endswith(".pdb"):
            continue

        pdb_path = os.path.join(input_dir, pdb_file)

        # Count models in the file
        model_count = sum(1 for line in open(pdb_path) if line.startswith("MODEL"))
        if model_count == 0:
            model_count = 1

        # Create processing task for each model
        for model_idx in range(model_count):
            processing_tasks.append(
                (
                    pdb_path,
                    model_idx,
                    reference_pdb,
                    ligand_chain,
                    target_chains,
                    clash_cutoff,
                    write_pdbs,
                    output_dir,
                )
            )

    # Use multiprocessing Pool with tqdm
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Process models in parallel with progress bar
        model_results = list(
            tqdm(
                pool.imap(process_single_model, processing_tasks),
                total=len(processing_tasks),
                desc="Processing Models",
                unit="model",
            )
        )

    # Filter out None results (failed processing)
    model_results = [item for item in model_results if item is not None]

    # Rest of the implementation remains the same as in the previous script
    # (Grouping results, generating summaries, etc.)

    # Group results by filename for summary statistics
    results_by_file = {}
    for result in model_results:
        filename = result["Filename"]
        if filename not in results_by_file:
            results_by_file[filename] = []
        results_by_file[filename].append(result)

        # Process summary for each file (keep the rest of the implementation same as previous script)
        # Process summary for each file
        summary_data = []
        for filename, file_results in results_by_file.items():
            # Convert results to DataFrame for easier analysis
            conformer_df = pd.DataFrame(file_results)

            # Skip if no results
            if conformer_df.empty:
                continue

            # Get first non-null compound and sequence
            compound_id = (
                conformer_df["Compound"].dropna().iloc[0]
                if not conformer_df["Compound"].isna().all()
                else filename
            )
            sequence = (
                conformer_df["Sequence"].dropna().iloc[0]
                if not conformer_df["Sequence"].isna().all()
                else None
            )

            # Best RMSD row
            best_rmsd_row = conformer_df.loc[conformer_df["RMSD"].idxmin()]

            # Non-clashing models
            non_clashing_df = conformer_df[~conformer_df["Has_Clashes"]]

            # Best non-clashing model info
            if len(non_clashing_df) > 0:
                best_non_clashing_row = non_clashing_df.loc[
                    non_clashing_df["RMSD"].idxmin()
                ]
                best_non_clashing_info = {
                    "Best_Non_Clashing_RMSD": best_non_clashing_row["RMSD"],
                    "Best_Non_Clashing_Model": best_non_clashing_row["Model"],
                    "Best_Non_Clashing_Frame": best_non_clashing_row["Frame"],
                }
            else:
                best_non_clashing_info = {
                    "Best_Non_Clashing_RMSD": float("inf"),
                    "Best_Non_Clashing_Model": None,
                    "Best_Non_Clashing_Frame": None,
                }

            # Create summary statistics
            summary_stats = {
                "Compound": compound_id,
                "Sequence": sequence,
                "Num_Conformers": len(conformer_df),
                # RMSD statistics
                "Min_RMSD": conformer_df["RMSD"].min(),
                "Max_RMSD": conformer_df["RMSD"].max(),
                "Mean_RMSD": conformer_df["RMSD"].mean(),
                "Median_RMSD": conformer_df["RMSD"].median(),
                "StdDev_RMSD": conformer_df["RMSD"].std(),
                # Atom matching statistics
                "Min_Matched_Atoms": conformer_df["Matched_Atoms"].min(),
                "Max_Matched_Atoms": conformer_df["Matched_Atoms"].max(),
                "Best_RMSD_Model": best_rmsd_row["Model"],
                "Best_RMSD_Frame": best_rmsd_row["Frame"],
                # Best non-clashing model info
                **best_non_clashing_info,
                # Enhanced clash statistics
                "Num_Clashing_Conformers": conformer_df["Has_Clashes"].sum(),
                "Min_Clashes": conformer_df["Num_Clashes"].min(),
                "Max_Clashes": conformer_df["Num_Clashes"].max(),
                "Mean_Clashes": conformer_df["Num_Clashes"].mean(),
                "Median_Clashes": conformer_df["Num_Clashes"].median(),
                "StdDev_Clashes": conformer_df["Num_Clashes"].std(),
            }

            # Save per-compound results
            results_csv_path = os.path.join(output_dir, f"{compound_id}_results.csv")
            conformer_df.to_csv(results_csv_path, index=False)

            summary_data.append(summary_stats)

            # Print summary
            print(f"\nSummary for {filename} (Compound {compound_id}):")
            print(f"Number of conformers processed: {summary_stats['Num_Conformers']}")
            print(
                f"RMSD range: {summary_stats['Min_RMSD']:.4f} - {summary_stats['Max_RMSD']:.4f}"
            )
            print(
                f"Best RMSD: {summary_stats['Min_RMSD']:.4f} (Model {summary_stats['Best_RMSD_Model']}, Frame {summary_stats['Best_RMSD_Frame']})"
            )
            if summary_stats["Best_Non_Clashing_Model"] is not None:
                print(
                    f"Best non-clashing RMSD: {summary_stats['Best_Non_Clashing_RMSD']:.4f} "
                    f"(Model {summary_stats['Best_Non_Clashing_Model']}, "
                    f"Frame {summary_stats['Best_Non_Clashing_Frame']})"
                )
            else:
                print("No non-clashing conformers found")
            print(
                f"Number of clashing conformers: {summary_stats['Num_Clashing_Conformers']}"
            )
            print(f"Clash statistics:")
            print(
                f"  Range: {summary_stats['Min_Clashes']} - {summary_stats['Max_Clashes']} clashes"
            )
            print(f"  Mean: {summary_stats['Mean_Clashes']:.2f}")
            print(f"  Median: {summary_stats['Median_Clashes']:.2f}")
            print(f"  StdDev: {summary_stats['StdDev_Clashes']:.2f}")

        # Write overall summary CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = os.path.join(output_dir, "summary_statistics.csv")
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"\nSummary statistics saved to {summary_csv_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Align molecular structures and detect clashes."
    )
    parser.add_argument("input_dir", help="Directory containing PDB files to align")
    parser.add_argument("output_dir", help="Directory for output files")
    parser.add_argument("reference_pdb", help="Reference PDB file")
    parser.add_argument("ligand_chain", help="Chain ID in the reference PDB")
    parser.add_argument(
        "--target_chains", nargs="+", help="Chain IDs to check for clashes against"
    )
    parser.add_argument(
        "--clash_cutoff",
        type=float,
        default=2.0,
        help="Distance cutoff for clash detection (Angstroms)",
    )
    parser.add_argument(
        "--write_pdbs", action="store_true", help="Write aligned PDB files"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of parallel processes to use (defaults to CPU cores - 1)",
    )

    args = parser.parse_args()

    process_directory(
        args.input_dir,
        args.output_dir,
        args.reference_pdb,
        args.ligand_chain,
        args.target_chains,
        args.clash_cutoff,
        args.write_pdbs,
        args.processes,
    )


if __name__ == "__main__":
    # Protect the entry point for Windows multiprocessing
    multiprocessing.freeze_support()
    main()

"""
Example usage:
python breadth_first_isomorphism_superimposer.py \
    /Users/Adam/Desktop/test \
    /Users/Adam/Desktop/test_data \
    /Users/Adam/Desktop/vhl1.pdb \
    D \
    --target_chains A B C \
    --clash_cutoff 2.0 \
    --write_pdbs

python breadth_first_isomorphism_superimposer.py \
    /Users/Adam/Desktop/hexamers_water \
    /Users/Adam/Desktop/hexamers_water_data \
    /Users/Adam/Desktop/vhl1.pdb \
    D \
    --target_chains A B C \
    --clash_cutoff 2.0 \
    --write_pdbs

python breadth_first_isomorphism_superimposer.py \
    /Users/Adam/Desktop/heptamers_water \
    /Users/Adam/Desktop/heptamers_water_data \
    /Users/Adam/Desktop/vhl1.pdb \
    D \
    --target_chains A B C \
    --clash_cutoff 2.0 \
    --write_pdbs

python breadth_first_isomorphism_superimposer.py \
    /Users/Adam/Desktop/hexamers_chc13 \
    /Users/Adam/Desktop/hexamers_chc13_data \
    /Users/Adam/Desktop/vhl1.pdb \
    D \
    --target_chains A B C \
    --clash_cutoff 2.0 \
    --write_pdbs

python breadth_first_isomorphism_superimposer.py \
    /Users/Adam/Desktop/heptamers_chcl3 \
    /Users/Adam/Desktop/heptamers_chcl3_data \
    /Users/Adam/Desktop/vhl1.pdb \
    D \
    --target_chains A B C \
    --clash_cutoff 2.0 \
    --write_pdbs
"""
