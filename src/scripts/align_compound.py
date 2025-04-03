# src/scripts/align_compound.py

"""
Script for aligning compound structures to a reference ligand.
Processes either a single compound directory or a directory containing multiple compounds.
"""

import argparse
import os
import logging
from pathlib import Path
import pandas as pd
from typing import List, Optional, Dict, Tuple, Any, Union
from tqdm import tqdm
import numpy as np
import MDAnalysis as mda
from scipy.spatial.distance import pdist, squareform
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO, Select
from Bio.PDB.PDBParser import PDBParser

from ..core.services.alignment_service import AlignmentService
from ..core.domain.models.molecular_graph import MolecularGraph
from ..core.domain.models.atom import Atom
from ..core.domain.models.bond import Bond, BondType
from ..infrastructure.repositories.structure_repository import StructureRepository
from ..core.domain.implementations.rdkit_mcs_superimposer import RDKitMCSSuperimposer


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO if verbose else logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "alignment.log"),
            logging.StreamHandler() if verbose else logging.NullHandler(),
        ],
    )


def infer_bonds(atoms: List[Atom], max_bond_length: float = 2.0) -> List[Bond]:
    """
    Infer bonds between atoms based on distance.

    Args:
        atoms: List of Atom objects
        max_bond_length: Maximum distance to consider for bonding (Angstroms)

    Returns:
        List of Bond objects
    """
    # Extract coordinates
    coords = np.array([atom.coordinates for atom in atoms])

    # Calculate pairwise distances
    distances = squareform(pdist(coords))

    # Find potential bonds (pairs of atoms within max_bond_length)
    bonds = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if distances[i, j] <= max_bond_length:
                # Create bond between atoms i and j
                bonds.append(
                    Bond(
                        atom1_id=atoms[i].atom_id,
                        atom2_id=atoms[j].atom_id,
                        bond_type=BondType.SINGLE,
                        bond_order=1.0,
                    )
                )

    return bonds


def parse_conect_records(pdb_path: str) -> Dict[int, List[int]]:
    """
    Parse CONECT records from a PDB file.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Dictionary mapping atom serial numbers to lists of connected atom serial numbers
    """
    conect_dict = {}
    conect_count = 0
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("CONECT"):
                conect_count += 1
                # CONECT records contain the atom serial number followed by the serial numbers of bonded atoms
                numbers = [
                    int(line[i : i + 5]) for i in range(6, len(line.rstrip()), 5)
                ]
                if numbers:
                    atom_serial = numbers[0]
                    connected_atoms = numbers[1:]
                    conect_dict[atom_serial] = connected_atoms
                    logging.debug(
                        f"Found CONECT record: {atom_serial} -> {connected_atoms}"
                    )

    if conect_count == 0:
        logging.error(f"No CONECT records found in {pdb_path}")
    else:
        logging.info(f"Found {conect_count} CONECT records in {pdb_path}")
        logging.info(f"Total atoms with connectivity: {len(conect_dict)}")
        logging.info(
            f"Total bonds: {sum(len(bonds) for bonds in conect_dict.values())}"
        )

    return conect_dict


def parse_conect_records_by_model(pdb_path: str) -> Dict[int, Dict[int, List[int]]]:
    """
    Parse CONECT records for all models from a PDB file.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Dictionary mapping model indices to dictionaries of CONECT records
        {model_idx: {atom_serial: [connected_atom_serials]}}

    Raises:
        ValueError: If no CONECT records found
    """
    model_conect_dict = {}
    current_model = -1
    current_conect = {}
    total_conect_count = 0

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                if current_model >= 0 and current_conect:
                    model_conect_dict[current_model] = current_conect
                current_model += 1
                current_conect = {}
            elif line.startswith("CONECT"):
                total_conect_count += 1
                numbers = [
                    int(line[i : i + 5]) for i in range(6, len(line.rstrip()), 5)
                ]
                if numbers:
                    atom_serial = numbers[0]
                    connected_atoms = numbers[1:]
                    current_conect[atom_serial] = connected_atoms

        # Don't forget to save the last model's CONECT records
        if current_model >= 0 and current_conect:
            model_conect_dict[current_model] = current_conect

    if total_conect_count == 0:
        raise ValueError(f"No CONECT records found in {pdb_path}")

    logging.info(
        f"Found {total_conect_count} total CONECT records across {len(model_conect_dict)} models"
    )
    return model_conect_dict


def create_molecular_graph_with_conect(
    universe: mda.Universe,
    frame_idx: int,
    chain_id: str,
    conect_dict: Optional[Dict[int, List[int]]] = None,
) -> MolecularGraph:
    """
    Create a MolecularGraph from an MDAnalysis Universe using CONECT records.
    Raises an error if CONECT records are not available.

    Args:
        universe: MDAnalysis Universe
        frame_idx: Frame index to use
        chain_id: Chain ID to filter by
        conect_dict: Dictionary of CONECT records mapping atom serial numbers to lists of connected atoms

    Returns:
        MolecularGraph object

    Raises:
        ValueError: If CONECT records are not available
    """
    # Set frame
    universe.trajectory[frame_idx]

    # Handle special case where MDAnalysis assigns 'SYSTEM' as chain ID
    if chain_id == "SYSTEM":
        # Get all atoms since they're all in the SYSTEM chain
        atoms = universe.atoms
    else:
        # Get atoms for specified chain
        atoms = universe.select_atoms(f"segid {chain_id} or chainID {chain_id}")

    if len(atoms) == 0:
        raise ValueError(f"No atoms found for chain {chain_id}")

    logging.info(f"Processing {len(atoms)} atoms for chain {chain_id}")

    # Create Atom objects
    atom_objects = []
    serial_to_id = {}  # Map from atom serial number to our atom ID

    # Log atom serials for debugging
    atom_serials = [atom.id for atom in atoms]
    logging.info(f"Atom serials in structure: {atom_serials}")
    if conect_dict:
        logging.info(f"Atom serials in CONECT records: {list(conect_dict.keys())}")
        overlap = set(atom_serials) & set(conect_dict.keys())
        logging.info(f"Atoms with CONECT records: {len(overlap)} out of {len(atoms)}")

    for i, atom in enumerate(atoms):
        # Try to get element from name
        if hasattr(atom, "element"):
            element = atom.element
        else:
            name = atom.name
            if len(name) > 0:
                element = "".join(c for c in name if c.isalpha())[:2].capitalize()
            else:
                element = "X"  # Unknown element

        atom_obj = Atom(
            atom_id=i,
            element=element,
            coordinates=(atom.position[0], atom.position[1], atom.position[2]),
            residue_name=atom.resname,
            residue_id=atom.resid,
            chain_id=chain_id,
            atom_name=atom.name,
            serial=atom.id,
        )
        atom_objects.append(atom_obj)
        serial_to_id[atom.id] = i

    # Create bonds from CONECT records
    if not conect_dict:
        raise ValueError("CONECT records are required but none were provided")

    bonds = []
    bonds_found = 0
    for atom in atoms:
        if atom.id in conect_dict:
            for connected_serial in conect_dict[atom.id]:
                # Only create bonds between atoms in our chain
                if connected_serial in serial_to_id:
                    bonds_found += 1
                    bonds.append(
                        Bond(
                            atom1_id=serial_to_id[atom.id],
                            atom2_id=serial_to_id[connected_serial],
                            bond_type=BondType.SINGLE,
                            bond_order=1.0,
                        )
                    )

    if not bonds:
        logging.error(f"Chain {chain_id} atoms: {atom_serials}")
        logging.error(f"CONECT records: {conect_dict}")
        raise ValueError(
            f"No valid bonds found in CONECT records for chain {chain_id}. Found {bonds_found} bonds but none were valid."
        )

    logging.info(
        f"Created molecular graph with {len(atom_objects)} atoms and {len(bonds)} bonds"
    )
    return MolecularGraph(atoms=atom_objects, bonds=bonds)


def process_compound_directory(
    compound_dir: Path,
    reference_path: Path,
    reference_chain: str = "D",  # Default to chain D for ligand
    protein_chains: List[str] = ["A", "B", "C"],  # Default to chains A,B,C for protein
    clash_cutoff: float = 2.0,
    verbose: bool = False,
) -> None:
    """
    Process a single compound directory.

    Args:
        compound_dir: Path to compound directory
        reference_path: Path to reference structure
        reference_chain: Chain ID of reference ligand (default: D)
        protein_chains: List of chain IDs for protein target (default: [A,B,C])
        clash_cutoff: Distance cutoff for clash detection
        verbose: Whether to show detailed output

    Raises:
        ValueError: If CONECT records are not available in the reference or compound PDB files
    """
    # Create output directory
    output_dir = compound_dir
    output_dir.mkdir(exist_ok=True)

    # Parse CONECT records from reference PDB
    logging.info("Parsing CONECT records from reference PDB")
    reference_conect = parse_conect_records(str(reference_path))
    if not reference_conect:
        raise ValueError("No CONECT records found in reference PDB")

    # Initialize services with RDKitMCSSuperimposer
    superimposer = RDKitMCSSuperimposer(timeout=60.0, match_valences=True)
    service = AlignmentService(superimposer=superimposer)

    try:
        # Load reference structure
        ref_universe = mda.Universe(str(reference_path))

        # Create molecular graph for reference ligand with connectivity
        reference = create_molecular_graph_with_conect(
            ref_universe,
            frame_idx=0,
            chain_id=reference_chain,
            conect_dict=reference_conect,
        )
        logging.info(f"Loaded reference ligand with {len(reference.atoms)} atoms")

        # Store protein chains for clash detection
        protein_structure = None
        if protein_chains:
            protein_atoms = ref_universe.select_atoms(
                " or ".join(
                    f"(segid {chain} or chainID {chain})" for chain in protein_chains
                )
            )
            if len(protein_atoms) > 0:
                logging.info(f"Found {len(protein_atoms)} atoms in protein target")
                protein_structure = protein_atoms
            else:
                logging.warning("No atoms found in specified protein chains")

        # Load compound structure
        compound_pdb = compound_dir / "md_Ref.pdb"
        compound_universe = mda.Universe(str(compound_pdb))
        logging.info(
            f"Loaded compound structure with {len(compound_universe.atoms)} atoms"
        )

        # Parse CONECT records for all models at once
        logging.info("Parsing CONECT records from compound PDB")
        model_conect_dict = parse_conect_records_by_model(str(compound_pdb))

        # Get available chains in compound structure
        available_chains = set()
        for atom in compound_universe.atoms:
            if hasattr(atom, "segid") and atom.segid:
                available_chains.add(atom.segid)
            if hasattr(atom, "chainID") and atom.chainID:
                available_chains.add(atom.chainID)

        if not available_chains:
            logging.warning(
                "No chain IDs found in compound structure, using default chain 'A'"
            )
            compound_chain = "A"
        else:
            compound_chain = list(available_chains)[0]
            logging.info(f"Using chain {compound_chain} from compound structure")

        # Process each model
        metrics = []
        for model_idx in tqdm(
            range(len(compound_universe.trajectory)), desc="Processing models"
        ):
            try:
                # Get CONECT records for current model
                if model_idx not in model_conect_dict:
                    raise ValueError(f"No CONECT records found for model {model_idx}")
                compound_conect = model_conect_dict[model_idx]

                # Create molecular graph for current model with CONECT records
                model = create_molecular_graph_with_conect(
                    compound_universe,
                    frame_idx=model_idx,
                    chain_id=compound_chain,
                    conect_dict=compound_conect,
                )

                # Align using RDKitMCSSuperimposer
                result = service.align_structures(reference, model)

                # Initialize metrics with alignment results
                metrics_entry = {
                    "model": model_idx + 1,
                    "rmsd": result.rmsd,
                    "matched_atoms": result.matched_atoms,
                    "has_clash": False,
                }

                # Only perform clash detection if we have both transformation matrix and protein structure
                if (
                    result.transformation_matrix is not None
                    and protein_structure is not None
                ):
                    # Check for clashes with protein
                    rotation, translation = result.transformation_matrix

                    # Transform compound coordinates
                    model_coords = model.get_coordinates()
                    transformed_coords = np.dot(model_coords, rotation.T) + translation

                    # Check distances to protein atoms
                    protein_coords = protein_structure.positions
                    has_clash = False

                    for coord in transformed_coords:
                        distances = np.linalg.norm(protein_coords - coord, axis=1)
                        min_model_dist = np.min(distances)
                        if min_model_dist < clash_cutoff:
                            has_clash = True

                    metrics_entry["has_clash"] = has_clash

                metrics.append(metrics_entry)

            except Exception as e:
                logging.error(f"Error processing model {model_idx + 1}: {str(e)}")
                metrics.append(
                    {
                        "model": model_idx + 1,
                        "rmsd": float("inf"),
                        "matched_atoms": 0,
                        "has_clash": False,
                    }
                )

        # Save metrics
        if metrics:
            metrics_df = pd.DataFrame(metrics)
            metrics_file = output_dir / "superimposition_metrics.csv"
            metrics_df.to_csv(metrics_file, index=False)
            logging.info(f"Saved metrics to {metrics_file}")
        else:
            logging.warning("No results to save")

    except Exception as e:
        logging.error(f"Error processing compound directory: {str(e)}")
        raise


def process_directory(
    input_dir: Path,
    reference_path: Path,
    reference_chain: str,
    target_chains: Optional[List[str]] = None,
    clash_cutoff: float = 2.0,
    verbose: bool = False,
) -> None:
    """
    Process all compound directories in a directory.

    Args:
        input_dir: Path to directory containing compound directories
        reference_path: Path to reference structure
        reference_chain: Chain ID in reference structure
        target_chains: List of chain IDs to check for clashes
        clash_cutoff: Distance cutoff for clash detection
        verbose: Whether to show detailed output
    """
    # Find all compound directories (those containing md_Ref.pdb)
    compound_dirs = []
    for path in input_dir.rglob("md_Ref.pdb"):
        compound_dirs.append(path.parent)

    if not compound_dirs:
        print(f"No compound directories found in {input_dir}")
        return

    print(f"Found {len(compound_dirs)} compound directories to process")

    # If only one compound directory, process it directly without compound progress bar
    if len(compound_dirs) == 1:
        try:
            process_compound_directory(
                compound_dir=compound_dirs[0],
                reference_path=reference_path,
                reference_chain=reference_chain,
                protein_chains=target_chains,
                clash_cutoff=clash_cutoff,
                verbose=verbose,
            )
        except Exception as e:
            logging.error(f"Error processing {compound_dirs[0]}: {str(e)}")
    else:
        # Process multiple compound directories with progress bar
        for compound_dir in tqdm(compound_dirs, desc="Processing compounds"):
            try:
                process_compound_directory(
                    compound_dir=compound_dir,
                    reference_path=reference_path,
                    reference_chain=reference_chain,
                    protein_chains=target_chains,
                    clash_cutoff=clash_cutoff,
                    verbose=verbose,
                )
            except Exception as e:
                logging.error(f"Error processing {compound_dir}: {str(e)}")
                continue


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Align compound structures to reference"
    )
    parser.add_argument("input_dir", help="Directory containing compound to align")
    parser.add_argument("reference", help="Reference structure for alignment")
    parser.add_argument("reference_chain", help="Chain ID in reference structure")
    parser.add_argument(
        "--target-chains", nargs="+", help="Chain IDs to check for clashes"
    )
    parser.add_argument(
        "--clash-cutoff",
        type=float,
        default=2.0,
        help="Distance cutoff for clash detection (Angstroms)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Convert paths
    input_path = Path(args.input_dir)
    reference_path = Path(args.reference)

    # Process directory
    process_directory(
        input_dir=input_path,
        reference_path=reference_path,
        reference_chain=args.reference_chain,
        target_chains=args.target_chains,
        clash_cutoff=args.clash_cutoff,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
