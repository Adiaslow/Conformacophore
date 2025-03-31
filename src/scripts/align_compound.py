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
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
import numpy as np
import MDAnalysis as mda
from scipy.spatial.distance import pdist, squareform

from ..core.services.alignment_service import AlignmentService
from ..core.domain.models.molecular_graph import MolecularGraph
from ..core.domain.models.atom import Atom
from ..core.domain.models.bond import Bond, BondType
from ..infrastructure.repositories.structure_repository import StructureRepository


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


def create_molecular_graph(
    universe: mda.Universe, frame_idx: int = 0, chain_id: Optional[str] = None
) -> MolecularGraph:
    """
    Create a MolecularGraph from an MDAnalysis Universe.

    Args:
        universe: MDAnalysis Universe
        frame_idx: Frame index to use
        chain_id: Optional chain ID to filter atoms

    Returns:
        MolecularGraph instance
    """
    # Set frame
    universe.trajectory[frame_idx]

    # Filter atoms by chain if specified
    atoms = universe.atoms
    if chain_id:
        try:
            # Try chainID first
            atoms = atoms.select_atoms(f"chainID {chain_id}")
            if len(atoms) == 0:
                # If no atoms found, try segid
                atoms = universe.atoms.select_atoms(f"segid {chain_id}")
            if len(atoms) == 0:
                logging.warning(f"No atoms found for chain {chain_id}, using all atoms")
                atoms = universe.atoms
        except Exception as e:
            logging.warning(
                f"Error selecting chain {chain_id}: {str(e)}, using all atoms"
            )
            atoms = universe.atoms

    # Create Atom objects
    atom_objects = []
    atom_id_map = {}  # Map original atom indices to new indices

    for i, atom in enumerate(atoms, 1):
        # Try to get element from various sources
        if hasattr(atom, "element"):
            element = atom.element
        else:
            # Try to get from name
            name = atom.name.strip()
            if len(name) > 0:
                element = "".join(c for c in name if c.isalpha())[:2].capitalize()
            else:
                element = "X"  # Unknown element

        atom_objects.append(
            Atom(
                atom_id=i,
                element=element,
                coordinates=(atom.position[0], atom.position[1], atom.position[2]),
                residue_name=atom.resname,
                residue_id=atom.resid,
                chain_id=chain_id
                or (atom.chainID if hasattr(atom, "chainID") else (atom.segid or "A")),
                atom_name=atom.name,
                serial=atom.id,
            )
        )
        atom_id_map[atom.id] = i

    # Try to get bonds from topology, if not available, infer them
    try:
        # Create Bond objects from topology
        bond_objects = []
        for bond in atoms.bonds:
            # Only create bonds between atoms we've included
            if bond.atoms[0].id in atom_id_map and bond.atoms[1].id in atom_id_map:
                bond_objects.append(
                    Bond(
                        atom1_id=atom_id_map[bond.atoms[0].id],
                        atom2_id=atom_id_map[bond.atoms[1].id],
                        bond_type=BondType.SINGLE,  # Default to single bonds
                        bond_order=1.0,
                    )
                )
    except (AttributeError, ValueError):
        # If bonds are not available in topology, infer them
        logging.info(
            "Bond information not available in topology, inferring bonds based on distance"
        )
        bond_objects = infer_bonds(atom_objects)

    if len(atom_objects) == 0:
        raise ValueError(f"No atoms found in structure")

    return MolecularGraph(atoms=atom_objects, bonds=bond_objects)


def process_compound_directory(
    compound_dir: Path,
    reference_path: Path,
    reference_chain: str,
    target_chains: Optional[List[str]] = None,
    clash_cutoff: float = 2.0,
    verbose: bool = False,
) -> None:
    """
    Process a single compound directory.

    Args:
        compound_dir: Path to compound directory
        reference_path: Path to reference structure
        reference_chain: Chain ID in reference structure
        target_chains: List of chain IDs to check for clashes
        clash_cutoff: Distance cutoff for clash detection
        verbose: Whether to show detailed output
    """
    # Create output directory
    output_dir = compound_dir / f"{compound_dir.name}_superimpositions"
    output_dir.mkdir(exist_ok=True)

    # Initialize services
    from ..core.domain.implementations.breadth_first_isomorphism_superimposer import (
        BreadthFirstIsomorphismSuperimposer,
    )

    service = AlignmentService(superimposer=BreadthFirstIsomorphismSuperimposer())

    try:
        # Load reference structure
        ref_universe = mda.Universe(str(reference_path))
        reference = create_molecular_graph(ref_universe, chain_id=reference_chain)
        logging.info(f"Loaded reference structure with {len(reference.atoms)} atoms")

        # Load compound structure
        compound_pdb = compound_dir / "md_Ref.pdb"
        compound_universe = mda.Universe(str(compound_pdb))
        logging.info(
            f"Loaded compound structure with {len(compound_universe.atoms)} atoms"
        )
        logging.info(f"Found {len(compound_universe.trajectory)} models")

        # Process each model
        results = []
        for model_idx in tqdm(
            range(len(compound_universe.trajectory)),
            desc="Processing models",
            disable=not verbose,
        ):
            try:
                # Create MolecularGraph for this model
                model = create_molecular_graph(compound_universe, frame_idx=model_idx)
                logging.info(
                    f"Processing model {model_idx + 1} with {len(model.atoms)} atoms"
                )

                # Align structures
                result = service.align_structures(
                    reference=reference,
                    target=model,
                    clash_cutoff=clash_cutoff,
                )
                logging.info(
                    f"Aligned model {model_idx + 1} with RMSD {result.rmsd:.3f} Å, "
                    f"matched {result.matched_atoms} atoms"
                )

                # Check for clashes with target chains if specified
                total_clashes = 0
                min_distance = float("inf")
                has_clashes = False

                if target_chains:
                    clash_results = []
                    for chain in target_chains:
                        try:
                            target_chain = create_molecular_graph(
                                ref_universe, chain_id=chain
                            )
                            clash_result = service._detect_clashes(
                                reference=target_chain,
                                target=model,
                                alignment=result,
                                cutoff=clash_cutoff,
                            )
                            clash_results.append(clash_result)
                            logging.info(
                                f"Checked clashes with chain {chain}: "
                                f"{clash_result.num_clashes} clashes found"
                            )
                        except Exception as e:
                            logging.error(
                                f"Error checking clashes with chain {chain}: {str(e)}"
                            )
                            continue

                    # Combine clash results
                    total_clashes = sum(r.num_clashes for r in clash_results)
                    min_distance = min(r.min_distance for r in clash_results)
                    has_clashes = any(r.has_clashes for r in clash_results)

                # Save aligned structure if alignment was successful
                if result.transformation_matrix is not None:
                    output_pdb = output_dir / f"model_{model_idx + 1}_aligned.pdb"
                    save_aligned_structure(
                        model,
                        result.transformation_matrix,
                        str(output_pdb),
                    )
                    logging.info(f"Saved aligned structure to {output_pdb}")

                # Collect metrics
                metrics = {
                    "Model": model_idx + 1,
                    "RMSD": result.rmsd,
                    "Matched_Atoms": result.matched_atoms,
                    "Has_Clashes": has_clashes,
                    "Num_Clashes": total_clashes,
                    "Min_Distance": min_distance,
                    "Isomorphic_Match": (
                        result.isomorphic_match
                        if hasattr(result, "isomorphic_match")
                        else False
                    ),
                    "Atom_Mapping": (
                        str(result.matched_pairs) if result.matched_pairs else ""
                    ),
                }
                results.append(metrics)

            except Exception as e:
                logging.error(f"Error processing model {model_idx + 1}: {str(e)}")
                continue

        # Save metrics to CSV
        if results:
            metrics_df = pd.DataFrame(results)
            metrics_df.to_csv(output_dir / "alignment_metrics.csv", index=False)
            logging.info(
                f"Saved metrics for {len(results)} models to "
                f"{output_dir / 'alignment_metrics.csv'}"
            )
        else:
            logging.warning("No results to save")

    except Exception as e:
        logging.error(f"Error processing compound directory {compound_dir}: {str(e)}")
        raise


def save_aligned_structure(
    structure: MolecularGraph,
    transformation: tuple[np.ndarray, np.ndarray],
    output_path: str,
) -> None:
    """
    Save aligned structure to PDB file.

    Args:
        structure: Structure to save
        transformation: (rotation_matrix, translation_vector)
        output_path: Path to save PDB file
    """
    rotation, translation = transformation

    # Get coordinates and apply transformation
    coords = structure.get_coordinates()
    aligned_coords = np.dot(coords, rotation.T) + translation

    # Update coordinates in structure
    for i, atom in enumerate(structure.atoms):
        atom["x"] = aligned_coords[i, 0]
        atom["y"] = aligned_coords[i, 1]
        atom["z"] = aligned_coords[i, 2]

    # Write PDB file
    with open(output_path, "w") as f:
        f.write("TITLE     Aligned structure\n")
        f.write("MODEL     1\n")

        # Write ATOM records
        for i, atom in enumerate(structure.atoms, 1):
            f.write(
                f"ATOM  {i:5d}  {atom['atom_name']:<4s}{atom['residue_name']:3s} "
                f"{atom['chain_id'] or 'A':1s}{atom['residue_num']:4d}    "
                f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
                f"  1.00  0.00           {atom['element']:>2s}\n"
            )

        # Write connectivity if available
        for i, atom in enumerate(structure.atoms, 1):
            connections = structure.get_connections(i - 1)
            if connections:
                f.write(
                    f"CONECT{i:5d}" + "".join(f"{j+1:5d}" for j in connections) + "\n"
                )

        f.write("ENDMDL\n")
        f.write("END\n")


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

    # Process each compound directory
    for compound_dir in tqdm(compound_dirs, desc="Processing compounds"):
        try:
            process_compound_directory(
                compound_dir=compound_dir,
                reference_path=reference_path,
                reference_chain=reference_chain,
                target_chains=target_chains,
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
