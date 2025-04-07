#!/usr/bin/env python3
# match_and_superimpose.py
"""
Script to match atoms between reference ligand and test molecules, then superimpose.

This script:
1. Loads a reference PDB file containing protein (chains A,B,C) and ligand (chain D)
2. Loads a test molecule (frame PDB file)
3. Finds matching atoms between reference ligand and test molecule
4. Calculates optimal transformation using Kabsch algorithm
5. Applies transformation to the entire test molecule
6. Saves the superimposed structure with the protein, ref ligand, and transformed test
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import glob
import multiprocessing
import logging
import time


def parse_pdb_atoms(pdb_file):
    """Parse atoms from a PDB file."""
    atoms_by_chain = defaultdict(list)
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_serial = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                chain_id = line[21:22]
                residue_num = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip() if len(line) >= 78 else atom_name[0]

                atom = {
                    "serial": atom_serial,
                    "name": atom_name,
                    "residue_name": residue_name,
                    "chain_id": chain_id,
                    "residue_num": residue_num,
                    "coord": np.array([x, y, z]),
                    "element": element,
                    "line": line,
                }
                atoms_by_chain[chain_id].append(atom)
    return atoms_by_chain


def get_vdw_radius(element):
    """Get van der Waals radius for an element in Angstroms."""
    # Standard VDW radii (in Angstroms)
    radii = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "S": 1.80,
        "P": 1.80,
        "F": 1.47,
        "CL": 1.75,
        "BR": 1.85,
        "I": 1.98,
        "FE": 1.80,
        "ZN": 1.39,
        "MG": 1.73,
        "CA": 1.74,
    }

    # Standardize element name to uppercase
    element = element.upper() if element else ""

    # Return radius or default to carbon if unknown
    return radii.get(element, 1.70)


def find_matching_atoms(ref_atoms, test_atoms, match_by="element", verbose=False):
    """Find matching atoms between reference and test molecules.

    Args:
        ref_atoms: List of reference atoms (from chain D)
        test_atoms: List of test atoms
        match_by: Method for matching atoms ("element" or "name")
        verbose: Whether to print detailed information

    Returns:
        List of matching atom index pairs [(ref_idx, test_idx), ...]
    """
    matches = []

    # Create dictionaries for quick lookup
    if match_by == "element":
        # Group atoms by element type
        ref_by_element = defaultdict(list)
        for i, atom in enumerate(ref_atoms):
            element = atom.get("element", "").upper()
            if element:
                ref_by_element[element].append((i, atom))

        # Match atoms by element
        used_test_indices = set()

        # For each reference atom, find closest test atom of same element
        for ref_element, ref_atoms_list in ref_by_element.items():
            # Find all test atoms with matching element
            matching_test_atoms = []
            for j, test_atom in enumerate(test_atoms):
                if j in used_test_indices:
                    continue  # Skip already matched test atoms

                test_element = test_atom.get("element", "").upper()
                if test_element == ref_element:
                    matching_test_atoms.append((j, test_atom))

            # For each reference atom of this element, find closest test atom
            for ref_idx, ref_atom in ref_atoms_list:
                ref_coord = ref_atom.get("coord")

                best_dist = float("inf")
                best_test_idx = None

                for test_idx, test_atom in matching_test_atoms:
                    if test_idx in used_test_indices:
                        continue

                    test_coord = test_atom.get("coord")
                    dist = np.linalg.norm(ref_coord - test_coord)

                    if dist < best_dist:
                        best_dist = dist
                        best_test_idx = test_idx

                if best_test_idx is not None:
                    matches.append((ref_idx, best_test_idx))
                    used_test_indices.add(best_test_idx)

    else:  # match by name
        # Create dictionaries for quick lookup
        ref_by_name = {}
        for i, atom in enumerate(ref_atoms):
            name = atom.get("name", "").strip()
            if name:
                ref_by_name[name] = (i, atom)

        # Match atoms by name
        for j, test_atom in enumerate(test_atoms):
            name = test_atom.get("name", "").strip()
            if name in ref_by_name:
                ref_idx, _ = ref_by_name[name]
                matches.append((ref_idx, j))

    if verbose:
        print(f"Found {len(matches)} matching atoms using {match_by} matching")

    return matches


def kabsch_align(ref_coords, test_coords):
    """Calculate rotation and translation using Kabsch algorithm.

    Args:
        ref_coords: Reference coordinates (Nx3 array)
        test_coords: Test coordinates to align (Nx3 array)

    Returns:
        Tuple of (rotation_matrix, translation_vector, rmsd)
    """
    # First, center both structures
    ref_center = np.mean(ref_coords, axis=0)
    test_center = np.mean(test_coords, axis=0)

    ref_centered = ref_coords - ref_center
    test_centered = test_coords - test_center

    # Calculate covariance matrix
    covariance = np.dot(test_centered.T, ref_centered)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(covariance)

    # Ensure proper rotation (no reflection)
    d = np.sign(np.linalg.det(np.dot(Vt.T, U.T)))
    correction = np.diag([1, 1, d])

    # Calculate rotation matrix
    rotation = np.dot(U, np.dot(correction, Vt))

    # Calculate translation
    translation = ref_center - np.dot(test_center, rotation)

    # Calculate RMSD
    aligned_test = np.dot(test_centered, rotation) + ref_center
    rmsd = np.sqrt(np.mean(np.sum((ref_coords - aligned_test) ** 2, axis=1)))

    return rotation, translation, rmsd


def apply_transformation(coord, rotation, translation):
    """Apply transformation to coordinates."""
    return np.dot(coord, rotation) + translation


def check_clashes(test_atoms, protein_atoms, overlap_cutoff=0.60, verbose=False):
    """Check for steric clashes between test molecule and protein.

    Args:
        test_atoms: List of test molecule atoms
        protein_atoms: List of protein atoms
        overlap_cutoff: VDW overlap threshold for clash detection
        verbose: Whether to print detailed clash information

    Returns:
        Tuple of (has_clashes, num_clashing_atoms, clash_details)
    """
    clashes = []
    atom_clashes = set()  # Track which test atoms clash

    # For each atom in test molecule
    for i, test_atom in enumerate(test_atoms):
        coord = test_atom.get("coord")
        element = test_atom.get("element", "C")
        test_vdw = get_vdw_radius(element)

        # Check against all protein atoms
        for j, protein_atom in enumerate(protein_atoms):
            p_coord = protein_atom.get("coord")
            p_element = protein_atom.get("element", "C")
            protein_vdw = get_vdw_radius(p_element)

            # Calculate center-to-center distance
            distance = np.linalg.norm(coord - p_coord)

            # Calculate VDW overlap
            vdw_overlap = test_vdw + protein_vdw - distance

            # If overlap exceeds cutoff, it's a clash
            if vdw_overlap >= overlap_cutoff:
                atom_clashes.add(i)
                clashes.append(
                    {
                        "test_atom": test_atom,
                        "protein_atom": protein_atom,
                        "distance": distance,
                        "overlap": vdw_overlap,
                    }
                )

    # Report clashes
    if verbose and clashes:
        print(
            f"\nFound {len(atom_clashes)} clashing atoms with {len(clashes)} total clashes"
        )
        print(f"VDW overlap cutoff: {overlap_cutoff}Å")

        # Show details for worst clashes
        sorted_clashes = sorted(clashes, key=lambda x: x["overlap"], reverse=True)
        for i, clash in enumerate(sorted_clashes[:5]):
            test_atom = clash["test_atom"]
            protein_atom = clash["protein_atom"]
            print(f"  {i+1}. Overlap: {clash['overlap']:.2f}Å between:")
            print(
                f"     Test: {test_atom.get('name')} {test_atom.get('residue_name')}{test_atom.get('residue_num')}"
            )
            print(
                f"     Protein: {protein_atom.get('name')} {protein_atom.get('residue_name')}{protein_atom.get('residue_num')}"
            )

    return len(atom_clashes) > 0, len(atom_clashes), clashes


def save_superimposed_structure(
    frame_file,
    ref_pdb_path,
    output_file,
    rotation=None,
    translation=None,
    metrics_file=None,
    match_by="element",
    clash_cutoff=0.60,
    verbose=False,
):
    """Save a superimposed structure using atom matching and Kabsch algorithm.

    Args:
        frame_file: Path to test molecule PDB file
        ref_pdb_path: Path to reference PDB file
        output_file: Path to save superimposed structure
        rotation: Optional preset rotation matrix
        translation: Optional preset translation vector
        metrics_file: Optional path to metrics file with transformations
        match_by: Method for matching atoms ("element" or "name")
        clash_cutoff: VDW overlap threshold for clash detection
        verbose: Whether to print detailed information

    Returns:
        Dictionary with results
    """
    if verbose:
        print(
            f"\nSuperimposing {os.path.basename(frame_file)} onto {os.path.basename(ref_pdb_path)}"
        )

    # Parse reference structure
    ref_atoms_by_chain = parse_pdb_atoms(ref_pdb_path)

    # Get reference ligand (chain D)
    ref_ligand = ref_atoms_by_chain.get("D", [])
    if not ref_ligand:
        print(f"ERROR: No ligand found in chain D of reference structure")
        return None

    # Get protein atoms (chains A, B, C)
    protein_atoms = []
    for chain_id in ["A", "B", "C"]:
        if chain_id in ref_atoms_by_chain:
            protein_atoms.extend(ref_atoms_by_chain[chain_id])

    # Parse test structure
    test_atoms_by_chain = parse_pdb_atoms(frame_file)

    # Combine all chains from test structure
    test_atoms = []
    for chain in test_atoms_by_chain.values():
        test_atoms.extend(chain)

    if verbose:
        print(f"Reference ligand: {len(ref_ligand)} atoms")
        print(f"Test molecule: {len(test_atoms)} atoms")
        print(f"Protein: {len(protein_atoms)} atoms")

    # Try to use provided rotation/translation
    if rotation is not None and translation is not None:
        if verbose:
            print(f"Using provided transformation matrices")

    # Try to get transformation from metrics file
    elif metrics_file and os.path.exists(metrics_file):
        try:
            frame_name = Path(frame_file).stem
            frame_num = frame_name.split("_")[1]

            with open(metrics_file, "r") as f:
                all_metrics = json.load(f)

                if frame_num in all_metrics:
                    metrics = all_metrics[frame_num]
                    if "rotation" in metrics and "translation" in metrics:
                        rotation = np.array(metrics["rotation"])
                        translation = np.array(metrics["translation"])
                        if verbose:
                            print(f"Using transformation from metrics file")
                            print(f"RMSD from metrics: {metrics.get('rmsd', 0.0):.4f}")
        except Exception as e:
            print(f"Error loading from metrics file: {str(e)}")

    # If no transformation provided, calculate it
    if rotation is None or translation is None:
        if verbose:
            print(f"Calculating transformation based on atom matching")

        # Find matching atoms
        atom_matches = find_matching_atoms(
            ref_ligand, test_atoms, match_by=match_by, verbose=verbose
        )

        if len(atom_matches) < 3:
            print(
                f"ERROR: Not enough matching atoms found ({len(atom_matches)}). At least 3 required."
            )
            return None

        # Extract coordinates for matching atoms
        ref_coords = np.array(
            [ref_ligand[ref_idx]["coord"] for ref_idx, _ in atom_matches]
        )
        test_coords = np.array(
            [test_atoms[test_idx]["coord"] for _, test_idx in atom_matches]
        )

        # Calculate transformation
        rotation, translation, rmsd = kabsch_align(ref_coords, test_coords)

        if verbose:
            print(f"Calculated RMSD: {rmsd:.4f} over {len(atom_matches)} matched atoms")

    # Apply transformation to all test atoms
    for atom in test_atoms:
        coord = atom["coord"]
        atom["coord"] = apply_transformation(coord, rotation, translation)

    # Check for clashes with protein
    has_clashes, num_clashes, clash_details = check_clashes(
        test_atoms, protein_atoms, overlap_cutoff=clash_cutoff, verbose=verbose
    )

    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Create the superimposed PDB file
    with open(output_file, "w") as out_f:
        # Add header
        out_f.write(f"REMARK   4 Superimposed structure\n")
        out_f.write(f"REMARK   4 Reference: {ref_pdb_path}\n")
        out_f.write(f"REMARK   4 Test: {frame_file}\n")
        # Always write clash information, regardless of rotation/translation
        out_f.write(f"REMARK   4 Has clashes: {has_clashes}\n")
        out_f.write(f"REMARK   4 Number of clashing atoms: {num_clashes}\n")
        out_f.write(f"REMARK   4 Total clash count: {len(clash_details)}\n")

        # Write the protein chains and reference ligand
        for chain_id in ["A", "B", "C", "D"]:
            if chain_id in ref_atoms_by_chain:
                for atom in ref_atoms_by_chain[chain_id]:
                    out_f.write(atom["line"])
                out_f.write("TER\n")

        # Write the transformed test atoms
        for i, atom in enumerate(test_atoms):
            coord = atom["coord"]
            serial = 10001 + i  # Start with high serial numbers
            name = atom["name"]
            residue_name = atom["residue_name"]
            residue_num = atom["residue_num"]
            element = atom["element"]

            # Format the atom line with new chain (E)
            line = (
                f"ATOM  {serial:5d} {name:4s} {residue_name:3s} E{residue_num:4d}    "
            )
            line += f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
            line += f"  1.00  0.00          {element:>2s}  \n"
            out_f.write(line)

        # End TER record
        out_f.write("TER\n")
        out_f.write("END\n")

    if verbose:
        print(f"Successfully saved superimposed structure to {output_file}")

    # Return results
    return {
        "output_file": output_file,
        "rotation": rotation.tolist() if rotation is not None else None,
        "translation": translation.tolist() if translation is not None else None,
        "has_clashes": has_clashes,
        "num_clashes": num_clashes,
        "total_clashes": len(clash_details),
        "matched_atoms": len(atom_matches) if "atom_matches" in locals() else 0,
        "rmsd": rmsd if "rmsd" in locals() else 0.0,
    }


# New function for processing a trajectory using the original command syntax
def process_trajectory_directory(
    base_dir,
    reference_pdb,
    num_processes=8,
    force=False,
    save_structures=False,
    output_dir=None,
    verbose=False,
):
    """Process a trajectory directory using the original command syntax."""
    print(f"\nProcessing trajectory in {base_dir}")

    # Find pdb_frames directories
    base_path = Path(base_dir)
    frame_dirs = list(base_path.rglob("pdb_frames"))

    if not frame_dirs:
        print(f"No pdb_frames directories found in {base_dir}")
        return 1

    print(f"Found {len(frame_dirs)} pdb_frames directories")

    # Process each directory
    for frame_dir in frame_dirs:
        print(f"\nProcessing directory: {frame_dir}")

        # Get all frame files
        frame_files = sorted(frame_dir.glob("frame_*.pdb"))
        if not frame_files:
            print(f"No frame files found in {frame_dir}")
            continue

        print(f"Found {len(frame_files)} frame files")

        # Create metrics file path
        metrics_file = frame_dir / "superimposition_metrics.json"

        # Check if already processed
        if not force and metrics_file.exists():
            print(f"Directory already processed. Use --force to reprocess.")
            continue

        # Create output directory for superimposed structures
        if save_structures:
            if output_dir:
                out_dir = Path(output_dir)
            else:
                out_dir = frame_dir.parent / "superimposed"

            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Will save superimposed structures to {out_dir}")
        else:
            out_dir = None

        # Process frames in parallel
        print(f"Processing {len(frame_files)} frames with {num_processes} processes")

        # Set limit for saving structures (default: save first 5)
        save_limit = 5

        metrics = {}

        # Create arguments for each frame
        for i, frame_file in enumerate(frame_files):
            frame_name = frame_file.stem
            frame_num = frame_name.split("_")[1]

            save_this_one = save_structures and (i < save_limit)

            # Skip frames we're not processing with verbose details
            if verbose and i >= 5:
                continue

            print(f"Processing frame {i+1}/{len(frame_files)}: {frame_name}")

            # Process this frame
            if save_this_one and out_dir:
                output_file = out_dir / f"superimposed_{frame_name}.pdb"
                result = save_superimposed_structure(
                    str(frame_file),
                    reference_pdb,
                    str(output_file),
                    None,  # rotation
                    None,  # translation
                    None,  # metrics_file - we're calculating fresh
                    "element",  # match_by
                    0.6,  # clash_cutoff
                    verbose,  # Only show verbose for first few
                )
            else:
                # Just calculate metrics, don't save structure
                ref_atoms_by_chain = parse_pdb_atoms(reference_pdb)
                test_atoms_by_chain = parse_pdb_atoms(str(frame_file))

                # Get reference ligand (chain D)
                ref_ligand = ref_atoms_by_chain.get("D", [])

                # Get protein atoms (chains A, B, C)
                protein_atoms = []
                for chain_id in ["A", "B", "C"]:
                    if chain_id in ref_atoms_by_chain:
                        protein_atoms.extend(ref_atoms_by_chain[chain_id])

                # Combine all chains from test structure
                test_atoms = []
                for chain in test_atoms_by_chain.values():
                    test_atoms.extend(chain)

                # Find matching atoms
                atom_matches = find_matching_atoms(
                    ref_ligand, test_atoms, match_by="element", verbose=verbose
                )

                if len(atom_matches) < 3:
                    print(f"Not enough matching atoms for {frame_name}")
                    continue

                # Extract coordinates for matching atoms
                ref_coords = np.array(
                    [ref_ligand[ref_idx]["coord"] for ref_idx, _ in atom_matches]
                )
                test_coords = np.array(
                    [test_atoms[test_idx]["coord"] for _, test_idx in atom_matches]
                )

                # Calculate transformation
                rotation, translation, rmsd = kabsch_align(ref_coords, test_coords)

                # Apply transformation to all test atoms
                for atom in test_atoms:
                    coord = atom["coord"]
                    atom["coord"] = apply_transformation(coord, rotation, translation)

                # Check for clashes with protein
                has_clashes, num_clashes, clash_details = check_clashes(
                    test_atoms, protein_atoms, overlap_cutoff=0.6, verbose=verbose
                )

                result = {
                    "rmsd": float(rmsd),
                    "rotation": rotation.tolist(),
                    "translation": translation.tolist(),
                    "has_clashes": has_clashes,
                    "num_clashes": num_clashes,
                    "total_clashes": len(clash_details),
                    "matched_atoms": len(atom_matches),
                }

            if result:
                # Store metrics
                metrics[frame_num] = {
                    "rmsd": result.get("rmsd", 0.0),
                    "rotation": result.get("rotation"),
                    "translation": result.get("translation"),
                    "has_clashes": result.get("has_clashes", False),
                    "num_clashes": result.get("num_clashes", 0),
                    "total_clashes": result.get("total_clashes", 0),
                    "matched_atoms": result.get("matched_atoms", 0),
                }
            else:
                print(f"Failed to process frame {frame_name}")

        # Save metrics
        try:
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"Processed {len(metrics)} frames successfully")
            if save_structures:
                print(f"Saved up to {save_limit} superimposed structures")
            print(f"Metrics file saved to {metrics_file}")
        except Exception as e:
            print(f"Error saving metrics file: {str(e)}")
            return 1

    return 0


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Superimpose ligands onto a reference structure"
    )

    # For compatibility with superimpose_trajectories.py
    parser.add_argument(
        "base_dir",
        help="Base directory containing pdb_frames directories",
        nargs="?",  # Make positional argument optional
        default="",
    )
    parser.add_argument(
        "reference_pdb",
        help="Reference PDB file with protein (chains A,B,C) and ligand (chain D)",
        nargs="?",  # Make positional argument optional
        default="",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of parallel processes (default: 8)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already processed files",
    )
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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Args specific to individual structure processing
    parser.add_argument(
        "--test-pdb",
        help="Test PDB file (instead of using base_dir)",
    )
    parser.add_argument(
        "--reference-pdb",
        help="Reference PDB file for single structure processing",
    )
    parser.add_argument(
        "--output-file",
        help="Output PDB file for a single structure",
    )
    parser.add_argument(
        "--metrics-file",
        help="Path to metrics file with transformations",
    )
    parser.add_argument(
        "--match-by",
        choices=["element", "name"],
        default="element",
        help="Method for matching atoms ('element' or 'name')",
    )
    parser.add_argument(
        "--clash-cutoff",
        type=float,
        default=0.60,
        help="VDW overlap threshold for clash detection",
    )

    args = parser.parse_args()

    # Process either a single structure or a trajectory
    if args.test_pdb and args.reference_pdb:
        # Process a single structure using named parameters
        output_file = args.output_file
        if not output_file:
            # Default output file path
            output_file = f"superimposed_{Path(args.test_pdb).stem}.pdb"

        result = save_superimposed_structure(
            frame_file=args.test_pdb,
            ref_pdb_path=args.reference_pdb,
            output_file=output_file,
            metrics_file=args.metrics_file,
            match_by=args.match_by,
            clash_cutoff=args.clash_cutoff,
            verbose=args.verbose,
        )

        if result:
            print(f"Successfully processed structure")
            return 0
        else:
            print(f"Failed to process structure")
            return 1
    elif args.base_dir and args.reference_pdb:
        # Process a trajectory directory using positional parameters
        return process_trajectory_directory(
            args.base_dir,
            args.reference_pdb,
            num_processes=args.num_processes,
            force=args.force,
            save_structures=args.save_structures,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    else:
        print("Error: You must specify either:")
        print("  1. --test-pdb and --reference-pdb for processing a single structure")
        print(
            "  2. base_dir and reference_pdb positional arguments for processing a trajectory"
        )
        return 1


if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f} seconds")
    sys.exit(exit_code)
