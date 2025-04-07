#!/usr/bin/env python3
# src/scripts/superimpose_basic.py
"""Basic script to superimpose and save PDB structures with minimal dependencies."""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
from collections import defaultdict


def parse_pdb_atoms(pdb_file):
    """Parse atoms from a PDB file.

    Returns:
        dict: Dictionary with chain IDs as keys and lists of atom dictionaries as values
    """
    atoms_by_chain = defaultdict(list)

    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Extract atom information
                atom_serial = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                chain_id = line[21:22]
                residue_num = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                atom = {
                    "serial": atom_serial,
                    "name": atom_name,
                    "residue_name": residue_name,
                    "chain_id": chain_id,
                    "residue_num": residue_num,
                    "coord": np.array([x, y, z]),
                    "element": line[76:78].strip() if len(line) >= 78 else atom_name[0],
                    "line": line,
                }

                atoms_by_chain[chain_id].append(atom)

    return atoms_by_chain


def parse_pdb_conect(pdb_file):
    """Parse CONECT records from a PDB file.

    Returns:
        dict: Dictionary with atom serials as keys and lists of connected atom serials as values
    """
    connections = defaultdict(list)

    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("CONECT"):
                parts = line.split()
                if len(parts) < 2:
                    continue

                # First number is the atom serial
                try:
                    atom_serial = int(parts[1])
                    # Remaining numbers are the connected atoms
                    for i in range(2, len(parts)):
                        try:
                            connected_serial = int(parts[i])
                            connections[atom_serial].append(connected_serial)
                        except ValueError:
                            continue
                except ValueError:
                    continue

    return connections


def match_atoms(ref_atoms, test_atoms, match_by="name"):
    """Match atoms between reference and test structures.

    Args:
        ref_atoms: List of reference atoms
        test_atoms: List of test atoms
        match_by: Criterion to match atoms ('name', 'position', or 'both')

    Returns:
        List of (ref_idx, test_idx) pairs of matched atoms
    """
    matched_pairs = []

    # Create dictionaries for quick lookup
    if match_by in ["name", "both"]:
        # Match by atom name
        ref_by_name = {}
        for i, atom in enumerate(ref_atoms):
            ref_by_name[atom.get("name", "")] = i

        for j, atom in enumerate(test_atoms):
            atom_name = atom.get("name", "")
            if atom_name in ref_by_name:
                matched_pairs.append((ref_by_name[atom_name], j))
                # Remove from reference to avoid duplicate matches
                ref_by_name.pop(atom_name)

    elif match_by == "position":
        # Simple position-based matching using distance cutoff
        cutoff = 2.0  # Angstroms
        for i, ref_atom in enumerate(ref_atoms):
            for j, test_atom in enumerate(test_atoms):
                ref_coord = ref_atom.get("coord", np.zeros(3))
                test_coord = test_atom.get("coord", np.zeros(3))
                dist = np.linalg.norm(ref_coord - test_coord)
                if dist < cutoff:
                    matched_pairs.append((i, j))
                    break  # Move to next reference atom

    return matched_pairs


def calculate_rmsd(coords1, coords2):
    """Calculate RMSD between two sets of coordinates.

    Args:
        coords1: First set of coordinates (N x 3)
        coords2: Second set of coordinates (N x 3)

    Returns:
        RMSD value
    """
    if len(coords1) != len(coords2):
        raise ValueError("Coordinate sets must have same length")

    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff * diff, axis=1)))


def superimpose_coordinates(ref_coords, mov_coords):
    """Superimpose one set of coordinates onto another using SVD.

    Args:
        ref_coords: Reference coordinates (N x 3)
        mov_coords: Moving coordinates (N x 3)

    Returns:
        Tuple of (rotation matrix, translation vector, RMSD)
    """
    if len(ref_coords) != len(mov_coords):
        raise ValueError("Coordinate sets must have same length")

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


def save_superimposed_structure(
    frame_file,
    ref_pdb_path,
    output_dir,
    metrics_file=None,
    match_by="name",
    verbose=False,
):
    """Save a superimposed structure as a PDB file using a simple approach.

    This function:
    1. Identifies matching atoms between reference ligand (chain D) and test molecule
    2. Calculates optimal superimposition transformation
    3. Creates a PDB with protein, reference ligand, and superimposed test molecule

    Args:
        frame_file: Path to the test frame file
        ref_pdb_path: Path to the reference PDB file
        output_dir: Directory to save the superimposed structure
        metrics_file: Optional path to a JSON file with transformation metrics
        match_by: Criterion to match atoms ('name', 'position', or 'both')
        verbose: Whether to print verbose output

    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get frame number from filename
    frame_name = Path(frame_file).stem

    # Create output file path
    output_file = Path(output_dir) / f"superimposed_{frame_name}.pdb"

    if verbose:
        print(f"Saving superimposed structure for {frame_name} to {output_file}")

    # Parse reference and test structures
    ref_atoms_by_chain = parse_pdb_atoms(ref_pdb_path)
    test_atoms_by_chain = parse_pdb_atoms(frame_file)

    # Parse CONECT records
    ref_connections = parse_pdb_conect(ref_pdb_path)
    test_connections = parse_pdb_conect(frame_file)

    if verbose:
        print(f"Found {len(ref_connections)} CONECT records in reference structure")
        print(f"Found {len(test_connections)} CONECT records in test structure")

    # Get reference ligand atoms (chain D)
    ref_ligand_atoms = ref_atoms_by_chain.get("D", [])
    if not ref_ligand_atoms:
        print(f"Warning: No atoms found in chain D of reference structure")
    elif verbose:
        print(f"Found {len(ref_ligand_atoms)} atoms in reference ligand (chain D)")

    # Get all test atoms (we'll match them against the reference ligand)
    test_atoms = []
    for chain in test_atoms_by_chain.values():
        test_atoms.extend(chain)

    if not test_atoms:
        print(f"Warning: No atoms found in test structure")
        return None
    elif verbose:
        print(f"Found {len(test_atoms)} atoms in test structure")

    # Match atoms between reference ligand and test molecule
    matched_pairs = match_atoms(ref_ligand_atoms, test_atoms, match_by)

    if len(matched_pairs) < 3:
        print(
            f"Warning: Not enough matched atoms ({len(matched_pairs)}) for reliable superimposition"
        )
        if verbose:
            print(
                "Ref ligand atom names: "
                + ", ".join(a.get("name", "") for a in ref_ligand_atoms[:10])
            )
            print(
                "Test atom names: "
                + ", ".join(a.get("name", "") for a in test_atoms[:10])
            )
        return None

    if verbose:
        print(
            f"Found {len(matched_pairs)} matching atoms between reference ligand and test molecule"
        )
        for i, (ref_idx, test_idx) in enumerate(matched_pairs[:5]):
            print(
                f"  Match {i+1}: {ref_ligand_atoms[ref_idx].get('name', '')} - {test_atoms[test_idx].get('name', '')}"
            )

    # Extract matched coordinates for superimposition
    ref_coords = np.array(
        [ref_ligand_atoms[i].get("coord", np.zeros(3)) for i, _ in matched_pairs]
    )
    test_coords = np.array(
        [test_atoms[j].get("coord", np.zeros(3)) for _, j in matched_pairs]
    )

    # Try to get transformation from metrics file if available
    rotation = None
    translation = None
    metrics = None
    using_metrics_transformation = False

    if metrics_file and os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                all_metrics = json.load(f)

                # Extract frame number
                frame_num = frame_name.split("_")[1]
                if frame_num in all_metrics:
                    metrics = all_metrics[frame_num]
                    if "rotation" in metrics and "translation" in metrics:
                        # Instead of using metrics directly, verify if it gives good alignment
                        met_rotation = np.array(metrics["rotation"])
                        met_translation = np.array(metrics["translation"])

                        # Test if this transformation aligns well
                        aligned_test = (
                            np.dot(test_coords, met_rotation) + met_translation
                        )
                        test_rmsd = calculate_rmsd(ref_coords, aligned_test)

                        if test_rmsd < 2.0:  # Use if reasonable alignment
                            rotation = met_rotation
                            translation = met_translation
                            using_metrics_transformation = True
                            if verbose:
                                print(
                                    f"Using transformation from metrics file (RMSD: {test_rmsd:.4f} Å)"
                                )
                        else:
                            if verbose:
                                print(
                                    f"Metrics transformation gives poor alignment (RMSD: {test_rmsd:.4f} Å), calculating new one"
                                )
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            if verbose:
                print(f"Error reading metrics file: {str(e)}")

    # If no transformation from metrics or poor alignment, calculate it ourselves
    if rotation is None or translation is None:
        if verbose:
            print("Calculating transformation matrix from matched atomic coordinates")

        # Calculate optimal superimposition using the matched atoms
        rotation, translation, rmsd = superimpose_coordinates(ref_coords, test_coords)

        if verbose:
            print(f"Calculated RMSD: {rmsd:.4f} Å")
            print(f"Rotation matrix:\n{rotation}")
            print(f"Translation vector: {translation}")

    # Create the superimposed PDB file
    with open(output_file, "w") as out_f:
        # Add header
        out_f.write(
            f"REMARK   4 Superimposed structure generated by superimpose_basic.py\n"
        )
        out_f.write(f"REMARK   4 Reference: {ref_pdb_path}\n")
        out_f.write(f"REMARK   4 Test: {frame_file}\n")
        out_f.write(f"REMARK   4 Matched atoms: {len(matched_pairs)}\n")

        if using_metrics_transformation and metrics:
            out_f.write(f"REMARK   4 Using transformation from metrics file\n")
            out_f.write(
                f"REMARK   4 RMSD from metrics: {metrics.get('rmsd', 0.0):.4f}\n"
            )
            out_f.write(
                f"REMARK   4 Has clashes: {metrics.get('has_clashes', False)}\n"
            )
            out_f.write(f"REMARK   4 Num clashes: {metrics.get('num_clashes', 0)}\n")
        else:
            # Calculate and write RMSD based on the current transformation
            aligned_test = np.dot(test_coords, rotation) + translation
            final_rmsd = calculate_rmsd(ref_coords, aligned_test)
            out_f.write(f"REMARK   4 Calculated RMSD: {final_rmsd:.4f}\n")

        # Copy reference PDB content (chains A, B, C, D) directly
        with open(ref_pdb_path, "r") as ref_f:
            for line in ref_f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    chain_id = line[21:22]
                    if chain_id in ["A", "B", "C", "D"]:
                        out_f.write(line)
                elif line.startswith("TER"):
                    out_f.write(line)
                elif line.startswith("CONECT"):
                    # Keep CONECT records from reference
                    out_f.write(line)

        # Add transformed test molecule as chain E
        atom_id_map = {}  # Maps original atom IDs to new ones
        atom_num = 10001  # Start with a high number to avoid conflicts

        for atom in test_atoms:
            old_id = atom.get("serial", 0)
            atom_id_map[old_id] = atom_num

            # Get coordinates and apply transformation
            coord = atom.get("coord", np.zeros(3))
            # Apply rotation first, then translation (correct order)
            new_coord = np.dot(coord, rotation) + translation

            # Create a new PDB line
            atom_name = atom.get("name", "")
            residue_name = atom.get("residue_name", "UNK")
            residue_num = atom.get("residue_num", 1)
            element = atom.get("element", "")

            line = f"ATOM  {atom_num:5d} {atom_name:4s} {residue_name:3s} E{residue_num:4d}    "
            line += f"{new_coord[0]:8.3f}{new_coord[1]:8.3f}{new_coord[2]:8.3f}"
            line += f"  1.00  0.00          {element:>2s}  \n"

            out_f.write(line)
            atom_num += 1

        # Add TER record
        out_f.write("TER\n")

        # Add CONECT records for the test molecule using the new atom IDs
        for old_id, connections in test_connections.items():
            if old_id in atom_id_map:
                new_id = atom_id_map[old_id]
                conect_line = f"CONECT{new_id:5d}"

                for connected_id in connections:
                    if connected_id in atom_id_map:
                        conect_line += f"{atom_id_map[connected_id]:5d}"

                conect_line += "\n"
                out_f.write(conect_line)

        # Add special CONECT record for cyclizing the peptide if needed
        # This is a simplified approach - in practice you'd need to identify
        # the correct atoms to connect
        if verbose:
            print("Checking for cyclic peptide structure...")

        # Find first and last residue numbers
        residue_nums = [atom.get("residue_num", 0) for atom in test_atoms]
        if residue_nums:
            min_res = min(residue_nums)
            max_res = max(residue_nums)

            # Find N atom in first residue and C atom in last residue
            first_N = None
            last_C = None

            for atom in test_atoms:
                res_num = atom.get("residue_num", 0)
                atom_name = atom.get("name", "")
                atom_id = atom.get("serial", 0)

                if res_num == min_res and atom_name == "N":
                    first_N = atom_id
                if res_num == max_res and atom_name == "C":
                    last_C = atom_id

            # Add cyclic connection if both atoms found and not already connected
            if first_N is not None and last_C is not None:
                if first_N in atom_id_map and last_C in atom_id_map:
                    # Check if connection already exists
                    if last_C not in test_connections.get(
                        first_N, []
                    ) and first_N not in test_connections.get(last_C, []):
                        if verbose:
                            print(
                                f"Adding cyclic peptide connection between residues {min_res} and {max_res}"
                            )

                        # Add CONECT record
                        new_first_N = atom_id_map[first_N]
                        new_last_C = atom_id_map[last_C]
                        out_f.write(f"CONECT{new_first_N:5d}{new_last_C:5d}\n")
                        out_f.write(f"CONECT{new_last_C:5d}{new_first_N:5d}\n")

        # End the file
        out_f.write("END\n")

    if verbose:
        print(f"Successfully saved superimposed structure to {output_file}")

        # Verify the superimposition quality
        test_idx_list = [j for _, j in matched_pairs]
        final_transformed_coords = []
        for j in test_idx_list:
            orig_coord = test_atoms[j].get("coord", np.zeros(3))
            trans_coord = np.dot(orig_coord, rotation) + translation
            final_transformed_coords.append(trans_coord)

        final_transformed_coords = np.array(final_transformed_coords)
        final_ref_coords = np.array(
            [ref_ligand_atoms[i].get("coord", np.zeros(3)) for i, _ in matched_pairs]
        )

        verification_rmsd = calculate_rmsd(final_ref_coords, final_transformed_coords)
        print(f"Verification RMSD for matched atoms: {verification_rmsd:.4f} Å")

        if verification_rmsd > 0.5:
            print(
                "WARNING: Superimposition may not be optimal. Please check the output structure."
            )

    return output_file


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create superimposed PDB files without requiring BioPython."
    )

    parser.add_argument("frame_file", help="Path to the frame PDB file")
    parser.add_argument("ref_pdb_path", help="Path to the reference PDB file")
    parser.add_argument(
        "--metrics-file",
        help="Path to JSON file with transformation metrics (optional)",
    )
    parser.add_argument(
        "--output-dir",
        default="superimposed",
        help="Directory to save superimposed structures",
    )
    parser.add_argument(
        "--match-by",
        choices=["name", "position", "both"],
        default="name",
        help="Criterion to match atoms (name, position, or both)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--add-cyclic-bond",
        action="store_true",
        help="Add bond between first and last residue to make cyclic peptide",
    )

    args = parser.parse_args()

    # Ensure paths are valid
    if not os.path.exists(args.frame_file):
        print(f"Error: Frame file not found: {args.frame_file}")
        return 1

    if not os.path.exists(args.ref_pdb_path):
        print(f"Error: Reference PDB file not found: {args.ref_pdb_path}")
        return 1

    # Save superimposed structure
    try:
        save_superimposed_structure(
            args.frame_file,
            args.ref_pdb_path,
            args.output_dir,
            args.metrics_file,
            args.match_by,
            args.verbose,
        )
    except Exception as e:
        print(f"Error saving superimposed structure: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
