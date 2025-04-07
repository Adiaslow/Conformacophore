#!/usr/bin/env python3
# src/scripts/simple_match.py
"""Simple script to superimpose PDB structures using metrics file data."""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


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


def parse_pdb_conect(pdb_file):
    """Parse CONECT records from a PDB file."""
    connections = defaultdict(list)
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("CONECT"):
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    atom_serial = int(parts[1])
                    for i in range(2, len(parts)):
                        try:
                            connected_serial = int(parts[i])
                            connections[atom_serial].append(connected_serial)
                        except ValueError:
                            continue
                except ValueError:
                    continue
    return connections


def get_vdw_radius(element):
    """Get van der Waals radius for an element in Angstroms.

    Based on values used in molecular visualization programs like ChimeraX.
    """
    # Standard VDW radii for common elements (in Angstroms)
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


def check_clashes(test_coords, protein_atoms, overlap_cutoff=0.60, verbose=False):
    """Check for steric clashes between a molecule and protein using VDW overlap (ChimeraX method).

    Args:
        test_coords: List of test molecule coordinates with elements
        protein_atoms: List of protein atoms with coordinates and elements
        overlap_cutoff: VDW overlap threshold to count as a clash (in Angstroms)
        verbose: Whether to print detailed diagnostic information

    Returns:
        Tuple of (has_clashes, num_clashing_atoms, detailed_clashes)
    """
    # Check for empty arrays to avoid numpy errors
    if len(test_coords) == 0 or len(protein_atoms) == 0:
        print("WARNING: Cannot check clashes - empty coordinate array provided")
        return False, 0, []

    clashes = []
    atom_clashes = set()  # Keep track of which test atoms clash
    all_overlaps = []  # Store all overlap values for statistics

    if verbose:
        print(
            f"Checking for clashes between {len(test_coords)} test atoms and {len(protein_atoms)} protein atoms"
        )
        print(f"VDW overlap threshold: {overlap_cutoff} Å")
        print("Sample atom info:")
        if test_coords:
            test_sample = test_coords[0]
            print(
                f"  Test atom 0: {test_sample.get('name')} {test_sample.get('element')} VDW: {get_vdw_radius(test_sample.get('element', 'C'))}"
            )
        if protein_atoms:
            protein_sample = protein_atoms[0]
            print(
                f"  Protein atom 0: {protein_sample.get('name')} {protein_sample.get('element')} VDW: {get_vdw_radius(protein_sample.get('element', 'C'))}"
            )

    # For each atom in the test molecule
    for i, test_atom in enumerate(test_coords):
        coord = test_atom.get("coord")
        element = test_atom.get("element", "C")
        test_vdw = get_vdw_radius(element)

        if verbose and i < 5:  # Print details for first few atoms
            print(f"Test atom {i}: {test_atom.get('name')} {element} at {coord}")

        # Check against all protein atoms
        for j, protein_atom in enumerate(protein_atoms):
            p_coord = protein_atom.get("coord")
            p_element = protein_atom.get("element", "C")
            protein_vdw = get_vdw_radius(p_element)

            # Calculate center-to-center distance
            distance = np.sqrt(np.sum((coord - p_coord) ** 2))

            # Calculate VDW overlap (sum of radii minus distance)
            vdw_overlap = test_vdw + protein_vdw - distance

            # Keep track of all overlaps for statistics
            all_overlaps.append(vdw_overlap)

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

                if verbose and len(clashes) <= 3:  # Print details for first few clashes
                    print(
                        f"CLASH FOUND: {test_atom.get('name')} - {protein_atom.get('name')} overlap: {vdw_overlap:.2f}Å"
                    )
                    print(
                        f"  Test: {test_atom.get('residue_name')}{test_atom.get('residue_num')} at {coord}"
                    )
                    print(
                        f"  Protein: {protein_atom.get('residue_name')}{protein_atom.get('residue_num')} at {p_coord}"
                    )
                    print(
                        f"  Distance: {distance:.2f}Å, VDW sum: {test_vdw + protein_vdw:.2f}Å"
                    )

    # Report overlap statistics
    if verbose and all_overlaps:
        all_overlaps = np.array(all_overlaps)
        max_overlap = np.max(all_overlaps)
        print(f"\nOverlap statistics:")
        print(f"  Maximum overlap: {max_overlap:.2f}Å")
        print(f"  Near-clash overlaps (> 0.4Å): {np.sum(all_overlaps > 0.4)}")
        print(f"  Overlap histogram:")
        bins = [
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.8,
            1.0,
            1.5,
            2.0,
        ]
        hist, edges = np.histogram(all_overlaps, bins=bins)
        for i in range(len(hist)):
            print(f"    {edges[i]:.1f} to {edges[i+1]:.1f}Å: {hist[i]}")

    return len(atom_clashes) > 0, len(atom_clashes), clashes


def save_superimposed_structure(
    frame_file,
    ref_pdb_path,
    output_dir,
    metrics_file=None,
    verbose=False,
    clash_cutoff=0.60,
    debug_clashes=False,
):
    """Save a superimposed structure using transformation from metrics file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get frame number from filename
    frame_name = Path(frame_file).stem
    frame_num = frame_name.split("_")[1]

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

    # Get protein atoms (chains A, B, C)
    protein_atoms = []
    for chain_id in ["A", "B", "C"]:
        if chain_id in ref_atoms_by_chain:
            protein_atoms.extend(ref_atoms_by_chain[chain_id])

    # Get all test atoms
    test_atoms = []
    for chain in test_atoms_by_chain.values():
        test_atoms.extend(chain)

    # Get transformation from metrics file
    rotation = None
    translation = None

    if metrics_file and os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                all_metrics = json.load(f)

                if frame_num in all_metrics:
                    metrics = all_metrics[frame_num]
                    if "rotation" in metrics and "translation" in metrics:
                        rotation = np.array(metrics["rotation"])
                        translation = np.array(metrics["translation"])
                        if verbose:
                            print(
                                f"Found transformation data in metrics file for frame {frame_num}"
                            )
                            print(f"RMSD from metrics: {metrics.get('rmsd', 0.0):.4f}")
        except Exception as e:
            if verbose:
                print(f"Error reading metrics file: {str(e)}")

    if rotation is None or translation is None:
        print(f"Error: No transformation data found for frame {frame_num}")
        return None

    # Apply transformation to test atom coordinates
    for atom in test_atoms:
        coord = atom.get("coord", np.zeros(3))
        new_coord = np.dot(coord, rotation) + translation
        atom["coord"] = new_coord

    if verbose:
        print(f"Test molecule: {len(test_atoms)} atoms")
        print(f"Protein: {len(protein_atoms)} atoms")
        print(f"Reference ligand: {len(ref_ligand_atoms)} atoms")

        # Check format of protein and test atoms
        if protein_atoms:
            first_protein = protein_atoms[0]
            print(
                f"First protein atom: {first_protein.get('name')}, element: {first_protein.get('element')}"
            )
            print(f"  Coordinate type: {type(first_protein.get('coord'))}")
        if test_atoms:
            first_test = test_atoms[0]
            print(
                f"First test atom: {first_test.get('name')}, element: {first_test.get('element')}"
            )
            print(f"  Coordinate type: {type(first_test.get('coord'))}")

        if len(protein_atoms) == 0:
            print("WARNING: No protein atoms found for clash detection!")

    # Check for clashes using VDW overlap
    has_clashes, num_clashes, clash_details = check_clashes(
        test_atoms, protein_atoms, overlap_cutoff=clash_cutoff, verbose=debug_clashes
    )

    if verbose:
        print(f"VDW overlap clash detection with cutoff {clash_cutoff} Å:")
        print(f"  Has clashes: {has_clashes}")
        print(f"  Number of clashing atoms: {num_clashes}")
        if num_clashes > 0 and len(clash_details) > 0:
            print(f"  Showing up to 5 worst clashes:")
            # Sort clashes by overlap amount (worst first)
            sorted_clashes = sorted(
                clash_details, key=lambda x: x["overlap"], reverse=True
            )
            for i, clash in enumerate(sorted_clashes[:5]):
                test_atom = clash["test_atom"]
                protein_atom = clash["protein_atom"]
                print(f"    {i+1}. Overlap: {clash['overlap']:.2f}Å between:")
                print(
                    f"       Test atom: {test_atom.get('name')} {test_atom.get('residue_name')} {test_atom.get('residue_num')}"
                )
                print(
                    f"       Protein atom: {protein_atom.get('name')} {protein_atom.get('residue_name')} {protein_atom.get('residue_num')}"
                )
                print(f"       Distance: {clash['distance']:.2f}Å")
                print(
                    f"       VDW radii: {get_vdw_radius(test_atom.get('element', 'C')):.2f}Å + {get_vdw_radius(protein_atom.get('element', 'C')):.2f}Å = {get_vdw_radius(test_atom.get('element', 'C')) + get_vdw_radius(protein_atom.get('element', 'C')):.2f}Å"
                )

    # Create the superimposed PDB file
    with open(output_file, "w") as out_f:
        # Add header
        out_f.write(f"REMARK   4 Superimposed structure generated by simple_match.py\n")
        out_f.write(f"REMARK   4 Reference: {ref_pdb_path}\n")
        out_f.write(f"REMARK   4 Test: {frame_file}\n")
        out_f.write(
            f"REMARK   4 VDW overlap clash detection cutoff: {clash_cutoff} Å\n"
        )
        out_f.write(f"REMARK   4 Has clashes: {has_clashes}\n")
        out_f.write(f"REMARK   4 Number of clashing atoms: {num_clashes}\n")

        # Add detailed clash information
        if num_clashes > 0 and len(clash_details) > 0:
            out_f.write(f"REMARK   4 \n")
            out_f.write(f"REMARK   4 Clash details (up to 5 worst):\n")
            sorted_clashes = sorted(
                clash_details, key=lambda x: x["overlap"], reverse=True
            )
            for i, clash in enumerate(sorted_clashes[:5]):
                test_atom = clash["test_atom"]
                protein_atom = clash["protein_atom"]
                out_f.write(
                    f"REMARK   4   {i+1}. Overlap: {clash['overlap']:.2f}Å between:\n"
                )
                out_f.write(
                    f"REMARK   4      Test: {test_atom.get('name')} {test_atom.get('residue_name')}{test_atom.get('residue_num')}\n"
                )
                out_f.write(
                    f"REMARK   4      Protein: {protein_atom.get('name')} {protein_atom.get('residue_name')}{protein_atom.get('residue_num')}\n"
                )
                out_f.write(f"REMARK   4      Distance: {clash['distance']:.2f}Å\n")

        # Copy reference PDB content (chains A, B, C, D)
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

            # Get the transformed coordinate
            new_coord = atom.get("coord")

            # Create new PDB line
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

        # Add CONECT records for the test molecule
        for old_id, connections in test_connections.items():
            if old_id in atom_id_map:
                new_id = atom_id_map[old_id]
                conect_line = f"CONECT{new_id:5d}"

                for connected_id in connections:
                    if connected_id in atom_id_map:
                        conect_line += f"{atom_id_map[connected_id]:5d}"

                conect_line += "\n"
                out_f.write(conect_line)

        # End file with END
        out_f.write("END\n")

    if verbose:
        print(f"Successfully saved superimposed structure to {output_file}")

    # Return a dictionary with results
    return {
        "output_file": str(output_file),
        "has_clashes": has_clashes,
        "num_clashes": num_clashes,
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create superimposed PDB files using metrics file data."
    )

    parser.add_argument("frame_file", help="Path to the frame PDB file")
    parser.add_argument("ref_pdb_path", help="Path to the reference PDB file")
    parser.add_argument(
        "--metrics-file",
        required=True,
        help="Path to JSON file with transformation metrics",
    )
    parser.add_argument(
        "--output-dir",
        default="superimposed",
        help="Directory to save superimposed structures",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--clash-cutoff",
        type=float,
        default=0.60,
        help="VDW overlap threshold for clash detection in Angstroms",
    )
    parser.add_argument(
        "--debug-clashes",
        action="store_true",
        help="Print detailed diagnostics about clash detection",
    )

    args = parser.parse_args()

    # Ensure paths are valid
    if not os.path.exists(args.frame_file):
        print(f"Error: Frame file not found: {args.frame_file}")
        return 1

    if not os.path.exists(args.ref_pdb_path):
        print(f"Error: Reference PDB file not found: {args.ref_pdb_path}")
        return 1

    if not os.path.exists(args.metrics_file):
        print(f"Error: Metrics file not found: {args.metrics_file}")
        return 1

    # Save superimposed structure
    try:
        save_superimposed_structure(
            args.frame_file,
            args.ref_pdb_path,
            args.output_dir,
            args.metrics_file,
            args.verbose,
            args.clash_cutoff,
            args.debug_clashes,
        )
    except Exception as e:
        print(f"Error saving superimposed structure: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
