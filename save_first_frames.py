#!/usr/bin/env python3
"""Script to save the first 5 frames of a trajectory using the simplified simple_match.py script."""

import os
import sys
import subprocess
import json
from pathlib import Path
import glob
import numpy as np  # Add numpy for direct clash detection

# Configuration - adjust these paths as needed
BASE_DIR = "/Volumes/LokeyLabShared/Adam/chads_library_conf/Hex/CHCl3/x_177"
REF_PDB = "/Volumes/LokeyLabShared/Adam/chads_library_conf/ref/vhl1.pdb"
OUTPUT_DIR = os.path.join(BASE_DIR, "superimposed")
METRICS_FILE = os.path.join(BASE_DIR, "pdb_frames", "superimposition_metrics.json")
VDW_OVERLAP_CUTOFF = 0.60  # VDW overlap threshold, matching ChimeraX

# Debug settings
DEBUG_MODE = True
DUMP_COORDS = True  # Whether to write coordinates to text files for analysis


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


def parse_pdb_atoms(pdb_file):
    """Parse atoms from a PDB file."""
    atoms_by_chain = {}
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21:22]
                if chain_id not in atoms_by_chain:
                    atoms_by_chain[chain_id] = []

                atom_serial = int(line[6:11].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                residue_num = int(line[22:26].strip())
                element = line[76:78].strip() if len(line) >= 78 else atom_name[0]

                atoms_by_chain[chain_id].append(
                    {
                        "serial": atom_serial,
                        "coord": np.array([x, y, z]),
                        "element": element,
                        "name": atom_name,
                        "residue_name": residue_name,
                        "residue_num": residue_num,
                        "line": line,
                    }
                )

    return atoms_by_chain


def check_vdw_clashes(test_atoms, protein_atoms, overlap_cutoff=0.60):
    """Check for clashes between molecules using VDW overlap (ChimeraX method)."""
    clashes = []
    atom_clashes = set()  # Keep track of which test atoms clash

    # For each atom in the test molecule
    for i, test_atom in enumerate(test_atoms):
        coord = test_atom["coord"]
        element = test_atom["element"]
        test_vdw = get_vdw_radius(element)

        # Check against all protein atoms
        for j, protein_atom in enumerate(protein_atoms):
            p_coord = protein_atom["coord"]
            p_element = protein_atom["element"]
            protein_vdw = get_vdw_radius(p_element)

            # Calculate center-to-center distance
            distance = np.sqrt(np.sum((coord - p_coord) ** 2))

            # Calculate VDW overlap (sum of radii minus distance)
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

    return len(atom_clashes) > 0, len(atom_clashes), clashes


def dump_coordinates_to_file(atoms, filename):
    """Write atom coordinates to a text file for external analysis."""
    with open(filename, "w") as f:
        f.write("serial,name,residue,x,y,z,element\n")
        for atom in atoms:
            coord = atom["coord"]
            f.write(
                f"{atom['serial']},{atom['name']},{atom['residue_name']}{atom['residue_num']},"
                f"{coord[0]:.3f},{coord[1]:.3f},{coord[2]:.3f},{atom['element']}\n"
            )


def main():
    """Main function to save the first 5 frames."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find PDB frame files
    pdb_frames_dir = os.path.join(BASE_DIR, "pdb_frames")
    frame_files = sorted(glob.glob(os.path.join(pdb_frames_dir, "frame_*.pdb")))

    if not frame_files:
        print(f"No frame files found in {pdb_frames_dir}")
        return 1

    print(f"Found {len(frame_files)} frame files")

    # Take the first 5 frames (or all if less than 5)
    frames_to_process = frame_files[: min(5, len(frame_files))]
    print(f"Processing {len(frames_to_process)} frames")

    # First, check if metrics file exists
    if not os.path.exists(METRICS_FILE):
        print(f"Error: Metrics file not found: {METRICS_FILE}")
        print("This script requires the metrics file with transformation data.")
        return 1

    # Load protein coordinates from reference PDB
    print("\nAnalyzing reference PDB structure...")
    ref_atoms = parse_pdb_atoms(REF_PDB)
    protein_atoms = []
    for chain_id in ["A", "B", "C"]:
        if chain_id in ref_atoms:
            protein_atoms.extend(ref_atoms[chain_id])

    print(f"Reference PDB: {REF_PDB}")
    print(f"Found chains: {list(ref_atoms.keys())}")
    for chain_id in ref_atoms:
        print(f"  Chain {chain_id}: {len(ref_atoms[chain_id])} atoms")
    print(f"Total protein atoms (chains A,B,C): {len(protein_atoms)}")

    if len(protein_atoms) == 0:
        print("WARNING: No protein atoms found in reference PDB!")

    # If debugging, dump protein coordinates
    if DUMP_COORDS:
        dump_file = os.path.join(OUTPUT_DIR, "protein_coords.csv")
        dump_coordinates_to_file(protein_atoms, dump_file)
        print(f"Dumped protein coordinates to {dump_file}")

    # Load metrics for transformations
    with open(METRICS_FILE, "r") as f:
        all_metrics = json.load(f)
        print(f"Loaded metrics for {len(all_metrics)} frames")

    # Process each frame
    for i, frame_file in enumerate(frames_to_process):
        print(f"\n{'='*80}")
        print(f"Processing frame {i+1}/{len(frames_to_process)}: {frame_file}")
        frame_name = Path(frame_file).stem
        frame_num = frame_name.split("_")[1]

        if frame_num not in all_metrics:
            print(f"No metrics found for frame {frame_num}, skipping")
            continue

        # Get transformation matrices
        metrics = all_metrics[frame_num]
        if "rotation" not in metrics or "translation" not in metrics:
            print("Missing transformation data, skipping")
            continue

        rotation = np.array(metrics["rotation"])
        translation = np.array(metrics["translation"])
        print(f"Transformation loaded - RMSD: {metrics.get('rmsd', 0.0):.4f}")

        # Load test compound and transform coordinates
        test_atoms_by_chain = parse_pdb_atoms(frame_file)
        test_atoms = []
        for chain in test_atoms_by_chain.values():
            test_atoms.extend(chain)

        print(f"Test compound: {len(test_atoms)} atoms")

        # Transform test coordinates
        for atom in test_atoms:
            coord = atom["coord"]
            new_coord = np.dot(coord, rotation) + translation
            atom["coord"] = new_coord

        # If debugging, dump transformed test coordinates
        if DUMP_COORDS:
            dump_file = os.path.join(OUTPUT_DIR, f"{frame_name}_transformed_coords.csv")
            dump_coordinates_to_file(test_atoms, dump_file)
            print(f"Dumped transformed test coordinates to {dump_file}")

        # Check for clashes using ChimeraX-style VDW overlap
        has_clashes, num_clashes, clash_details = check_vdw_clashes(
            test_atoms, protein_atoms, overlap_cutoff=VDW_OVERLAP_CUTOFF
        )

        print(
            f"\nVDW CLASH DETECTION (ChimeraX style, ≥ {VDW_OVERLAP_CUTOFF}Å overlap):"
        )
        print(f"  Result: {num_clashes} clashing atoms")

        # Show details for worst clashes
        if num_clashes > 0 and len(clash_details) > 0:
            print("  Top 5 worst clashes:")
            sorted_clashes = sorted(
                clash_details, key=lambda x: x["overlap"], reverse=True
            )
            for j, clash in enumerate(sorted_clashes[:5]):
                test_atom = clash["test_atom"]
                protein_atom = clash["protein_atom"]
                print(f"    {j+1}. VDW overlap: {clash['overlap']:.2f}Å between:")
                print(
                    f"       Test: {test_atom['name']} {test_atom['residue_name']}{test_atom['residue_num']}"
                )
                print(
                    f"       Protein: {protein_atom['name']} {protein_atom['residue_name']}{protein_atom['residue_num']}"
                )
                print(f"       Distance: {clash['distance']:.2f}Å")
                print(
                    f"       VDW radii: {get_vdw_radius(test_atom['element']):.2f}Å + "
                    f"{get_vdw_radius(protein_atom['element']):.2f}Å = "
                    f"{get_vdw_radius(test_atom['element']) + get_vdw_radius(protein_atom['element']):.2f}Å"
                )

        if DEBUG_MODE:
            print("\nRunning simple_match.py with debug clashes enabled:")

        try:
            cmd = [
                "python3",
                "src/scripts/simple_match.py",
                frame_file,
                REF_PDB,
                "--metrics-file",
                METRICS_FILE,
                "--output-dir",
                OUTPUT_DIR,
                "--verbose",
            ]

            if DEBUG_MODE:
                cmd.extend(
                    ["--debug-clashes", "--clash-cutoff", str(VDW_OVERLAP_CUTOFF)]
                )
            else:
                cmd.extend(["--clash-cutoff", str(VDW_OVERLAP_CUTOFF)])

            # Run the command
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Print output
            print(result.stdout)

            # Check for errors
            if result.returncode != 0:
                print(f"Error running command: {result.stderr}")
                continue

            print(f"Successfully processed frame {i+1}")

        except Exception as e:
            print(f"Error processing frame {frame_file}: {str(e)}")

    # List created files
    created_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "superimposed_*.pdb")))
    print(f"\nCreated {len(created_files)} superimposed structure files:")
    for file in created_files:
        print(f"  {file}")

    print("\nTo visualize these PDB files:")
    print("1. Open them in PyMOL, ChimeraX, or similar")
    print("2. Chain A, B, C: Protein target")
    print("3. Chain D: Reference ligand")
    print("4. Chain E: Superimposed test molecule")
    print(
        "\nYou should see the test molecule (E) properly superimposed onto the reference ligand (D)"
    )

    print("\nClash detection summary:")
    print(
        f"VDW overlap method (ChimeraX style): using cutoff ≥ {VDW_OVERLAP_CUTOFF}Å overlap"
    )
    print(f"Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")
    print(f"Coordinate dumps: {'ON' if DUMP_COORDS else 'OFF'}")

    if DUMP_COORDS:
        print("\nCoordinate files were dumped for external analysis in:")
        print(f"  {OUTPUT_DIR}/protein_coords.csv")
        print(f"  {OUTPUT_DIR}/<frame>_transformed_coords.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
