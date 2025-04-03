#!/usr/bin/env python3
# simple_extract.py

"""
Simple script to extract PDB frames from MD trajectory with proper bonds.

This script extracts frames from an XTC trajectory and saves them as PDB files
with proper bond information from the topology file.

Usage:
    python simple_extract.py /path/to/gro /path/to/xtc /path/to/top output_dir [options]

Options:
    --start N       Start from frame N (default: 0)
    --stop N        Stop at frame N (default: last frame)
    --step N        Extract every Nth frame (default: 1)
    --force         Overwrite existing output directory
"""

import MDAnalysis as mda
import numpy as np
import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import re


def extract_pdbs(
    gro_file, xtc_file, top_file, output_dir, start=0, stop=None, step=1, force=False
):
    """
    Extract PDB files from trajectory with proper bond information.

    Args:
        gro_file (str): Path to reference structure file (.gro)
        xtc_file (str): Path to trajectory file (.xtc)
        top_file (str): Path to topology file (.top)
        output_dir (str): Directory to save PDB files
        start (int): First frame to extract (0-indexed)
        stop (int): Last frame to extract (None for all frames)
        step (int): Extract every nth frame
        force (bool): Whether to overwrite existing output directory
    """
    # Handle output directory
    output_path = Path(output_dir)
    if output_path.exists():
        if force:
            print(f"Removing existing directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(
                f"Error: Output directory {output_dir} already exists. Use --force to overwrite."
            )
            return

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading trajectory: {xtc_file}")
    print(f"Using topology: {gro_file}")

    # Load trajectory without using lock files (important for network drives)
    u = mda.Universe(gro_file, xtc_file, use_lock=False)

    # Load a TPR file if available (better for preserving bonds)
    tpr_file = os.path.join(os.path.dirname(gro_file), "topol.tpr")
    if os.path.exists(tpr_file):
        print(f"Using TPR file for bond information: {tpr_file}")
        try:
            # Create a universe with the TPR to get bond information
            u_tpr = mda.Universe(tpr_file)
            # Copy bond information
            u.add_bonds(u_tpr.bonds)
            print(f"Added {len(u_tpr.bonds)} bonds from TPR file")
        except Exception as e:
            print(f"Warning: Could not load TPR file: {e}")

    # Check for cyclic peptide (common in your structures)
    print("Analyzing structure for cyclic bonds...")
    # Get atoms from first and last residue that might form a bond
    if len(u.residues) > 1:
        first_res = u.residues[0]
        last_res = u.residues[-1]

        # In a typical peptide, we'd connect C of last residue to N of first
        c_atoms = last_res.atoms.select_atoms("name C")
        n_atoms = first_res.atoms.select_atoms("name N")

        # Check distance to verify if they should be bonded
        if len(c_atoms) > 0 and len(n_atoms) > 0:
            c_atom = c_atoms[0]
            n_atom = n_atoms[0]
            distance = np.linalg.norm(c_atom.position - n_atom.position)

            # If atoms are close enough, they're probably bonded
            # Typical C-N bond length is around 1.3-1.5 Å
            if distance < 2.0:
                print(
                    f"Detected potential cyclic bond: {last_res.resname}-{c_atom.name} to {first_res.resname}-{n_atom.name} (distance: {distance:.2f} Å)"
                )
                try:
                    # Add the bond to the universe
                    u.add_bonds([(c_atom, n_atom)])
                    print("Added cyclic bond between first and last residue")
                except Exception as e:
                    print(f"Warning: Could not add cyclic bond: {e}")
            else:
                print(
                    f"No cyclic bond detected (distance between terminal residues: {distance:.2f} Å)"
                )

    # Set frame range
    total_frames = len(u.trajectory)
    if stop is None:
        stop = total_frames
    else:
        stop = min(stop, total_frames)

    print(f"Total frames in trajectory: {total_frames}")
    print(f"Selected frame range: {start} to {stop-1}, step {step}")

    # Validate frame range
    if start >= total_frames:
        print(f"Error: Start frame {start} exceeds total frames {total_frames}")
        return

    # Get frame indices to extract
    frame_indices = range(start, stop, step)
    n_frames = len(frame_indices)
    print(f"Extracting {n_frames} frames...")

    # Process each frame
    for i, ts_idx in tqdm(
        enumerate(frame_indices), total=n_frames, desc="Extracting frames"
    ):
        # Jump to the specified frame
        u.trajectory[ts_idx]
        frame_num = u.trajectory.frame

        # Save PDB with connectivity information
        pdb_path = os.path.join(output_dir, f"frame_{frame_num}.pdb")

        # Save PDB with bonds
        try:
            # Write PDB with all bond information
            u.atoms.write(pdb_path, bonds="all")

            # Check if we need to manually add CONECT records for cyclic bond
            if hasattr(u, "_topology") and hasattr(u._topology, "bonds"):
                # Check specifically for the bond between first and last residue
                first_res = u.residues[0]
                last_res = u.residues[-1]

                # Get atoms that might be involved in a cyclic bond
                c_atoms = last_res.atoms.select_atoms("name C")
                n_atoms = first_res.atoms.select_atoms("name N")

                if len(c_atoms) > 0 and len(n_atoms) > 0:
                    c_atom = c_atoms[0]
                    n_atom = n_atoms[0]

                    # Check if MDAnalysis properly wrote this bond
                    # If not, we'll add it manually
                    bond_found = False

                    # Open the file to check if bond exists and append if needed
                    with open(pdb_path, "r") as f:
                        content = f.read()
                        # Check if any CONECT record includes both atom serial numbers
                        c_serial = c_atom.id
                        n_serial = n_atom.id
                        bond_pattern = f"CONECT.*{c_serial}.*{n_serial}|CONECT.*{n_serial}.*{c_serial}"
                        if not re.search(bond_pattern, content):
                            # Need to add CONECT record
                            with open(pdb_path, "a") as f_out:
                                f_out.write(f"CONECT{c_serial:5d}{n_serial:5d}\n")
                            if i == 0:  # Only print for first frame
                                print(
                                    f"Added missing CONECT record between atoms {c_serial} and {n_serial}"
                                )

        except Exception as e:
            print(f"Warning on frame {frame_num}: {e}")
            # Fallback to simple write
            u.atoms.write(pdb_path)

    print(f"Successfully extracted {n_frames} PDB files to {output_dir}")

    # Create a simple script to load all frames in ChimeraX
    chimera_script = os.path.join(output_dir, "open_in_chimerax.cxc")
    with open(chimera_script, "w") as f:
        f.write("# ChimeraX script to load and view frames\n")
        f.write(f"open {output_dir}/frame_*.pdb\n")
        f.write("# To view as movie\n")
        f.write("coordset #1 play direction forward loop true maxFrameRate 15\n")

    print(f"Created ChimeraX script: {chimera_script}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract PDB frames from trajectory with bonds"
    )
    parser.add_argument("gro_file", help="Path to GRO structure file")
    parser.add_argument("xtc_file", help="Path to XTC trajectory file")
    parser.add_argument("top_file", help="Path to TOP topology file")
    parser.add_argument("output_dir", help="Directory to save PDB files")

    # Frame selection options
    parser.add_argument(
        "--start", type=int, default=0, help="First frame to extract (0-indexed)"
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Last frame to extract (default: last frame)",
    )
    parser.add_argument("--step", type=int, default=1, help="Extract every Nth frame")
    parser.add_argument(
        "--force", action="store_true", help="Force overwrite of existing directories"
    )

    args = parser.parse_args()

    extract_pdbs(
        args.gro_file,
        args.xtc_file,
        args.top_file,
        args.output_dir,
        start=args.start,
        stop=args.stop,
        step=args.step,
        force=args.force,
    )


if __name__ == "__main__":
    main()
