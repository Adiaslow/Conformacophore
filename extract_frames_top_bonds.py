#!/usr/bin/env python3
# extract_frames_top_bonds.py

"""
Script to extract PDB frames from MD trajectory with correct bond information from TOP file.

This script processes trajectory files (.xtc) along with structure (.gro) and
topology (.top) files to extract frames as PDB files with complete bond information
from the original topology, ensuring all bonds are correctly represented.

Usage:
    python extract_frames_top_bonds.py /path/to/gro /path/to/xtc /path/to/top output_dir [options]

Options:
    --start N       Start from frame N (default: 0)
    --stop N        Stop at frame N (default: last frame)
    --step N        Extract every Nth frame (default: 1)
    --force         Overwrite existing output directory
"""

import MDAnalysis as mda
import os
import shutil
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def parse_top_bonds(top_file):
    """
    Parse bond information directly from GROMACS topology file.

    Args:
        top_file (str): Path to topology file (.top)

    Returns:
        list: List of tuples containing atom indices for each bond
    """
    bonds = []
    reading_bonds = False

    print(f"Parsing bond information from {top_file}...")

    try:
        with open(top_file, "r") as f:
            for line in f:
                line = line.strip()

                # Look for the [ bonds ] section
                if line.startswith("[ bonds ]"):
                    reading_bonds = True
                    continue

                # Stop reading when we reach a new section
                if reading_bonds and line.startswith("["):
                    reading_bonds = False

                # Parse bond entries
                if reading_bonds and line and not line.startswith(";"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            # GROMACS bonds are 1-indexed, so subtract 1
                            atom1 = int(parts[0]) - 1
                            atom2 = int(parts[1]) - 1
                            bonds.append((atom1, atom2))
                        except ValueError:
                            continue
    except Exception as e:
        print(f"Error parsing TOP file: {e}")

    print(f"Found {len(bonds)} bonds in topology file")
    return bonds


def extract_pdbs(
    gro_file, xtc_file, top_file, output_dir, start=0, stop=None, step=1, force=False
):
    """
    Extract PDB files from trajectory with bond information from TOP file.

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
    print(f"Using structure: {gro_file}")

    # Load trajectory without using lock files (important for network drives)
    u = mda.Universe(gro_file, xtc_file, use_lock=False)

    # Get bond information from TOP file
    if os.path.exists(top_file):
        # Parse bonds from the topology file
        top_bonds = parse_top_bonds(top_file)

        # Check for a cyclic bond (between first and last residue)
        if top_bonds:
            # Add the bonds to the universe
            try:
                # Convert bond atom indices to actual atoms
                bond_atoms = [
                    (u.atoms[i], u.atoms[j])
                    for i, j in top_bonds
                    if i < len(u.atoms) and j < len(u.atoms)
                ]
                u.add_bonds(bond_atoms)
                print(f"Added {len(bond_atoms)} bonds from topology file")

                # Check if there's a bond between the first and last residue
                first_res_atoms = u.residues[0].atoms
                last_res_atoms = u.residues[-1].atoms

                cyclic_bond = False
                for bond in bond_atoms:
                    if (bond[0] in first_res_atoms and bond[1] in last_res_atoms) or (
                        bond[1] in first_res_atoms and bond[0] in last_res_atoms
                    ):
                        cyclic_bond = True
                        print(
                            f"Detected cyclic bond between residues {u.residues[0].resname} and {u.residues[-1].resname}"
                        )
                        break

                if not cyclic_bond:
                    print("No cyclic bond found between first and last residue")
            except Exception as e:
                print(f"Warning: Could not add bonds from topology: {e}")
    else:
        print(f"Warning: Topology file {top_file} not found")

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

    # Check if we have any bond information
    has_bonds = (
        hasattr(u, "_topology")
        and hasattr(u._topology, "bonds")
        and len(u._topology.bonds) > 0
    )
    print(
        f"Bond information available: {has_bonds} ({len(u._topology.bonds.bondlist) if has_bonds else 0} bonds)"
    )

    # Process each frame
    for i, ts_idx in tqdm(
        enumerate(frame_indices), total=n_frames, desc="Extracting frames"
    ):
        # Jump to the specified frame
        u.trajectory[ts_idx]
        frame_num = u.trajectory.frame

        # Save PDB with original bond information
        pdb_path = os.path.join(output_dir, f"frame_{frame_num}.pdb")

        # Write PDB with bond information
        try:
            # Use bonds="all" to include all bond information
            u.atoms.write(pdb_path, bonds="all")
        except Exception as e:
            print(f"Warning on frame {frame_num}: {e}")
            # Fallback to simple write without bonds
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
        description="Extract PDB frames from trajectory with TOP file bonds"
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
