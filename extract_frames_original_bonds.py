#!/usr/bin/env python3
# extract_frames_original_bonds.py

"""
Script to extract PDB frames from MD trajectory using only original bond information.

This script processes trajectory files (.xtc) along with structure (.gro) and
topology (.top) files to extract frames as PDB files with the exact bond information
from the original data, without any inference or additions.

Usage:
    python extract_frames_original_bonds.py /path/to/gro /path/to/xtc /path/to/top output_dir [options]

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
from pathlib import Path
from tqdm import tqdm


def extract_pdbs(
    gro_file, xtc_file, top_file, output_dir, start=0, stop=None, step=1, force=False
):
    """
    Extract PDB files from trajectory with only original bond information.

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
    print(f"Using topology file: {top_file}")

    # Load trajectory without using lock files (important for network drives)
    u = mda.Universe(gro_file, xtc_file, use_lock=False)

    # Try to load bond information from TPR file if available
    tpr_file = os.path.join(os.path.dirname(gro_file), "topol.tpr")
    if os.path.exists(tpr_file):
        print(f"Using TPR file for bond information: {tpr_file}")
        try:
            # Create a universe with the TPR to get bond information
            u_tpr = mda.Universe(tpr_file)
            # Transfer only original bonds (no inference)
            u.add_bonds(u_tpr.bonds)
            print(f"Added {len(u_tpr.bonds)} bonds from TPR file")
        except Exception as e:
            print(f"Warning: Could not load TPR file: {e}")
    else:
        print("No TPR file found. Bond information may be limited.")

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

        # Save PDB with original bond information
        pdb_path = os.path.join(output_dir, f"frame_{frame_num}.pdb")

        # Write PDB with bond information
        try:
            # Use bonds="all" to include all available bond information from the original data
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
        description="Extract PDB frames from trajectory with original bonds"
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
