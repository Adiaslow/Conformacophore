#!/usr/bin/env python3
# extract_frames.py

"""
Script to extract frames from MD trajectory for visualization and analysis.

This script processes MD trajectory files (.xtc) along with topology files (.gro, .top)
to extract individual frames. It automatically detects whether the directory structure
is nested and processes all directories containing MD files accordingly.

Usage:
    python extract_frames.py /path/to/directory [options]

Options:
    --start N       Start from frame N (default: 0)
    --stop N        Stop at frame N (default: last frame)
    --step N        Extract every Nth frame (default: 1)
    --force         Overwrite existing output directory
    --verbose       Show detailed processing information

Author: Claude
Date: 2024
"""

import MDAnalysis as mda
import numpy as np
import os
import pickle
from pathlib import Path
import argparse
import glob
import shutil
from tqdm import tqdm


def is_nested_structure(base_path):
    """
    Detect if the directory structure contains nested MD files.

    Args:
        base_path (str): Path to search for MD files

    Returns:
        bool: True if nested structure detected, False otherwise
    """
    base = Path(base_path)

    # Get all subdirectories
    subdirs = [d for d in base.glob("**/") if d != base]

    for subdir in subdirs:
        # Check if any subdirectory contains MD files
        if (
            list(subdir.glob("*.gro"))
            or list(subdir.glob("*.xtc"))
            or list(subdir.glob("*.top"))
        ):
            return True

    return False


def find_md_file_sets(base_path):
    """
    Find sets of MD files (.gro, .xtc, .top) in the given directory.

    Args:
        base_path (str): Path to search for MD files

    Returns:
        list: List of tuples containing (directory, gro_file, xtc_file, top_file)
    """
    md_sets = []

    # Convert base_path to Path object for easier manipulation
    base = Path(base_path)

    # Determine if we need to search recursively
    recursive = is_nested_structure(base_path)
    search_pattern = "**/" if recursive else ""

    # Find all .gro files
    for gro_file in base.glob(f"{search_pattern}*.gro"):
        dir_path = gro_file.parent

        # Look for matching .xtc and .top files in the same directory
        xtc_files = list(dir_path.glob("*.xtc"))
        top_files = list(dir_path.glob("*.top"))

        if xtc_files and top_files:  # Only include if we have all required files
            md_sets.append(
                (
                    str(dir_path),
                    str(gro_file),
                    str(xtc_files[0]),  # Take first matching file if multiple exist
                    str(top_files[0]),
                )
            )

    return md_sets


def extract_frames(
    gro_file,
    xtc_file,
    output_dir,
    start=0,
    stop=None,
    step=1,
    force=False,
    verbose=False,
):
    """
    Extract frames from trajectory while preserving all molecular information.

    Args:
        gro_file (str): Path to reference structure file (.gro)
        xtc_file (str): Path to trajectory file (.xtc)
        output_dir (str): Directory to save extracted frames
        start (int): First frame to extract (0-indexed)
        stop (int): Last frame to extract (None for all frames)
        step (int): Extract every nth frame
        force (bool): Whether to overwrite existing output directory
        verbose (bool): Whether to show detailed output

    Returns:
        dict: Dictionary containing frame data and metadata
    """
    # Handle output directory
    output_path = Path(output_dir)
    if output_path.exists():
        if force:
            if verbose:
                print(f"Removing existing directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"Warning: Output directory {output_dir} already exists.")
            print("Use --force to overwrite or choose a different output location.")
            return None

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the trajectory
    if verbose:
        print(f"Loading trajectory: {xtc_file}")
        print(f"Using topology: {gro_file}")

    # Set use_lock=False to prevent issues with network filesystems
    u = mda.Universe(gro_file, xtc_file, use_lock=False)

    # Load topology file separately if provided
    top_file = os.path.join(os.path.dirname(gro_file), "topol.top")
    if os.path.exists(top_file):
        if verbose:
            print(f"Reading connectivity information from topology: {top_file}")
        try:
            # Create a temporary universe with the topology to extract connectivity
            u_top = mda.Universe(gro_file, topology_format="gro", use_lock=False)

            # Parse the topology file to extract bond information
            with open(top_file, "r") as f:
                top_content = f.read()

            # Store information about cyclic bonds or special connectivity
            # This is a simplistic approach - full topology parsing would be more complex
            if verbose:
                print("Checking for cyclic bonds or special connectivity...")

            # Adding this information to metadata for later use
            cyclic_structure = (
                "cyclic" in top_content.lower() or "ring" in top_content.lower()
            )
            if cyclic_structure and verbose:
                print("Detected potential cyclic structure")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not process topology file: {e}")

    # Set frame range
    total_frames = len(u.trajectory)
    if stop is None:
        stop = total_frames
    else:
        stop = min(stop, total_frames)

    if verbose:
        print(f"Total frames in trajectory: {total_frames}")
        print(f"Selected frame range: {start} to {stop-1}, step {step}")

    # Validate frame range
    if start >= total_frames:
        print(f"Error: Start frame {start} exceeds total frames {total_frames}")
        return None

    # Dictionary to store frame data
    trajectory_data = {
        "metadata": {
            "n_frames": total_frames,
            "n_atoms": len(u.atoms),
            "n_selected_frames": len(range(start, stop, step)),
            "selected_range": (start, stop, step),
            "residues": [(r.resname, r.resid) for r in u.residues],
            "atom_names": [atom.name for atom in u.atoms],
            "cyclic_structure": (
                cyclic_structure if "cyclic_structure" in locals() else False
            ),
        },
        "frames": {},
    }

    # Extract each frame
    frame_indices = range(start, stop, step)
    if verbose:
        print(f"Extracting {len(frame_indices)} frames...")

    # Create progress bar
    progress_bar = tqdm(
        total=len(frame_indices),
        desc="Extracting Frames",
        unit="frame",
        disable=not verbose,
    )

    # Flag for CONECT records
    add_conect = True

    for i, ts_idx in enumerate(frame_indices):
        # Jump to the specified frame
        u.trajectory[ts_idx]
        ts = u.trajectory
        frame_num = ts.frame

        # Save PDB with connectivity information
        pdb_path = os.path.join(output_dir, f"frame_{frame_num}.pdb")

        # Try to write with connectivity information
        try:
            # First attempt - write standard PDB
            u.atoms.write(pdb_path)

            # Add CONECT records for bonds between first and last residue if needed
            if add_conect and trajectory_data["metadata"]["cyclic_structure"]:
                # Get first and last residue atoms
                first_res_atoms = u.residues[0].atoms
                last_res_atoms = u.residues[-1].atoms

                # Find potential connection points
                potential_bonds = []
                for a1 in last_res_atoms:
                    for a2 in first_res_atoms:
                        # Check distance between atoms - this is a heuristic
                        # Better would be to use actual bond information from topology
                        dist = np.linalg.norm(a1.position - a2.position)
                        if dist < 2.0:  # Typical bond length threshold
                            potential_bonds.append((a1.id, a2.id))

                if potential_bonds and verbose:
                    print(
                        f"Adding {len(potential_bonds)} CONECT records between first and last residue"
                    )

                # Append CONECT records to the PDB file
                if potential_bonds:
                    with open(pdb_path, "a") as f:
                        for a1_id, a2_id in potential_bonds:
                            f.write(f"CONECT{a1_id:5d}{a2_id:5d}\n")

                # Only try to add CONECT records for the first PDB file
                # If it works, keep doing it; if it fails, skip for remaining frames
                add_conect = True
        except Exception as e:
            if verbose and i == 0:
                print(f"Warning: Could not add connectivity information: {e}")
            add_conect = False

            # Fallback to basic PDB write
            u.atoms.write(pdb_path)

        # Store comprehensive frame data
        frame_data = {
            "positions": u.atoms.positions.copy(),
            "velocities": (
                u.atoms.velocities.copy() if hasattr(u.atoms, "velocities") else None
            ),
            "forces": u.atoms.forces.copy() if hasattr(u.atoms, "forces") else None,
            "dimensions": u.dimensions.copy(),
            "time": ts.time,
            "step": ts.frame,
            "original_frame_idx": ts_idx,
        }

        trajectory_data["frames"][i] = frame_data

        # Update progress bar
        progress_bar.update(1)

        if verbose and (i + 1) % 10 == 0:
            progress_bar.set_postfix({"Current Frame": frame_num})

    # Close progress bar
    progress_bar.close()

    # Save the complete trajectory data
    pickle_path = os.path.join(output_dir, "trajectory_data.pkl")
    if verbose:
        print(f"Saving trajectory data to {pickle_path}")

    with open(pickle_path, "wb") as f:
        pickle.dump(trajectory_data, f)

    # Additionally save a ChimeraX script to help visualize the structure properly
    chimera_script = os.path.join(output_dir, "load_in_chimerax.cxc")
    with open(chimera_script, "w") as f:
        f.write("# ChimeraX script to load and display the trajectory\n")
        f.write(f"open {output_dir}/frame_*.pdb\n")
        f.write("movie record supersample 3\n")
        f.write("coordset #1 range all step 1\n")
        f.write("movie encode H264 bitrate 4000000 output trajectory.mp4\n")
        f.write("# Bond the first and last residue if needed\n")
        f.write("select #1/1 #1:1,end\n")
        f.write(
            "# Uncomment the following line if you need to create bonds between residues\n"
        )
        f.write("# bond sel relativeLength 1.5\n")

    if verbose:
        print(f"Created ChimeraX script: {chimera_script}")

    return trajectory_data


def calculate_rmsd(frame1_data, frame2_data, selection="all"):
    """
    Calculate RMSD between two frames.

    Args:
        frame1_data (dict): Data for first frame
        frame2_data (dict): Data for second frame
        selection (str): Atom selection string

    Returns:
        float: RMSD value
    """
    positions1 = frame1_data["positions"]
    positions2 = frame2_data["positions"]

    # Calculate RMSD
    diff = positions1 - positions2
    rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
    return rmsd


def process_directory(dir_path, gro_file, xtc_file, top_file, args):
    """
    Process a single directory containing MD files.

    Args:
        dir_path (str): Path to directory containing files
        gro_file (str): Path to .gro file
        xtc_file (str): Path to .xtc file
        top_file (str): Path to .top file
        args: Command line arguments
    """
    output_dir = os.path.join(dir_path, "extracted_frames")

    print(f"\nProcessing directory: {dir_path}")
    print(f"GRO file: {os.path.basename(gro_file)}")
    print(f"XTC file: {os.path.basename(xtc_file)}")
    print(f"TOP file: {os.path.basename(top_file)}")

    # Extract frames
    trajectory_data = extract_frames(
        gro_file,
        xtc_file,
        output_dir,
        start=args.start,
        stop=args.stop,
        step=args.step,
        force=args.force,
        verbose=args.verbose,
    )

    if trajectory_data:
        print(f"Extracted {len(trajectory_data['frames'])} frames")
        print(f"Files saved to: {output_dir}")
    else:
        print("Frame extraction failed or was skipped")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from MD trajectories")
    parser.add_argument("path", help="Path to directory containing MD files")

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

    # Processing options
    parser.add_argument(
        "--force", action="store_true", help="Force overwrite of existing directories"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed progress information"
    )

    args = parser.parse_args()

    # Find all sets of MD files (automatically handles nested directories)
    md_sets = find_md_file_sets(args.path)

    if not md_sets:
        print(f"No complete sets of MD files found in {args.path}")
        return

    print(f"Found {len(md_sets)} sets of MD files to process")

    # Process each set of files
    for dir_path, gro_file, xtc_file, top_file in md_sets:
        process_directory(dir_path, gro_file, xtc_file, top_file, args)


if __name__ == "__main__":
    main()
