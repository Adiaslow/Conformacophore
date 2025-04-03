#!/usr/bin/env python3
# read_tpr_bonds_recursive.py

"""
Script to extract PDB frames from MD trajectory using GROMACS tools to properly read bonds.
This version can recursively process subdirectories.

This script uses a combination of MDAnalysis and GROMACS tools to extract PDB frames
with correct bond information from the original topology.

Usage:
    python read_tpr_bonds_recursive.py /path/to/directory [options]

Options:
    --start N       Start from frame N (default: 0)
    --stop N        Stop at frame N (default: last frame)
    --step N        Extract every Nth frame (default: 1)
    --force         Overwrite existing output directories
    --recursive     Process subdirectories recursively
"""

import MDAnalysis as mda
import os
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm


def check_gmx_command():
    """Check if GROMACS command-line tools are available"""
    try:
        result = subprocess.run(
            ["gmx", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            print("Found GROMACS command-line tools")
            return True
    except FileNotFoundError:
        pass

    print(
        "Warning: GROMACS command-line tools not found. Bond information may be limited."
    )
    return False


def generate_tpr_file(gro_file, top_file, output_dir):
    """
    Generate a TPR file from GRO and TOP files using GROMACS.

    Args:
        gro_file (str): Path to structure file
        top_file (str): Path to topology file
        output_dir (str): Directory to save the TPR file

    Returns:
        str: Path to the created TPR file, or None if failed
    """
    tpr_path = os.path.join(output_dir, "temp.tpr")

    print(f"Generating TPR file from {gro_file} and {top_file}...")

    try:
        # Create a minimal mdp file
        mdp_path = os.path.join(output_dir, "temp.mdp")
        with open(mdp_path, "w") as f:
            f.write("integrator = md\n")
            f.write("nsteps = 0\n")
            f.write("dt = 0.001\n")

        # Run grompp to generate the TPR file
        cmd = [
            "gmx",
            "grompp",
            "-f",
            mdp_path,
            "-c",
            gro_file,
            "-p",
            top_file,
            "-o",
            tpr_path,
            "-maxwarn",
            "10",
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode == 0:
            print(f"Successfully generated TPR file: {tpr_path}")
            return tpr_path
        else:
            print(f"Error generating TPR file: {result.stderr}")
            return None
    except Exception as e:
        print(f"Exception while generating TPR file: {e}")
        return None


def find_md_directories(base_dir, recursive=False):
    """
    Find directories containing MD files (GRO, XTC, TOP).

    Args:
        base_dir (str): Base directory to search
        recursive (bool): Whether to search subdirectories recursively

    Returns:
        list: List of directories containing MD files
    """
    md_dirs = []
    base_path = Path(base_dir)

    if recursive:
        # Search all subdirectories recursively
        for dirpath, dirnames, filenames in os.walk(base_dir):
            if any(f.endswith(".gro") for f in filenames) and any(
                f.endswith(".xtc") for f in filenames
            ):
                md_dirs.append(dirpath)
    else:
        # Only search immediate subdirectories
        # First check if the base directory itself has MD files
        if any(f.suffix == ".gro" for f in base_path.glob("*.gro")) and any(
            f.suffix == ".xtc" for f in base_path.glob("*.xtc")
        ):
            md_dirs.append(str(base_path))

        # Then check immediate subdirectories
        for subdir in base_path.iterdir():
            if subdir.is_dir():
                if any(f.suffix == ".gro" for f in subdir.glob("*.gro")) and any(
                    f.suffix == ".xtc" for f in subdir.glob("*.xtc")
                ):
                    md_dirs.append(str(subdir))

    return md_dirs


def process_md_directory(
    md_dir, output_dir=None, start=0, stop=None, step=1, force=False
):
    """
    Process a directory containing MD files (GRO, XTC, TOP).

    Args:
        md_dir (str): Path to directory with MD files
        output_dir (str): Directory to save extracted frames (default: md_dir/pdb_frames)
        start (int): First frame to extract
        stop (int): Last frame to extract
        step (int): Extract every Nth frame
        force (bool): Whether to overwrite existing output directory
    """
    # Find GRO, XTC and TOP files
    md_path = Path(md_dir)
    gro_files = list(md_path.glob("*.gro"))
    xtc_files = list(md_path.glob("*.xtc"))
    top_files = list(md_path.glob("*.top"))
    tpr_files = list(md_path.glob("*.tpr"))

    if not gro_files:
        print(f"Error: No GRO file found in {md_dir}")
        return

    if not xtc_files:
        print(f"Error: No XTC file found in {md_dir}")
        return

    # Use the first files found
    gro_file = str(gro_files[0])
    xtc_file = str(xtc_files[0])
    top_file = str(top_files[0]) if top_files else None
    tpr_file = str(tpr_files[0]) if tpr_files else None

    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(md_dir, "pdb_frames")

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

    # Check if GROMACS is available
    has_gmx = check_gmx_command()

    # If we have GROMACS and a TOP file but no TPR file, generate one
    if has_gmx and top_file and not tpr_file:
        tpr_file = generate_tpr_file(gro_file, top_file, output_dir)

    print(f"Loading trajectory: {xtc_file}")
    print(f"Using structure: {gro_file}")

    # Load universe with TPR file for bond information if available
    if tpr_file:
        print(f"Using TPR file for topology: {tpr_file}")
        try:
            u = mda.Universe(tpr_file, xtc_file, use_lock=False)
            print(f"Loaded universe with bond information from TPR file")
        except Exception as e:
            print(f"Error loading TPR file: {e}")
            print("Falling back to GRO file for structure")
            u = mda.Universe(gro_file, xtc_file, use_lock=False)
    else:
        u = mda.Universe(gro_file, xtc_file, use_lock=False)

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

    # Check if we have bond information
    has_bonds = hasattr(u, "bonds") and len(u.bonds) > 0
    print(
        f"Bond information available: {has_bonds} ({len(u.bonds) if has_bonds else 0} bonds)"
    )

    # Detect if we have a cyclic structure
    is_cyclic = False
    if has_bonds:
        # Check for bonds between first and last residue
        first_res_atoms = u.residues[0].atoms
        last_res_atoms = u.residues[-1].atoms

        for bond in u.bonds:
            if (bond[0] in first_res_atoms and bond[1] in last_res_atoms) or (
                bond[1] in first_res_atoms and bond[0] in last_res_atoms
            ):
                is_cyclic = True
                print(
                    f"Detected cyclic structure: bond between {bond[0].resname}-{bond[0].name} and {bond[1].resname}-{bond[1].name}"
                )
                break

    # Process each frame
    for ts_idx in tqdm(frame_indices, desc="Extracting frames"):
        # Jump to the specified frame
        u.trajectory[ts_idx]
        frame_num = u.trajectory.frame

        # Save PDB with bond information
        pdb_path = os.path.join(output_dir, f"frame_{frame_num}.pdb")

        # Write PDB
        try:
            # Use bonds="all" to include all bond information
            u.atoms.write(pdb_path, bonds="all")
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

        if is_cyclic:
            # If cyclic, add command to connect first and last residue
            f.write(
                "\n# This appears to be a cyclic structure. Uncomment to add bond:\n"
            )
            f.write("# select #1/1:{1,end}\n")
            f.write("# bond sel relativeLength 1.4\n")

        # Add movie commands
        f.write("\n# To view as movie:\n")
        f.write("coordset #1 play direction forward loop true maxFrameRate 15\n")

    print(f"Created ChimeraX script: {chimera_script}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract PDB frames from MD trajectory with proper bonds"
    )
    parser.add_argument(
        "base_dir", help="Path to directory containing MD files (GRO, XTC, TOP)"
    )
    parser.add_argument(
        "--output", "-o", help="Base directory to save extracted frames"
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Process subdirectories recursively",
    )

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

    # Find all directories with MD files
    md_dirs = find_md_directories(args.base_dir, args.recursive)

    if not md_dirs:
        print(f"No directories with MD files found in {args.base_dir}")
        return

    print(f"Found {len(md_dirs)} directories with MD files:")
    for md_dir in md_dirs:
        print(f"  {md_dir}")

    # Process each directory
    successful = 0
    for md_dir in md_dirs:
        print(f"\n{'='*80}\nProcessing {md_dir}\n{'='*80}")

        # Define output directory
        if args.output:
            # If user specified a base output directory, create subdirectories based on input structure
            rel_path = os.path.relpath(md_dir, args.base_dir)
            if rel_path == ".":
                # If processing the base directory itself
                output_dir = args.output
            else:
                # If processing a subdirectory
                output_dir = os.path.join(args.output, rel_path)
        else:
            # Use default (pdb_frames in each directory)
            output_dir = None

        # Process the directory
        result = process_md_directory(
            md_dir,
            output_dir,
            start=args.start,
            stop=args.stop,
            step=args.step,
            force=args.force,
        )

        if result:
            successful += 1

    print(f"\nProcessed {successful} out of {len(md_dirs)} directories successfully")


if __name__ == "__main__":
    main()
