#!/usr/bin/env python3
# parallel_extract.py

"""
Parallel script to extract PDB frames from MD trajectories using multiple CPU cores.

This script processes multiple directories in parallel to significantly speed up frame extraction.
It preserves all bond information from the original topology files.

Usage:
    python parallel_extract.py /path/to/directory [options]

Options:
    --start N       Start from frame N (default: 0)
    --stop N        Stop at frame N (default: last frame)
    --step N        Extract every Nth frame (default: 1)
    --force         Overwrite existing output directories
    --recursive     Process subdirectories recursively
    --processes N   Number of parallel processes to use (default: 4)
    --no-tpr        Skip TPR file generation (faster but may lose bond information)
"""

import MDAnalysis as mda
import os
import shutil
import argparse
import subprocess
import tempfile
import multiprocessing
from pathlib import Path
from tqdm import tqdm
import time


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
            return True
    except FileNotFoundError:
        pass

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

        # Run command and suppress output
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode == 0:
            return tpr_path
        else:
            return None
    except Exception:
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


def process_directory(args):
    """
    Process a single directory with MD files.

    Args:
        args: Tuple containing (md_dir, output_dir, start, stop, step, force, use_tpr)

    Returns:
        tuple: (md_dir, success, n_frames, error_message)
    """
    md_dir, output_dir, start, stop, step, force, use_tpr = args

    try:
        # Find GRO, XTC and TOP files
        md_path = Path(md_dir)
        gro_files = list(md_path.glob("*.gro"))
        xtc_files = list(md_path.glob("*.xtc"))
        top_files = list(md_path.glob("*.top"))
        tpr_files = list(md_path.glob("*.tpr"))

        if not gro_files or not xtc_files:
            return (md_dir, False, 0, "Missing GRO or XTC files")

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
                shutil.rmtree(output_dir)
            else:
                return (md_dir, False, 0, "Output directory already exists")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate TPR file if needed and requested
        has_gmx = check_gmx_command()
        if use_tpr and has_gmx and top_file and not tpr_file:
            tpr_file = generate_tpr_file(gro_file, top_file, output_dir)

        # Load universe with TPR file for bond information if available
        if tpr_file and use_tpr:
            try:
                u = mda.Universe(tpr_file, xtc_file, use_lock=False)
            except Exception:
                u = mda.Universe(gro_file, xtc_file, use_lock=False)
        else:
            u = mda.Universe(gro_file, xtc_file, use_lock=False)

        # Set frame range
        total_frames = len(u.trajectory)
        if stop is None:
            stop = total_frames
        else:
            stop = min(stop, total_frames)

        if start >= total_frames:
            return (
                md_dir,
                False,
                0,
                f"Start frame {start} exceeds total frames {total_frames}",
            )

        # Get frame indices to extract
        frame_indices = range(start, stop, step)
        n_frames = len(frame_indices)

        # Check if we have bond information
        has_bonds = hasattr(u, "bonds") and len(u.bonds) > 0

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
                    break

        # Process each frame
        for ts_idx in frame_indices:
            # Jump to the specified frame
            u.trajectory[ts_idx]
            frame_num = u.trajectory.frame

            # Save PDB with bond information
            pdb_path = os.path.join(output_dir, f"frame_{frame_num}.pdb")

            # Write PDB
            try:
                # Use bonds="all" to include all bond information
                u.atoms.write(pdb_path, bonds="all")
            except Exception:
                # Fallback to simple write
                u.atoms.write(pdb_path)

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

        return (md_dir, True, n_frames, "")

    except Exception as e:
        return (md_dir, False, 0, str(e))


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
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=4,
        help="Number of parallel processes to use",
    )
    parser.add_argument(
        "--no-tpr",
        action="store_true",
        help="Skip TPR file generation (faster but may lose bond information)",
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

    start_time = time.time()

    print(f"Using {args.processes} parallel processes")
    print(f"{'Using' if not args.no_tpr else 'Skipping'} TPR file generation")

    # Find all directories with MD files
    print(f"Searching for MD files in {args.base_dir}...")
    md_dirs = find_md_directories(args.base_dir, args.recursive)

    if not md_dirs:
        print(f"No directories with MD files found in {args.base_dir}")
        return

    print(f"Found {len(md_dirs)} directories with MD files")

    # Prepare arguments for parallel processing
    process_args = []
    for md_dir in md_dirs:
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

        process_args.append(
            (
                md_dir,
                output_dir,
                args.start,
                args.stop,
                args.step,
                args.force,
                not args.no_tpr,
            )
        )

    # Process directories in parallel
    print(f"Processing {len(md_dirs)} directories in parallel...")
    with multiprocessing.Pool(processes=args.processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_directory, process_args),
                total=len(process_args),
                desc="Directories Processed",
            )
        )

    # Summarize results
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    total_frames = sum(r[2] for r in successful)

    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"Total time: {time.time() - start_time:.1f} seconds")
    print(
        f"Successfully processed {len(successful)} directories ({total_frames} frames)"
    )

    if failed:
        print(f"Failed to process {len(failed)} directories:")
        for md_dir, _, _, error in failed:
            print(f"  {md_dir}: {error}")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
