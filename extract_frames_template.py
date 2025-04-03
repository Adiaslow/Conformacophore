#!/usr/bin/env python3
# extract_frames_template.py

"""
Script to extract PDB frames from MD trajectories using template-based approach.

This script creates a template of the molecule structure once, then extracts
coordinates frame by frame, significantly speeding up the extraction process.

Usage:
    python extract_frames_template.py /path/to/directory [options]

Options:
    --start N       Start from frame N (default: 0)
    --stop N        Stop at frame N (default: last frame)
    --step N        Extract every Nth frame (default: 1)
    --force         Overwrite existing output directories
    --recursive     Process subdirectories recursively
    --processes N   Number of parallel processes to use (default: 4)
"""

import os
import re
import shutil
import argparse
import subprocess
import multiprocessing
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Try to import MDAnalysis for XTC trajectory reading
# This is mainly used for reading trajectories efficiently
try:
    import MDAnalysis as mda
except ImportError:
    print("MDAnalysis not found. Please install with: pip install MDAnalysis")
    exit(1)


def parse_gro_file(gro_file):
    """
    Parse a GRO file to extract atom and residue information.

    Args:
        gro_file (str): Path to GRO file

    Returns:
        tuple: (atoms, residues) where:
            atoms is a list of dicts with atom info
            residues is a dict mapping residue numbers to residue names
    """
    atoms = []
    residues = {}

    try:
        with open(gro_file, "r") as f:
            lines = f.readlines()

            # Skip the first line (title) and the last line (box dimensions)
            # Second line contains the number of atoms
            num_atoms = int(lines[1].strip())

            # Parse atom lines
            for i in range(2, 2 + num_atoms):
                line = lines[i]
                # GRO format: residue number, residue name, atom name, atom number, x, y, z
                # The format is fixed-width columns
                try:
                    res_nr = int(line[0:5].strip())
                    res_name = line[5:10].strip()
                    atom_name = line[10:15].strip()
                    atom_nr = int(line[15:20].strip())
                    x = float(line[20:28].strip())
                    y = float(line[28:36].strip())
                    z = float(line[36:44].strip())

                    # Store residue information
                    residues[res_nr] = res_name

                    # Store atom information
                    atoms.append(
                        {
                            "atom_nr": atom_nr,
                            "atom_name": atom_name,
                            "res_nr": res_nr,
                            "res_name": res_name,
                            "x": x,
                            "y": y,
                            "z": z,
                        }
                    )
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line {i+1} in GRO file: {line.strip()}")
                    print(f"Error: {e}")
                    continue

        return atoms, residues

    except Exception as e:
        print(f"Error parsing GRO file {gro_file}: {e}")
        return [], {}


def parse_topology_bonds(top_file):
    """
    Parse a GROMACS topology file to extract bond information.

    Args:
        top_file (str): Path to topology file

    Returns:
        list: List of bond pairs (atom1, atom2)
    """
    bonds = []
    reading_bonds = False

    try:
        with open(top_file, "r") as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if line.startswith(";") or not line:
                    continue

                # Look for the bonds section
                if line == "[ bonds ]":
                    reading_bonds = True
                    continue

                # Stop when we reach another section
                if reading_bonds and line.startswith("["):
                    reading_bonds = False
                    continue

                # Parse bond lines
                if reading_bonds:
                    # Split by whitespace and remove comments
                    parts = line.split(";")[0].strip().split()
                    if len(parts) >= 2:
                        try:
                            atom1 = int(parts[0])
                            atom2 = int(parts[1])
                            bonds.append((atom1, atom2))
                        except ValueError:
                            continue

        return bonds

    except Exception as e:
        print(f"Error parsing topology file {top_file}: {e}")
        return []


def create_pdb_template(atoms, bonds, pdb_lines=None):
    """
    Create a PDB template with atom and bond information.

    Args:
        atoms (list): List of atom dictionaries
        bonds (list): List of bond pairs
        pdb_lines (list, optional): Existing PDB lines to modify

    Returns:
        tuple: (atom_lines, conect_lines) where:
            atom_lines is a list of ATOM record template strings
            conect_lines is a list of CONECT record strings
    """
    atom_lines = []
    atom_indices = {}  # Maps atom numbers to their index in the PDB (1-based)
    pdb_idx = 1

    # Create ATOM records
    for atom in atoms:
        if pdb_lines is None:
            # Create template line with fixed-width fields following PDB standard format
            # PDB format requires specific column positions
            atom_line = f"ATOM  {pdb_idx:5d} {atom['atom_name']:^4s} {atom['res_name']:3s} X{atom['res_nr']:4d}    "
            # Add coordinates with proper spacing
            atom_line += f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
            # Add occupancy and temperature factor with proper spacing
            atom_line += f"  1.00  0.00           {atom['atom_name'][0]:1s}"
        else:
            # Use existing template but replace atom index
            atom_line = (
                pdb_lines[pdb_idx - 1][:6]
                + f"{pdb_idx:5d}"
                + pdb_lines[pdb_idx - 1][11:]
            )

        atom_lines.append(atom_line)
        atom_indices[atom["atom_nr"]] = pdb_idx
        pdb_idx += 1

    # Create CONECT records
    conect_lines = []
    for bond in bonds:
        atom1, atom2 = bond
        if atom1 in atom_indices and atom2 in atom_indices:
            idx1 = atom_indices[atom1]
            idx2 = atom_indices[atom2]
            conect_lines.append(f"CONECT{idx1:5d}{idx2:5d}")

    return atom_lines, conect_lines


def update_pdb_coordinates(atom_lines, coordinates):
    """
    Update the coordinates in the PDB ATOM lines.

    Args:
        atom_lines (list): List of PDB ATOM record strings
        coordinates (numpy.ndarray): 2D array of (N, 3) coordinates

    Returns:
        list: Updated ATOM record strings
    """
    updated_lines = []

    for i, line in enumerate(atom_lines):
        if i < len(coordinates):
            # Extract the coordinates for this atom
            x, y, z = coordinates[i]

            # Replace the coordinates in the ATOM line
            # PDB format: coordinates in columns 31-38, 39-46, 47-54
            updated_line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
            updated_lines.append(updated_line)
        else:
            updated_lines.append(line)

    return updated_lines


def extract_frames_for_directory(args):
    """
    Extract frames from a trajectory file using a template-based approach.

    Args:
        args: Tuple containing (md_dir, output_dir, start, stop, step, force)

    Returns:
        tuple: (md_dir, success, n_frames, error_message)
    """
    md_dir, output_dir, start, stop, step, force = args

    try:
        # Find GRO, XTC and TOP files
        md_path = Path(md_dir)
        gro_files = list(md_path.glob("*.gro"))
        xtc_files = list(md_path.glob("*.xtc"))
        top_files = list(md_path.glob("*.top"))

        if not gro_files or not xtc_files:
            return (md_dir, False, 0, "Missing GRO or XTC files")

        # Use the first files found
        gro_file = str(gro_files[0])
        xtc_file = str(xtc_files[0])
        top_file = str(top_files[0]) if top_files else None

        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(md_dir, "pdb_frames")

        # Handle output directory
        output_path = Path(output_dir)
        try:
            if output_path.exists():
                if force:
                    # Try to remove directory with error handling
                    try:
                        shutil.rmtree(output_dir)
                    except Exception as e:
                        # If can't remove the directory, delete all files inside it
                        print(f"Warning: Couldn't remove directory: {e}")
                        print("Deleting individual files inside the directory...")
                        try:
                            # Delete all files in the directory individually
                            for file_path in output_path.glob("*"):
                                try:
                                    if file_path.is_file():
                                        file_path.unlink()
                                    elif file_path.is_dir():
                                        shutil.rmtree(file_path)
                                except Exception as file_err:
                                    print(
                                        f"Warning: Couldn't remove {file_path}: {file_err}"
                                    )
                        except Exception as dir_err:
                            print(f"Warning: Error cleaning directory: {dir_err}")
                        print("Continuing with existing directory...")
                else:
                    return (md_dir, False, 0, "Output directory already exists")

            # Create output directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return (md_dir, False, 0, f"Error handling output directory: {e}")

        # Step 1: Parse GRO file to get atom and residue information
        atoms, residues = parse_gro_file(gro_file)

        if not atoms:
            return (md_dir, False, 0, "Failed to parse GRO file")

        # Step 2: Parse topology file to get bond information
        bonds = []
        if top_file:
            bonds = parse_topology_bonds(top_file)

        # Step 3: Create PDB template
        atom_lines, conect_lines = create_pdb_template(atoms, bonds)

        # Step 4: Load trajectory for coordinate extraction
        u = mda.Universe(gro_file, xtc_file)

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

        # Check if the structure is cyclic (bond between first and last residue)
        is_cyclic = False
        first_res = min(residues.keys())
        last_res = max(residues.keys())

        first_res_atoms = [a["atom_nr"] for a in atoms if a["res_nr"] == first_res]
        last_res_atoms = [a["atom_nr"] for a in atoms if a["res_nr"] == last_res]

        for bond in bonds:
            atom1, atom2 = bond
            if (atom1 in first_res_atoms and atom2 in last_res_atoms) or (
                atom2 in first_res_atoms and atom1 in last_res_atoms
            ):
                is_cyclic = True
                break

        # Step 5: Process each frame
        extracted_frames = 0
        for ts_idx in frame_indices:
            try:
                # Jump to the specified frame
                u.trajectory[ts_idx]
                frame_num = u.trajectory.frame

                # Get coordinates for this frame
                # Check if u.atoms exists to prevent attribute error
                if hasattr(u, "atoms"):
                    # Access positions safely with getattr to avoid linter errors
                    positions = getattr(u.atoms, "positions", None)
                    if positions is not None:
                        coordinates = positions
                    else:
                        print(
                            f"Warning: Could not access atom positions for frame {frame_num}."
                        )
                        continue
                else:
                    print(f"Warning: No atoms attribute for frame {frame_num}.")
                    continue

                # Safety check for frames near the end of trajectory
                # Some trajectories have corrupt frames at the end
                if ts_idx > total_frames - 20:
                    # This is very close to the end, be extra careful
                    try:
                        # Check if the coordinates look valid
                        if np.isnan(coordinates).any() or np.isinf(coordinates).any():
                            print(
                                f"Warning: Invalid coordinates in frame {frame_num}, skipping."
                            )
                            continue

                        # Check if coordinates are within reasonable bounds
                        if np.max(np.abs(coordinates)) > 999:
                            print(
                                f"Warning: Extreme coordinates in frame {frame_num}, skipping."
                            )
                            continue
                    except Exception as check_err:
                        print(
                            f"Warning: Error validating frame {frame_num}: {check_err}"
                        )
                        continue

                # Update the PDB template with coordinates
                updated_atom_lines = update_pdb_coordinates(atom_lines, coordinates)

                # Write PDB file for this frame
                pdb_path = os.path.join(output_dir, f"frame_{frame_num}.pdb")
                with open(pdb_path, "w") as f:
                    # Write header
                    f.write(
                        f"TITLE     Frame {frame_num} from {os.path.basename(xtc_file)}\n"
                    )
                    f.write(
                        f"REMARK    EXTRACTED FROM TRAJECTORY USING TEMPLATE-BASED APPROACH\n"
                    )

                    # Write ATOM records
                    for line in updated_atom_lines:
                        f.write(line + "\n")

                    # Write CONECT records
                    for line in conect_lines:
                        f.write(line + "\n")

                    # Write END
                    f.write("END\n")

                extracted_frames += 1
            except Exception as e:
                print(f"Error processing frame {ts_idx}: {str(e)}")
                continue

        if extracted_frames == 0:
            return (md_dir, False, 0, "No frames were successfully extracted")

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

        return (md_dir, True, extracted_frames, "")

    except Exception as e:
        return (md_dir, False, 0, str(e))


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


def main():
    parser = argparse.ArgumentParser(
        description="Extract PDB frames from MD trajectory using a template-based approach"
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
        "--safe",
        "-s",
        action="store_true",
        help="Skip frames in last 5%% of trajectory to avoid common errors",
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

    print(f"Using template-based approach with {args.processes} parallel processes")

    # Define safe stop percentage even if not used
    safe_stop_percentage = 0.95  # Skip the last 5% of frames

    # If safe mode is enabled, adjust stop parameter
    if args.safe and args.stop is None:
        print("Safe mode enabled: Will skip frames in last 5% of trajectories")

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

        # If safe mode is enabled, calculate stop frame for this directory
        safe_stop = args.stop
        if args.safe and args.stop is None:
            try:
                # Try to determine trajectory length to set safe stop point
                u = mda.Universe(
                    str(list(Path(md_dir).glob("*.gro"))[0]),
                    str(list(Path(md_dir).glob("*.xtc"))[0]),
                )
                total_frames = len(u.trajectory)
                safe_stop = int(total_frames * safe_stop_percentage)
                print(
                    f"Safe mode: For {md_dir}, processing frames 0-{safe_stop} out of {total_frames}"
                )
            except Exception as e:
                print(f"Could not determine trajectory length for {md_dir}: {e}")
                # Fall back to default behavior
                safe_stop = args.stop

        process_args.append(
            (md_dir, output_dir, args.start, safe_stop, args.step, args.force)
        )

    # Process directories in parallel
    print(f"Processing {len(md_dirs)} directories in parallel...")
    pool = None
    try:
        with multiprocessing.Pool(processes=args.processes) as pool:
            results = list(
                tqdm(
                    pool.imap(extract_frames_for_directory, process_args),
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
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Partial results may have been saved.")
        if pool:
            pool.terminate()
            pool.join()
    except Exception as e:
        print(f"\nAn error occurred during processing: {str(e)}")
        if pool:
            pool.terminate()
            pool.join()


if __name__ == "__main__":
    main()
