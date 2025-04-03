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
import sys
import uuid
import datetime
import fcntl
import errno
import signal
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Try to import MDAnalysis for XTC trajectory reading
# This is mainly used for reading trajectories efficiently
try:
    import MDAnalysis as mda
    import warnings

    # Suppress all MDAnalysis warnings
    warnings.filterwarnings("ignore", module="MDAnalysis")

    # More aggressive lock file prevention
    print("Configuring MDAnalysis for network storage compatibility...")

    # Define a function for complete lock file prevention
    def disable_locks_completely():
        """
        Apply aggressive methods to prevent MDAnalysis from creating lock files.
        Patches multiple levels of the API to ensure no locks are created.
        """
        # Method 1: Patch the coordination module directly if available
        lock_disabled = False
        try:
            # Override the entire lock acquiring function if possible
            if hasattr(mda, "lib") and hasattr(mda.lib, "mdamath"):
                # Create a dummy coordination object
                class DummyCoordination:
                    @staticmethod
                    def acquire_lock(*args, **kwargs):
                        return None

                    @staticmethod
                    def release_lock(*args, **kwargs):
                        pass

                # Try to find the coordination module and replace it
                for module_path in ["lib.mdamath.coordination", "coordinates.base"]:
                    parts = module_path.split(".")
                    target = mda
                    for part in parts[:-1]:
                        if hasattr(target, part):
                            target = getattr(target, part)
                        else:
                            target = None
                            break

                    if target and hasattr(target, parts[-1]):
                        # Replace the coordination module with our dummy
                        setattr(target, parts[-1], DummyCoordination)
                        lock_disabled = True
                        print(f"Disabled lock coordination in {module_path}")
        except Exception as e:
            print(f"Note: Could not patch coordination module: {e}")

        # Method 2: Monkey patch all trajectory format classes
        try:
            # Define the patch function that prevents lock checking
            def _patched_check_exist_path(self, path):
                return False

            # Define a completely disabled lock constructor
            def _disabled_lock_init(self, *args, **kwargs):
                pass

            # Find and patch all relevant classes
            patched_formats = []
            for module_name in ["lib.formats", "coordinates.formats", "coordinates"]:
                try:
                    module = mda
                    for part in module_name.split("."):
                        module = getattr(module, part, None)
                        if module is None:
                            break

                    if module is not None:
                        # Try to patch all format classes we can find
                        for attr_name in dir(module):
                            try:
                                attr = getattr(module, attr_name)
                                if isinstance(attr, type):  # If it's a class
                                    # Patch _check_exist_path if it exists
                                    if hasattr(attr, "_check_exist_path"):
                                        attr._check_exist_path = (
                                            _patched_check_exist_path
                                        )
                                        patched_formats.append(
                                            f"{attr_name}._check_exist_path"
                                        )

                                    # Look for any lock-related methods and disable them
                                    for method_name in dir(attr):
                                        if "lock" in method_name.lower():
                                            try:
                                                # Replace with empty method
                                                setattr(
                                                    attr,
                                                    method_name,
                                                    lambda *args, **kwargs: None,
                                                )
                                                patched_formats.append(
                                                    f"{attr_name}.{method_name}"
                                                )
                                            except:
                                                pass
                            except:
                                continue
                except:
                    continue

            if patched_formats:
                print(f"Patched {len(patched_formats)} lock-related methods")
                lock_disabled = True
        except Exception as e:
            print(f"Note: Error during method patching: {e}")

        # Method 3: Set environment variables that might affect locking
        try:
            import os

            os.environ["MDA_NO_LOCK"] = "1"
            os.environ["NO_LOCK"] = "1"
            print("Set environment variables to disable locks")
        except:
            pass

        return lock_disabled

    # Call our aggressive lock disabling function
    locks_disabled = disable_locks_completely()

    # Original lock prevention method as fallback
    if not locks_disabled:
        # Try to disable lock file creation by monkey patching
        def patch_trajectory_readers():
            # Define the patch function
            def _patched_check_exist_path(self, path):
                return False

            patched_formats = []

            # Get potentially relevant modules dynamically
            modules_to_check = []
            for module_name in ["lib.formats", "coordinates.formats", "coordinates"]:
                try:
                    module = mda
                    for part in module_name.split("."):
                        module = getattr(module, part, None)
                        if module is None:
                            break
                    if module is not None:
                        modules_to_check.append(module)
                except Exception:
                    pass

            # Look for trajectory format classes in each module
            for module in modules_to_check:
                for format_name in ["XTC", "TRR", "DCD"]:
                    try:
                        format_class = getattr(module, format_name, None)
                        if format_class and hasattr(format_class, "_check_exist_path"):
                            # Save original function
                            original_func = format_class._check_exist_path

                            # Apply the patch
                            format_class._check_exist_path = _patched_check_exist_path
                            patched_formats.append(format_name)
                    except Exception:
                        continue

            return patched_formats

        # Apply the patches
        patched_formats = patch_trajectory_readers()
        if patched_formats:
            print(f"Disabled lock file creation for: {', '.join(patched_formats)}")
        else:
            print("Note: Could not disable lock file creation for trajectories")

        # Configure periodic boundary conditions
        try:
            # Try accessing through config dictionary first (newer versions)
            if hasattr(mda, "config"):
                try:
                    mda_config = getattr(mda, "config")
                    mda_config["use_periodic_selections"] = True
                    mda_config["use_pbc"] = True
                    print("Enabled periodic selections via config")
                except:
                    pass

            # Try through core.flags for older versions
            elif hasattr(mda, "core"):
                try:
                    mda_core = getattr(mda, "core")
                    if hasattr(mda_core, "flags"):
                        mda_core.flags["use_periodic_selections"] = True
                        mda_core.flags["use_pbc"] = True
                        print("Enabled periodic selections via core.flags")
                except:
                    pass
        except Exception as e:
            print(f"Note: Could not configure periodic selections: {e}")

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


# Function to check if a file is currently busy
def is_file_busy(file_path):
    """
    Check if a file is currently busy (locked by another process).

    Args:
        file_path (str or Path): Path to file to check

    Returns:
        bool: True if file is busy, False otherwise
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return False

    try:
        # Try to open file with non-blocking exclusive access
        fd = os.open(str(file_path), os.O_RDWR | os.O_NONBLOCK)
        try:
            # Try to get an exclusive lock
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # If we get here, file is not busy
            fcntl.flock(fd, fcntl.LOCK_UN)
        except (IOError, OSError) as e:
            # Resource temporarily unavailable means file is busy
            if e.errno == errno.EAGAIN:
                os.close(fd)
                return True
        finally:
            os.close(fd)
        return False
    except (IOError, OSError):
        # If we can't open the file at all, consider it busy
        return True


# Function to check if a directory has busy files
def has_busy_files(dir_path, sample_size=10):
    """
    Check if a directory has busy files by sampling some files.

    Args:
        dir_path (str or Path): Path to directory
        sample_size (int): Number of files to sample for busy check

    Returns:
        bool: True if directory has busy files, False otherwise
    """
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        return False

    # Get a list of files in the directory
    try:
        files = list(dir_path.glob("frame_*.pdb"))
        # If there are too many files, just sample some
        if len(files) > sample_size:
            import random

            files = random.sample(files, sample_size)

        # Check if any files are busy
        for file in files:
            if is_file_busy(file):
                return True
        return False
    except Exception:
        # If we can't access the directory, consider it busy
        return True


def retry_on_busy(func, *args, retries=3, delay=1.0, **kwargs):
    """
    Retry a function if it fails due to busy resources.

    Args:
        func: Function to call
        *args: Arguments to pass to function
        retries (int): Number of retries
        delay (float): Delay between retries in seconds
        **kwargs: Keyword arguments to pass to function

    Returns:
        Result of function call
    """
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except (IOError, OSError) as e:
            # Check if error is due to busy resource
            if attempt < retries - 1 and (
                getattr(e, "errno", None) in (errno.EBUSY, errno.EAGAIN)
                or "busy" in str(e).lower()
            ):
                time.sleep(delay * (attempt + 1))  # Exponential backoff
                continue
            raise
    # If we get here, all retries failed
    return False


def safe_write_file(file_path, content):
    """
    Safely write content to a file with handling for network drives.

    Args:
        file_path (Path or str): Path to file
        content (str): Content to write

    Returns:
        bool: Success
    """
    file_path = Path(file_path)

    # Check if file exists and is busy with our specialized function
    if is_file_busy(file_path):
        # File is busy, skip it
        return False

    # Use a temporary file with a specific pattern to avoid conflicts
    temp_name = f".tmp_{int(time.time())}_{os.getpid()}_{file_path.name}"
    temp_path = file_path.parent / temp_name

    try:
        # Write to temporary file
        with open(temp_path, "w") as f:
            f.write(content)

        # Use our retry function for the replace operation
        success = retry_on_busy(
            lambda: temp_path.replace(file_path) or True, retries=3, delay=0.5
        )

        return success
    except Exception as e:
        print(f"Error writing file {file_path}: {e}")
        try:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        except:
            pass
        return False


def handle_output_directory(output_path, force):
    """
    Handle output directory creation with robust network drive support.

    Args:
        output_path (Path): Path object for output directory
        force (bool): Whether to force overwrite existing directory

    Returns:
        tuple: (success, message, output_path)
    """
    try:
        # If directory doesn't exist, create it
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            return True, "", output_path

        # Directory exists but force is not enabled
        if not force:
            return False, "Output directory already exists", output_path

        # If filename contains timestamp, it's a new directory so we don't need to check for busy files
        if "_202" in output_path.name:  # Simple check for timestamp pattern
            # This is a timestamped directory, so it should be new - just create it
            output_path.mkdir(parents=True, exist_ok=True)
            return True, "Created new directory", output_path

        # Check if directory has busy files
        if has_busy_files(output_path):
            print(f"Note: Directory {output_path} has busy files")
            print("Will skip existing busy files and continue with available files.")
        else:
            # No busy files, try to delete the directory
            try:
                shutil.rmtree(output_path)
                output_path.mkdir(parents=True, exist_ok=True)
                return True, "Recreated directory", output_path
            except Exception as e:
                print(f"Note: Couldn't remove directory: {e}")
                print(
                    "Will skip existing busy files and continue with available files."
                )

        # Make sure directory exists
        output_path.mkdir(parents=True, exist_ok=True)

        return True, "Using existing directory", output_path

    except Exception as e:
        return False, f"Error handling output directory: {e}", output_path


def extract_frames_for_directory(args):
    """
    Extract frames from a trajectory file using a template-based approach.

    Args:
        args: Tuple containing (md_dir, output_dir, start, stop, step, force, inner_processes, no_offsets, skip_ranges, process_all)

    Returns:
        tuple: (md_dir, success, n_frames, error_message)
    """
    (
        md_dir,
        output_dir,
        start,
        stop,
        step,
        force,
        inner_processes,
        no_offsets,
        skip_ranges,
        process_all,
    ) = args

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

        # Handle output directory with improved network drive support
        output_path = Path(output_dir)
        directory_success, directory_message, output_path = handle_output_directory(
            output_path, force
        )

        if not directory_success:
            return (md_dir, False, 0, directory_message)

        output_dir = str(output_path)  # Update output_dir to possibly modified path

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
        # Configure MDAnalysis options for network share compatibility
        if no_offsets:
            # Try to load without using offsets (needed for network shares)
            u = mda.Universe(
                gro_file,
                xtc_file,
                in_memory=False,
                refresh_offsets=False,
                dt="guess",
                is_periodic=True,
            )
        else:
            # Regular loading with offsets
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
        frame_indices = list(range(start, stop, step))

        # Filter out frames in skip ranges only if not using process_all
        if skip_ranges and not process_all:
            filtered_indices = []
            for idx in frame_indices:
                skip_this_frame = False
                for skip_start, skip_end in skip_ranges:
                    if skip_start <= idx <= skip_end:
                        skip_this_frame = True
                        break
                if not skip_this_frame:
                    filtered_indices.append(idx)

            skipped_count = len(frame_indices) - len(filtered_indices)
            if skipped_count > 0:
                print(
                    f"Skipped {skipped_count} frames in specified ranges for {md_dir}"
                )
            frame_indices = filtered_indices

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
        # Improve performance by using chunk-based processing
        extracted_frames = 0

        # Pre-allocate buffer for PDB filenames to avoid repeated string operations
        pdb_files = [
            os.path.join(output_dir, f"frame_{idx}.pdb") for idx in frame_indices
        ]

        # Track skipped frames for better reporting
        busy_files_skipped = 0
        error_frames_skipped = 0
        problematic_ranges_skipped = 0

        # Add progress bar for frame extraction within each directory
        for i, ts_idx in enumerate(
            tqdm(frame_indices, desc=f"Extracting {md_dir}", leave=False)
        ):
            # Check if we should skip this file
            pdb_path = pdb_files[i]

            # Skip busy files - check if the file exists and is busy
            # If process_all is true, keep retrying with backoff instead of skipping
            if is_file_busy(pdb_path):
                if process_all:
                    # Try multiple times with backoff
                    for attempt in range(5):  # Try up to 5 times
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        if not is_file_busy(pdb_path):
                            break  # File is no longer busy
                    # If still busy after retries, we'll try to process anyway
                else:
                    # Standard behavior: skip busy files
                    busy_files_skipped += 1
                    extracted_frames += 1
                    continue

            # Check frame number against problematic ranges - never skip if process_all is true
            if not process_all:
                is_in_skip_range = False
                for skip_start, skip_end in skip_ranges:
                    if skip_start <= ts_idx <= skip_end:
                        is_in_skip_range = True
                        problematic_ranges_skipped += 1
                        extracted_frames += 1
                        break

                if is_in_skip_range:
                    continue

            try:
                # Jump to the specified frame
                try:
                    u.trajectory[ts_idx]
                except (IOError, ValueError) as e:
                    print(f"Error accessing frame {ts_idx}: {str(e)}")
                    error_frames_skipped += 1
                    continue

                frame_num = u.trajectory.frame

                # Get coordinates for this frame
                if not hasattr(u, "atoms"):
                    print(f"Warning: No atoms attribute for frame {frame_num}.")
                    continue

                try:
                    positions = getattr(u.atoms, "positions", None)
                    if positions is None:
                        print(
                            f"Warning: Could not access atom positions for frame {frame_num}."
                        )
                        continue

                    coordinates = (
                        positions.copy()
                    )  # Make a copy to avoid reference issues
                except (AttributeError, IOError, ValueError) as e:
                    print(
                        f"Warning: Error accessing atom positions for frame {frame_num}: {e}"
                    )
                    continue

                # Safety check for frames near the end of trajectory
                if ts_idx > total_frames - 20:
                    # Check if the coordinates look valid
                    if np.isnan(coordinates).any() or np.isinf(coordinates).any():
                        print(
                            f"Warning: Invalid coordinates in frame {frame_num}, skipping."
                        )
                        error_frames_skipped += 1
                        continue

                    # Check if coordinates are within reasonable bounds
                    if np.max(np.abs(coordinates)) > 999:
                        print(
                            f"Warning: Extreme coordinates in frame {frame_num}, skipping."
                        )
                        error_frames_skipped += 1
                        continue

                # Update the PDB template with coordinates
                updated_atom_lines = update_pdb_coordinates(atom_lines, coordinates)

                # Write PDB file for this frame using safe write method
                file_content = ""
                # Build header
                file_content += (
                    f"TITLE     Frame {frame_num} from {os.path.basename(xtc_file)}\n"
                )
                file_content += f"REMARK    EXTRACTED FROM TRAJECTORY USING TEMPLATE-BASED APPROACH\n"

                # Add ATOM records
                for line in updated_atom_lines:
                    file_content += line + "\n"

                # Add CONECT records
                for line in conect_lines:
                    file_content += line + "\n"

                # Add END
                file_content += "END\n"

                # Write using safer method
                if safe_write_file(pdb_path, file_content):
                    extracted_frames += 1
                else:
                    busy_files_skipped += 1

            except Exception as e:
                print(f"Error processing frame {ts_idx}: {str(e)}")
                error_frames_skipped += 1
                continue

        # Report on skipped frames
        if busy_files_skipped > 0:
            print(f"Skipped {busy_files_skipped} busy files in {md_dir}")
        if problematic_ranges_skipped > 0:
            print(
                f"Skipped {problematic_ranges_skipped} frames in problematic ranges for {md_dir}"
            )
        if error_frames_skipped > 0:
            print(
                f"Skipped {error_frames_skipped} frames due to processing errors in {md_dir}"
            )

        # Clean up Universe object to free memory
        del u

        if extracted_frames == 0:
            return (md_dir, False, 0, "No frames were successfully extracted")

        # Create a simple script to load all frames in ChimeraX
        chimera_script = os.path.join(output_dir, "open_in_chimerax.cxc")
        chimera_content = "# ChimeraX script to load and view frames\n"
        chimera_content += f"open {output_dir}/frame_*.pdb\n"

        if is_cyclic:
            # If cyclic, add command to connect first and last residue
            chimera_content += (
                "\n# This appears to be a cyclic structure. Uncomment to add bond:\n"
            )
            chimera_content += "# select #1/1:{1,end}\n"
            chimera_content += "# bond sel relativeLength 1.4\n"

        # Add movie commands
        chimera_content += "\n# To view as movie:\n"
        chimera_content += (
            "coordset #1 play direction forward loop true maxFrameRate 15\n"
        )

        # Write chimera script using safer method
        safe_write_file(chimera_script, chimera_content)

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
        "root_dir", help="Root directory containing MD files or directories"
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
        "--inner-processes",
        "-i",
        type=int,
        default=0,
        help="Number of processes to use for frame extraction within each trajectory (0=auto)",
    )
    parser.add_argument(
        "--fast",
        "-f",
        action="store_true",
        help="Use optimized frame-level parallelism for faster extraction",
    )
    parser.add_argument(
        "--buffer-size",
        "-b",
        type=int,
        default=0,
        help="Memory buffer size in MB (0=auto)",
    )
    parser.add_argument(
        "--safe",
        "-s",
        action="store_true",
        help="Skip frames in last 5%% of trajectory to avoid common errors",
    )
    parser.add_argument(
        "--no-offsets",
        "-n",
        action="store_true",
        help="Disable XTC offset building (for network storage)",
    )
    parser.add_argument(
        "--skip-range",
        type=str,
        default="",
        help="Skip frame ranges known to cause issues (format: 'start-stop,start-stop')",
    )
    parser.add_argument(
        "--no-skip-range",
        action="store_true",
        help="Don't automatically skip problematic frames (3700-3999)",
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

    # Add wider skip range option
    parser.add_argument(
        "--wider-skip-range",
        action="store_true",
        help="Use a wider range for skipping problematic frames (3500-3999 instead of default 3700-3999)",
    )

    # Add process-all option
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Process all frames without skipping any frames, even problematic ones (3700-3999)",
    )

    # Add new diagnostic argument
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run diagnostics on the output directory to find busy files and processes",
    )

    # Add no-locks option for even more aggressive lock prevention
    parser.add_argument(
        "--no-locks",
        action="store_true",
        help="Use even more aggressive methods to prevent lock file creation (may impact performance)",
    )

    # Add option to create a new directory each time
    parser.add_argument(
        "--new-dir",
        action="store_true",
        help="Create a new output directory with timestamp suffix instead of reusing pdb_frames",
    )

    args = parser.parse_args()

    # Apply additional lock prevention if requested
    if args.no_locks:
        print("Applying additional lock prevention measures")
        # Delete any existing lock files in /tmp
        try:
            import glob
            import os

            for lock_file in glob.glob("/tmp/MDA_*.lock"):
                try:
                    os.remove(lock_file)
                    print(f"Removed lock file: {lock_file}")
                except:
                    pass
        except:
            pass

        # Modify the MDAnalysis configuration if possible
        try:
            import MDAnalysis as mda

            # Try different approaches to disable locks
            if hasattr(mda, "config"):
                mda.config["use_lock"] = False
                mda.config["use_locks"] = False
                print("Disabled locks in MDAnalysis configuration")
        except:
            pass

    # Parse skip ranges if provided
    skip_ranges = []
    if args.skip_range:
        try:
            for range_str in args.skip_range.split(","):
                if "-" in range_str:
                    start, stop = map(int, range_str.split("-"))
                    skip_ranges.append((start, stop))
        except ValueError:
            print(f"Warning: Invalid skip range format: {args.skip_range}")
            print("Using format: 'start-stop,start-stop'")

    # Add default problematic frame range if none specified and not explicitly disabled
    if not skip_ranges and not args.no_skip_range:
        skip_ranges.append((3700, 3999))  # Common problematic frame range
        print("Using default skip range: 3700-3999 (common problematic frames)")
    elif skip_ranges:
        print(f"Will skip the following frame ranges: {skip_ranges}")

    # Set up skip_ranges for problematic frames - process-all takes precedence
    skip_ranges = []
    if args.process_all:
        print("Processing ALL frames without skipping any ranges")
    elif not args.no_skip_range:
        if args.wider_skip_range:
            # Use a wider range to avoid more problematic frames
            skip_start = 3500
            skip_end = 3999
            print(
                f"Using wider skip range: {skip_start}-{skip_end} (common problematic frames)"
            )
        else:
            # Use default range
            skip_start = 3700
            skip_end = 3999
            print(
                f"Using default skip range: {skip_start}-{skip_end} (common problematic frames)"
            )

        skip_ranges.append((skip_start, skip_end))

    # Run diagnostics if requested
    if args.diagnose:
        # If there's an output directory specified, diagnose that
        if args.output:
            diagnose_filesystem(args.output)
        else:
            # Otherwise check the default output directory in the first MD directory
            md_dirs = find_md_directories(args.root_dir, args.recursive)
            if md_dirs:
                for md_dir in md_dirs:
                    output_dir = os.path.join(md_dir, "pdb_frames")
                    diagnose_filesystem(output_dir)
                    # Only diagnose the first directory
                    break
            else:
                print("No MD directories found to diagnose")
        # Exit after diagnostics
        return

    start_time = time.time()

    print(f"Using template-based approach with {args.processes} directory processes")

    # Define inner_processes here to ensure it's always defined
    inner_processes = 0

    # Configure inner processes for frame-level parallelism
    if args.fast:
        inner_processes = args.inner_processes
        if inner_processes <= 0:
            # Auto-configure: use 2 processes for frame extraction by default
            inner_processes = 2
        print(f"Fast mode enabled: Using {inner_processes} processes per trajectory")

    # Configure memory buffer
    buffer_size_mb = args.buffer_size
    if buffer_size_mb <= 0:
        # Auto-configure based on available memory
        try:
            import psutil

            mem = psutil.virtual_memory()
            # Use 10% of available memory
            buffer_size_mb = int(mem.available / (1024 * 1024) * 0.1)
            # Cap at 1GB
            buffer_size_mb = min(buffer_size_mb, 1024)
        except ImportError:
            # Default to 100MB if psutil not available
            buffer_size_mb = 100

    # Print optimized configuration
    if args.fast:
        print(f"Memory buffer size: {buffer_size_mb}MB")

    if args.no_offsets:
        print("Network storage mode: XTC offset building disabled")

    # Define safe stop percentage even if not used
    safe_stop_percentage = 0.95  # Skip the last 5% of frames

    # If safe mode is enabled, adjust stop parameter
    if args.safe and args.stop is None:
        print("Safe mode enabled: Will skip frames in last 5% of trajectories")

    # Find all directories with MD files
    print(f"Searching for MD files in {args.root_dir}...")
    md_dirs = find_md_directories(args.root_dir, args.recursive)

    if not md_dirs:
        print(f"No directories with MD files found in {args.root_dir}")
        return

    print(f"Found {len(md_dirs)} directories with MD files")

    # Prepare arguments for parallel processing
    process_args = []
    for md_dir in md_dirs:
        # Define output directory
        if args.output:
            # If user specified a base output directory, create subdirectories based on input structure
            rel_path = os.path.relpath(md_dir, args.root_dir)
            if rel_path == ".":
                # If processing the base directory itself
                output_dir = args.output
            else:
                # If processing a subdirectory
                output_dir = os.path.join(args.output, rel_path)
        else:
            # Use default directory name
            base_name = "pdb_frames"

            # If new-dir option is enabled, add timestamp
            if args.new_dir:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:6]
                output_dir = os.path.join(
                    md_dir, f"{base_name}_{timestamp}_{unique_id}"
                )
                print(f"Will create new directory: {os.path.basename(output_dir)}")
            else:
                # Use default (pdb_frames in each directory)
                output_dir = os.path.join(md_dir, base_name)

        # If safe mode is enabled, calculate stop frame for this directory
        safe_stop = args.stop
        if args.safe and args.stop is None:
            try:
                # Try to determine trajectory length to set safe stop point
                # Use no_offsets mode for reading length to avoid lock errors
                u = mda.Universe(
                    str(list(Path(md_dir).glob("*.gro"))[0]),
                    str(list(Path(md_dir).glob("*.xtc"))[0]),
                    refresh_offsets=False,  # Always disable offset building for length check
                    in_memory=False,
                    is_periodic=True,
                )
                total_frames = len(u.trajectory)
                safe_stop = int(total_frames * safe_stop_percentage)
                print(
                    f"Safe mode: For {md_dir}, processing frames 0-{safe_stop} out of {total_frames}"
                )

                # Clean up Universe object to free memory
                del u
            except Exception as e:
                print(f"Could not determine trajectory length for {md_dir}: {e}")
                # Fall back to default behavior
                safe_stop = args.stop

        # Add process_all parameter to args tuple
        args_tuple = (
            md_dir,
            output_dir,
            args.start,
            safe_stop,
            args.step,
            args.force,
            inner_processes if args.fast else None,
            args.no_offsets,
            skip_ranges,
            args.process_all,  # Add this parameter
        )
        process_args.append(args_tuple)

    # Process directories in parallel
    start_dir_time = time.time()
    results = []

    # Choose which function to use for processing based on fast mode
    process_func = (
        extract_frames_parallel if args.fast else extract_frames_for_directory
    )

    # Add a lock for managing access to shared resources
    lock = multiprocessing.Manager().Lock()

    # Set a larger timeout for network operations
    timeout = 600  # 10 minutes

    # Initialize pool at the module level to ensure it's accessible in exception handlers
    pool = None

    try:
        pool = multiprocessing.Pool(processes=args.processes)
        # Use a more resilient approach with explicit progress tracking
        results = []
        total = len(process_args)
        completed = 0

        with tqdm(total=total, desc="Processing directories") as pbar:
            # Start workers
            result_iter = pool.imap_unordered(process_func, process_args)

            # Process as results come in
            while completed < total:
                try:
                    # Try to get next result with timeout
                    result = result_iter.next(timeout=10)
                    dir_path, success, n_frames, error = result
                    results.append(result)

                    # Update progress and display
                    if success:
                        print(f"✓ {dir_path}: Extracted {n_frames} frames")
                    else:
                        print(f"✗ {dir_path}: Error - {error}")

                    # Force flush stdout to ensure progress is shown
                    sys.stdout.flush()

                    # Update progress bar
                    completed += 1
                    pbar.update(1)

                except multiprocessing.TimeoutError:
                    # Just continue waiting
                    continue
                except StopIteration:
                    # No more results
                    break
                except KeyboardInterrupt:
                    # Handle keyboard interrupt more gracefully
                    print(
                        "\nInterrupted by user. Waiting for current tasks to complete..."
                    )
                    # Wait for current workers to finish (with timeout)
                    if pool:
                        pool.close()
                        try:
                            pool.join()
                        except:
                            pass
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Partial results may have been saved.")
        if pool:
            try:
                pool.terminate()
                pool.join()
            except:
                pass
    except Exception as e:
        print(f"\nAn error occurred during processing: {str(e)}")
        if pool:
            try:
                pool.terminate()
                pool.join()
            except:
                pass

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


def extract_frame_parallel(args):
    """
    Process a single frame in parallel.

    Args:
        args: Tuple containing (frame_idx, universe, atom_lines, conect_lines,
              output_dir, xtc_filename, total_frames)

    Returns:
        tuple: (frame_num, success) where success is a boolean
    """
    (
        ts_idx,
        universe,
        atom_lines,
        conect_lines,
        output_dir,
        xtc_filename,
        total_frames,
    ) = args

    try:
        # Jump to the specified frame
        try:
            universe.trajectory[ts_idx]
        except (IOError, ValueError) as e:
            return (ts_idx, False)

        frame_num = universe.trajectory.frame

        # Get coordinates for this frame
        if not hasattr(universe, "atoms"):
            return (frame_num, False)

        try:
            positions = getattr(universe.atoms, "positions", None)
            if positions is None:
                return (frame_num, False)

            coordinates = positions.copy()  # Make a copy to avoid reference issues
        except (AttributeError, IOError, ValueError) as e:
            return (frame_num, False)

        # Safety check for frames near the end of trajectory
        if ts_idx > total_frames - 20:
            # Check if the coordinates look valid
            if np.isnan(coordinates).any() or np.isinf(coordinates).any():
                return (frame_num, False)

            # Check if coordinates are within reasonable bounds
            if np.max(np.abs(coordinates)) > 999:
                return (frame_num, False)

        # Update the PDB template with coordinates
        updated_atom_lines = update_pdb_coordinates(atom_lines, coordinates)

        # Write PDB file for this frame
        pdb_path = os.path.join(output_dir, f"frame_{frame_num}.pdb")
        with open(pdb_path, "w") as f:
            # Write header
            f.write(
                f"TITLE     Frame {frame_num} from {os.path.basename(xtc_filename)}\n"
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

            # Add END
            f.write("END\n")

        return (frame_num, True)

    except Exception as e:
        return (ts_idx, False)


def extract_frames_parallel(args):
    """
    Extract frames from a trajectory file using a template-based approach with
    frame-level parallelism for faster processing.

    Args:
        args: Tuple containing (md_dir, output_dir, start, stop, step, force, inner_processes, no_offsets, skip_ranges, process_all)

    Returns:
        tuple: (md_dir, success, n_frames, error_message)
    """
    (
        md_dir,
        output_dir,
        start,
        stop,
        step,
        force,
        inner_processes,
        no_offsets,
        skip_ranges,
        process_all,
    ) = args

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

        # Handle output directory with improved network drive support
        output_path = Path(output_dir)
        directory_success, directory_message, output_path = handle_output_directory(
            output_path, force
        )

        if not directory_success:
            return (md_dir, False, 0, directory_message)

        output_dir = str(output_path)  # Update output_dir to possibly modified path

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
        # Configure MDAnalysis options for network share compatibility
        if no_offsets:
            # Try to load without using offsets (needed for network shares)
            u = mda.Universe(
                gro_file,
                xtc_file,
                in_memory=False,
                refresh_offsets=False,
                dt="guess",
                is_periodic=True,
            )
        else:
            # Regular loading with offsets
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
        frame_indices = list(range(start, stop, step))

        # Filter out frames in skip ranges - but only if not using process_all
        if skip_ranges and not process_all:
            filtered_indices = []
            for idx in frame_indices:
                skip_this_frame = False
                for skip_start, skip_end in skip_ranges:
                    if skip_start <= idx <= skip_end:
                        skip_this_frame = True
                        break
                if not skip_this_frame:
                    filtered_indices.append(idx)

            skipped_count = len(frame_indices) - len(filtered_indices)
            if skipped_count > 0:
                print(
                    f"Skipped {skipped_count} frames in specified ranges for {md_dir}"
                )
            frame_indices = filtered_indices

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

        # Process frames in chunks to avoid creating nested process pools
        # which is not allowed with multiprocessing
        print(f"Processing {n_frames} frames with {inner_processes} worker threads...")

        # Use threading instead of multiprocessing to avoid daemonic process issue
        import threading
        import queue

        # Create a queue for results and a queue for tasks
        result_queue = queue.Queue()
        task_queue = queue.Queue()

        # Put all tasks in the queue
        for ts_idx in frame_indices:
            task_queue.put(ts_idx)

        # Track skipped frames for better reporting
        busy_files_skipped = multiprocessing.Value("i", 0)
        error_frames_skipped = multiprocessing.Value("i", 0)
        problematic_ranges_skipped = multiprocessing.Value("i", 0)

        # Define a worker function with improved file handling
        def worker():
            while not task_queue.empty():
                try:
                    # Get a task
                    ts_idx = task_queue.get(block=False)

                    # Check frame number against problematic ranges - but only if not processing all
                    if not process_all:
                        is_in_skip_range = False
                        for skip_start, skip_end in skip_ranges:
                            if skip_start <= ts_idx <= skip_end:
                                is_in_skip_range = True
                                with problematic_ranges_skipped.get_lock():
                                    problematic_ranges_skipped.value += 1
                                result_queue.put((ts_idx, True))
                                task_queue.task_done()
                                break

                        if is_in_skip_range:
                            continue

                    # Process the frame
                    try:
                        # Jump to the specified frame
                        try:
                            u.trajectory[ts_idx]
                        except (IOError, ValueError) as e:
                            with error_frames_skipped.get_lock():
                                error_frames_skipped.value += 1
                            result_queue.put((ts_idx, False))
                            task_queue.task_done()
                            continue

                        frame_num = u.trajectory.frame

                        # Check if file exists and is busy - if process_all, retry instead of skip
                        pdb_path = os.path.join(output_dir, f"frame_{frame_num}.pdb")
                        if is_file_busy(pdb_path):
                            if process_all:
                                # Try multiple times with backoff
                                for attempt in range(5):  # Try up to 5 times
                                    time.sleep(
                                        0.5 * (attempt + 1)
                                    )  # Exponential backoff
                                    if not is_file_busy(pdb_path):
                                        break  # File is no longer busy
                                # If still busy after retries, we'll try to process anyway
                            else:
                                # Standard behavior: skip busy files
                                with busy_files_skipped.get_lock():
                                    busy_files_skipped.value += 1
                                result_queue.put((frame_num, True))
                                task_queue.task_done()
                                continue

                        # Get coordinates for this frame
                        if not hasattr(u, "atoms"):
                            result_queue.put((frame_num, False))
                            task_queue.task_done()
                            continue

                        try:
                            positions = getattr(u.atoms, "positions", None)
                            if positions is None:
                                result_queue.put((frame_num, False))
                                task_queue.task_done()
                                continue

                            coordinates = (
                                positions.copy()
                            )  # Make a copy to avoid reference issues
                        except (AttributeError, IOError, ValueError) as e:
                            result_queue.put((frame_num, False))
                            task_queue.task_done()
                            continue

                        # Safety check for frames near the end of trajectory
                        if ts_idx > total_frames - 20:
                            # Additional validation for frames near the end
                            if (
                                np.isnan(coordinates).any()
                                or np.isinf(coordinates).any()
                            ):
                                with error_frames_skipped.get_lock():
                                    error_frames_skipped.value += 1
                                result_queue.put((frame_num, False))
                                task_queue.task_done()
                                continue

                            # Check if coordinates are within reasonable bounds
                            if np.max(np.abs(coordinates)) > 999:
                                with error_frames_skipped.get_lock():
                                    error_frames_skipped.value += 1
                                result_queue.put((frame_num, False))
                                task_queue.task_done()
                                continue

                        # Update the PDB template with coordinates
                        updated_atom_lines = update_pdb_coordinates(
                            atom_lines, coordinates
                        )

                        # Write PDB file for this frame using safe write method
                        file_content = ""
                        # Build header
                        file_content += f"TITLE     Frame {frame_num} from {os.path.basename(xtc_file)}\n"
                        file_content += f"REMARK    EXTRACTED FROM TRAJECTORY USING TEMPLATE-BASED APPROACH\n"

                        # Add ATOM records
                        for line in updated_atom_lines:
                            file_content += line + "\n"

                        # Add CONECT records
                        for line in conect_lines:
                            file_content += line + "\n"

                        # Add END
                        file_content += "END\n"

                        # Write using safer method
                        if safe_write_file(pdb_path, file_content):
                            result_queue.put((frame_num, True))
                        else:
                            # If write failed, it's likely busy - count as skipped
                            with busy_files_skipped.get_lock():
                                busy_files_skipped.value += 1
                            # We still count this as processed to avoid retries
                            result_queue.put((frame_num, True))

                    except Exception as e:
                        with error_frames_skipped.get_lock():
                            error_frames_skipped.value += 1
                        result_queue.put((ts_idx, False))

                    finally:
                        # Mark the task as done
                        task_queue.task_done()

                except queue.Empty:
                    break

        # Create and start worker threads
        threads = []
        for _ in range(inner_processes):
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()
            threads.append(t)

        # Setup progress bar
        progress_bar = tqdm(
            total=n_frames, desc=f"Extracting frames from {md_dir}", leave=False
        )

        # Process results as they come in
        extracted_frames = 0
        completed = 0

        # Wait for all tasks to be processed
        while completed < n_frames:
            try:
                # Get a result (with timeout to allow checking if all threads died)
                frame_num, success = result_queue.get(timeout=0.1)
                if success:
                    extracted_frames += 1
                completed += 1
                progress_bar.update(1)
            except queue.Empty:
                # Check if all threads are dead
                if all(not t.is_alive() for t in threads):
                    # If all threads died but we haven't processed all frames, there was an error
                    if completed < n_frames:
                        print(
                            f"Warning: All worker threads died before completing all frames."
                        )
                        break

        # Close progress bar
        progress_bar.close()

        # Clean up Universe object to free memory
        del u

        # After processing, report on skipped frames
        if busy_files_skipped.value > 0:
            print(f"Skipped {busy_files_skipped.value} busy files in {md_dir}")
        if problematic_ranges_skipped.value > 0:
            print(
                f"Skipped {problematic_ranges_skipped.value} frames in problematic ranges for {md_dir}"
            )
        if error_frames_skipped.value > 0:
            print(
                f"Skipped {error_frames_skipped.value} frames due to processing errors in {md_dir}"
            )

        if extracted_frames == 0:
            return (md_dir, False, 0, "No frames were successfully extracted")

        # Create a simple script to load all frames in ChimeraX
        chimera_script = os.path.join(output_dir, "open_in_chimerax.cxc")
        chimera_content = "# ChimeraX script to load and view frames\n"
        chimera_content += f"open {output_dir}/frame_*.pdb\n"

        if is_cyclic:
            # If cyclic, add command to connect first and last residue
            chimera_content += (
                "\n# This appears to be a cyclic structure. Uncomment to add bond:\n"
            )
            chimera_content += "# select #1/1:{1,end}\n"
            chimera_content += "# bond sel relativeLength 1.4\n"

        # Add movie commands
        chimera_content += "\n# To view as movie:\n"
        chimera_content += (
            "coordset #1 play direction forward loop true maxFrameRate 15\n"
        )

        # Write chimera script using safer method
        safe_write_file(chimera_script, chimera_content)

        return (md_dir, True, extracted_frames, "")

    except Exception as e:
        return (md_dir, False, 0, str(e))


# Function to diagnose file system access issues
def diagnose_filesystem(dir_path):
    """
    Diagnose filesystem access issues.

    Args:
        dir_path (str or Path): Path to directory to diagnose
    """
    dir_path = Path(dir_path)

    print("\n========== FILESYSTEM DIAGNOSIS ==========")
    print(f"Checking directory: {dir_path}")

    # Check if directory exists
    if not dir_path.exists():
        print(f"Directory does not exist: {dir_path}")
        return

    # Check if directory is accessible
    try:
        files = list(dir_path.glob("*"))
        print(f"Directory is accessible, contains {len(files)} files/dirs")
    except Exception as e:
        print(f"Directory is not accessible: {e}")
        return

    # Check for busy files
    busy_files = []
    for file in dir_path.glob("frame_*.pdb"):
        if is_file_busy(file):
            busy_files.append(file.name)

    if busy_files:
        print(f"Found {len(busy_files)} busy files:")
        for i, filename in enumerate(busy_files[:10]):  # Show first 10
            print(f"  {i+1}. {filename}")
        if len(busy_files) > 10:
            print(f"  ... and {len(busy_files) - 10} more")
    else:
        print("No busy files found in directory")

    # Try to find processes accessing the directory on MacOS
    try:
        print("\nAttempting to identify processes accessing the directory...")
        # Run lsof command to find processes
        cmd = ["lsof", "+D", str(dir_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

        if result.stdout.strip():
            print("Processes accessing the directory:")
            print(result.stdout)
        else:
            print("No processes found accessing the directory via lsof")

        # Check for MDAnalysis lock files
        lock_files = list(Path("/tmp").glob("*.lock"))
        mda_locks = [f for f in lock_files if "MDA_" in f.name]
        if mda_locks:
            print(f"\nFound {len(mda_locks)} MDAnalysis lock files in /tmp:")
            for lock in mda_locks[:5]:  # Show first 5
                print(f"  {lock}")
            if len(mda_locks) > 5:
                print(f"  ... and {len(mda_locks) - 5} more")
    except Exception as e:
        print(f"Error running diagnostics: {e}")

    print("\nRecommendations:")
    print("1. If there are busy files, try running:")
    print("   pkill -9 python")
    print("2. To clear MDAnalysis lock files:")
    print("   rm /tmp/MDA_*.lock")
    print("3. Restart Finder if using macOS:")
    print("   killall Finder")
    print("4. Try unmounting and remounting the network volume")

    print("===========================================\n")


if __name__ == "__main__":
    main()
