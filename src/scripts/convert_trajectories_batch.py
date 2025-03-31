#!/usr/bin/env python3
# src/scripts/batch_convert_trajectories.py

"""
Script to convert molecular dynamics trajectories between different formats.
Currently supports converting XTC to multi-model PDB files using GROMACS.
"""

import os
import shutil
import tempfile
import argparse
import logging
import json
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm.auto import tqdm
import datetime

from src.core.services.gmx_converter import GromacsPDBConverter


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_progress(progress_file: Path) -> Set[str]:
    """Load set of successfully processed trajectories."""
    if progress_file.exists():
        with open(progress_file, "r") as f:
            return set(json.load(f))
    return set()


def save_progress(progress_file: Path, processed: Set[str]) -> None:
    """Save set of successfully processed trajectories."""
    with open(progress_file, "w") as f:
        json.dump(list(processed), f)


def copy_to_temp(file_path: str) -> tuple[Path, Path]:
    """Copy a file to a temporary directory and return both paths."""
    src_path = Path(file_path)
    temp_dir = Path(tempfile.mkdtemp(prefix="traj_convert_"))
    temp_path = temp_dir / src_path.name

    # Read source file in binary mode and write to destination
    with open(src_path, "rb") as src, open(temp_path, "wb") as dst:
        dst.write(src.read())

    return src_path, temp_path


def is_already_processed(output_dir: Path, xtc_path: str) -> bool:
    """
    Check if the trajectory has already been processed successfully.

    Args:
        output_dir: Directory containing output files
        xtc_path: Path to the input XTC file

    Returns:
        bool: True if both PDB and summary files exist and are valid
    """
    pdb_file = output_dir / "md_Ref.pdb"
    summary_file = output_dir / "conversion_summary.json"

    # Check if both files exist
    if not (pdb_file.exists() and summary_file.exists()):
        return False

    try:
        # Check if summary file is valid and matches current input
        with open(summary_file, "r") as f:
            summary = json.load(f)

        # Verify summary contains expected information
        if not all(
            key in summary for key in ["input_xtc", "timestamp", "n_frames", "n_atoms"]
        ):
            return False

        # Check if the summary matches the current input file
        if summary["input_xtc"] != str(xtc_path):
            return False

        # Optionally, verify PDB file size is non-zero
        if pdb_file.stat().st_size == 0:
            return False

        return True

    except (json.JSONDecodeError, KeyError, OSError):
        return False


def write_conversion_summary(
    output_dir: Path,
    xtc_path: str,
    n_frames: int,
    n_atoms: int,
) -> None:
    """
    Write a summary of the conversion process.

    Args:
        output_dir: Directory to write summary file
        xtc_path: Path to input XTC file
        n_frames: Number of frames processed
        n_atoms: Number of atoms
    """
    summary = {
        "input_xtc": str(xtc_path),
        "timestamp": datetime.datetime.now().isoformat(),
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "conversion_version": "1.0",  # Add version to track format changes
    }

    summary_file = output_dir / "conversion_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


def create_temp_dir() -> Path:
    """Create a temporary directory in the user's home directory."""
    home = Path.home()
    temp_base = home / ".traj_convert_temp"
    temp_base.mkdir(exist_ok=True)
    temp_dir = temp_base / f"traj_{time.time_ns()}"
    temp_dir.mkdir(parents=True)
    return temp_dir


def process_trajectory(args: Tuple[str, str, str, str, bool]) -> Optional[str]:
    """Process a single trajectory file.

    Args:
        args: Tuple containing (xtc_path, topology_path, top_path, output_base, verbose)

    Returns:
        Optional error message if processing failed, None if successful
    """
    xtc_path, topology_path, top_path, output_base, verbose = args

    # Skip macOS hidden files
    if os.path.basename(xtc_path).startswith("._"):
        return None

    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory(prefix="traj_convert_") as temp_dir:
        temp_dir_path = Path(temp_dir)

        try:
            # Use the directory containing the XTC file as the output directory
            output_dir = Path(xtc_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Check if already processed
            if is_already_processed(output_dir, xtc_path):
                return None

            # Initialize converter
            converter = GromacsPDBConverter(verbose=verbose)

            # Get trajectory information first
            traj_info = converter.get_trajectory_info(xtc_path, topology_path, top_path)
            n_frames = traj_info.get("n_frames", 0)
            n_atoms = traj_info.get("n_atoms", 0)

            # Write initial conversion summary
            write_conversion_summary(
                output_dir,
                xtc_path,
                n_frames=n_frames,
                n_atoms=n_atoms,
            )

            # Convert trajectory
            output_files = converter.convert_trajectory(
                xtc_path=xtc_path,
                gro_path=topology_path,
                top_path=top_path,
                output_dir=str(output_dir),
                start=0,
                end=None,
                stride=1,
            )

            return None

        except Exception as e:
            # Clean up partial output on error
            try:
                for file in output_dir.glob("conversion_summary.json"):
                    file.unlink()
                for file in output_dir.glob("md_Ref.pdb"):
                    file.unlink()
            except:
                pass
            return f"Error processing {xtc_path}: {str(e)}"


def main():
    """Main function for batch converting trajectories."""
    parser = argparse.ArgumentParser(
        description="Batch convert XTC trajectories to PDB"
    )
    parser.add_argument(
        "base_dir", type=str, help="Base directory containing trajectory files"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of parallel processes to use",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already converted trajectories",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress non-essential output"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed processing information"
    )

    args = parser.parse_args()

    # Configure logging to file
    log_file = Path("trajectory_conversion.log")

    # Remove any existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure file logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)

    # Set logging level based on verbose flag
    root_logger.setLevel(logging.INFO if args.verbose else logging.ERROR)

    # Set environment variable for child processes
    if args.quiet:
        os.environ["QUIET"] = "1"
    if args.verbose:
        os.environ["VERBOSE"] = "1"

    # Find all trajectory sets
    trajectory_sets = []
    base_dir = Path(args.base_dir)

    print("Scanning for trajectory files...")
    for xtc_file in base_dir.rglob(
        "300K_fit_4000.xtc"
    ):  # Only process 4000-frame files
        # Skip macOS hidden files
        if xtc_file.name.startswith("._"):
            continue

        gro_file = xtc_file.parent / "md_Ref.gro"
        top_file = xtc_file.parent / "topol.top"

        if gro_file.exists() and top_file.exists():
            trajectory_sets.append(
                (
                    str(xtc_file),
                    str(gro_file),
                    str(top_file),
                    str(base_dir),
                    args.verbose,
                )
            )

    if not trajectory_sets:
        print(f"No valid trajectory sets found in {base_dir}")
        return

    total = len(trajectory_sets)
    print(f"Found {total} trajectory sets to process")

    # Initialize progress bar
    with tqdm(
        total=total,
        desc="Converting trajectories",
        unit="traj",
        disable=args.quiet,
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar:
        # Process trajectories in parallel
        with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
            futures = []
            for traj_set in trajectory_sets:
                xtc_path = traj_set[0]
                # Check if already processed in the xtc's parent directory
                if not args.force and is_already_processed(
                    Path(xtc_path).parent, xtc_path
                ):
                    pbar.update(1)
                    continue
                futures.append(executor.submit(process_trajectory, traj_set))

            # Process results as they complete
            errors = []
            for future in as_completed(futures):
                error = future.result()
                if error:
                    errors.append(error)
                pbar.update(1)

    if errors:
        print(f"\nCompleted with {len(errors)} errors. See {log_file} for details.")


if __name__ == "__main__":
    main()
