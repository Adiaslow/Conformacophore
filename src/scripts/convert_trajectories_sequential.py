#!/usr/bin/env python3
# src/scripts/convert_trajectories_sequential.py

"""
Script to convert molecular dynamics trajectories between different formats sequentially.
Currently supports converting XTC to multi-model PDB files.
"""

import os
import argparse
import logging
from pathlib import Path
from tqdm.auto import tqdm
import warnings
from typing import List

# Disable MDAnalysis file locking
os.environ["MDANALYSIS_NO_FD_CACHE"] = "1"

from src.core.services.trajectory_converter import TrajectoryConverter


def write_conversion_summary(
    output_dir: Path,
    xtc_path: str,
    n_frames: int,
    n_atoms: int,
    residue_sequence: List[str],
) -> None:
    """
    Write a summary of the conversion process.

    Args:
        output_dir: Directory to write summary file
        xtc_path: Path to input XTC file
        n_frames: Number of frames processed
        n_atoms: Number of atoms
        residue_sequence: List of residue names
    """
    import datetime
    import json

    summary = {
        "input_xtc": str(xtc_path),
        "timestamp": datetime.datetime.now().isoformat(),
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "residue_sequence": residue_sequence,
        "conversion_version": "1.0",
    }

    summary_file = output_dir / "conversion_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


def process_trajectory(
    xtc_path: Path,
    gro_path: Path,
    top_path: Path,
    output_base: Path,
    verbose: bool = False,
) -> None:
    """
    Process a single trajectory file.

    Args:
        xtc_path: Path to XTC trajectory file
        gro_path: Path to GRO topology file
        top_path: Path to TOP topology file
        output_base: Base directory for output
        verbose: Whether to show detailed output
    """
    # Use the directory containing the XTC file as the output directory
    output_dir = xtc_path.parent

    # Skip if already processed
    pdb_file = output_dir / "md_Ref.pdb"
    summary_file = output_dir / "conversion_summary.json"
    if pdb_file.exists() and pdb_file.stat().st_size > 0 and summary_file.exists():
        return

    # Initialize converter
    converter = TrajectoryConverter(
        topology_file=str(gro_path), top_file=str(top_path), verbose=verbose
    )

    # Get trajectory information first
    traj_info = converter.get_trajectory_info(str(xtc_path))
    n_frames = traj_info.get("n_frames", 0)
    n_atoms = traj_info.get("n_atoms", 0)
    residue_sequence = traj_info.get("residue_sequence", ["UNK"])

    # Write initial conversion summary
    write_conversion_summary(
        output_dir,
        xtc_path,
        n_frames=n_frames,
        n_atoms=n_atoms,
        residue_sequence=residue_sequence,
    )

    # Convert trajectory
    converter.xtc_to_multimodel_pdb(
        str(xtc_path), str(output_dir), start=0, end=None, stride=1
    )

    # Update conversion summary with completion timestamp
    write_conversion_summary(
        output_dir,
        xtc_path,
        n_frames=n_frames,
        n_atoms=n_atoms,
        residue_sequence=residue_sequence,
    )


def main():
    """Main function for sequentially converting trajectories."""
    parser = argparse.ArgumentParser(
        description="Sequentially convert XTC trajectories to PDB"
    )
    parser.add_argument(
        "base_dir", type=str, help="Base directory containing trajectory files"
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
    base_dir = Path(args.base_dir)

    # Configure logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "trajectory_conversion.log"
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if args.verbose else logging.NullHandler(),
        ],
    )

    # Disable MDAnalysis warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Find all trajectory sets
    print("Scanning for trajectory files...")
    trajectory_sets = []

    for xtc_file in base_dir.rglob("*.xtc"):
        # Skip macOS hidden files
        if xtc_file.name.startswith("._"):
            continue

        gro_file = xtc_file.parent / "md_Ref.gro"
        top_file = xtc_file.parent / "topol.top"

        if gro_file.exists() and top_file.exists():
            trajectory_sets.append((xtc_file, gro_file, top_file))

    if not trajectory_sets:
        print(f"No valid trajectory sets found in {base_dir}")
        return

    total = len(trajectory_sets)
    print(f"Found {total} trajectory sets to process")

    # Process trajectories sequentially with progress bar
    errors = []
    with tqdm(
        total=total,
        desc="Converting trajectories",
        unit="traj",
        disable=args.quiet,
        ncols=100,
    ) as pbar:
        for xtc_path, gro_path, top_path in trajectory_sets:
            try:
                process_trajectory(
                    xtc_path, gro_path, top_path, base_dir, verbose=args.verbose
                )
            except Exception as e:
                error_msg = f"Error processing {xtc_path}: {str(e)}"
                errors.append(error_msg)
                logging.error(error_msg)
            finally:
                pbar.update(1)

    if errors:
        print(f"\nCompleted with {len(errors)} errors.")
        print(f"Log file location: {log_file.absolute()}")
        print("\nFirst few errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"- {error}")
        if len(errors) > 5:
            print(
                f"... and {len(errors) - 5} more errors. See log file for complete details."
            )


if __name__ == "__main__":
    main()
