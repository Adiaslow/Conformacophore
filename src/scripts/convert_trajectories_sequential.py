#!/usr/bin/env python3
# src/scripts/convert_trajectories_sequential.py

"""
Script to convert molecular dynamics trajectories between different formats sequentially.
Currently supports converting XTC to multi-model PDB files using GROMACS.
"""

import os
import argparse
import logging
from pathlib import Path
from tqdm.auto import tqdm
import warnings
from typing import List
import json
import tempfile

# Disable MDAnalysis file locking
os.environ["MDANALYSIS_NO_FD_CACHE"] = "1"

from src.core.services.gmx_converter import GromacsPDBConverter


def check_conversion_summary(output_dir: Path) -> bool:
    """
    Check if a directory has been successfully converted by examining the summary file.

    Args:
        output_dir: Directory to check for conversion summary

    Returns:
        True if directory has been successfully converted, False otherwise
    """
    summary_file = output_dir / "conversion_summary.json"
    pdb_file = output_dir / "md_Ref.pdb"

    if not (
        summary_file.exists() and pdb_file.exists() and pdb_file.stat().st_size > 0
    ):
        return False

    try:
        with open(summary_file) as f:
            summary = json.load(f)

        # Check for required fields
        required_fields = ["n_frames", "n_atoms", "timestamp", "conversion_version"]
        if not all(field in summary for field in required_fields):
            return False

        # Check if the PDB file size is reasonable given the number of frames and atoms
        expected_min_size = (
            summary["n_frames"] * summary["n_atoms"] * 50
        )  # Rough estimate of minimum expected size
        if pdb_file.stat().st_size < expected_min_size:
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
    import datetime
    import json

    summary = {
        "input_xtc": str(xtc_path),
        "timestamp": datetime.datetime.now().isoformat(),
        "n_frames": n_frames,
        "n_atoms": n_atoms,
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
    force: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Process a single trajectory file.

    Args:
        xtc_path: Path to XTC trajectory file
        gro_path: Path to GRO topology file
        top_path: Path to TOP topology file
        output_base: Base directory for output
        force: Whether to force reprocessing
        verbose: Whether to show detailed output

    Returns:
        True if processing was needed and successful, False if skipped
    """
    # Use the directory containing the XTC file as the output directory
    output_dir = xtc_path.parent

    # Skip if already processed successfully
    if not force and check_conversion_summary(output_dir):
        logging.info(f"Skipping {xtc_path} - already successfully converted")
        return False

    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory(prefix="traj_convert_") as temp_dir:
        temp_dir_path = Path(temp_dir)

        try:
            # Initialize converter
            converter = GromacsPDBConverter(verbose=verbose)

            # Get trajectory information first
            traj_info = converter.get_trajectory_info(
                str(xtc_path), str(gro_path), str(top_path)
            )
            n_frames = traj_info.get("n_frames", 0)
            n_atoms = traj_info.get("n_atoms", 0)

            # Write initial conversion summary
            write_conversion_summary(
                output_dir,
                str(xtc_path),
                n_frames=n_frames,
                n_atoms=n_atoms,
            )

            # Convert trajectory
            output_files = converter.convert_trajectory(
                xtc_path=str(xtc_path),
                gro_path=str(gro_path),
                top_path=str(top_path),
                output_dir=str(output_dir),
                start=0,
                end=None,
                stride=1,
            )

            # Update conversion summary with completion timestamp
            write_conversion_summary(
                output_dir,
                str(xtc_path),
                n_frames=n_frames,
                n_atoms=n_atoms,
            )

            return True

        except Exception as e:
            # Clean up partial output on error
            try:
                for file in output_dir.glob("conversion_summary.json"):
                    file.unlink()
                for file in output_dir.glob("md_Ref.pdb"):
                    file.unlink()
            except:
                pass
            raise  # Re-raise the exception after cleanup


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

    # Set up file logging only
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )

    # Disable MDAnalysis warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Find all trajectory sets
    if not args.quiet:
        print("Scanning for trajectory files...")
    trajectory_sets = []

    for xtc_file in base_dir.rglob("300K_fit_4000.xtc"):  # Only look for 4000.xtc files
        # Skip macOS hidden files
        if xtc_file.name.startswith("._"):
            continue

        gro_file = xtc_file.parent / "md_Ref.gro"
        top_file = xtc_file.parent / "topol.top"

        if gro_file.exists() and top_file.exists():
            trajectory_sets.append((xtc_file, gro_file, top_file))

    if not trajectory_sets:
        if not args.quiet:
            print(f"No valid 300K_fit_4000.xtc trajectory sets found in {base_dir}")
        return

    if not args.quiet:
        print(f"Found {len(trajectory_sets)} trajectory sets to process")

    # Process trajectories sequentially with progress bar
    errors = []
    processed = 0
    skipped = 0

    with tqdm(
        total=len(trajectory_sets),
        desc="Converting trajectories (processed=0, skipped=0, errors=0)",
        unit="traj",
        disable=args.quiet,
        ncols=120,
    ) as pbar:
        for xtc_path, gro_path, top_path in trajectory_sets:
            try:
                if process_trajectory(
                    xtc_path,
                    gro_path,
                    top_path,
                    base_dir,
                    force=args.force,
                    verbose=args.verbose,
                ):
                    processed += 1
                else:
                    skipped += 1
            except Exception as e:
                error_msg = f"Error processing {xtc_path}: {str(e)}"
                errors.append(error_msg)
                logging.error(error_msg)
            finally:
                # Update progress bar description with current counts
                pbar.set_description(
                    f"Converting trajectories (processed={processed}, skipped={skipped}, errors={len(errors)})"
                )
                pbar.update(1)

    # Only show summary if verbose or there were errors
    if args.verbose or errors:
        print(f"\nProcessing complete:")
        print(f"- Successfully processed: {processed}")
        print(f"- Skipped (already converted): {skipped}")
        print(f"- Errors: {len(errors)}")

        if errors:
            print(f"\nLog file location: {log_file.absolute()}")
            if args.verbose:
                print("\nFirst few errors:")
                for error in errors[:5]:  # Show first 5 errors
                    print(f"- {error}")
                if len(errors) > 5:
                    print(
                        f"... and {len(errors) - 5} more errors. See log file for complete details."
                    )


if __name__ == "__main__":
    main()
