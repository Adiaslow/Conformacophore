#!/usr/bin/env python3
# src/scripts/batch_convert_trajectories.py

"""
Script to convert molecular dynamics trajectories between different formats.
Currently supports converting XTC to multi-model PDB files.
"""

import os
import shutil
import tempfile
import argparse
import logging
import json
from pathlib import Path
from typing import Optional, Dict, List, Set
from concurrent.futures import ProcessPoolExecutor
import time
from tqdm import tqdm

from src.core.services.trajectory_converter import TrajectoryConverter


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


def process_trajectory(args: Dict) -> Dict:
    """Process a single trajectory with local caching and error handling."""
    xtc_path = args["xtc_path"]
    gro_path = args["gro_path"]
    top_path = args.get("top_path")
    output_dir = args["output_dir"]
    max_retries = args.get("max_retries", 3)
    retry_delay = args.get("retry_delay", 1.0)

    logger = setup_logging()
    result = {"success": False, "xtc_path": xtc_path, "error": None}

    try:
        # Copy files to temporary directory
        logger.info(f"Copying files to temporary directory for {xtc_path}...")
        _, temp_xtc = copy_to_temp(xtc_path)
        _, temp_gro = copy_to_temp(gro_path)
        temp_top = None
        if top_path:
            _, temp_top = copy_to_temp(top_path)

        temp_dir = temp_xtc.parent
        logger.info(f"Using temporary directory: {temp_dir}")

        # Create a temporary output directory for local writing
        with tempfile.TemporaryDirectory(prefix="traj_output_") as temp_output_dir:
            temp_output_path = Path(temp_output_dir)
            logger.info(f"Using temporary output directory: {temp_output_path}")

            # Initialize converter with temporary files
            converter = TrajectoryConverter(
                topology_file=str(temp_gro),
                top_file=str(temp_top) if temp_top else None,
            )

            try:
                # Get trajectory info
                logger.info("Getting trajectory information...")
                info = converter.get_trajectory_info(str(temp_xtc))
                logger.info(
                    f"Found {info.get('n_frames', 0)} frames, {info.get('n_atoms', 0)} atoms"
                )

                # Convert trajectory
                logger.info("Converting trajectory...")
                output_files = converter.xtc_to_multimodel_pdb(
                    xtc_path=str(temp_xtc),
                    output_path=str(temp_output_path),
                )

                # Create output directory
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                # Copy output files to final destination
                logger.info("Copying output files to final destination...")
                for temp_file in output_files:
                    dest_file = output_path / temp_file.name
                    try:
                        # Read source file in binary mode and write to destination
                        with open(temp_file, "rb") as src, open(dest_file, "wb") as dst:
                            dst.write(src.read())
                        logger.info(f"  - Successfully copied {dest_file}")
                    except Exception as e:
                        logger.error(
                            f"Error copying {temp_file} to {dest_file}: {str(e)}"
                        )
                        raise

                result["success"] = True

            except Exception as e:
                result["error"] = str(e)
                logger.error(f"Error converting trajectory: {str(e)}")

        # Clean up temporary directory
        logger.info("Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error during file processing: {str(e)}")

    return result


def main():
    """Main function to convert trajectories."""
    logger = setup_logging()

    parser = argparse.ArgumentParser(
        description="Convert molecular dynamics trajectories"
    )
    parser.add_argument(
        "root_dir", type=str, help="Root directory containing trajectory files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be processed without converting",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,  # Reduced default to minimize contention
        help="Number of parallel processes to use for conversion",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed operations",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Delay between retries in seconds",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last successful conversion"
    )

    args = parser.parse_args()

    # Set up progress tracking
    progress_file = Path(args.root_dir) / ".conversion_progress.json"
    processed_trajectories = load_progress(progress_file) if args.resume else set()

    try:
        # Find all trajectory sets
        logger.info(f"Scanning for trajectory sets...")

        trajectory_sets = []
        for root, _, files in os.walk(args.root_dir):
            if "300K_fit_4000.xtc" in files:
                xtc_path = os.path.join(root, "300K_fit_4000.xtc")
                gro_path = os.path.join(root, "md_Ref.gro")
                top_path = os.path.join(root, "topol.top")

                if os.path.exists(gro_path):
                    if xtc_path not in processed_trajectories:
                        trajectory_sets.append(
                            {
                                "xtc_path": xtc_path,
                                "gro_path": gro_path,
                                "top_path": (
                                    top_path if os.path.exists(top_path) else None
                                ),
                                "output_dir": os.path.join(root + "_xtc_to_pdb"),
                                "max_retries": args.max_retries,
                                "retry_delay": args.retry_delay,
                            }
                        )

        logger.info(f"Found {len(trajectory_sets)} trajectory sets to process")

        if args.dry_run:
            logger.info("Dry run - would process:")
            for ts in trajectory_sets:
                logger.info(f"  {ts['xtc_path']} -> {ts['output_dir']}")
            return

        # Process trajectories in parallel
        num_processes = min(args.num_processes, os.cpu_count() or 1)
        logger.info(f"Using {num_processes} processes for conversion")

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for ts in trajectory_sets:
                futures.append(executor.submit(process_trajectory, ts))

            # Track progress with tqdm
            successful = 0
            failed = 0
            for future in tqdm(
                futures, total=len(futures), desc="Converting trajectories"
            ):
                result = future.result()
                if result["success"]:
                    successful += 1
                    processed_trajectories.add(result["xtc_path"])
                    # Save progress periodically
                    if successful % 10 == 0:
                        save_progress(progress_file, processed_trajectories)
                else:
                    failed += 1
                    logger.error(
                        f"Failed to process {result['xtc_path']}: {result['error']}"
                    )

        # Save final progress
        save_progress(progress_file, processed_trajectories)

        logger.info(f"Conversion complete. Successful: {successful}, Failed: {failed}")

    except Exception as e:
        logger.error(f"Error during batch processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
