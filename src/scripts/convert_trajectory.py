#!/usr/bin/env python3
# src/scripts/convert_trajectory.py

"""
Script to convert molecular dynamics trajectories between different formats.
Currently supports converting XTC to multi-model PDB files using GROMACS.
"""

import os
import shutil
import tempfile
import argparse
import logging
from pathlib import Path
from typing import Optional

from src.core.services.gmx_converter import GromacsPDBConverter


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(message)s"
            if not verbose
            else "%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def copy_to_temp(file_path: str) -> tuple[Path, Path]:
    """Copy a file to a temporary directory and return both paths."""
    src_path = Path(file_path)
    temp_dir = Path(tempfile.mkdtemp(prefix="traj_convert_"))
    temp_path = temp_dir / src_path.name

    # Read source file in binary mode and write to destination
    with open(src_path, "rb") as src, open(temp_path, "wb") as dst:
        dst.write(src.read())

    return src_path, temp_path


def safe_copy(src: Path, dst: Path) -> None:
    """Safely copy a file with error handling."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except Exception as e:
        raise RuntimeError(f"Failed to copy {src} to {dst}: {str(e)}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert molecular dynamics trajectories between formats."
    )
    parser.add_argument(
        "--topology",
        type=str,
        required=True,
        help="Path to topology file (GRO)",
    )
    parser.add_argument(
        "--top",
        type=str,
        required=True,
        help="Path to GROMACS topology file (.top)",
    )
    parser.add_argument(
        "--xtc",
        type=str,
        required=True,
        help="Path to XTC trajectory file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting frame (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending frame (default: None = all frames)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output",
    )
    return parser.parse_args()


def main():
    """Main function to convert trajectories."""
    args = parse_args()
    logger = setup_logging(args.verbose)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize converter
        converter = GromacsPDBConverter(verbose=args.verbose)

        # Get trajectory info
        logger.info("Getting trajectory information...")
        info = converter.get_trajectory_info(args.xtc, args.topology, args.top)
        logger.info(f"Found {info['n_frames']} frames, {info['n_atoms']} atoms")

        if args.end is not None and args.end > info["n_frames"]:
            logger.warning(
                f"Specified end frame ({args.end}) is greater than number of frames "
                f"({info['n_frames']}). Will process until last frame."
            )
            args.end = info["n_frames"]

        # Convert trajectory
        logger.info("Converting trajectory...")
        output_files = converter.convert_trajectory(
            xtc_path=args.xtc,
            gro_path=args.topology,
            top_path=args.top,
            output_dir=str(output_dir),
            start=args.start,
            end=args.end,
            stride=args.stride,
        )

        # Log success
        for output_file in output_files:
            logger.info(f"Successfully created {output_file}")

    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        raise


if __name__ == "__main__":
    main()
