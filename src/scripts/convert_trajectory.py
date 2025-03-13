#!/usr/bin/env python3
# src/scripts/convert_trajectory.py

"""
Script to convert molecular dynamics trajectories between different formats.
Currently supports converting XTC to multi-model PDB files.
"""

import os
import shutil
import tempfile
import argparse
import logging
from pathlib import Path
from typing import Optional

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
    """Safely copy a file from source to destination."""
    # First copy to a temporary file in the destination directory
    temp_dst = dst.parent / f".{dst.name}.tmp"
    try:
        # Copy in chunks to handle large files
        with open(src, "rb") as fsrc, open(temp_dst, "wb") as fdst:
            while True:
                chunk = fsrc.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                fdst.write(chunk)
                fdst.flush()  # Ensure data is written to disk
                os.fsync(fdst.fileno())  # Force write to disk

        # Ensure the temporary file is fully written
        os.sync()

        # Rename the temporary file to the final name
        # This is an atomic operation on most filesystems
        temp_dst.rename(dst)

    except Exception as e:
        # Clean up temporary file if something went wrong
        if temp_dst.exists():
            temp_dst.unlink()
        raise e


def write_pdb_header(
    output_path: Path, xtc_path: str, gro_path: str, info: dict
) -> None:
    """Write a properly formatted PDB header."""
    with open(output_path, "w") as f:
        f.write("TITLE     TRAJECTORY CONVERTED FROM XTC\n")
        f.write(f"REMARK   1 REFERENCE STRUCTURE: {os.path.basename(gro_path)}\n")
        f.write(f"REMARK   2 TRAJECTORY FILE: {os.path.basename(xtc_path)}\n")
        f.write(f"REMARK   3 NUMBER OF FRAMES: {info['n_frames']}\n")
        f.write(f"REMARK   4 NUMBER OF ATOMS: {info['n_atoms']}\n")
        if "sequence" in info:
            f.write(f"REMARK   5 SEQUENCE: {info['sequence']}\n")
        f.write(
            "CRYST1    0.000    0.000    0.000  90.00  90.00  90.00 P 1           1\n"
        )


def main():
    """Main function to convert trajectories."""
    logger = setup_logging()

    parser = argparse.ArgumentParser(
        description="Convert molecular dynamics trajectories"
    )
    parser.add_argument("--xtc", type=str, required=True, help="Path to input XTC file")
    parser.add_argument("--topology", type=str, help="Path to topology file (GRO/PDB)")
    parser.add_argument("--top", type=str, help="Path to GROMACS topology file (.top)")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="First frame to convert (0-based)"
    )
    parser.add_argument(
        "--end", type=int, help="Last frame to convert (None = all frames)"
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Step size for frame selection"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Copy files to temporary directory
        logger.info("Copying files to temporary directory...")
        _, temp_xtc = copy_to_temp(args.xtc)
        _, temp_gro = copy_to_temp(args.topology) if args.topology else (None, None)
        _, temp_top = copy_to_temp(args.top) if args.top else (None, None)

        temp_dir = temp_xtc.parent
        logger.info(f"Using temporary directory: {temp_dir}")

        # Create a temporary output directory for local writing
        with tempfile.TemporaryDirectory(prefix="traj_output_") as temp_output_dir:
            temp_output_path = Path(temp_output_dir)
            logger.info(f"Using temporary output directory: {temp_output_path}")

            # Initialize converter with temporary files
            converter = TrajectoryConverter(
                topology_file=str(temp_gro) if temp_gro else None,
                top_file=str(temp_top) if temp_top else None,
            )

            try:
                # Get trajectory info
                logger.info("Getting trajectory information...")
                info = converter.get_trajectory_info(str(temp_xtc))
                logger.info(f"Found {info['n_frames']} frames, {info['n_atoms']} atoms")

                if args.end is not None and args.end > info["n_frames"]:
                    logger.warning(
                        f"Specified end frame ({args.end}) is greater than number of frames "
                        f"({info['n_frames']}). Will process until last frame."
                    )
                    args.end = info["n_frames"]

                # Convert trajectory to temporary directory
                logger.info("Converting trajectory...")
                output_files = converter.xtc_to_multimodel_pdb(
                    xtc_path=str(temp_xtc),
                    output_path=str(temp_output_path),
                    start=args.start,
                    end=args.end,
                    stride=args.stride,
                )

                # Copy output files to final destination
                logger.info("Copying output files to final destination...")
                for temp_file in output_files:
                    dest_file = output_dir / temp_file.name
                    try:
                        safe_copy(temp_file, dest_file)
                        logger.info(f"  - Successfully copied {dest_file}")
                    except Exception as e:
                        logger.error(
                            f"Error copying {temp_file} to {dest_file}: {str(e)}"
                        )
                        raise

            except Exception as e:
                logger.error(f"Error converting trajectory: {str(e)}")
                raise

        # Clean up temporary directory
        logger.info("Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        logger.error(f"Error during file processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
