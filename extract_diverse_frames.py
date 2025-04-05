#!/usr/bin/env python3
# extract_diverse_frames.py

"""
Extract and copy the most diverse frames based on RMSD analysis.
"""

import os
import sys
import csv
import argparse
import shutil
from pathlib import Path


def extract_diverse_frames(
    csv_file, frames_dir, output_dir=None, num_frames=10, only_list=False
):
    """Extract the most diverse frames based on RMSD and copy to a new directory.

    Args:
        csv_file: Path to CSV file with ranked frames
        frames_dir: Directory containing the PDB frames (original or superimposed)
        output_dir: Directory to copy the selected frames to
        num_frames: Number of frames to extract
        only_list: If True, just list the diverse frames without copying them

    Returns:
        List of tuples with information about the diverse frames
    """
    # Create output directory if it doesn't exist and we're copying files
    if output_dir and not only_list:
        os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    frames_data = []
    try:
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                rank = int(row[0])
                frame_name = row[1]
                rmsd = float(row[2])
                frames_data.append((rank, frame_name, rmsd))
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

    # Sort by rank (just in case)
    frames_data.sort(key=lambda x: x[0])

    # Limit to requested number of frames
    frames_data = frames_data[:num_frames]

    if only_list:
        # Just print the diverse frames without copying them
        print(f"\nTop {len(frames_data)} diverse frames:")
        for rank, frame_name, rmsd in frames_data:
            print(f"{rank}. {frame_name}: RMSD = {rmsd:.4f}")
        return frames_data

    # Find and copy the frames
    copied_frames = []
    for rank, frame_name, rmsd in frames_data:
        # Try various path patterns to find the frame
        source_paths = [
            Path(frames_dir)
            / f"aligned_{frame_name}",  # Check superimposed frames first
            Path(frames_dir) / frame_name,  # Then check original frames
            Path(frames_dir).parent
            / "pdb_frames"
            / frame_name,  # Check original frames in parent/pdb_frames
        ]

        # Find the first existing path
        source_path = None
        for path in source_paths:
            if path.exists():
                source_path = path
                break

        # Skip if file doesn't exist
        if not source_path:
            print(f"Warning: Frame file not found for: {frame_name}")
            continue

        # Create target filename with rank
        target_filename = f"diverse_{rank}_{frame_name}"
        target_path = Path(output_dir) / target_filename

        # Copy the file
        shutil.copy2(source_path, target_path)
        print(f"Copied {frame_name}: {rmsd:.4f} -> {target_filename}")

        copied_frames.append((rank, frame_name, rmsd, target_filename))

    # Create a README file with information about the frames
    if copied_frames:
        readme_path = Path(output_dir) / "README.txt"
        with open(readme_path, "w") as f:
            f.write("Diverse Frames Dataset\n")
            f.write("=====================\n\n")
            f.write(
                f"This directory contains {len(copied_frames)} diverse conformations selected based on RMSD values.\n\n"
            )
            f.write("Frame Details:\n")
            f.write("--------------\n")
            for rank, frame_name, rmsd, target_filename in copied_frames:
                f.write(
                    f"{rank}. {target_filename} - Original: {frame_name}, RMSD: {rmsd:.4f}\n"
                )

        print(f"\nExtracted {len(copied_frames)} diverse frames to {output_dir}")
        print(f"Created README with frame details at {readme_path}")

    return copied_frames


def main():
    parser = argparse.ArgumentParser(
        description="Extract diverse frames based on RMSD ranking"
    )
    parser.add_argument(
        "csv_file", help="CSV file with ranked frames (from analyze_rmsd.py)"
    )
    parser.add_argument("frames_dir", help="Directory containing PDB frames", nargs="?")
    parser.add_argument("--output", "-o", help="Output directory for diverse frames")
    parser.add_argument(
        "--num", "-n", type=int, default=10, help="Number of frames to extract"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Just list diverse frames without copying them",
    )

    args = parser.parse_args()

    # List-only mode doesn't require output directory
    if args.list_only and not args.output:
        pass  # Output directory not needed
    elif not args.list_only and not args.output:
        # Default output directory
        csv_path = Path(args.csv_file)
        args.output = str(csv_path.parent.parent / "diverse_frames")

    # Determine frames directory if not provided
    frames_dir = args.frames_dir
    if frames_dir is None:
        # Try to determine from the CSV file path
        csv_path = Path(args.csv_file)

        # Check different possible locations
        possible_dirs = [
            csv_path.parent.parent
            / "superimposed_frames",  # Check for superimposed frames first
            csv_path.parent.parent / "pdb_frames",  # Then check for original frames
        ]

        for directory in possible_dirs:
            if directory.exists():
                frames_dir = directory
                print(f"Using frames directory: {frames_dir}")
                break

        if frames_dir is None:
            print(
                f"Error: Could not determine frames directory. Please specify it explicitly."
            )
            return 1

    extract_diverse_frames(
        args.csv_file, frames_dir, args.output, args.num, args.list_only
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
