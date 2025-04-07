#!/usr/bin/env python3
# process_frames.py
"""
Process multiple frame files from a directory using match_and_superimpose.py

This script:
1. Finds all frame PDB files in a directory
2. Processes each frame using match_and_superimpose.py
3. Reports on success/failure and clash detection results
"""

import os
import sys
import subprocess
import glob
from pathlib import Path
import argparse
import re


def main():
    """Main function to process all frame PDB files."""
    parser = argparse.ArgumentParser(
        description="Process multiple frame files using match_and_superimpose.py"
    )

    parser.add_argument("frames_dir", help="Directory containing frame PDB files")
    parser.add_argument("ref_pdb", help="Path to reference PDB file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save superimposed structures (defaults to <frames_dir>/superimposed)",
    )
    parser.add_argument(
        "--metrics-file",
        default=None,
        help="Path to metrics file with transformation data (defaults to <frames_dir>/superimposition_metrics.json)",
    )
    parser.add_argument(
        "--match-by",
        choices=["element", "name"],
        default="element",
        help="Method for matching atoms ('element' or 'name')",
    )
    parser.add_argument(
        "--clash-cutoff",
        type=float,
        default=0.60,
        help="VDW overlap threshold for clash detection",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all frames)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Check if frames directory exists
    frames_dir = args.frames_dir
    if not os.path.isdir(frames_dir):
        print(f"ERROR: Frames directory not found: {frames_dir}")
        return 1

    # Check if reference PDB exists
    ref_pdb = args.ref_pdb
    if not os.path.isfile(ref_pdb):
        print(f"ERROR: Reference PDB file not found: {ref_pdb}")
        return 1

    # Set default output directory if not provided
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(frames_dir, "superimposed")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set default metrics file if not provided
    metrics_file = args.metrics_file
    if not metrics_file:
        metrics_file = os.path.join(frames_dir, "superimposition_metrics.json")
        # Check if default metrics file exists
        if not os.path.isfile(metrics_file):
            print(f"No metrics file found at {metrics_file}")
            print("Will calculate transformations based on atom matching")
            metrics_file = None
    elif not os.path.isfile(metrics_file):
        print(f"WARNING: Specified metrics file not found: {metrics_file}")
        print("Will calculate transformations based on atom matching")
        metrics_file = None

    # Find all frame PDB files
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.pdb")))

    if not frame_files:
        print(f"No frame files found in {frames_dir}")
        return 1

    print(f"Found {len(frame_files)} frame files")

    # Limit number of frames if requested
    if args.max_frames and args.max_frames > 0:
        frame_files = frame_files[: args.max_frames]
        print(f"Processing first {len(frame_files)} frames")

    # Process each frame
    successful = 0
    failed = 0
    with_clashes = 0

    for i, frame_file in enumerate(frame_files):
        frame_name = Path(frame_file).stem
        output_file = os.path.join(output_dir, f"superimposed_{frame_name}.pdb")

        print(f"\nProcessing frame {i+1}/{len(frame_files)}: {frame_name}")

        # Build command
        cmd = [
            "python3",
            "match_and_superimpose.py",
            "--test-pdb",
            frame_file,
            "--reference-pdb",
            ref_pdb,
            "--output-file",
            output_file,
            "--match-by",
            args.match_by,
            "--clash-cutoff",
            str(args.clash_cutoff),
        ]

        if metrics_file:
            cmd.extend(["--metrics-file", metrics_file])

        if args.verbose:
            cmd.append("--verbose")

        # Run the command
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                successful += 1

                # Check for clash information in output by looking for the "Found X clashing atoms" pattern
                clash_pattern = r"Found (\d+) clashing atoms with (\d+) total clashes"
                clash_match = re.search(clash_pattern, result.stdout)

                if clash_match:
                    num_clashing_atoms = int(clash_match.group(1))
                    total_clashes = int(clash_match.group(2))
                    if num_clashing_atoms > 0:
                        with_clashes += 1
                        print(
                            f"✓ Success (with {num_clashing_atoms} clashing atoms, {total_clashes} total clashes)"
                        )
                    else:
                        print(f"✓ Success (no clashes)")
                else:
                    # If pattern not found, check if "Has clashes: True" is present for backward compatibility
                    if "Has clashes: True" in result.stdout:
                        with_clashes += 1
                        print(f"✓ Success (with clashes)")
                    else:
                        print(f"✓ Success (no clashes)")
            else:
                failed += 1
                print(f"✗ Failed: {result.stderr.strip()}")

            if args.verbose:
                print(result.stdout)

        except Exception as e:
            failed += 1
            print(f"✗ Error running command: {str(e)}")

    # Summary
    print(f"\n{'='*80}")
    print(f"Processing complete: {successful} successful, {failed} failed")
    print(f"Structures with clashes: {with_clashes}")
    print(f"All superimposed structures saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
