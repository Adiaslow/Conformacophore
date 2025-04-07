#!/usr/bin/env python3
# src/scripts/superimpose.py
"""
Command-line interface for molecular superimposition.

This script provides a command-line interface for superimposing
molecular structures and analyzing trajectories.
"""

import os
import sys
import argparse
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: NumPy is required. Install with: pip install numpy")
    sys.exit(1)

# Add the project root directory to path if running as script
if __name__ == "__main__":
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.path.insert(0, project_root)

try:
    from src.conformacophore.superimposition import (
        save_superimposed_structure,
        load_metrics_file,
        save_metrics_file,
    )
    from src.conformacophore.batch import (
        process_trajectory,
        analyze_trajectory_clashes,
    )
except ImportError as e:
    print(f"ERROR: Failed to import modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Superimpose molecules and analyze trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single structure superimposition
    single_parser = subparsers.add_parser(
        "single", help="Superimpose a single structure"
    )
    single_parser.add_argument(
        "test_pdb", help="Test PDB file (structure to be transformed)"
    )
    single_parser.add_argument(
        "ref_pdb",
        help="Reference PDB file with protein (chains A,B,C) and ligand (chain D)",
    )
    single_parser.add_argument(
        "--output-file", help="Output PDB file (default: superimposed_<test>.pdb)"
    )
    single_parser.add_argument(
        "--metrics-file", help="Optional path to metrics file with transformations"
    )
    single_parser.add_argument(
        "--match-by",
        choices=["element", "name"],
        default="element",
        help="Method for matching atoms",
    )
    single_parser.add_argument(
        "--clash-cutoff",
        type=float,
        default=0.60,
        help="VDW overlap threshold for clash detection",
    )
    single_parser.add_argument(
        "--verbose", action="store_true", help="Print detailed information"
    )

    # Batch trajectory processing
    batch_parser = subparsers.add_parser(
        "batch", help="Process a batch of structures from a trajectory"
    )
    batch_parser.add_argument("frame_dir", help="Directory containing frame PDB files")
    batch_parser.add_argument(
        "ref_pdb",
        help="Reference PDB file with protein (chains A,B,C) and ligand (chain D)",
    )
    batch_parser.add_argument(
        "--output-dir",
        help="Directory to save superimposed structures (default: <parallel_to_frame_dir>/superimposed_frames)",
    )
    batch_parser.add_argument(
        "--metrics-file",
        help="Path to metrics file (default: <frame_dir>/superimposition_metrics.json)",
    )
    batch_parser.add_argument(
        "--csv-file",
        help="Path to CSV results file (default: <parallel_to_frame_dir>/superimposition_results.csv)",
    )
    batch_parser.add_argument(
        "--match-by",
        choices=["element", "name"],
        default="element",
        help="Method for matching atoms",
    )
    batch_parser.add_argument(
        "--clash-cutoff",
        type=float,
        default=0.60,
        help="VDW overlap threshold for clash detection",
    )
    batch_parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of parallel processes",
    )
    batch_parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to process",
    )
    batch_parser.add_argument(
        "--save-structures",
        action="store_true",
        help="Save superimposed structures",
    )
    batch_parser.add_argument(
        "--save-limit",
        type=int,
        default=5,
        help="Maximum number of structures to save",
    )
    batch_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already processed frames",
    )
    batch_parser.add_argument(
        "--verbose", action="store_true", help="Print detailed information"
    )

    # Analyze trajectory clashes
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze clash statistics from a trajectory"
    )
    analyze_parser.add_argument("metrics_file", help="Path to metrics file")
    analyze_parser.add_argument(
        "--verbose", action="store_true", help="Print detailed information"
    )

    # Legacy compatibility with superimpose_trajectories.py
    legacy_parser = subparsers.add_parser(
        "legacy", help="Legacy mode compatible with superimpose_trajectories.py"
    )
    legacy_parser.add_argument(
        "base_dir", help="Base directory containing pdb_frames directories"
    )
    legacy_parser.add_argument(
        "reference_pdb", help="Reference PDB file with protein and ligand"
    )
    legacy_parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of parallel processes",
    )
    legacy_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already processed files",
    )
    legacy_parser.add_argument(
        "--save-structures",
        action="store_true",
        help="Save the first 5 superimposed structures as PDB files",
    )
    legacy_parser.add_argument(
        "--output-dir",
        help="Directory to save superimposed structures",
    )
    legacy_parser.add_argument(
        "--verbose", action="store_true", help="Print detailed information"
    )

    return parser.parse_args()


def process_single_structure(args):
    """Process a single structure."""
    # Set default output file if not provided
    output_file = args.output_file
    if not output_file:
        output_file = f"superimposed_{Path(args.test_pdb).stem}.pdb"

    # Process the structure
    result = save_superimposed_structure(
        frame_file=args.test_pdb,
        ref_pdb_path=args.ref_pdb,
        output_file=output_file,
        metrics_file=args.metrics_file,
        match_by=args.match_by,
        clash_cutoff=args.clash_cutoff,
        verbose=args.verbose,
    )

    if result:
        if args.verbose:
            # Detailed output was already printed by the function
            pass
        else:
            # Print a simple summary
            has_clashes = result.get("has_clashes", False)
            num_clashes = result.get("num_clashes", 0)
            total_clashes = result.get("total_clashes", 0)
            rmsd = result.get("rmsd", 0.0)

            print(f"Successfully superimposed structure:")
            print(f"  - Output file: {result['output_file']}")
            print(f"  - RMSD: {rmsd:.4f}")

            if has_clashes:
                print(
                    f"  - Clashes: {num_clashes} atoms with {total_clashes} total overlaps"
                )
            else:
                print(f"  - No clashes detected")

        return 0
    else:
        print("Failed to superimpose structure")
        return 1


def process_batch(args):
    """Process a batch of structures."""
    try:
        summary = process_trajectory(
            frame_dir=args.frame_dir,
            ref_pdb_path=args.ref_pdb,
            output_dir=args.output_dir,
            metrics_file=args.metrics_file,
            csv_file=args.csv_file,
            match_by=args.match_by,
            clash_cutoff=args.clash_cutoff,
            num_processes=args.num_processes,
            max_frames=args.max_frames,
            save_structures=args.save_structures,
            save_limit=args.save_limit,
            force=args.force,
            verbose=args.verbose,
        )

        return 0
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return 1


def analyze_clashes(args):
    """Analyze clash statistics from a trajectory."""
    stats = analyze_trajectory_clashes(args.metrics_file)

    if "error" in stats:
        print(stats["error"])
        return 1

    print(f"Clash analysis for {args.metrics_file}:")
    print(f"  - Total frames: {stats['total_frames']}")
    print(
        f"  - Frames with clashes: {stats['frames_with_clashes']} ({stats['clash_percentage']:.1f}%)"
    )
    print(f"  - Average clashing atoms per frame: {stats['avg_clashing_atoms']:.1f}")
    print(f"  - Maximum clashing atoms: {stats['max_clashing_atoms']}")
    print(f"  - Average total clashes per frame: {stats['avg_total_clashes']:.1f}")
    print(f"  - Maximum total clashes: {stats['max_total_clashes']}")

    if "max_clash_frame" in stats:
        print(
            f"  - Frame with most clashes: {stats['max_clash_frame']} ({stats['max_clash_frame_count']} atoms)"
        )

    return 0


def process_legacy(args):
    """Process in legacy mode compatible with superimpose_trajectories.py."""
    # Find all pdb_frames directories
    base_path = Path(args.base_dir)
    frame_dirs = list(base_path.rglob("pdb_frames"))

    if not frame_dirs:
        print(f"No pdb_frames directories found in {args.base_dir}")
        return 1

    print(f"Found {len(frame_dirs)} pdb_frames directories")

    # Process each directory
    success_count = 0
    error_count = 0

    for frame_dir in frame_dirs:
        print(f"\nProcessing directory: {frame_dir}")

        # Set output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = frame_dir.parent / "superimposed_frames"

        # Set metrics file
        metrics_file = frame_dir / "superimposition_metrics.json"

        # Set CSV file
        csv_file = frame_dir.parent / "superimposition_results.csv"

        try:
            summary = process_trajectory(
                frame_dir=str(frame_dir),
                ref_pdb_path=args.reference_pdb,
                output_dir=str(output_dir),
                metrics_file=str(metrics_file),
                csv_file=str(csv_file),
                num_processes=args.num_processes,
                save_structures=args.save_structures,
                save_limit=5,  # Hard-coded in legacy mode
                force=args.force,
                verbose=args.verbose,
            )

            success_count += 1
        except Exception as e:
            print(f"Error processing directory {frame_dir}: {str(e)}")
            error_count += 1

    print(
        f"\nProcessed {len(frame_dirs)} directories: {success_count} successful, {error_count} failed"
    )
    return 0 if error_count == 0 else 1


def main():
    """Main function."""
    args = parse_args()

    # Handle different commands
    if args.command == "single":
        return process_single_structure(args)
    elif args.command == "batch":
        return process_batch(args)
    elif args.command == "analyze":
        return analyze_clashes(args)
    elif args.command == "legacy":
        return process_legacy(args)
    else:
        print("Please specify a command: single, batch, analyze, or legacy")
        print("Use --help for more information")
        return 1


if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f} seconds")
    sys.exit(exit_code)
