#!/usr/bin/env python3
# src/scripts/convert_trajectory.py

"""
Command-line script for converting XTC trajectories to multi-model PDB files.
"""

import argparse
from pathlib import Path
import warnings
from src.core.services.trajectory_converter import TrajectoryConverter


def main():
    """Main function to run the trajectory conversion."""
    parser = argparse.ArgumentParser(
        description="Convert XTC trajectory to multi-model PDB"
    )
    parser.add_argument("xtc_path", type=str, help="Path to input XTC file")
    parser.add_argument("output_path", type=str, help="Path for output PDB file")

    # Structure and topology options
    parser.add_argument(
        "--topology",
        type=str,
        help="Optional structure file (e.g., GRO, PDB)",
        default=None,
    )
    parser.add_argument(
        "--top",
        type=str,
        help="Optional GROMACS topology file (.top)",
        default=None,
    )

    # Frame selection options
    parser.add_argument(
        "--start", type=int, default=0, help="First frame to convert (0-based indexing)"
    )
    parser.add_argument("--end", type=int, default=None, help="Last frame to convert")
    parser.add_argument(
        "--stride", type=int, default=1, help="Step size for frame selection"
    )

    # Metadata options (used if not available from topology)
    parser.add_argument(
        "--resname", type=str, default="UNK", help="Default residue name for all atoms"
    )
    parser.add_argument(
        "--chain", type=str, default="A", help="Default chain ID for all atoms"
    )
    parser.add_argument(
        "--element", type=str, default="C", help="Default element type for all atoms"
    )

    args = parser.parse_args()

    # Prepare metadata (will be overridden by topology file if available)
    metadata = {
        "resnames": [args.resname] * 10000,  # Large enough for most cases
        "chainIDs": [args.chain] * 10000,
        "elements": [args.element] * 10000,
    }

    # Create converter instance
    converter = TrajectoryConverter(
        topology_file=args.topology, metadata=metadata, top_file=args.top
    )

    # Print trajectory information
    try:
        info = converter.get_trajectory_info(args.xtc_path)
        print("\nTrajectory Information:")
        print(f"Number of frames: {info['n_frames']}")
        print(f"Number of atoms: {info['n_atoms']}")
        print(
            f"Time range: {info['time_range'][0]:.2f} to {info['time_range'][1]:.2f} ps"
        )
        print(f"Box dimensions: {info['dimensions']}")
        print(f"Structure source: {info['topology_source']}")
        print(f"Topology file: {info['top_file'] or 'None'}")

        # Print metadata status
        print("\nMetadata Status:")
        for key, has_data in info["metadata_status"].items():
            status = "Available" if has_data else "Using defaults"
            print(f"{key}: {status}")
        print()
    except ValueError as e:
        print(f"Warning: Could not get trajectory info: {e}")

    # Convert trajectory
    try:
        output_file = converter.xtc_to_multimodel_pdb(
            args.xtc_path,
            args.output_path,
            start=args.start,
            end=args.end,
            stride=args.stride,
        )
        print(f"Successfully converted trajectory to: {output_file}")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
