#!/usr/bin/env python3
# src/scripts/convert_traj.py

"""
Script to convert trajectory files to PDB format.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import os

# Add the project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from src.core.services.trajectory_converter import TrajectoryConverter


def main(args: Optional[argparse.Namespace] = None) -> int:
    """Convert trajectory file to PDB."""
    if args is None:
        parser = argparse.ArgumentParser(description="Convert trajectory to PDB")
        parser.add_argument("trajectory", help="Path to trajectory file")
        parser.add_argument("output", help="Output directory path")
        parser.add_argument("--topology", help="Path to topology file")
        parser.add_argument(
            "--start", type=int, default=0, help="First frame to convert"
        )
        parser.add_argument("--end", type=int, help="Last frame to convert")
        parser.add_argument("--stride", type=int, default=1, help="Frame stride")
        args = parser.parse_args()

    try:
        converter = TrajectoryConverter(topology_file=args.topology)

        # Get trajectory info
        info = converter.get_trajectory_info(args.trajectory)
        print("\nTrajectory Information:")
        print(f"Number of frames: {info['n_frames']}")
        print(f"Number of atoms: {info['n_atoms']}")
        print(
            f"Time range: {info['time_range'][0]:.2f} to {info['time_range'][1]:.2f} ps"
        )
        print(f"Box dimensions: {info['dimensions']}")
        print(f"Structure source: {info['topology_source']}")
        print(f"Topology file: {info['top_file']}\n")

        print("Metadata Status:")
        for key, value in info["metadata_status"].items():
            print(f"{key}: {'Available' if value else 'Using defaults'}")
        print()

        # Convert trajectory
        output_files = converter.xtc_to_multimodel_pdb(
            args.trajectory,
            args.output,
            start=args.start,
            end=args.end,
            stride=args.stride,
        )
        print(f"\nSuccessfully converted trajectory to: {output_files}")
        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
