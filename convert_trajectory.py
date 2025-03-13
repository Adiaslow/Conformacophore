#!/usr/bin/env python3
"""
Script to convert XTC to PDB using MDAnalysis with preserved connectivity information.
Explicitly handles bond information from GROMACS topology files.
"""

from pathlib import Path
import sys
from src.core.services.trajectory_converter import TrajectoryConverter


def convert_trajectory(xtc_path: str, output_path: str, topology_path: str) -> None:
    """
    Convert XTC trajectory to PDB with preserved connectivity information.

    Args:
        xtc_path: Path to the XTC trajectory file
        output_path: Path for the output PDB file
        topology_path: Path to the topology file
    """
    # Look for a GROMACS topology file with the same base name
    top_file = Path(topology_path).with_suffix(".top")
    if not top_file.exists():
        top_file = Path(topology_path).parent / "topol.top"

    # Create converter instance with topology file and optional .top file
    converter = TrajectoryConverter(
        topology_file=topology_path,
        top_file=str(top_file) if top_file.exists() else None,
    )

    try:
        # Get trajectory info
        info = converter.get_trajectory_info(xtc_path)
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

        # Convert trajectory with connectivity information preserved
        output_files = converter.xtc_to_multimodel_pdb(xtc_path, output_path)
        print(f"\nSuccessfully converted trajectory to: {output_files}")

        if not top_file.exists():
            print(
                "\nNote: No GROMACS topology (.top) file found. Bond information may be incomplete."
            )
            print(
                f"To preserve all connectivity, provide a topology file at: {top_file}"
            )

    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python convert_trajectory.py <xtc_file> <output_pdb> <topology_file>"
        )
        sys.exit(1)

    xtc_path = sys.argv[1]
    output_path = sys.argv[2]
    topology_path = sys.argv[3]

    convert_trajectory(xtc_path, output_path, topology_path)
