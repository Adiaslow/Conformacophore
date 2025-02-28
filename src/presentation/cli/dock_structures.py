# src/presentation/cli/dock_structures.py
"""Command-line interface for molecular docking."""

import argparse
import os
from typing import List

from ...core.services.docking_service import DockingService
from ...infrastructure.repositories.structure_repository import StructureRepository
from ....rosetta_adapter import RosettaAdapter


def setup_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Perform molecular docking")
    parser.add_argument("input_dir", help="Directory containing structure files")
    parser.add_argument("output_dir", help="Directory for docking results")
    parser.add_argument(
        "--receptor",
        required=True,
        help="Receptor structure identifier",
    )
    parser.add_argument(
        "--n-poses",
        type=int,
        default=100,
        help="Number of docking poses to generate",
    )
    return parser


def main() -> None:
    """Main entry point for molecular docking CLI."""
    parser = setup_parser()
    args = parser.parse_args()

    # Setup services and repositories
    repository = StructureRepository(args.input_dir)
    backend = RosettaAdapter()
    service = DockingService(backend)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load receptor
    receptor = repository.get(args.receptor)
    if not receptor:
        raise ValueError(f"Receptor structure {args.receptor} not found")

    # Process all ligands
    structures = repository.list()
    for structure in structures:
        results = service.dock_molecule(
            receptor=receptor,
            ligand=structure,
            n_poses=args.n_poses,
        )
        # Save results (implementation depends on output format requirements)


if __name__ == "__main__":
    main()
