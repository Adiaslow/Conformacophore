"""Command-line interface for structure alignment."""

import argparse
import os
from typing import List

from ...core.services.alignment_service import AlignmentService
from ...infrastructure.repositories.structure_repository import StructureRepository


def setup_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Align molecular structures")
    parser.add_argument("input_dir", help="Directory containing PDB files")
    parser.add_argument("output_dir", help="Directory for output files")
    parser.add_argument(
        "--reference", required=True, help="Reference structure for alignment"
    )
    parser.add_argument(
        "--clash-cutoff",
        type=float,
        default=2.0,
        help="Distance cutoff for clash detection (Angstroms)",
    )
    return parser


def main() -> None:
    """Main entry point for structure alignment CLI."""
    parser = setup_parser()
    args = parser.parse_args()

    # Setup services and repositories
    repository = StructureRepository(args.input_dir)
    service = AlignmentService()  # Configure with appropriate strategy

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load reference structure
    reference = repository.get(args.reference)
    if not reference:
        raise ValueError(f"Reference structure {args.reference} not found")

    # Process all structures
    structures = repository.list()
    for structure in structures:
        result = service.align_structures(
            reference=reference,
            target=structure,
            clash_cutoff=args.clash_cutoff,
        )
        # Save results (implementation depends on output format requirements)


if __name__ == "__main__":
    main()
