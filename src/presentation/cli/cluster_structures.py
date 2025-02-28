"""Command-line interface for structure clustering."""

import argparse
import os
from typing import List

from ...core.services.clustering_service import ClusteringService
from ...infrastructure.repositories.structure_repository import StructureRepository


def setup_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Cluster molecular structures")
    parser.add_argument("input_dir", help="Directory containing aligned structures")
    parser.add_argument("output_dir", help="Directory for clustering results")
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters to generate",
    )
    return parser


def main() -> None:
    """Main entry point for structure clustering CLI."""
    parser = setup_parser()
    args = parser.parse_args()

    # Setup services and repositories
    repository = StructureRepository(args.input_dir)
    service = ClusteringService()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and process structures
    structures = repository.list()

    # Calculate RMSD matrix
    rmsd_matrix = service.calculate_rmsd_matrix(structures)

    # Perform clustering
    clusters, linkage = service.perform_clustering(
        rmsd_matrix=rmsd_matrix, n_clusters=args.n_clusters
    )

    # Find representatives
    representatives = service.find_cluster_representatives(
        rmsd_matrix=rmsd_matrix, clusters=clusters
    )

    # Save results (implementation depends on output format requirements)


if __name__ == "__main__":
    main()
