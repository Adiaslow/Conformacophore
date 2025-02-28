"""Test script for molecular structure processing pipeline."""

import os
import logging
import json
import numpy as np
from pathlib import Path
from src.infrastructure.repositories.structure_repository import StructureRepository
from src.core.services.alignment_service import AlignmentService
from src.core.services.frame_processing_service import FrameProcessingService


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging for debugging."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup file handler
    file_handler = logging.FileHandler(output_dir / "pipeline.log")
    console_handler = logging.StreamHandler()

    # Setup formatting
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_results(results: list, output_path: Path, prefix: str) -> None:
    """Save results to JSON file."""
    output_data = {
        "rmsd_values": [r.rmsd for r in results],
        "matched_atoms": [r.matched_atoms for r in results],
        "matched_pairs": [r.matched_pairs for r in results],
    }

    with open(output_path / f"{prefix}_results.json", "w") as f:
        json.dump(output_data, f, indent=2)


def test_pipeline():
    """Run test pipeline on sample data."""
    # Setup paths
    base_dir = Path("test_data")
    input_dir = base_dir / "hexamers_water"
    ref_dir = base_dir / "reference"
    output_dir = Path("test_output")

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting pipeline test")
    logger.debug(f"Input directory: {input_dir}")
    logger.debug(f"Reference directory: {ref_dir}")
    logger.debug(f"Output directory: {output_dir}")

    # Initialize services
    repository = StructureRepository(str(input_dir))
    alignment_service = AlignmentService()
    frame_processor = FrameProcessingService(
        alignment_service=alignment_service,
        structure_repository=repository,
    )
    logger.info("Initialized services")

    # Load reference structure
    logger.info("Loading reference structure")
    ref_repo = StructureRepository(str(ref_dir))
    reference = ref_repo.get("vhl1", chain_id="D")[0]
    logger.debug(f"Reference structure loaded: {len(reference.atoms)} atoms")
    logger.debug(
        "Reference atom types: "
        + ", ".join(
            f"{a['atom_name']}({a['residue_name']})" for a in reference.atoms[:5]
        )
        + "..."
    )

    # Process structures
    for compound_id in ["943"]:
        logger.info(f"\nProcessing compound {compound_id}")
        compound_output = output_dir / compound_id
        compound_output.mkdir(exist_ok=True)

        frames = repository.get(compound_id, chain_id="A")
        if frames:
            logger.debug(f"Loaded {len(frames)} frames")
            first_frame = frames[0]
            logger.debug(f"First frame has {len(first_frame.atoms)} atoms")
            logger.debug(
                "First frame atom types: "
                + ", ".join(
                    f"{a['atom_name']}({a['residue_name']})"
                    for a in first_frame.atoms[:5]
                )
                + "..."
            )

            # Get metadata
            metadata = repository.get_metadata(compound_id)
            if metadata:
                logger.info(f"Sequence: {metadata.sequence}")
                logger.debug(f"Source file: {metadata.source_file}")
                logger.debug(f"Number of frames: {metadata.num_frames}")
            else:
                logger.warning("No metadata available")

            # Process frames
            logger.info("Starting frame processing")

            # Step 1: Initial alignment
            results = frame_processor.process_frames(
                compound_id=compound_id,
                reference=reference,
                frames=frames,
                clash_cutoff=2.0,
            )
            save_results(results, compound_output, "initial")

            # Step 2: Filter by RMSD
            rmsd_threshold = 5.0  # Adjust as needed
            filtered_results = [r for r in results if r.rmsd < rmsd_threshold]
            save_results(filtered_results, compound_output, "filtered")

            logger.info("Alignment Statistics:")
            logger.info(f"Total frames: {len(results)}")
            logger.info(f"Frames after RMSD filtering: {len(filtered_results)}")

            # Calculate RMSD statistics
            rmsds = np.array([r.rmsd for r in filtered_results])
            logger.info("\nRMSD Statistics:")
            logger.info(f"  Average: {np.mean(rmsds):.2f}")
            logger.info(f"  Std Dev: {np.std(rmsds):.2f}")
            logger.info(f"  Minimum: {np.min(rmsds):.2f}")
            logger.info(f"  Maximum: {np.max(rmsds):.2f}")

            # Step 3: Clustering (if implemented)
            try:
                from src.core.services.clustering_service import ClusteringService

                clustering_service = ClusteringService()
                clusters = clustering_service.cluster_frames(
                    filtered_results,
                    rmsd_cutoff=2.0,  # Adjust based on your needs
                    method="complete",
                )

                # Get representatives
                representatives = clustering_service.get_cluster_representatives(
                    clusters, filtered_results
                )

                logger.info("\nClustering Results:")
                logger.info(f"Number of clusters: {len(clusters)}")
                for i, cluster in enumerate(clusters):
                    logger.info(f"Cluster {i+1}: {len(cluster)} frames")
                    cluster_rmsds = [filtered_results[idx].rmsd for idx in cluster]
                    logger.info(f"  Average RMSD: {np.mean(cluster_rmsds):.2f}")
                    logger.info(
                        f"  Representative: Frame {representatives[i]} (RMSD: {filtered_results[representatives[i]].rmsd:.2f})"
                    )

                # Save clustering results
                cluster_data = {
                    "num_clusters": len(clusters),
                    "cluster_sizes": [len(c) for c in clusters],
                    "cluster_indices": [list(c) for c in clusters],
                }
                with open(compound_output / "clustering_results.json", "w") as f:
                    json.dump(cluster_data, f, indent=2)

            except ImportError:
                logger.warning("Clustering service not implemented")

            # Save PDB files for best poses
            top_n = 5
            sorted_results = sorted(filtered_results, key=lambda x: x.rmsd)[:top_n]

            for i, result in enumerate(sorted_results):
                frame_idx = results.index(result)
                frame = frames[frame_idx]
                # Save aligned frame as PDB
                output_path = compound_output / f"pose_{i+1}_rmsd_{result.rmsd:.2f}.pdb"
                frame_processor.save_aligned_frame(frame, result, output_path)
                logger.info(f"Saved top pose {i+1} to {output_path}")

        else:
            logger.error(f"Failed to load compound {compound_id}")


if __name__ == "__main__":
    test_pipeline()
