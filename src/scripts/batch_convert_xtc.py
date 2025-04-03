# src/scripts/batch_convert_xtc.py

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from src.core.services.xtc_converter import XTCConverter
from src.core.utils.benchmarking import PerformanceStats, timer


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def find_xtc_files(
    directory: str, force: bool = False
) -> List[Tuple[str, str, str, str]]:
    """Find XTC files and their associated TOP and GRO files.

    Args:
        directory: Directory to search in
        force: Whether to include files that already have output

    Returns:
        List of tuples (xtc_path, top_path, gro_path, compound_name)
    """
    file_sets = []
    logger = logging.getLogger(__name__)

    for root, _, files in os.walk(directory):
        # Look for XTC files
        xtc_files = [f for f in files if f == "300K_fit_4000.xtc"]

        for xtc_file in xtc_files:
            xtc_path = os.path.join(root, xtc_file)
            top_path = os.path.join(root, "topol.top")
            gro_path = os.path.join(root, "md_Ref.gro")

            # Check if all required files exist
            if not all(os.path.exists(p) for p in [xtc_path, top_path, gro_path]):
                missing = [
                    p for p in [xtc_path, top_path, gro_path] if not os.path.exists(p)
                ]
                logger.warning(
                    f"Skipping incomplete set in {root}. Missing: {', '.join(missing)}"
                )
                continue

            # Check output file
            output_path = os.path.join(root, "conformers.pdb")
            if os.path.exists(output_path) and not force:
                logger.info(f"Skipping existing output: {output_path}")
                continue

            # Get compound name from parent directory
            compound_name = os.path.basename(root)

            # Get file sizes for benchmarking
            xtc_size = os.path.getsize(xtc_path)
            top_size = os.path.getsize(top_path)
            gro_size = os.path.getsize(gro_path)

            file_sets.append(
                (
                    xtc_path,
                    top_path,
                    gro_path,
                    compound_name,
                    xtc_size + top_size + gro_size,
                )
            )

    return file_sets


def process_file_set(
    args: Tuple[
        str, str, str, str, int, Optional[int], Optional[int], Optional[int], bool
    ],
) -> Tuple[bool, dict]:
    """Process a single file set.

    Args:
        args: Tuple of (xtc_path, top_path, gro_path, compound_name, total_size, start, stop, step, verbose)

    Returns:
        Tuple of (success, timing_stats)
    """
    (
        xtc_path,
        top_path,
        gro_path,
        compound_name,
        total_size,
        start,
        stop,
        step,
        verbose,
    ) = args
    logger = logging.getLogger(__name__)
    stats = PerformanceStats()

    try:
        with timer("total_conversion", stats, total_size):
            output_path = os.path.join(os.path.dirname(xtc_path), "conformers.pdb")
            converter = XTCConverter(verbose=verbose)

            # Time file loading
            with timer("file_loading", stats, total_size):
                converter.convert(
                    xtc_path=xtc_path,
                    top_path=top_path,
                    gro_path=gro_path,
                    output_path=output_path,
                    compound_name=compound_name,
                    start=start,
                    stop=stop,
                    step=step,
                )

            if verbose:
                logger.info(
                    f"\nPerformance stats for {compound_name}:\n{stats.report()}"
                )

            return True, {name: stat.__dict__ for name, stat in stats.stats.items()}

    except Exception as e:
        logger.error(f"Error processing {xtc_path}: {str(e)}")
        return False, {}


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert XTC trajectories to PDB format"
    )
    parser.add_argument("directory", help="Directory to process")
    parser.add_argument("--start", type=int, help="First frame to convert (0-based)")
    parser.add_argument("--stop", type=int, help="Last frame to convert (exclusive)")
    parser.add_argument("--step", type=int, help="Step size between frames")
    parser.add_argument(
        "--force", action="store_true", help="Force overwrite existing files"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=mp.cpu_count(),
        help="Number of parallel processes to use",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    stats = PerformanceStats()

    with timer("total_processing", stats):
        # Find file sets to process
        with timer("find_files", stats):
            file_sets = find_xtc_files(args.directory, args.force)
            total_sets = len(file_sets)

            if total_sets == 0:
                logger.error(f"No valid file sets found in {args.directory}")
                return

            logger.info(f"Found {total_sets} file sets to process")

        # Calculate total data size
        total_size = sum(size for _, _, _, _, size in file_sets)
        stats.get_stats("find_files").bytes_processed = total_size

        # Prepare arguments for parallel processing
        process_args = [
            (xtc, top, gro, name, size, args.start, args.stop, args.step, args.verbose)
            for xtc, top, gro, name, size in file_sets
        ]

        # Process files in parallel with progress bar
        successful = 0
        all_stats = []

        with timer("parallel_processing", stats, total_size):
            with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
                futures = [
                    executor.submit(process_file_set, arg_set)
                    for arg_set in process_args
                ]

                with tqdm(total=total_sets, desc="Converting") as pbar:
                    for future in as_completed(futures):
                        success, timing_stats = future.result()
                        if success:
                            successful += 1
                            all_stats.append(timing_stats)
                        pbar.update(1)

        # Aggregate stats from all processes
        for proc_stats in all_stats:
            for name, stat_dict in proc_stats.items():
                if "times" in stat_dict:  # Only process TimingStats objects
                    for t in stat_dict["times"]:
                        stats.add_timing(
                            name,
                            t,
                            stat_dict.get("bytes_processed", 0)
                            // len(stat_dict["times"]),
                        )

        # Report results
        logger.info(f"Processing complete: {successful}/{total_sets} successful")
        logger.info(f"\nOverall performance:\n{stats.report()}")


if __name__ == "__main__":
    main()
