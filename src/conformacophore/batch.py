#!/usr/bin/env python3
# src/conformacophore/batch.py
"""
Batch processing of multiple frames for molecular superimposition.

This module provides functions for processing multiple frame files
and generating metrics across a trajectory.
"""

import os
import glob
from pathlib import Path
import json
import csv
from typing import Dict, List, Optional, Any, Tuple
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from .superimposition import (
    save_superimposed_structure,
    load_metrics_file,
    save_metrics_file,
)


def find_frame_files(directory: str, pattern: str = "frame_*.pdb") -> List[str]:
    """Find frame files in a directory.

    Args:
        directory: Directory to search in
        pattern: Glob pattern for frame files

    Returns:
        List of paths to frame files
    """
    return sorted(glob.glob(os.path.join(directory, pattern)))


def process_frame(
    args: Tuple[str, str, str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Process a single frame file.

    Args:
        args: Tuple containing (frame_file, ref_pdb_path, output_file, options)

    Returns:
        Dictionary with results or None if processing failed
    """
    frame_file, ref_pdb_path, output_file, options = args

    try:
        result = save_superimposed_structure(
            frame_file=frame_file,
            ref_pdb_path=ref_pdb_path,
            output_file=output_file,
            rotation=options.get("rotation"),
            translation=options.get("translation"),
            metrics_file=options.get("metrics_file"),
            match_by=options.get("match_by", "element"),
            clash_cutoff=options.get("clash_cutoff", 0.6),
            verbose=options.get("verbose", False),
        )

        if result:
            # Extract frame number from filename
            frame_name = Path(frame_file).stem
            frame_num = frame_name.split("_")[1]

            # Add frame info to result
            result["frame_num"] = frame_num
            result["frame_file"] = frame_file

            return result

    except Exception as e:
        print(f"Error processing frame {frame_file}: {str(e)}")

    return None


def save_metrics_to_csv(metrics: Dict[str, Any], csv_file: str) -> bool:
    """Save metrics to a CSV file.

    Args:
        metrics: Dictionary of metrics by frame number
        csv_file: Path to CSV file to save

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(csv_file)), exist_ok=True)

        fieldnames = [
            "frame_number",
            "rmsd",
            "has_clashes",
            "num_clashes",
            "total_clashes",
            "matched_atoms",
        ]

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Sort by frame number
            for frame_num in sorted(
                metrics.keys(), key=lambda x: int(x) if x.isdigit() else float("inf")
            ):
                frame_data = metrics[frame_num]
                writer.writerow(
                    {
                        "frame_number": frame_num,
                        "rmsd": frame_data.get("rmsd", 0.0),
                        "has_clashes": frame_data.get("has_clashes", False),
                        "num_clashes": frame_data.get("num_clashes", 0),
                        "total_clashes": frame_data.get("total_clashes", 0),
                        "matched_atoms": frame_data.get("matched_atoms", 0),
                    }
                )
        return True
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        return False


def process_trajectory(
    frame_dir: str,
    ref_pdb_path: str,
    output_dir: Optional[str] = None,
    metrics_file: Optional[str] = None,
    csv_file: Optional[str] = None,
    match_by: str = "element",
    clash_cutoff: float = 0.6,
    num_processes: int = 4,
    max_frames: Optional[int] = None,
    save_structures: bool = False,
    save_limit: int = 5,
    force: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Process a trajectory of frames.

    Args:
        frame_dir: Directory containing frame files
        ref_pdb_path: Path to reference PDB file
        output_dir: Directory to save output files (default: parallel to frame_dir/superimposed_frames)
        metrics_file: Path to metrics file (default: frame_dir/superimposition_metrics.json)
        csv_file: Path to CSV results file (default: parallel to frame_dir/superimposition_results.csv)
        match_by: Atom matching method ("element" or "name")
        clash_cutoff: Cutoff for clash detection
        num_processes: Number of parallel processes to use
        max_frames: Maximum number of frames to process
        save_structures: Whether to save superimposed structures
        save_limit: Maximum number of structures to save
        force: Whether to force reprocessing of already processed frames
        verbose: Whether to print detailed information

    Returns:
        Dictionary with processing results
    """
    # Set default paths if not provided
    if not os.path.isdir(frame_dir):
        raise ValueError(f"Frame directory not found: {frame_dir}")

    if not os.path.isfile(ref_pdb_path):
        raise ValueError(f"Reference PDB file not found: {ref_pdb_path}")

    # Calculate parent directory (where pdb_frames is located)
    parent_dir = os.path.dirname(frame_dir)

    if output_dir is None:
        output_dir = os.path.join(parent_dir, "superimposed_frames")

    if metrics_file is None:
        metrics_file = os.path.join(frame_dir, "superimposition_metrics.json")

    if csv_file is None:
        csv_file = os.path.join(parent_dir, "superimposition_results.csv")

    # Create output directory if saving structures
    if save_structures:
        os.makedirs(output_dir, exist_ok=True)

    # Find frame files
    frame_files = find_frame_files(frame_dir)
    if not frame_files:
        raise ValueError(f"No frame files found in {frame_dir}")

    if verbose:
        print(f"Found {len(frame_files)} frame files in {frame_dir}")

    # Limit number of frames if requested
    if max_frames and max_frames > 0 and max_frames < len(frame_files):
        frame_files = frame_files[:max_frames]
        if verbose:
            print(f"Processing first {len(frame_files)} frames")

    # Check if metrics file exists
    existing_metrics = {}
    if os.path.exists(metrics_file) and not force:
        existing_metrics = load_metrics_file(metrics_file) or {}
        if verbose:
            print(
                f"Loaded {len(existing_metrics)} existing metrics from {metrics_file}"
            )

    # Prepare tasks for parallel processing
    tasks = []
    for i, frame_file in enumerate(frame_files):
        frame_name = Path(frame_file).stem
        frame_num = frame_name.split("_")[1]

        # Skip if already processed and not forced
        if not force and frame_num in existing_metrics:
            if verbose and i < 10:
                print(f"Skipping already processed frame {frame_name}")
            continue

        # Determine if we should save this structure
        should_save = save_structures and (i < save_limit)
        output_file = (
            os.path.join(output_dir, f"superimposed_{frame_name}.pdb")
            if should_save
            else None
        )

        # Create options dictionary
        options = {
            "match_by": match_by,
            "clash_cutoff": clash_cutoff,
            "verbose": verbose and i < 10,  # Only be verbose for first few frames
        }

        tasks.append((frame_file, ref_pdb_path, output_file, options))

    if verbose:
        print(f"Processing {len(tasks)} frames with {num_processes} parallel processes")

    # Process frames in parallel
    results = []
    if tasks:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = {
                executor.submit(process_frame, task): i for i, task in enumerate(tasks)
            }

            for future in as_completed(futures):
                task_idx = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        if verbose and task_idx < 10:
                            frame_file = tasks[task_idx][0]
                            frame_name = Path(frame_file).stem
                            has_clashes = result.get("has_clashes", False)
                            num_clashes = result.get("num_clashes", 0)
                            total_clashes = result.get("total_clashes", 0)

                            if has_clashes:
                                print(
                                    f"✓ Frame {frame_name}: {num_clashes} clashing atoms, {total_clashes} total clashes"
                                )
                            else:
                                print(f"✓ Frame {frame_name}: No clashes")
                except Exception as e:
                    if verbose:
                        frame_file = tasks[task_idx][0]
                        frame_name = Path(frame_file).stem
                        print(f"✗ Failed to process {frame_name}: {str(e)}")

    # Combine results with existing metrics
    metrics = existing_metrics.copy()
    structures_saved = 0

    for result in results:
        frame_num = result.pop("frame_num", None)
        frame_file = result.pop("frame_file", None)

        if frame_num and frame_file:
            metrics[frame_num] = {
                "rmsd": result.get("rmsd", 0.0),
                "rotation": result.get("rotation"),
                "translation": result.get("translation"),
                "has_clashes": result.get("has_clashes", False),
                "num_clashes": result.get("num_clashes", 0),
                "total_clashes": result.get("total_clashes", 0),
                "matched_atoms": result.get("matched_atoms", 0),
            }

            if result.get("output_file"):
                structures_saved += 1

    # Save metrics to JSON
    if metrics:
        if save_metrics_file(metrics_file, metrics):
            if verbose:
                print(f"Saved metrics for {len(metrics)} frames to {metrics_file}")
        else:
            print(f"Failed to save metrics to {metrics_file}")

        # Save metrics to CSV
        if save_metrics_to_csv(metrics, csv_file):
            if verbose:
                print(f"Saved metrics for {len(metrics)} frames to {csv_file}")
        else:
            print(f"Failed to save metrics to {csv_file}")

    # Summary
    summary = {
        "total_frames": len(frame_files),
        "processed_frames": len(results),
        "total_metrics": len(metrics),
        "structures_saved": structures_saved,
        "metrics_file": metrics_file,
        "csv_file": csv_file,
        "output_dir": output_dir,
    }

    if verbose:
        print(f"\nProcessing complete:")
        print(f"  - Total frames: {summary['total_frames']}")
        print(f"  - Processed in this run: {summary['processed_frames']}")
        print(f"  - Total frames with metrics: {summary['total_metrics']}")
        print(f"  - Structures saved: {summary['structures_saved']}")
        if summary["structures_saved"] > 0:
            print(f"  - Output directory: {summary['output_dir']}")
        print(f"  - Metrics file: {summary['metrics_file']}")
        print(f"  - CSV results: {summary['csv_file']}")

    return summary


def analyze_trajectory_clashes(metrics_file: str) -> Dict[str, Any]:
    """Analyze clash statistics across a trajectory.

    Args:
        metrics_file: Path to metrics file

    Returns:
        Dictionary with clash statistics
    """
    metrics = load_metrics_file(metrics_file)
    if not metrics:
        return {"error": f"Failed to load metrics file: {metrics_file}"}

    total_frames = len(metrics)
    frames_with_clashes = sum(
        1 for m in metrics.values() if m.get("has_clashes", False)
    )
    clash_counts = [m.get("num_clashes", 0) for m in metrics.values()]
    total_clash_counts = [m.get("total_clashes", 0) for m in metrics.values()]

    result = {
        "total_frames": total_frames,
        "frames_with_clashes": frames_with_clashes,
        "clash_percentage": (
            (frames_with_clashes / total_frames * 100) if total_frames > 0 else 0
        ),
        "max_clashing_atoms": max(clash_counts) if clash_counts else 0,
        "avg_clashing_atoms": (
            sum(clash_counts) / total_frames if total_frames > 0 else 0
        ),
        "max_total_clashes": max(total_clash_counts) if total_clash_counts else 0,
        "avg_total_clashes": (
            sum(total_clash_counts) / total_frames if total_frames > 0 else 0
        ),
    }

    # Find frames with most clashes
    if clash_counts:
        max_clash_frame = max(metrics.items(), key=lambda x: x[1].get("num_clashes", 0))
        result["max_clash_frame"] = max_clash_frame[0]
        result["max_clash_frame_count"] = max_clash_frame[1].get("num_clashes", 0)

    return result
