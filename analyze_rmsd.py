#!/usr/bin/env python3
# analyze_rmsd.py

"""
Analyze RMSD results from superimposition to identify the most diverse frames.
"""

import os
import csv
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_rmsd_data(csv_file):
    """Load RMSD data from a CSV file.

    Args:
        csv_file: Path to CSV file containing superimposition results

    Returns:
        List of dictionaries with frame data
    """
    frames = []
    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip rows with no RMSD value
                if not row["rmsd"] or row["rmsd"] == "None":
                    continue

                # Convert RMSD to float
                row["rmsd"] = float(row["rmsd"])

                # Add to frames list
                frames.append(row)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

    # Sort by RMSD (descending)
    frames.sort(key=lambda x: x["rmsd"], reverse=True)
    return frames


def generate_visualizations(frames, output_dir):
    """Generate visualizations of RMSD distribution.

    Args:
        frames: List of dictionaries with frame data
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract RMSD values
    rmsd_values = [frame["rmsd"] for frame in frames]

    # Generate histogram
    plt.figure(figsize=(10, 6))
    plt.hist(rmsd_values, bins=30, alpha=0.7, color="blue")
    plt.xlabel("RMSD (Å)")
    plt.ylabel("Frequency")
    plt.title("Distribution of RMSD Values")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "rmsd_distribution.png"), dpi=300)
    plt.close()

    # Generate sorted RMSD plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(len(rmsd_values)), sorted(rmsd_values, reverse=True), "-o", markersize=3
    )
    plt.xlabel("Frame Rank")
    plt.ylabel("RMSD (Å)")
    plt.title("Sorted RMSD Values")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "rmsd_sorted.png"), dpi=300)
    plt.close()


def save_top_diverse_frames(frames, output_dir, top_n=10):
    """Save the top diverse frames to a CSV file.

    Args:
        frames: List of dictionaries with frame data
        output_dir: Directory to save output
        top_n: Number of top diverse frames to save
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Limit to top n frames
    top_frames = frames[:top_n]

    # Create CSV file
    output_file = os.path.join(output_dir, "top_diverse_frames.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Frame", "RMSD"])
        for i, frame in enumerate(top_frames):
            writer.writerow([i + 1, os.path.basename(frame["frame"]), frame["rmsd"]])

    return output_file


def analyze_rmsd(csv_file, output_dir, top_n=10):
    """Analyze RMSD results from superimposition.

    Args:
        csv_file: Path to CSV file containing superimposition results
        output_dir: Directory to save output
        top_n: Number of top diverse frames to report
    """
    # Load RMSD data
    frames = load_rmsd_data(csv_file)
    if not frames:
        print("No valid RMSD data found")
        return

    # Calculate statistics
    rmsd_values = [frame["rmsd"] for frame in frames]
    min_rmsd = float(np.min(rmsd_values))
    max_rmsd = float(np.max(rmsd_values))
    mean_rmsd = float(np.mean(rmsd_values))
    median_rmsd = float(np.median(rmsd_values))
    std_rmsd = float(np.std(rmsd_values))

    # Print statistics
    print(f"Total frames: {len(frames)}")
    print(f"Minimum RMSD: {min_rmsd:.4f}")
    print(f"Maximum RMSD: {max_rmsd:.4f}")
    print(f"Mean RMSD: {mean_rmsd:.4f}")
    print(f"Median RMSD: {median_rmsd:.4f}")
    print(f"Standard deviation: {std_rmsd:.4f}")

    # Generate visualizations
    generate_visualizations(frames, output_dir)

    # Save top diverse frames
    save_top_diverse_frames(frames, output_dir, top_n)

    # Print top diverse frames
    print("\nTop diverse frames:")
    for i, frame in enumerate(frames[:top_n]):
        print(f"{i+1}. {frame['frame']}: {frame['rmsd']:.4f}")

    print(f"\nAnalysis results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RMSD results from superimposition"
    )
    parser.add_argument("csv_file", help="CSV file containing superimposition results")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=10,
        help="Number of top diverse frames to report",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} does not exist")
        return 1

    # Set default output directory to be in the parent directory of the CSV file
    output_dir = args.output
    if output_dir is None:
        csv_path = Path(args.csv_file)
        parent_dir = csv_path.parent
        output_dir = str(parent_dir / "rmsd_analysis")

    analyze_rmsd(args.csv_file, output_dir, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
