import os
import pandas as pd
import shutil
from typing import List, Dict
import argparse


class PDBHeaderHandler:
    """Handles reading and writing of PDB header information."""

    def __init__(self):
        self.headers = []
        self.model_headers = {}  # Headers specific to each model

    def read_headers(self, pdb_path: str):
        """Read header information from PDB file."""
        self.headers = []
        self.model_headers = {}

        # Header keywords to capture
        header_keywords = [
            "HEADER",
            "TITLE",
            "COMPND",
            "SOURCE",
            "KEYWDS",
            "EXPDTA",
            "AUTHOR",
            "REVDAT",
            "REMARK",
            "SEQRES",
        ]

        with open(pdb_path, "r") as f:
            lines = f.readlines()

        # First pass: capture global headers before any MODEL
        for line in lines:
            if any(line.startswith(keyword) for keyword in header_keywords):
                self.headers.append(line)

            if line.startswith("MODEL"):
                break

        # Second pass: capture model-specific headers
        current_model = -1
        reading_model = False

        for line in lines:
            if line.startswith("MODEL"):
                current_model += 1
                reading_model = True
                self.model_headers[current_model] = []

            elif line.startswith("ENDMDL"):
                reading_model = False

            elif reading_model:
                # Capture model-specific headers
                if any(
                    line.startswith(keyword)
                    for keyword in ["COMPND", "REMARK", "SEQRES"]
                ):
                    self.model_headers[current_model].append(line)

        # If no models found, treat all headers as global
        if not self.model_headers:
            self.model_headers[0] = []

    def write_headers(self, file_handle):
        """Write header information to file."""
        # Write global headers
        for header in self.headers:
            file_handle.write(header)

    def write_model_headers(self, file_handle, model_num: int):
        """Write model-specific headers to file."""
        if model_num in self.model_headers:
            for header in self.model_headers[model_num]:
                file_handle.write(header)


def filter_results(
    input_dir: str,
    output_dir: str,
    rmsd_threshold: float,
    clash_percentage_threshold: float,
) -> None:
    """
    Filter superimposition results based on RMSD and percentage of clashing conformers.

    Args:
        input_dir: Directory containing original superimposition results
        output_dir: Directory to store filtered results
        rmsd_threshold: Maximum allowed RMSD
        clash_percentage_threshold: Maximum allowed percentage of conformers with clashes
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the main summary statistics file
    summary_path = os.path.join(input_dir, "summary_statistics.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary statistics file not found in {input_dir}")

    summary_df = pd.read_csv(summary_path)
    filtered_compounds: List[Dict] = []

    # Process each compound
    for _, row in summary_df.iterrows():
        compound_id = row["Compound"]
        compound_csv = os.path.join(input_dir, f"{compound_id}_results.csv")

        if not os.path.exists(compound_csv):
            print(f"Warning: Results CSV not found for compound {compound_id}")
            continue

        # Read compound-specific results
        compound_df = pd.read_csv(compound_csv)

        # Calculate percentage of conformers with clashes
        total_conformers = len(compound_df)
        clashing_conformers = compound_df["Has_Clashes"].sum()
        clash_percentage = (clashing_conformers / total_conformers) * 100

        # First filter: Check if compound meets clash percentage threshold
        if clash_percentage <= clash_percentage_threshold:
            # Second filter: Get only non-clashing conformers that meet RMSD threshold
            filtered_df = compound_df[
                (~compound_df["Has_Clashes"]) & (compound_df["RMSD"] <= rmsd_threshold)
            ]

            if len(filtered_df) > 0:
                # Save filtered compound results
                filtered_df.to_csv(
                    os.path.join(output_dir, f"{compound_id}_results.csv"), index=False
                )

                # Copy corresponding PDB files with headers preserved
                compound_dir = os.path.join(input_dir, str(compound_id))
                filtered_output_dir = os.path.join(output_dir, str(compound_id))
                os.makedirs(filtered_output_dir, exist_ok=True)

                for _, model_row in filtered_df.iterrows():
                    model_num = model_row["Model"]
                    pdb_name = f"aligned_conf_{model_num}.pdb"
                    src_path = os.path.join(compound_dir, pdb_name)
                    if os.path.exists(src_path):
                        # Read headers from the source file
                        header_handler = PDBHeaderHandler()
                        header_handler.read_headers(src_path)

                        # Write to new file with headers preserved
                        dest_path = os.path.join(filtered_output_dir, pdb_name)
                        with open(src_path, "r") as src_file, open(
                            dest_path, "w"
                        ) as dest_file:
                            # Write global headers
                            header_handler.write_headers(dest_file)

                            # Write model-specific headers for this model
                            header_handler.write_model_headers(dest_file, model_num - 1)

                            # Copy the rest of the file (contents after headers)
                            model_started = False
                            for line in src_file:
                                if line.startswith(f"MODEL        {model_num}"):
                                    model_started = True

                                if model_started:
                                    dest_file.write(line)

                # Calculate new summary statistics for this compound
                summary_stats = {
                    "Compound": compound_id,
                    "Sequence": row["Sequence"],
                    "Original_Num_Conformers": total_conformers,
                    "Original_Clash_Percentage": clash_percentage,
                    "Filtered_Num_Conformers": len(filtered_df),
                    "Min_RMSD": filtered_df["RMSD"].min(),
                    "Max_RMSD": filtered_df["RMSD"].max(),
                    "Mean_RMSD": filtered_df["RMSD"].mean(),
                    "Median_RMSD": filtered_df["RMSD"].median(),
                    "StdDev_RMSD": filtered_df["RMSD"].std(),
                    "Min_Matched_Atoms": filtered_df["Matched_Atoms"].min(),
                    "Max_Matched_Atoms": filtered_df["Matched_Atoms"].max(),
                    "Best_RMSD_Model": filtered_df.loc[
                        filtered_df["RMSD"].idxmin(), "Model"
                    ],
                    "Best_RMSD_Frame": filtered_df.loc[
                        filtered_df["RMSD"].idxmin(), "Frame"
                    ],
                }

                filtered_compounds.append(summary_stats)

                # Print progress
                print(f"\nFiltered results for compound {compound_id}:")
                print(f"Original conformers: {total_conformers}")
                print(f"Original clash percentage: {clash_percentage:.1f}%")
                print(f"Filtered clash-free conformers: {len(filtered_df)}")
                print(
                    f"RMSD range: {summary_stats['Min_RMSD']:.4f} - {summary_stats['Max_RMSD']:.4f}"
                )
        else:
            print(
                f"\nSkipping compound {compound_id} - clash percentage {clash_percentage:.1f}% exceeds threshold {clash_percentage_threshold}%"
            )

    # Create new summary statistics file
    if filtered_compounds:
        filtered_summary_df = pd.DataFrame(filtered_compounds)
        filtered_summary_df.to_csv(
            os.path.join(output_dir, "summary_statistics.csv"), index=False
        )
        print(f"\nFiltered {len(filtered_compounds)} compounds saved to {output_dir}")

        # Print overall statistics
        print("\nOverall statistics for filtered dataset:")
        print(f"Total compounds: {len(filtered_compounds)}")
        print(
            f"Average number of conformers per compound: {filtered_summary_df['Filtered_Num_Conformers'].mean():.1f}"
        )
        print(f"Average RMSD: {filtered_summary_df['Mean_RMSD'].mean():.4f}")
    else:
        print("\nNo compounds met the filtering criteria")


def main():
    parser = argparse.ArgumentParser(
        description="Filter superimposition results based on RMSD and clash percentage thresholds"
    )
    parser.add_argument(
        "input_dir", help="Directory containing original superimposition results"
    )
    parser.add_argument("output_dir", help="Directory to store filtered results")
    parser.add_argument(
        "--rmsd", type=float, required=True, help="Maximum allowed RMSD"
    )
    parser.add_argument(
        "--clash-percent",
        type=float,
        required=True,
        help="Maximum allowed percentage of conformers with clashes (0-100)",
    )

    args = parser.parse_args()

    if not 0 <= args.clash_percent <= 100:
        parser.error("Clash percentage must be between 0 and 100")

    filter_results(args.input_dir, args.output_dir, args.rmsd, args.clash_percent)


if __name__ == "__main__":
    main()

"""
Example Usage:
python filter_superimposition_results.py /Users/Adam/Desktop/hexamers_water_data /Users/Adam/Desktop/hexamers_water_data_filtered --rmsd 2 --clash-percent 50

python filter_superimposition_results.py /Users/Adam/Desktop/hexamers_chc13_data /Users/Adam/Desktop/hexamers_chc13_data_filtered --rmsd 2 --clash-percent 50

python filter_superimposition_results.py /Users/Adam/Desktop/heptamers_water_data /Users/Adam/Desktop/heptamers_water_data_filtered --rmsd 2 --clash-percent 75

python filter_superimposition_results.py /Users/Adam/Desktop/heptamers_chc13_data /Users/Adam/Desktop/heptamers_chc13_data_filtered --rmsd 2 --clash-percent 50
"""
