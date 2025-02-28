import pyrosetta
from pyrosetta.rosetta.protocols.docking import DockMCMProtocol
import os
import pandas as pd
from typing import List, Tuple, Dict
import glob
import json

def initialize_rosetta(extra_options: List[str] = None) -> None:
    """Initialize PyRosetta with optional extra options."""
    init_options = ["-ex1", "-ex2", "-use_input_sc", "-ignore_unrecognized_res",
                   "-restore_talaris_behavior", "-no_optH false"]
    if extra_options:
        init_options.extend(extra_options)
    pyrosetta.init(' '.join(init_options))

def setup_docking_protocol(
    score_function: str = "ref2015"
) -> Tuple[DockMCMProtocol, pyrosetta.ScoreFunction]:
    """Setup docking protocol and scoring function."""
    sf = pyrosetta.create_score_function(score_function)
    sf_pack = pyrosetta.create_score_function(score_function)
    dock_prot = DockMCMProtocol(1, sf, sf_pack)
    return dock_prot, sf

def perform_docking(
    complex_pose: pyrosetta.Pose,
    target_chains: str,
    molecule_chain: str,
    n_decoys: int = 150,
    compound_id: str = None
) -> List[Dict]:
    """
    Perform docking simulation and return sorted results with metadata.

    Args:
        complex_pose: Pose object containing both target and molecule
        target_chains: String of target protein chains (e.g., 'AB')
        molecule_chain: Chain ID of the molecule to dock (e.g., 'X')
        n_decoys: Number of docking decoys to generate
        compound_id: Identifier for the compound being docked
    """
    docking, sf = setup_docking_protocol()
    results = []

    # Set up docking partners
    docking.set_partners(f"{target_chains}_{molecule_chain}")

    # Generate decoys
    for i in range(n_decoys):
        current_pose = complex_pose.clone()
        docking.apply(current_pose)
        score = sf(current_pose)

        # Store result with metadata
        result = {
            'pose': current_pose.clone(),
            'score': score,
            'compound_id': compound_id,
            'decoy_number': i + 1
        }
        results.append(result)
        print(f"Completed decoy {i+1}/{n_decoys} for compound {compound_id}, Score: {score:.2f}")

    # Sort by score
    results.sort(key=lambda x: x['score'])
    return results

def save_docking_results(results: List[Dict], output_dir: str) -> None:
    """
    Save docking results to PDB files and create a summary CSV.

    Args:
        results: List of dictionaries containing docking results and metadata
        output_dir: Directory to save results
    """
    # Create compound-specific directory
    compound_id = results[0]['compound_id']
    compound_dir = os.path.join(output_dir, f"docking_compound_{compound_id}")
    os.makedirs(compound_dir, exist_ok=True)

    # Prepare summary data
    summary_data = []

    # Save all decoys
    for result in results:
        score = result['score']
        decoy_num = result['decoy_number']

        # Save PDB file
        pdb_path = os.path.join(compound_dir, f"decoy_{decoy_num}_score_{score:.2f}.pdb")
        result['pose'].dump_pdb(pdb_path)

        # Add to summary data
        summary_data.append({
            'compound_id': compound_id,
            'decoy_number': decoy_num,
            'score': score,
            'pdb_path': pdb_path
        })

    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(compound_dir, "docking_summary.csv"), index=False)

    print(f"\nSaved {len(results)} docking results for compound {compound_id}")
    print(f"Best score: {results[0]['score']:.2f}")
    print(f"Average score: {sum(r['score'] for r in results)/len(results):.2f}")

def process_clustering_results(
    clustering_dir: str,
    target_chains: List[str],
    molecule_chain: str,
    output_dir: str,
    n_decoys: int = 150
) -> None:
    """
    Process clustering results and perform docking on representative structures.

    Args:
        clustering_dir: Directory containing clustering results
        target_chains: List of chain IDs for the target protein
        molecule_chain: Chain ID for the molecule to dock
        output_dir: Directory to save docking results
        n_decoys: Number of docking decoys to generate per compound
    """
    # Initialize PyRosetta
    initialize_rosetta()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all representative structure PDBs
    complex_pdbs = glob.glob(os.path.join(clustering_dir, "compound_*_complex.pdb"))

    if not complex_pdbs:
        raise FileNotFoundError(f"No representative structures found in {clustering_dir}")

    # Process each structure
    all_summaries = []
    for pdb_path in complex_pdbs:
        # Extract compound ID from filename
        compound_id = os.path.basename(pdb_path).split("_")[1]
        print(f"\nProcessing compound {compound_id}")

        try:
            # Load the complex
            complex_pose = pyrosetta.pose_from_pdb(pdb_path)

            # Perform docking
            results = perform_docking(
                complex_pose,
                ''.join(target_chains),
                molecule_chain,
                n_decoys=n_decoys,
                compound_id=compound_id
            )

            # Save results
            save_docking_results(results, output_dir)

            # Collect summary data
            summary_df = pd.read_csv(os.path.join(output_dir, f"docking_compound_{compound_id}", "docking_summary.csv"))
            all_summaries.append(summary_df)

        except Exception as e:
            print(f"Error processing compound {compound_id}: {str(e)}")
            continue

    # Combine all summaries into one file
    if all_summaries:
        combined_df = pd.concat(all_summaries, ignore_index=True)
        combined_df.to_csv(os.path.join(output_dir, "complete_docking_summary.csv"), index=False)
        print("\nSaved complete docking summary")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Perform docking on representative structures')
    parser.add_argument('clustering_dir', help='Directory containing clustering results')
    parser.add_argument('output_dir', help='Directory to save docking results')
    parser.add_argument('--target-chains', nargs='+', required=True,
                      help='Chain IDs for target protein (e.g., A B)')
    parser.add_argument('--molecule-chain', required=True,
                      help='Chain ID for molecule to dock')
    parser.add_argument('--n-decoys', type=int, default=150,
                      help='Number of docking decoys to generate per compound')

    args = parser.parse_args()

    process_clustering_results(
        args.clustering_dir,
        args.target_chains,
        args.molecule_chain,
        args.output_dir,
        args.n_decoys
    )

if __name__ == "__main__":
    main()

"""
Example Usage:

python dock_representatives.py /Users/Adam/Desktop/hexamers_water_data_filtered_representatives /Users/Adam/Desktop/hexamers_water_data_filtered_representatives_docked --target-chains A B C --molecule-chain X
"""
