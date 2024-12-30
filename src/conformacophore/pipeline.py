import os
import pandas as pd
from typing import List
from scipy.cluster import hierarchy
from io.pdb_reader import PDBReader
from io.pdb_writer import CustomPDBIO
from contexts.clustering_context import ClusteringContext
from clustering.rmsd_clustering import RMSDClustering
from filters.representatives_finder import RepresentativeFinder
from metrics.optimal_cluster_count import OptimalClusterCount
from visualizers.cluster_visualizer import ClusterVisualizer
from utils.helpers import extract_chains

class Pipeline:
    def __init__(self, input_dir: str, output_dir: str, target_chains: List[str], ligand_chain: str, molecule_chain: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_chains = target_chains
        self.ligand_chain = ligand_chain
        self.molecule_chain = molecule_chain
        self.pdb_reader = PDBReader()
        self.pdb_writer = CustomPDBIO()
        self.clustering_context = ClusteringContext(RMSDClustering())
        self.representative_finder = RepresentativeFinder()
        self.optimal_cluster_count = OptimalClusterCount()
        self.visualizer = ClusterVisualizer()

    def run(self):
        self._prepare_output_directory()
        summary_df = self._read_summary_statistics()
        for _, row in summary_df.iterrows():
            self._process_compound(row['Compound'])
        self._write_analysis_summary()

    def _prepare_output_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _read_summary_statistics(self) -> pd.DataFrame:
        summary_file = os.path.join(self.input_dir, "summary_statistics.csv")
        if not os.path.exists(summary_file):
            raise FileNotFoundError(f"Summary statistics file not found in {self.input_dir}")
        return pd.read_csv(summary_file)

    def _process_compound(self, compound_id: str):
        compound_dir = os.path.join(self.input_dir, str(compound_id))
        if not os.path.exists(compound_dir):
            print(f"Warning: Directory not found for compound {compound_id}")
            return

        pdb_files = [os.path.join(compound_dir, f) for f in os.listdir(compound_dir) if f.endswith('.pdb')]
        if not pdb_files:
            print(f"Warning: No PDB files found for compound {compound_id}")
            return

        print(f"\nProcessing compound {compound_id}")
        print(f"Found {len(pdb_files)} structures")

        trajs = [extract_chains(pdb, [self.molecule_chain]) for pdb in pdb_files]
        rmsd_matrix = self.clustering_context.calculate_rmsd_matrix(trajs)
        if rmsd_matrix is None:
            print(f"Skipping compound {compound_id} due to RMSD calculation error")
            return

        metrics, suggestions, linkage_matrix = self.optimal_cluster_count.get_optimal_clusters(
            rmsd_matrix, max_clusters=10, output_dir=self.output_dir, compound_id=compound_id
        )
        n_clusters = max(set(suggestions.values()), key=list(suggestions.values()).count)
        clusters = hierarchy.fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

        representative, cluster_info = self.representative_finder.find_lowest_rmsd_structure(pdb_files, rmsd_matrix, clusters)
        compound_summary = self._create_compound_summary(compound_id, metrics, suggestions, clusters, rmsd_matrix, cluster_info)
        self._append_summary(compound_summary)

        output_path = os.path.join(self.output_dir, f"compound_{compound_id}_complex.pdb")
        self.pdb_writer.save(representative, output_path)
        self.visualizer.create_visualizations(rmsd_matrix, clusters, linkage_matrix, suggestions['elbow'], compound_id, self.output_dir)

    def _create_compound_summary(self, compound_id: str, metrics: dict, suggestions: dict, clusters: List[int], rmsd_matrix: List[List[float]], cluster_info: dict) -> dict:
        # Implementation of creating the summary dictionary goes here
        pass

    def _append_summary(self, summary: dict):
        if not hasattr(self, 'summaries'):
            self.summaries = []
        self.summaries.append(summary)

    def _write_analysis_summary(self):
        summary_path = os.path.join(self.output_dir, "analysis_summary.csv")
        df = pd.DataFrame(self.summaries)
        df = df.sort_values('compound_id')
        df.to_csv(summary_path, index=False)
        print(f"\nWrote complete analysis summary to {summary_path}")
        return summary_path
