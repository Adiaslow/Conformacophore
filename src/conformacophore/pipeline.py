from typing import List
import os
import pandas as pd
from src.conformacophore.handlers.pdb_handler import PDBHandler
from src.conformacophore.handlers.clustering_handler import ClusteringHandler
from src.conformacophore.handlers.visualization_handler import VisualizationHandler
from src.conformacophore.handlers.summary_handler import SummaryHandler
from src.conformacophore.handlers.alignment_handler import AlignmentHandler
from src.conformacophore.handlers.alignment_handler import AlignmentHandler
from src.conformacophore.contexts.alignment_context import AlignmentContext
from src.conformacophore.alignment.clash_detector import ClashDetector
from src.conformacophore.strategies.isomorphic_alignment_strategy import IsomorphicAlignmentStrategy

class Pipeline:
    """Main pipeline for conformer analysis."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        reference_pdb: str,
        target_chains: List[str],
        ligand_chain: str,
        molecule_chain: str,
        clash_cutoff: float = 2.0
    ):
        """
        Initialize pipeline with configuration.

        Args:
            input_dir: Directory containing input PDB files
            output_dir: Directory for output files
            reference_pdb: Path to reference PDB file containing target protein
            target_chains: List of protein chain IDs to check for clashes
            ligand_chain: Chain ID of ligand in reference PDB
            molecule_chain: Chain ID in input PDB files
            clash_cutoff: Distance threshold for clash detection (Angstroms)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.reference_pdb = reference_pdb
        self.target_chains = target_chains
        self.ligand_chain = ligand_chain
        self.molecule_chain = molecule_chain
        self.clash_cutoff = clash_cutoff

        # Initialize handlers
        self.pdb_handler = PDBHandler()
        self.clustering_handler = ClusteringHandler()
        self.visualization_handler = VisualizationHandler()
        self.summary_handler = SummaryHandler()

        # Initialize alignment components
        alignment_context = AlignmentContext(IsomorphicAlignmentStrategy())
        clash_detector = ClashDetector()
        self.alignment_handler = AlignmentHandler(
            alignment_context,
            clash_detector,
            self.pdb_handler
        )

    def run(self):
        """Execute the complete analysis pipeline."""
        self._prepare_output_directory()
        summary_df = self._read_summary_statistics()

        for _, row in summary_df.iterrows():
            self._process_compound(row['Compound'])

        self.summary_handler.write_analysis_summary(self.output_dir)

    def _prepare_output_directory(self):
        """Create output directory and subdirectories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'aligned_structures'), exist_ok=True)

    def _read_summary_statistics(self) -> pd.DataFrame:
        """Read summary statistics from input directory."""
        summary_file = os.path.join(self.input_dir, "summary_statistics.csv")
        if not os.path.exists(summary_file):
            raise FileNotFoundError(f"Summary statistics file not found in {self.input_dir}")
        return pd.read_csv(summary_file)

    def _process_compound(self, compound_id: str):
        """Process a single compound through the pipeline."""
        compound_dir = os.path.join(self.input_dir, str(compound_id))
        if not os.path.exists(compound_dir):
            print(f"Warning: Directory not found for compound {compound_id}")
            return

        pdb_files = [os.path.join(compound_dir, f) for f in os.listdir(compound_dir)
                    if f.endswith('.pdb')]

        if not pdb_files:
            print(f"Warning: No PDB files found for compound {compound_id}")
            return

        print(f"\nProcessing compound {compound_id}")
        print(f"Found {len(pdb_files)} structures")

        # Step 1: Alignment and clash detection
        aligned_structures = []
        for pdb_file in pdb_files:
            alignment_results, _ = self.alignment_handler.align_structures(
                self.reference_pdb,
                pdb_file,
                os.path.join(self.output_dir, 'aligned_structures'),
                self.ligand_chain,
                self.molecule_chain,
                'X',  # New chain ID
                self.target_chains,
                self.clash_cutoff,
                write_pdbs=True
            )

            # Filter valid alignments
            for result in alignment_results:
                if (result.rmsd < float('inf') and
                    result.clash_results and
                    not result.clash_results.has_clashes):
                    aligned_structures.append(pdb_file)

        if not aligned_structures:
            print(f"No valid alignments found for compound {compound_id}")
            return

        # Step 2: Extract chains and calculate RMSD matrix
        trajs = [self.pdb_handler.extract_chains(pdb, [self.molecule_chain])
                for pdb in aligned_structures]

        rmsd_matrix = self.clustering_handler.calculate_rmsd_matrix(trajs)

        if rmsd_matrix is None:
            print(f"Skipping compound {compound_id} due to RMSD calculation error")
            return

        # Step 3: Determine optimal clusters and cluster the data
        metrics, suggestions, linkage_matrix = self.clustering_handler.get_optimal_clusters(
            rmsd_matrix,
            max_clusters=10,
            output_dir=self.output_dir,
            compound_id=compound_id
        )

        n_clusters = max(set(suggestions.values()), key=list(suggestions.values()).count)
        clusters = self.clustering_handler.get_clusters(linkage_matrix, n_clusters)

        # Step 4: Find representative structure
        representative, cluster_info = self.clustering_handler.find_lowest_rmsd_structure(
            aligned_structures,
            rmsd_matrix,
            clusters
        )

        # Step 5: Create summary
        compound_summary = self.summary_handler.create_compound_summary(
            compound_id,
            metrics,
            suggestions,
            clusters,
            rmsd_matrix,
            cluster_info
        )
        self.summary_handler.append_summary(compound_summary)

        # Step 6: Save representative structure
        output_path = os.path.join(self.output_dir, f"compound_{compound_id}_complex.pdb")
        self.pdb_handler.save_representative_structure(representative, output_path)

        # Step 7: Create visualizations
        self.visualization_handler.create_visualizations(
            rmsd_matrix,
            clusters,
            linkage_matrix,
            suggestions['elbow'],
            compound_id,
            self.output_dir
        )
