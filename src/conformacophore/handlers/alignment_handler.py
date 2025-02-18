import numpy as np
from typing import List, Optional, Tuple, Dict
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from src.conformacophore.handlers.pdb_handler import PDBHandler
from src.conformacophore.contexts.alignment_context import AlignmentContext
from src.conformacophore.results.alignment_result import AlignmentResult
from src.conformacophore.strategies.alignment_strategy import AlignmentStrategy
from src.conformacophore.alignment.clash_detector import ClashDetector
from src.conformacophore.entities.molecular_graph import MolecularGraph


class AlignmentHandler:
    """Handles molecular structure alignment operations."""

    def __init__(
        self,
        alignment_context: AlignmentContext,
        clash_detector: ClashDetector,
        pdb_handler: PDBHandler
    ):
        """
        Initialize alignment handler.

        Args:
            alignment_context: Context for alignment strategies
            clash_detector: Detector for steric clashes
            pdb_handler: Handler for PDB file operations
        """
        self._context = alignment_context
        self._clash_detector = clash_detector
        self._pdb_handler = pdb_handler

    def set_strategy(self, strategy: AlignmentStrategy) -> None:
        """
        Set the alignment strategy.

        Args:
            strategy: New alignment strategy to use
        """
        self._context.set_strategy(strategy)

    def align_structures(
        self,
        ligand_path: str,
        target_path: str,
        output_dir: Optional[str] = None,
        ligand_chain: str = 'A',
        target_chain: str = 'A',
        new_chain: str = 'B',
        target_chains: Optional[List[str]] = None,
        clash_cutoff: float = 2.0,
        write_pdbs: bool = False,
        model_number: Optional[int] = None
    ) -> Tuple[List[AlignmentResult], Dict]:
        """
        Align structures and check for clashes.

        Args:
            ligand_path: Path to reference ligand PDB
            target_path: Path to target structure PDB
            output_dir: Directory for output files (optional)
            ligand_chain: Chain ID in ligand PDB
            target_chain: Chain ID in target PDB
            new_chain: Chain ID for aligned structure
            target_chains: Chain IDs to check for clashes
            clash_cutoff: Distance cutoff for clash detection
            write_pdbs: Whether to write aligned structures
            model_number: Specific model to align (optional)

        Returns:
            Tuple of (list of alignment results, model information dict)
        """
        try:
            # Extract specific model from target structure
            target_struct = self._pdb_handler.get_structure_from_model(
                target_path, model_number or 0
            )

            # Get ligand structure
            ligand_struct = self._pdb_handler.get_structure_from_model(
                ligand_path, 0  # Always use first model for reference
            )

            # Validate chains exist
            if target_chain not in target_struct[0]:
                raise ValueError(f"Target chain {target_chain} not found")
            if ligand_chain not in ligand_struct[0]:
                raise ValueError(f"Ligand chain {ligand_chain} not found")

            # Create molecular graphs
            ligand_graph = MolecularGraph.from_pdb_file(ligand_path)
            target_graph = MolecularGraph.from_pdb_file(target_path)

            # Perform alignment
            result = self._context.align(ligand_graph, target_graph)

            if result.transformation_matrix is not None and target_chains:
                # Create combined structure for clash detection
                combined = Structure('combined')
                model = self._create_combined_model(
                    ligand_struct,
                    target_struct,
                    result.transformation_matrix,
                    ligand_chain,
                    target_chain,
                    new_chain
                )
                combined.add(model)

                # Check for clashes
                new_chain_obj = model[new_chain]
                clash_result = self._clash_detector.detect_clashes(
                    combined,
                    new_chain_obj,
                    target_chains,
                    clash_cutoff
                )
                result.clash_results = clash_result

                # Write PDB if requested
                if write_pdbs and output_dir:
                    self._write_aligned_structure(
                        combined,
                        output_dir,
                        model_number,
                        target_path
                    )

            return [result], self._get_model_info(target_path, model_number or 0)

        except Exception as e:
            print(f"Error in alignment: {str(e)}")
            return [self._empty_result()], {}

    def _create_combined_model(
        self,
        ligand_struct: Structure,
        target_struct: Structure,
        transformation: Tuple[np.ndarray, np.ndarray],
        ligand_chain: str,
        target_chain: str,
        new_chain: str
    ) -> 'Model':
        """Create a combined model with transformed structures."""
        from Bio.PDB.Model import Model
        from Bio.PDB.Chain import Chain

        model = Model(0)

        # Add original chains
        for chain_id in ligand_struct[0]:
            model.add(ligand_struct[0][chain_id].copy())

        # Add transformed target chain with new ID
        new_chain_obj = Chain(new_chain)
        old_chain = target_struct[0][target_chain]

        for residue in old_chain:
            new_chain_obj.add(residue.copy())

        # Apply transformation
        rotation, translation = transformation
        for atom in new_chain_obj.get_atoms():
            atom.transform(rotation, translation)

        model.add(new_chain_obj)
        return model

    def _write_aligned_structure(
        self,
        structure: Structure,
        output_dir: str,
        model_number: Optional[int],
        source_path: str
    ) -> None:
        """Write aligned structure to file."""
        import os

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        output_path = os.path.join(
            output_dir,
            f"{base_name}_aligned_{model_number or 1}.pdb"
        )

        self._pdb_handler.save_structure(structure, output_path)

    @staticmethod
    def _empty_result() -> AlignmentResult:
        """Create an empty alignment result."""
        return AlignmentResult(
            rmsd=float('inf'),
            matched_atoms=0,
            transformation_matrix=None,
            matched_pairs=[],
            clash_results=None
        )

    def _get_model_info(self, pdb_path: str, model_num: int) -> Dict:
        """Get information about a specific model from PDB file."""
        # This would use your existing header handler functionality
        return self._pdb_handler.header_handler.get_model_info(pdb_path, model_num)
