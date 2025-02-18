from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
import os
import mdtraj as md
from typing import List, Optional, Tuple

from src.conformacophore.io.pdb_reader import PDBReader
from src.conformacophore.io.pdb_writer import CustomPDBIO
from src.conformacophore.handlers.pdb_header_handler import PDBHeaderHandler
from src.conformacophore.utils.helpers import extract_chains
from src.conformacophore.io.enhanced_pdb_io import EnhancedPDBIO


class PDBHandler:
    """Main handler for PDB file operations."""

    def __init__(self):
        self.writer = EnhancedPDBIO()
        self.header_handler = PDBHeaderHandler()

    def get_structure_from_model(self, filepath: str, model_num: int = 0) -> Structure:
        """Get a specific model from a PDB file while preserving metadata."""
        parser = PDBParser(QUIET=True)
        full_structure = parser.get_structure('model', filepath)

        if len(full_structure) <= model_num:
            raise ValueError(f"Model number {model_num} is out of range")

        # Create new structure with selected model
        selected_structure = Structure('selected')
        selected_model = Model(0)
        selected_structure.add(selected_model)

        # Copy the model's chains
        for chain in full_structure[model_num]:
            new_chain = chain.copy()
            selected_model.add(new_chain)

        # Store original filepath for metadata preservation
        selected_structure.original_filepath = filepath

        return selected_structure

    def extract_chains(self, pdb_file: str, chain_letters: List[str]) -> Tuple[Structure, md.Trajectory]:
        """Extract specified chains while preserving relevant metadata."""
        # Get the structure and trajectory
        structure = self.get_structure_from_model(pdb_file)
        traj = md.load(pdb_file)

        # Filter chains in the trajectory
        chain_indices = []
        for chain in structure.get_chains():
            if chain.id in chain_letters:
                chain_indices.extend([atom.get_serial_number() - 1
                                   for atom in chain.get_atoms()])

        filtered_traj = traj.atom_slice(chain_indices)

        return structure, filtered_traj

    def save_structure(self, structure: Structure, output_path: str, model_num: Optional[int] = None):
        """Save structure while preserving all relevant metadata."""
        self.writer.save(structure, output_path, model_num)
