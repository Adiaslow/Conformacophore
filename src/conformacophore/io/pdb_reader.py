from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from typing import Optional

class PDBReader:
    """Handles reading of PDB files."""

    @staticmethod
    def get_structure_from_model(filepath: str, model_num: int = 0) -> Structure:
        parser = PDBParser(QUIET=True)
        full_structure = parser.get_structure('model', filepath)

        if len(full_structure) <= model_num:
            return full_structure[0]

        selected_structure = Structure('selected')
        selected_model = Model(0)
        selected_structure.add(selected_model)

        for chain in full_structure[model_num]:
            new_chain = chain.copy()
            selected_model.add(new_chain)

        return selected_structure
