from Bio.PDB.Structure import Structure
import os
from typing import Optional
from src.conformacophore.handlers.pdb_header_handler import PDBHeaderHandler

class EnhancedPDBIO:
    """Enhanced PDB writer that preserves model-specific information."""

    def __init__(self):
        self.header_handler = PDBHeaderHandler()

    def save(self, structure: Structure, filepath: str, model_num: Optional[int] = None):
        """Save structure while preserving all relevant metadata."""
        # If original file exists, read its headers
        original_filepath = getattr(structure, 'original_filepath', None)
        if original_filepath and os.path.exists(original_filepath):
            self.header_handler.read_headers(original_filepath)

        with open(filepath, 'w') as f:
            # Write headers
            if model_num is not None:
                self.header_handler.write_model_information(f, model_num)

            # Write the structure
            model_count = 1
            for model in structure:
                f.write(f'MODEL{model_count:>9d}\n')
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            f.write(self._format_atom(atom))
                f.write('ENDMDL\n')
                model_count += 1

            # Write CONECT records if present
            for conect in self.header_handler.conect_records:
                f.write(conect)

            f.write('END\n')

    def _format_atom(self, atom) -> str:
        """Format atom record according to PDB specification."""
        # Handle chain ID differently since it might be a space
        chain_id = atom.parent.parent.id
        if chain_id == " ":
            chain_id = ""

        # Format coordinates with exactly 3 decimal places
        x = f"{atom.coord[0]:8.3f}"
        y = f"{atom.coord[1]:8.3f}"
        z = f"{atom.coord[2]:8.3f}"

        # Handle occupancy and B-factor (temperature factor)
        occupancy = getattr(atom, "occupancy", 1.0)
        bfactor = getattr(atom, "bfactor", 0.0)

        return (f"ATOM  {atom.serial_number:>5d} {atom.name:<4s}{atom.parent.resname:>3s} "
               f"{chain_id:1s}{atom.parent.id[1]:>4d}    "
               f"{x}{y}{z}"
               f"{occupancy:6.2f}{bfactor:6.2f}          "
               f"{atom.element:>2s}  \n")
