from Bio.PDB.PDBIO import PDBIO
from src.conformacophore.handlers.pdb_header_handler import PDBHeaderHandler
from typing import Optional

class CustomPDBIO(PDBIO):
    def __init__(self):
        super().__init__()
        self._header_handler: Optional[PDBHeaderHandler] = None

    @property
    def header_handler(self) -> Optional[PDBHeaderHandler]:
        return self._header_handler

    @header_handler.setter
    def header_handler(self, value: Optional[PDBHeaderHandler]):
        self._header_handler = value

    def save(self, file, select=None, write_end: bool = True, preserve_atom_numbering: bool = False):
        if isinstance(file, str):
            fhandle = open(file, "w")
            close_file = True
        else:
            fhandle = file
            close_file = False

        if isinstance(file, str) and self._header_handler:
            self._header_handler.write_headers(fhandle)

        try:
            super().save(fhandle, select=select, write_end=write_end, preserve_atom_numbering=preserve_atom_numbering)
            if self._header_handler:
                self._header_handler.write_connectivity(fhandle)
        finally:
            if close_file:
                fhandle.close()
