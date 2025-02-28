"""Utility for handling PDB file headers."""

from typing import Dict, List, Optional
import os


class PDBHeaderHandler:
    """Handles reading and writing of PDB header information."""

    def __init__(self):
        """Initialize header storage."""
        self.headers: List[str] = []
        self.model_headers: Dict[int, List[str]] = {}
        self.connectivity: List[str] = []
        self.remarks: Dict[int, List[str]] = {}
        self.other_records: Dict[str, List[str]] = {
            "SEQRES": [],
            "COMPND": [],
            "SOURCE": [],
            "KEYWDS": [],
            "EXPDTA": [],
            "TITLE": [],
            "AUTHOR": [],
            "REVDAT": [],
            "DBREF": [],
            "SEQADV": [],
            "HET": [],
            "HETNAM": [],
            "HETSYN": [],
            "FORMUL": [],
            "HELIX": [],
            "SHEET": [],
            "TURN": [],
            "SSBOND": [],
            "LINK": [],
            "MODRES": [],
        }

    def read_headers(self, pdb_path: str) -> None:
        """
        Read header information from PDB file.

        Args:
            pdb_path: Path to PDB file
        """
        with open(pdb_path, "r") as f:
            lines = f.readlines()

        current_model = -1
        reading_model = False

        for line in lines:
            record_type = line[:6].strip()

            # Handle model-specific headers
            if record_type == "MODEL":
                current_model += 1
                reading_model = True
                self.model_headers[current_model] = []
            elif record_type == "ENDMDL":
                reading_model = False
            elif reading_model and record_type in self.other_records:
                self.model_headers[current_model].append(line)

            # Handle global records
            if record_type in self.other_records:
                self.other_records[record_type].append(line)
            elif record_type == "REMARK":
                self._handle_remark(line)
            elif record_type == "CONECT":
                self.connectivity.append(line)

    def write_headers(self, file_handle) -> None:
        """Write headers to file."""
        # Write global headers
        for header in self.headers:
            file_handle.write(header)

        # Write other records
        for records in self.other_records.values():
            for record in records:
                file_handle.write(record)

    def write_model_headers(self, file_handle, model_num: int) -> None:
        """Write model-specific headers."""
        if model_num in self.model_headers:
            for header in self.model_headers[model_num]:
                file_handle.write(header)

    def _handle_remark(self, line: str) -> None:
        """Process REMARK records."""
        try:
            remark_num = int(line[6:10].strip())
            if remark_num not in self.remarks:
                self.remarks[remark_num] = []
            self.remarks[remark_num].append(line)
        except (ValueError, IndexError):
            pass
