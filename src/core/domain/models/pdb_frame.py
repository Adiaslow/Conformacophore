"""Model representing a single frame from a PDB file."""

from dataclasses import dataclass
from typing import List, Dict, Optional
import os


@dataclass
class PDBFrame:
    """Represents a single frame/model from a PDB file."""

    model_num: int
    compound_id: str
    frame_num: int
    sequence: str
    atoms: List[Dict]
    connectivity: List[tuple]
    chain_id: str = "A"  # Default chain ID


class PDBFrameCollection:
    """Collection of frames from a PDB file."""

    def __init__(self, file_path: str):
        """Initialize collection from PDB file."""
        self.file_path = file_path
        self.frames: List[PDBFrame] = []
        self._parse_file()

    def _parse_file(self) -> None:
        """Parse PDB file into frames."""
        current_frame = None
        atoms = []
        connectivity = []
        current_model = 0

        with open(self.file_path, "r") as f:
            for line in f:
                record_type = line[0:6].strip()

                if record_type == "MODEL":
                    # Start new frame
                    current_model = int(line.split()[1])
                    if current_frame and atoms:
                        self._add_frame(current_frame, atoms, connectivity)
                    atoms = []
                    connectivity = []
                    current_frame = {
                        "model_num": current_model,
                        "compound_id": os.path.splitext(
                            os.path.basename(self.file_path)
                        )[0],
                        "frame_num": 0,
                        "sequence": "UNK",
                    }
                elif record_type in ["ATOM", "HETATM"]:  # Handle both ATOM and HETATM
                    # Create frame if not exists (for files without MODEL records)
                    if current_frame is None:
                        current_frame = {
                            "model_num": 1,
                            "compound_id": os.path.splitext(
                                os.path.basename(self.file_path)
                            )[0],
                            "frame_num": 0,
                            "sequence": "UNK",
                        }
                    atom = self._parse_atom_line(line)
                    if atom:  # Only add if parsing was successful
                        atoms.append(atom)
                elif record_type == "CONECT":
                    conn = self._parse_connect_line(line)
                    if conn:  # Only add if parsing was successful
                        connectivity.append(conn)
                elif record_type == "ENDMDL" or record_type == "END":
                    if current_frame and atoms:
                        self._add_frame(current_frame, atoms, connectivity)
                        atoms = []
                        connectivity = []
                        current_frame = None

            # Add last frame if file doesn't end with ENDMDL/END
            if current_frame and atoms:
                self._add_frame(current_frame, atoms, connectivity)

    def _add_frame(
        self, frame_data: Dict, atoms: List[Dict], connectivity: List[tuple]
    ) -> None:
        """Add parsed frame to collection."""
        frame = PDBFrame(
            model_num=frame_data["model_num"],
            compound_id=frame_data["compound_id"],
            frame_num=frame_data["frame_num"],
            sequence=frame_data["sequence"],
            atoms=atoms,
            connectivity=connectivity,
        )
        self.frames.append(frame)

    @staticmethod
    def _parse_atom_line(line: str) -> Optional[Dict]:
        """Parse ATOM/HETATM record line."""
        try:
            return {
                "atom_num": int(line[6:11].strip()),
                "atom_name": line[12:16].strip(),
                "residue_name": line[17:20].strip(),
                "chain_id": line[21],
                "residue_num": int(line[22:26].strip()),
                "x": float(line[30:38].strip()),
                "y": float(line[38:46].strip()),
                "z": float(line[46:54].strip()),
            }
        except (ValueError, IndexError):
            # Skip malformed lines
            return None

    @staticmethod
    def _parse_connect_line(line: str) -> Optional[tuple]:
        """Parse CONECT record line."""
        try:
            numbers = [int(x) for x in line.split()[1:]]
            return tuple(numbers)
        except (ValueError, IndexError):
            return None
