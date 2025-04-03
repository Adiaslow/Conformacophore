# src/core/services/xtc_converter.py

import os
import logging
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Iterator,
    Union,
    Any,
    TypeVar,
    Generic,
    Sequence,
    cast,
)
import numpy as np
import mdtraj as md
from dataclasses import dataclass, field
from io import StringIO
import queue
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from ..utils.benchmarking import Timer, PerformanceStats
from ..models import Atom

# Increased chunk and buffer sizes for better performance
CHUNK_SIZE = 100  # Number of frames to process at once
BUFFER_SIZE = 8 * 1024 * 1024  # 8MB write buffer
MAX_QUEUE_SIZE = 10  # Maximum number of chunks to queue for writing
NUM_WORKERS = 4  # Number of worker threads for parallel processing

T = TypeVar("T")


@dataclass
class PDBTemplates:
    """Pre-formatted PDB record templates."""

    header: str = ""  # TITLE + SEQRES
    model_start: str = "MODEL     {model_num:>4d}\n"
    model_end: str = "ENDMDL\n"
    atom: str = (
        "ATOM  {atom_num:>5d} {atom_name:<4s}{res_name:>4s} {chain:>1s}{res_num:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2s}\n"
    )
    conect: str = ""  # All CONECT records


@dataclass
class ValidationError:
    """Represents a PDB format validation error."""

    line_number: int
    message: str
    severity: str  # 'ERROR' or 'WARNING'
    line_content: str


def parse_gro_file(gro_path: str) -> List[Atom]:
    """Parse a GROMACS GRO file.

    Args:
        gro_path: Path to GRO file

    Returns:
        List of Atom objects
    """
    with open(gro_path, "r") as f:
        lines = f.readlines()

    n_atoms = int(lines[1])
    atom_lines = lines[2 : n_atoms + 2]

    atoms = []
    for i, line in enumerate(atom_lines):
        residue_number = int(line[0:5])
        residue_name = line[5:10].strip()
        atom_name = line[10:15].strip()
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])

        element = atom_name[0] if not atom_name.startswith("H") else "H"
        atoms.append(
            Atom(
                index=i,
                name=atom_name,
                residue_name=residue_name,
                residue_number=residue_number,
                element=element,
                x=x,
                y=y,
                z=z,
            )
        )

    return atoms


def parse_top_file(top_path: str) -> List[Tuple[int, int]]:
    """Parse a GROMACS topology file to get bonds.

    Args:
        top_path: Path to topology file

    Returns:
        List of (atom1, atom2) bond tuples (1-based indices)
    """
    bonds = []
    in_bonds = False

    with open(top_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue

            if line.startswith("["):
                section = line.strip("[]").strip()
                if section == "bonds":
                    in_bonds = True
                elif in_bonds:
                    break
                continue

            if in_bonds:
                # Parse bond line (format: atom1 atom2 func [params])
                parts = line.split()
                if len(parts) >= 2:
                    atom1 = int(parts[0])  # Already 1-based in TOP file
                    atom2 = int(parts[1])
                    bonds.append((atom1, atom2))

    return bonds


def validate_pdb_file(pdb_path: str) -> List[ValidationError]:
    """Validate PDB file format.

    Args:
        pdb_path: Path to PDB file to validate

    Returns:
        List of ValidationError objects
    """
    errors = []
    model_open = False
    atom_count = 0
    model_count = 0
    seen_atoms = set()

    with open(pdb_path, "r") as f:
        for i, line in enumerate(f, 1):
            # Check basic line length
            if len(line.rstrip()) > 80:
                errors.append(
                    ValidationError(
                        i, "Line exceeds 80 characters", "ERROR", line.rstrip()
                    )
                )

            # Check record types
            record_type = line[:6].strip()

            if record_type == "TITLE":
                if i > 10:  # TITLE should be near the start
                    errors.append(
                        ValidationError(
                            i,
                            "TITLE record should appear near start of file",
                            "WARNING",
                            line.rstrip(),
                        )
                    )

            elif record_type == "SEQRES":
                # Check SEQRES format
                if len(line) < 70:
                    errors.append(
                        ValidationError(
                            i, "SEQRES record is too short", "ERROR", line.rstrip()
                        )
                    )
                try:
                    int(line[7:10])  # Serial number
                    int(line[13:17])  # Number of residues
                except ValueError:
                    errors.append(
                        ValidationError(
                            i,
                            "Invalid number format in SEQRES record",
                            "ERROR",
                            line.rstrip(),
                        )
                    )

            elif record_type == "MODEL":
                if model_open:
                    errors.append(
                        ValidationError(
                            i,
                            "New MODEL record before closing previous model",
                            "ERROR",
                            line.rstrip(),
                        )
                    )
                model_open = True
                model_count += 1
                atom_count = 0  # Reset atom count for new model
                seen_atoms.clear()

                # Check model number format
                try:
                    model_num = int(line[10:14])
                    if model_num != model_count:
                        errors.append(
                            ValidationError(
                                i,
                                f"Model number {model_num} does not match sequence {model_count}",
                                "WARNING",
                                line.rstrip(),
                            )
                        )
                except ValueError:
                    errors.append(
                        ValidationError(
                            i, "Invalid model number format", "ERROR", line.rstrip()
                        )
                    )

            elif record_type == "ATOM":
                if not model_open:
                    errors.append(
                        ValidationError(
                            i, "ATOM record outside MODEL", "ERROR", line.rstrip()
                        )
                    )

                # Check atom serial number
                try:
                    atom_num = int(line[6:11])
                    if atom_num in seen_atoms:
                        errors.append(
                            ValidationError(
                                i,
                                f"Duplicate atom number {atom_num}",
                                "ERROR",
                                line.rstrip(),
                            )
                        )
                    seen_atoms.add(atom_num)
                except ValueError:
                    errors.append(
                        ValidationError(
                            i, "Invalid atom serial number", "ERROR", line.rstrip()
                        )
                    )

                # Check coordinate format
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    # Check for reasonable coordinate ranges (in Angstroms)
                    if abs(x) > 999.999 or abs(y) > 999.999 or abs(z) > 999.999:
                        errors.append(
                            ValidationError(
                                i,
                                "Coordinate value exceeds reasonable range",
                                "WARNING",
                                line.rstrip(),
                            )
                        )
                except ValueError:
                    errors.append(
                        ValidationError(
                            i, "Invalid coordinate format", "ERROR", line.rstrip()
                        )
                    )

                atom_count += 1

            elif record_type == "ENDMDL":
                if not model_open:
                    errors.append(
                        ValidationError(
                            i,
                            "ENDMDL record without matching MODEL",
                            "ERROR",
                            line.rstrip(),
                        )
                    )
                model_open = False

            elif record_type == "CONECT":
                # Check CONECT record format
                atoms: List[int] = []
                try:
                    atoms = [int(line[6:11])]  # First atom
                    for pos in range(11, 31, 5):  # Check bonded atoms
                        if line[pos : pos + 5].strip():
                            atoms.append(int(line[pos : pos + 5]))
                except ValueError:
                    errors.append(
                        ValidationError(
                            i,
                            "Invalid atom number in CONECT record",
                            "ERROR",
                            line.rstrip(),
                        )
                    )

                # Check if referenced atoms exist
                for atom in atoms:
                    if (
                        atom not in seen_atoms and atom_count > 0
                    ):  # Only check if we've seen atoms
                        errors.append(
                            ValidationError(
                                i,
                                f"CONECT record references non-existent atom {atom}",
                                "ERROR",
                                line.rstrip(),
                            )
                        )

            elif record_type == "END":
                if model_open:
                    errors.append(
                        ValidationError(
                            i, "END record with unclosed MODEL", "ERROR", line.rstrip()
                        )
                    )

    if model_open:
        errors.append(ValidationError(-1, "File ends with unclosed MODEL", "ERROR", ""))

    return errors


class XTCConverter:
    """Converts XTC trajectories to PDB format with structure information."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        self.stats = PerformanceStats()
        self.templates = PDBTemplates()

    def _create_header(self, compound_name: str, unique_residues: List[str]) -> str:
        """Create PDB header with TITLE and SEQRES records.

        Args:
            compound_name: Name of the compound
            unique_residues: List of unique residue names in order

        Returns:
            Formatted header string
        """
        # TITLE record
        header = f"TITLE     {compound_name}\n"

        # SEQRES record (one line, all residues)
        n_residues = len(unique_residues)
        # Pad each residue name to 4 characters and join with spaces
        residue_str = "  ".join(f"{res:<3}" for res in unique_residues)
        # Format the full SEQRES record with proper spacing
        header += f"SEQRES   1 A {n_residues:>4d}  {residue_str:<51}\n"

        return header

    def _create_atom_template(self, atoms: List[Atom]) -> Dict[int, str]:
        """Create template for ATOM records and mapping of atom indices.

        Args:
            atoms: List of Atom objects

        Returns:
            Dict mapping atom index to record template string
        """
        templates = {}
        for i, atom in enumerate(atoms, 1):
            # Create template with coordinate placeholders
            template = self.templates.atom.format(
                atom_num=i,
                atom_name=atom.name,
                res_name=atom.residue_name,
                chain="A",
                res_num=atom.residue_number,
                x=0.0,  # Placeholder
                y=0.0,  # Placeholder
                z=0.0,  # Placeholder
                element=atom.element,
            )
            templates[i] = template

        return templates

    def _create_conect_records(self, bonds: List[Tuple[int, int]]) -> str:
        """Create all CONECT records.

        Args:
            bonds: List of (atom1, atom2) indices

        Returns:
            String containing all CONECT records
        """
        # Group bonds by first atom
        bond_groups: Dict[int, List[int]] = {}
        for a1, a2 in bonds:
            if a1 not in bond_groups:
                bond_groups[a1] = []
            bond_groups[a1].append(a2)

        # Create CONECT records
        records = []
        for atom1 in sorted(bond_groups.keys()):
            bonded = bond_groups[atom1]
            record = f"CONECT{atom1:>5d}"
            for atom2 in sorted(bonded):
                record += f"{atom2:>5d}"
            records.append(record)

        return "\n".join(records) + "\n"

    def _format_coordinates(
        self, coords: np.ndarray, atom_templates: Dict[int, str]
    ) -> List[str]:
        """Format coordinates into PDB ATOM records using templates.

        Args:
            coords: Array of shape (n_atoms, 3) containing XYZ coordinates
            atom_templates: Dict mapping atom index to record template

        Returns:
            List of formatted ATOM records
        """
        records = []
        for i, (x, y, z) in enumerate(coords, 1):
            template = atom_templates[i]
            # Format coordinates into the template
            record = template.format(x=x * 10, y=y * 10, z=z * 10)  # Convert nm to Å
            records.append(record)
        return records

    def convert(
        self,
        xtc_path: str,
        top_path: str,
        gro_path: str,
        output_path: str,
        compound_name: str,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        """Convert XTC trajectory to PDB format.

        Args:
            xtc_path: Path to XTC trajectory file
            top_path: Path to topology file
            gro_path: Path to GRO structure file
            output_path: Path to output directory (will create conformers subdirectory)
            compound_name: Name of the compound
            start: First frame to convert (0-based)
            stop: Last frame to convert (exclusive)
            step: Step size between frames
        """
        with Timer("total_conversion"):
            try:
                # Validate input files
                if not all(Path(p).exists() for p in [xtc_path, top_path, gro_path]):
                    missing = [
                        p
                        for p in [xtc_path, top_path, gro_path]
                        if not Path(p).exists()
                    ]
                    raise FileNotFoundError(
                        f"Missing input files: {', '.join(missing)}"
                    )

                # Create conformers directory
                output_dir = Path(output_path)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Parse structure and topology
                with Timer("parse_files"):
                    atoms = parse_gro_file(gro_path)
                    if not atoms:
                        raise ValueError(f"No atoms found in GRO file: {gro_path}")

                    bonds = parse_top_file(top_path)
                    if not bonds:
                        self.logger.warning(
                            f"No bonds found in topology file: {top_path}"
                        )

                if self.verbose:
                    self.logger.info(f"Parsed structure: {len(atoms)} atoms")
                    self.logger.info(f"Parsed topology: {len(bonds)} bonds")

                # Create templates
                unique_residues = []
                seen = set()
                for atom in atoms:
                    if atom.residue_name not in seen:
                        unique_residues.append(atom.residue_name)
                        seen.add(atom.residue_name)

                self.templates.header = self._create_header(
                    compound_name, unique_residues
                )
                atom_templates = self._create_atom_template(atoms)
                self.templates.conect = self._create_conect_records(bonds)

                # Load trajectory
                with Timer("load_trajectory"):
                    traj = md.load(xtc_path, top=gro_path)

                # Process frames
                frame_indices = range(len(traj))
                if start is not None:
                    frame_indices = range(start, len(traj))
                if stop is not None:
                    frame_indices = range(frame_indices.start, min(stop, len(traj)))
                if step is not None:
                    frame_indices = range(frame_indices.start, frame_indices.stop, step)

                # Process frames in chunks
                for frame_idx in frame_indices:
                    frame_coords = traj.xyz[frame_idx]
                    frame_path = output_dir / f"frame_{frame_idx}.pdb"

                    # Format coordinates for this frame
                    atom_records = self._format_coordinates(
                        frame_coords, atom_templates
                    )

                    # Write PDB file for this frame
                    with open(frame_path, "w") as f:
                        # Write header
                        f.write(self.templates.header)

                        # Write ATOM records
                        for record in atom_records:
                            f.write(record)

                        # Write CONECT records
                        f.write(self.templates.conect)

                    # Validate the frame
                    try:
                        errors = validate_pdb_file(str(frame_path))
                        if errors:
                            error_msg = "\n".join(str(error) for error in errors)
                            raise ValueError(
                                f"Frame {frame_idx} validation failed:\n{error_msg}"
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Frame {frame_idx} validation failed: {str(e)}"
                        )
                        raise

                if self.verbose:
                    self.logger.info(
                        f"Successfully converted {len(frame_indices)} frames to PDB format"
                    )

            except Exception as e:
                self.logger.error(f"Error converting trajectory: {str(e)}")
                raise
