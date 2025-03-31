#!/usr/bin/env python3
# src/core/services/trajectory_converter.py

"""
Service for converting molecular dynamics trajectories between different formats.
Supports both MDAnalysis and GROMACS-based conversion methods.
"""
# Standard library imports
import os
import warnings
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, NamedTuple, Sequence
from dataclasses import dataclass
import functools
import tempfile
import shutil
import subprocess
import json
import datetime

# Third party imports
import MDAnalysis as mda
from tqdm import tqdm
from MDAnalysis.coordinates.PDB import PDBWriter
import numpy as np

# Set up warning filter to ignore specific warning
warnings.filterwarnings(
    "ignore",
    message="Error applying metadata: Length of resids does not match number of residues. Expect: 7 Have: 121",
)


@dataclass
class MoleculeMetadata:
    """Container for molecule metadata"""

    compound_name: str
    frame_num: int
    residue_sequence: Sequence[str]
    numbered_sequence: Sequence[str]
    time: float
    box_dimensions: Sequence[float]

    @classmethod
    def create_default(
        cls, frame_num: int, compound_name: str = "unknown"
    ) -> "MoleculeMetadata":
        """Create a default metadata object when information is missing"""
        return cls(
            compound_name=compound_name,
            frame_num=frame_num,
            residue_sequence=["UNK"],
            numbered_sequence=["UNK1"],
            time=0.0,
            box_dimensions=[0.0, 0.0, 0.0, 90.0, 90.0, 90.0],
        )


class CustomPDBWriter:
    """Custom PDB writer that formats output according to specifications."""

    def __init__(self, filename: str):
        """Initialize the writer with output file path."""
        self.filename = filename
        self.model_number = 1
        self._file = None

    def __enter__(self):
        """Context manager entry."""
        self._file = open(self.filename, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file:
            self._file.close()

    def _write_atom_line(self, atom, coords) -> str:
        """Format a single atom line according to PDB specifications."""
        # Get the element from the atom name (first character)
        element = atom.name[0] if atom.name else ""

        return (
            f"ATOM  {atom.id:5d} {atom.name:<4s} {atom.resname:3s} X{atom.resid:4d}"
            f"    {coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}"
            f"  1.00  0.00      {element:>1s}   \n"
        )

    def _write_conect_records(self, universe):
        """Write CONECT records for bonds in the universe.

        Args:
            universe (MDAnalysis.Universe): Universe containing bond information

        Returns:
            List[str]: List of CONECT record strings
        """
        conect_records = []
        if not hasattr(universe, "bonds") or len(universe.bonds) == 0:
            if self.verbose:
                self.logger.warning("No bonds found in universe")
            return conect_records

        # Create a dictionary to store all bonds for each atom
        bond_dict = {}
        for bond in universe.bonds:
            # Convert back to 1-based indexing for PDB format
            atom1 = bond.atoms[0].ix + 1
            atom2 = bond.atoms[1].ix + 1

            # Add bond to both atoms' lists
            if atom1 not in bond_dict:
                bond_dict[atom1] = []
            if atom2 not in bond_dict:
                bond_dict[atom2] = []
            bond_dict[atom1].append(atom2)
            bond_dict[atom2].append(atom1)

        # Write CONECT records
        for atom_idx in sorted(bond_dict.keys()):
            # Sort bonded atoms for consistency
            bonded_atoms = sorted(bond_dict[atom_idx])
            # Format according to PDB standard (leading spaces important)
            record = f"CONECT{atom_idx:5d}"
            for bonded_atom in bonded_atoms:
                record += f"{bonded_atom:5d}"
            conect_records.append(record + "\n")

        if self.verbose:
            self.logger.info(f"Generated {len(conect_records)} CONECT records")

        return conect_records

    def write(self, universe, frame_number: int = 0, compound_number: int = 943):
        """Write a single frame/model to the PDB file."""
        # Write MODEL header
        self._file.write(f"MODEL {self.model_number}\n")
        self._file.write(f"COMPND {compound_number}\n")
        self._file.write(f"REMARK Frame {frame_number}\n")

        # Write SEQRES record
        if hasattr(universe, "residues") and len(universe.residues) > 0:
            residues = [res.resname for res in universe.residues]
            self._file.write(f"SEQRES   1 A {len(residues):4d}  {' '.join(residues)}\n")

        # Write atom records
        for atom in universe.atoms:
            self._file.write(self._write_atom_line(atom, atom.position))

        # Write connectivity records
        for conect_line in self._write_conect_records(universe):
            self._file.write(conect_line)

        # Write model end
        self._file.write("END\n")
        self._file.write("ENDMDL\n")
        self.model_number += 1


class GromacsPDBConverter:
    """Handles trajectory conversion using GROMACS commands."""

    def __init__(self, verbose: bool = False):
        """Initialize the converter.

        Args:
            verbose: Whether to show detailed output
        """
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.gmx_cmd = self._find_gmx_command()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the converter."""
        logger = logging.getLogger(__name__)

        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Set up file handler
        log_file = Path("trajectory_conversion.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        # Only log to console if verbose is True
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)

        return logger

    def _find_gmx_command(self) -> str:
        """Find the GROMACS executable command.

        Returns:
            str: Path to GROMACS executable

        Raises:
            RuntimeError: If GROMACS cannot be found
        """
        # Default command
        gmx_cmd = "gmx"

        # Check if gmx is in PATH
        which_gmx = subprocess.run(
            "which gmx", shell=True, capture_output=True, text=True
        )

        if which_gmx.returncode != 0:
            # Try common GROMACS installation locations
            potential_paths = [
                "/usr/local/gromacs/bin/gmx",
                "/opt/gromacs/bin/gmx",
                os.path.expanduser("~/gromacs/bin/gmx"),
                "/Applications/gromacs/bin/gmx",
            ]

            for path in potential_paths:
                if os.path.exists(path):
                    gmx_cmd = path
                    if self.verbose:
                        self.logger.info(f"Found GROMACS at: {gmx_cmd}")
                    break

            # If still not found, try sourcing GMXRC
            if gmx_cmd == "gmx":
                try:
                    if self.verbose:
                        self.logger.info("Trying to source GROMACS environment...")
                    gmxrc_paths = [
                        "/usr/local/gromacs/bin/GMXRC",
                        "/opt/gromacs/bin/GMXRC",
                        os.path.expanduser("~/gromacs/bin/GMXRC"),
                        "/Applications/gromacs/bin/GMXRC",
                    ]

                    for gmxrc in gmxrc_paths:
                        if os.path.exists(gmxrc):
                            source_cmd = f"source {gmxrc} && which gmx"
                            result = subprocess.run(
                                source_cmd,
                                shell=True,
                                executable="/bin/bash",
                                capture_output=True,
                                text=True,
                            )
                            if result.returncode == 0 and result.stdout.strip():
                                gmx_cmd = result.stdout.strip()
                                if self.verbose:
                                    self.logger.info(
                                        f"GROMACS sourced from {gmxrc}. Using: {gmx_cmd}"
                                    )
                                break
                except Exception as e:
                    self.logger.error(f"Error sourcing GROMACS: {e}")

        if gmx_cmd == "gmx" and which_gmx.returncode != 0:
            raise RuntimeError(
                "Could not find GROMACS executable. Please ensure GROMACS is installed and in your PATH."
            )

        return gmx_cmd

    def _create_minimal_mdp(self, output_path: Path) -> None:
        """Create a minimal MDP file for grompp.

        Args:
            output_path: Path to write the MDP file
        """
        with open(output_path, "w") as f:
            f.write("integrator = md\nnsteps = 0\n")

    def _copy_file_chunked(
        self, src_path: str, dst_path: Path, chunk_size: int = 1024 * 1024
    ) -> None:
        """Copy a file in chunks to handle large files.

        Args:
            src_path: Source file path
            dst_path: Destination file path
            chunk_size: Size of chunks to copy (default: 1MB)
        """
        with open(src_path, "rb") as fsrc:
            with open(dst_path, "wb") as fdst:
                while True:
                    chunk = fsrc.read(chunk_size)
                    if not chunk:
                        break
                    fdst.write(chunk)
                    fdst.flush()
                    os.fsync(fdst.fileno())

    def get_trajectory_info(
        self, xtc_path: str, gro_path: str, top_path: str
    ) -> Dict[str, Any]:
        """Get information about a trajectory file using GROMACS.

        Args:
            xtc_path: Path to XTC trajectory file
            gro_path: Path to GRO structure file
            top_path: Path to TOP topology file

        Returns:
            Dict containing trajectory information
        """
        with tempfile.TemporaryDirectory(prefix="traj_info_") as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Copy files to temporary directory
            local_xtc = temp_dir_path / "traj.xtc"
            local_gro = temp_dir_path / "conf.gro"
            local_top = temp_dir_path / "topol.top"
            local_mdp = temp_dir_path / "temp.mdp"
            local_tpr = temp_dir_path / "temp.tpr"

            self._copy_file_chunked(xtc_path, local_xtc)
            self._copy_file_chunked(gro_path, local_gro)
            self._copy_file_chunked(top_path, local_top)

            # Create minimal MDP file
            self._create_minimal_mdp(local_mdp)

            # Create TPR file
            cmd_grompp = f"{self.gmx_cmd} grompp -f {local_mdp} -c {local_gro} -p {local_top} -o {local_tpr} -maxwarn 100"
            result = subprocess.run(
                cmd_grompp, shell=True, capture_output=True, text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to create TPR file: {result.stderr}")

            # Get trajectory information
            cmd_check = f"{self.gmx_cmd} check -f {local_xtc}"
            result = subprocess.run(
                cmd_check, shell=True, capture_output=True, text=True
            )

            info = {
                "n_frames": 0,
                "n_atoms": 0,
                "time_range": (0, 0),
                "topology_source": gro_path,
                "top_file": top_path,
            }

            # Parse gmx check output
            for line in result.stdout.split("\n") + result.stderr.split("\n"):
                if "Number of frames" in line:
                    info["n_frames"] = int(line.split()[-1])
                elif "Number of atoms" in line:
                    info["n_atoms"] = int(line.split()[-1])
                elif "Last frame time" in line:
                    info["time_range"] = (0, float(line.split()[-1]))

            return info

    def convert_trajectory(
        self,
        xtc_path: str,
        gro_path: str,
        top_path: str,
        output_dir: str,
        start: int = 0,
        end: Optional[int] = None,
        stride: int = 1,
    ) -> List[Path]:
        """Convert XTC trajectory to PDB using GROMACS.

        Args:
            xtc_path: Path to XTC trajectory file
            gro_path: Path to GRO structure file
            top_path: Path to TOP topology file
            output_dir: Directory to write output files
            start: First frame to convert (0-based)
            end: Last frame to convert (None = all frames)
            stride: Step size for frame selection

        Returns:
            List of output file paths
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="traj_convert_") as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Copy files to temporary directory
            local_xtc = temp_dir_path / "traj.xtc"
            local_gro = temp_dir_path / "conf.gro"
            local_top = temp_dir_path / "topol.top"
            local_mdp = temp_dir_path / "temp.mdp"
            local_tpr = temp_dir_path / "temp.tpr"
            local_pdb = temp_dir_path / "md_Ref.pdb"

            if self.verbose:
                self.logger.info("Copying files to temporary directory...")

            self._copy_file_chunked(xtc_path, local_xtc)
            self._copy_file_chunked(gro_path, local_gro)
            self._copy_file_chunked(top_path, local_top)

            # Create minimal MDP file
            if self.verbose:
                self.logger.info("Creating minimal MDP file...")
            self._create_minimal_mdp(local_mdp)

            # Create TPR file
            if self.verbose:
                self.logger.info("Creating TPR file...")

            cmd_grompp = f"{self.gmx_cmd} grompp -f {local_mdp} -c {local_gro} -p {local_top} -o {local_tpr} -maxwarn 100"
            result = subprocess.run(
                cmd_grompp, shell=True, capture_output=True, text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to create TPR file: {result.stderr}")

            # Convert trajectory
            if self.verbose:
                self.logger.info("Converting trajectory to PDB...")

            # Build trjconv command with frame selection
            trjconv_cmd = f"echo 0 | {self.gmx_cmd} trjconv -s {local_tpr} -f {local_xtc} -o {local_pdb} -pbc mol -conect"

            if start > 0:
                trjconv_cmd += f" -b {start}"
            if end is not None:
                trjconv_cmd += f" -e {end}"
            if stride > 1:
                trjconv_cmd += f" -skip {stride}"

            result = subprocess.run(
                trjconv_cmd, shell=True, capture_output=True, text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to convert trajectory: {result.stderr}")

            # Get residue sequence from the topology file to add SEQRES
            residue_sequence = self._extract_residue_sequence(top_path)
            if residue_sequence and self.verbose:
                self.logger.info(f"Found residue sequence: {residue_sequence}")

            # Customize PDB header format
            self._customize_pdb_header(
                local_pdb, Path(xtc_path).parent.name, residue_sequence
            )

            # Copy result to output directory
            output_pdb = output_dir_path / "md_Ref.pdb"
            if self.verbose:
                self.logger.info(f"Copying result to {output_pdb}...")

            self._copy_file_chunked(local_pdb, output_pdb)

            # Verify CONECT records were written
            if self.verbose:
                with open(output_pdb, "r") as f:
                    content = f.read()
                    conect_lines = [
                        line
                        for line in content.split("\n")
                        if line.startswith("CONECT")
                    ]
                    self.logger.info(
                        f"Number of CONECT records written: {len(conect_lines)}"
                    )
                    if conect_lines:
                        self.logger.info("Sample CONECT records:")
                        for line in conect_lines[:5]:
                            self.logger.info(line)

            return [output_pdb]

    def _extract_residue_sequence(self, top_path: str) -> List[str]:
        """Extract residue sequence from a GROMACS topology file.

        Args:
            top_path: Path to the GROMACS topology file

        Returns:
            List of residue names in sequence
        """
        try:
            residues = []
            in_molecules = False
            in_atoms = False
            residue_types = {}

            # First pass: extract residue info from atoms section
            with open(top_path, "r") as f:
                for line in f:
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith(";"):
                        continue

                    # Check for sections
                    if line.startswith("["):
                        section = line.strip("[]").strip()
                        in_atoms = section == "atoms"
                        in_molecules = section == "molecules"
                        continue

                    # Process atoms section to get residue types
                    if in_atoms:
                        parts = line.split()
                        if len(parts) >= 5:  # Ensure we have enough fields
                            try:
                                # Extract residue id and name
                                resid = int(parts[2]) if parts[2].isdigit() else 0
                                resname = parts[3]
                                if resid > 0:
                                    residue_types[resid] = resname
                            except Exception as e:
                                if self.verbose:
                                    self.logger.warning(
                                        f"Error parsing atom line: {str(e)}"
                                    )

                    # Process molecules section as backup
                    elif in_molecules:
                        parts = line.split()
                        if len(parts) >= 2 and parts[0].lower() != "system":
                            try:
                                residue_name = parts[0]
                                count = int(parts[1])
                                # Add residue multiple times based on count
                                for _ in range(count):
                                    residues.append(residue_name)
                            except Exception as e:
                                if self.verbose:
                                    self.logger.warning(
                                        f"Error parsing molecule line: {str(e)}"
                                    )

            # If we found residue types from atoms section, use them
            if residue_types:
                # Sort by resid to get the correct sequence
                sorted_resids = sorted(residue_types.keys())
                residues = [residue_types[resid] for resid in sorted_resids]

            # Filter out system entries
            residues = [r for r in residues if r.lower() != "system"]

            # If no valid residues found, try to use file name
            if not residues:
                dir_name = os.path.basename(os.path.dirname(top_path))
                if dir_name.startswith("x_") and dir_name[2:].isdigit():
                    # For x_177 type directories, extract residue info from another source
                    try:
                        # Extract three-letter codes from gro file
                        gro_file = os.path.join(os.path.dirname(top_path), "md_Ref.gro")
                        if os.path.exists(gro_file):
                            res_set = set()
                            with open(gro_file, "r") as gf:
                                # Skip first two lines
                                next(gf)
                                next(gf)
                                for line in gf:
                                    if len(line) >= 10:
                                        res_name = line[5:8].strip()
                                        if res_name and len(res_name) == 3:
                                            res_set.add(res_name)
                            residues = list(res_set)
                    except Exception as e:
                        if self.verbose:
                            self.logger.warning(
                                f"Error extracting residue info from GRO file: {str(e)}"
                            )

            if self.verbose:
                self.logger.info(f"Extracted residue sequence: {residues}")
            return residues
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Error extracting residue sequence: {str(e)}")
            return []

    def _customize_pdb_header(
        self, pdb_path: Path, compound_name: str, residue_sequence: List[str]
    ) -> None:
        """Customize the PDB header format.

        Args:
            pdb_path: Path to the PDB file
            compound_name: Name of the compound (typically directory name)
            residue_sequence: List of residue names
        """
        try:
            with open(pdb_path, "r") as f:
                lines = f.readlines()

            # Extract model sections
            model_sections = []
            current_section = []
            in_model = False

            for line in lines:
                if line.startswith("MODEL"):
                    in_model = True
                    current_section = [line]
                elif line.startswith("ENDMDL"):
                    current_section.append(line)
                    model_sections.append(current_section)
                    in_model = False
                    current_section = []
                elif in_model:
                    current_section.append(line)

            # Compile new PDB content with custom headers
            new_lines = []
            frame_number = 0

            for section in model_sections:
                # Add custom header
                new_lines.append(f"TITLE {compound_name}\n")

                # Add SEQRES if available
                if residue_sequence:
                    residues_per_line = 13  # Max residues per SEQRES line
                    for i in range(0, len(residue_sequence), residues_per_line):
                        chunk = residue_sequence[i : i + residues_per_line]
                        line_num = i // residues_per_line + 1
                        seqres_line = f"SEQRES  {line_num:2d} A {len(residue_sequence):4d}  {' '.join(chunk)}\n"
                        new_lines.append(seqres_line)

                # Add frame remark
                new_lines.append(f"REMARK frame={frame_number}\n")

                # Add model line and atom records (skip CRYST1)
                found_model_line = False
                for line in section:
                    if line.startswith("MODEL"):
                        new_lines.append("MODEL 1\n")  # Always use MODEL 1
                        found_model_line = True
                    elif line.startswith(("ATOM", "HETATM", "TER", "CONECT", "END")):
                        new_lines.append(line)

                if not found_model_line:
                    # If no MODEL line found, insert one before first atom
                    for i, line in enumerate(new_lines):
                        if line.startswith("ATOM"):
                            new_lines.insert(i, "MODEL 1\n")
                            break

                frame_number += 1

            # Write back to file
            with open(pdb_path, "w") as f:
                f.writelines(new_lines)

            if self.verbose:
                self.logger.info(f"Customized PDB header format for {pdb_path}")

        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Error customizing PDB header: {str(e)}")

    def _ensure_seqres_in_pdb(
        self, pdb_path: Path, residue_sequence: List[str]
    ) -> None:
        """Ensure the PDB file has SEQRES records.

        Args:
            pdb_path: Path to the PDB file
            residue_sequence: List of residue names in sequence
        """
        try:
            with open(pdb_path, "r") as f:
                lines = f.readlines()

            # Check if SEQRES already exists
            has_seqres = any(line.startswith("SEQRES") for line in lines)

            if not has_seqres and residue_sequence:
                # Find where to insert SEQRES (after REMARK, TITLE, CRYST1 but before MODEL)
                insert_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith(("REMARK", "TITLE", "CRYST1")):
                        insert_idx = i + 1
                    elif line.startswith("MODEL"):
                        break

                # Format SEQRES record according to PDB format
                residues_per_line = 13  # Max residues per SEQRES line
                seqres_lines = []

                for i in range(0, len(residue_sequence), residues_per_line):
                    chunk = residue_sequence[i : i + residues_per_line]
                    line_num = i // residues_per_line + 1
                    seqres_line = f"SEQRES  {line_num:2d} A {len(residue_sequence):4d}  {' '.join(chunk)}\n"
                    seqres_lines.append(seqres_line)

                # Insert SEQRES lines
                for i, seqres_line in enumerate(seqres_lines):
                    lines.insert(insert_idx + i, seqres_line)

                # Write back to file
                with open(pdb_path, "w") as f:
                    f.writelines(lines)

                if self.verbose:
                    self.logger.info(
                        f"Added {len(seqres_lines)} SEQRES records to PDB file"
                    )

        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Error ensuring SEQRES records: {str(e)}")


# Keep the TrajectoryConverter class as a legacy option or for future use
