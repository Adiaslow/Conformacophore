#!/usr/bin/env python3
# src/core/services/gmx_converter.py

"""
Service for converting molecular dynamics trajectories using GROMACS commands.
"""
# Standard library imports
import os
import warnings
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set, Union
import tempfile
import shutil
import subprocess
import json
import datetime


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
        self,
        src_path: Union[str, Path],
        dst_path: Union[str, Path],
        chunk_size: int = 1024 * 1024,
    ) -> None:
        """Copy a file in chunks to handle large files.

        Args:
            src_path: Source file path
            dst_path: Destination file path
            chunk_size: Size of chunks to copy (default: 1MB)
        """
        # Convert to string if paths are Path objects
        if isinstance(src_path, Path):
            src_path = str(src_path)
        if isinstance(dst_path, Path):
            dst_path = str(dst_path)

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

    def _extract_residue_sequence(self, top_path: str, gro_path: str) -> List[str]:
        """Extract residue sequence from a GROMACS topology file.

        Args:
            top_path: Path to the GROMACS topology file
            gro_path: Path to the GRO structure file for fallback

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

            # If no valid residues found, try to extract from GRO file
            if not residues and gro_path and os.path.exists(gro_path):
                try:
                    res_set = set()
                    with open(gro_path, "r") as gf:
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

            # If still no residues, try to use directory name
            if not residues:
                dir_name = os.path.basename(os.path.dirname(top_path))
                if dir_name.startswith("x_") and dir_name[2:].isdigit():
                    if self.verbose:
                        self.logger.info(
                            f"Using directory name {dir_name} as compound identifier"
                        )
                    residues = [dir_name]

            if self.verbose:
                self.logger.info(f"Extracted residue sequence: {residues}")
            return residues
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Error extracting residue sequence: {str(e)}")
            return []

    def _extract_bonds_from_top(self, top_path: str) -> List[Tuple[int, int]]:
        """Extract bond information from topology file.

        Args:
            top_path: Path to the GROMACS topology file

        Returns:
            List of bond pairs (atom1, atom2) with 1-based indexing
        """
        bonds = []
        try:
            with open(top_path, "r") as f:
                in_bonds = False

                for line in f:
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith(";"):
                        continue

                    # Check for bonds section
                    if line.startswith("["):
                        in_bonds = line.strip("[]").strip() == "bonds"
                        continue

                    # Process bonds section
                    if in_bonds:
                        parts = line.split()
                        if (
                            len(parts) >= 5
                        ):  # GROMACS bond format has at least 5 columns
                            try:
                                atom1 = int(parts[0])
                                atom2 = int(parts[1])
                                bonds.append((atom1, atom2))
                            except Exception as e:
                                if self.verbose:
                                    self.logger.warning(
                                        f"Error parsing bond line: {str(e)}"
                                    )

            if self.verbose and bonds:
                self.logger.info(f"Extracted {len(bonds)} bonds from topology file")

            return bonds
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Error extracting bonds: {str(e)}")
            return []

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
            # The -conect flag must be used with the TPR file that has topology info
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

            # Check if CONECT records were created, if not, try to generate them
            with open(local_pdb, "r") as f:
                content = f.read()
                if "CONECT" not in content:
                    if self.verbose:
                        self.logger.warning(
                            "No CONECT records found in GROMACS output, generating manually..."
                        )

                    # Try using a different approach - write to a temporary PDB first
                    temp_pdb = temp_dir_path / "temp.pdb"
                    conect_cmd = f"echo 0 | {self.gmx_cmd} editconf -f {local_tpr} -o {temp_pdb} -conect"
                    try:
                        result = subprocess.run(
                            conect_cmd, shell=True, capture_output=True, text=True
                        )

                        # Extract CONECT records from the temporary PDB
                        if result.returncode == 0:
                            with open(temp_pdb, "r") as temp_f:
                                conect_lines = [
                                    line for line in temp_f if line.startswith("CONECT")
                                ]

                            # Append CONECT records to the main PDB
                            if conect_lines:
                                with open(local_pdb, "r") as main_f:
                                    main_content = main_f.readlines()

                                # Find where to insert CONECT records (before END)
                                insert_idx = len(main_content)
                                for i, line in enumerate(main_content):
                                    if line.startswith("END"):
                                        insert_idx = i
                                        break

                                # Insert CONECT records
                                main_content[insert_idx:insert_idx] = conect_lines

                                # Write back
                                with open(local_pdb, "w") as main_f:
                                    main_f.writelines(main_content)

                                if self.verbose:
                                    self.logger.info(
                                        f"Added {len(conect_lines)} CONECT records manually"
                                    )
                    except Exception as e:
                        if self.verbose:
                            self.logger.warning(
                                f"Failed to generate CONECT records manually: {str(e)}"
                            )

            # Get residue sequence from the topology file
            residue_sequence = self._extract_residue_sequence(
                str(local_top), str(local_gro)
            )
            if residue_sequence and self.verbose:
                self.logger.info(f"Found residue sequence: {residue_sequence}")

            # Customize PDB header format
            self._customize_pdb_header(
                local_pdb, Path(xtc_path).parent.name, residue_sequence, top_path
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

    def _customize_pdb_header(
        self,
        pdb_path: Path,
        compound_name: str,
        residue_sequence: List[str],
        top_path: Optional[str] = None,
    ) -> None:
        """Customize the PDB header format and add CONECT records.

        Args:
            pdb_path: Path to the PDB file
            compound_name: Name of the compound (typically directory name)
            residue_sequence: List of residue names
            top_path: Path to topology file for bond information
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

            # Check if we have CONECT records
            has_conect = any(
                line.startswith("CONECT")
                for section in model_sections
                for line in section
            )

            # If no CONECT records and we have a topology file, generate them
            bonds = []
            if not has_conect and top_path:
                bonds = self._extract_bonds_from_top(top_path)

            # Build bond dict for faster lookup
            bond_dict = {}
            for atom1, atom2 in bonds:
                if atom1 not in bond_dict:
                    bond_dict[atom1] = []
                if atom2 not in bond_dict:
                    bond_dict[atom2] = []
                bond_dict[atom1].append(atom2)
                bond_dict[atom2].append(atom1)

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
                atom_lines = []

                for line in section:
                    if line.startswith("MODEL"):
                        # Use sequential model numbers starting from 1
                        new_lines.append(f"MODEL {frame_number+1}\n")
                        found_model_line = True
                    elif line.startswith("ATOM") or line.startswith("HETATM"):
                        new_lines.append(line)
                        atom_lines.append(line)
                    elif line.startswith(("TER", "END")):
                        new_lines.append(line)
                    elif line.startswith("CONECT"):
                        # Keep existing CONECT records if any
                        new_lines.append(line)

                if not found_model_line:
                    # If no MODEL line found, insert one before first atom with proper number
                    for i, line in enumerate(new_lines):
                        if line.startswith("ATOM"):
                            new_lines.insert(i, f"MODEL {frame_number+1}\n")
                            break

                # Add CONECT records if we have bond information but no CONECT records in the file
                if not has_conect and bonds:
                    # Find position for CONECT records (before END/ENDMDL)
                    insert_idx = len(new_lines)
                    for i in range(len(new_lines) - 1, -1, -1):
                        if new_lines[i].startswith(("END", "ENDMDL")):
                            insert_idx = i
                            break

                    # Generate CONECT records
                    conect_lines = []
                    for atom_idx in sorted(bond_dict.keys()):
                        bonded_atoms = sorted(bond_dict[atom_idx])
                        if bonded_atoms:
                            record = f"CONECT{atom_idx:5d}"
                            for bonded_atom in bonded_atoms:
                                record += f"{bonded_atom:5d}"
                            conect_lines.append(record + "\n")

                    # Insert CONECT records
                    new_lines[insert_idx:insert_idx] = conect_lines

                    if self.verbose:
                        self.logger.info(
                            f"Added {len(conect_lines)} CONECT records from topology"
                        )

                frame_number += 1

            # Write back to file
            with open(pdb_path, "w") as f:
                f.writelines(new_lines)

            if self.verbose:
                self.logger.info(f"Customized PDB header format for {pdb_path}")

        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Error customizing PDB header: {str(e)}")
