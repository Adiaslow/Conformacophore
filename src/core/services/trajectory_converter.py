#!/usr/bin/env python3
# src/core/services/trajectory_converter.py

"""
Service for converting molecular dynamics trajectories between different formats.
Primarily focused on converting XTC trajectories to multi-model PDB files.
"""

from pathlib import Path
import MDAnalysis as mda
from typing import Optional, Dict, Any, List, Tuple, NamedTuple, Sequence
import os
from tqdm import tqdm
import warnings
import logging
from dataclasses import dataclass
import functools
import tempfile
from MDAnalysis.coordinates.PDB import PDBWriter


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


class TrajectoryConverter:
    """Service class for converting between different trajectory formats."""

    def __init__(
        self,
        topology_file: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        top_file: Optional[str] = None,
    ):
        """
        Initialize the TrajectoryConverter.

        Args:
            topology_file: Optional path to a topology file (e.g., GRO, PDB)
                         If not provided, will attempt to infer topology from trajectory
            metadata: Optional dictionary containing default metadata for PDB writing
                     Possible keys: resnames, chainIDs, resids, names, elements
            top_file: Optional path to a GROMACS topology (.top) file
        """
        self.topology_file = topology_file
        self.metadata = metadata or {}
        self.top_file = top_file
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the converter"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _find_topology_files(
        self, xtc_path: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Find topology and .top files in the same directory as the XTC file.

        Args:
            xtc_path: Path to the XTC file

        Returns:
            Tuple[Optional[str], Optional[str]]: Paths to topology and .top files
        """
        base_path = os.path.splitext(xtc_path)[0]
        base_dir = os.path.dirname(xtc_path)

        # Look for structure files
        struct_file = None
        for ext in [".gro", ".pdb", ".tpr"]:
            potential_top = base_path + ext
            if os.path.exists(potential_top):
                self.logger.info(f"Found structure file: {potential_top}")
                struct_file = potential_top
                break

        # Look for topology file
        top_file = os.path.join(base_dir, "topol.top")
        if os.path.exists(top_file):
            self.logger.info(f"Found topology file: {top_file}")
            return struct_file, top_file

        return struct_file, None

    def _create_universe(self, xtc_path: str) -> mda.Universe:
        """
        Create an MDAnalysis Universe from the trajectory file.

        Args:
            xtc_path: Path to the XTC file

        Returns:
            MDAnalysis Universe object

        Raises:
            ValueError: If the trajectory cannot be loaded
        """
        try:
            # Use explicitly provided topology file if available
            if self.topology_file:
                if not os.path.exists(self.topology_file):
                    raise ValueError(f"Topology file not found: {self.topology_file}")
                return mda.Universe(self.topology_file, xtc_path)

            # Try to find topology files in the same directory
            struct_file, found_top = self._find_topology_files(xtc_path)

            if struct_file:
                if found_top and not self.top_file:
                    self.top_file = found_top
                return mda.Universe(struct_file, xtc_path)

            # If no topology file found, try to load XTC directly
            self.logger.warning(
                "No structure file found. Some metadata will use default values."
            )
            return mda.Universe(xtc_path)

        except Exception as e:
            raise ValueError(f"Failed to create Universe from trajectory: {str(e)}")

    def _parse_top_file(self) -> Dict[str, Any]:
        """
        Parse GROMACS topology file for additional metadata.

        Returns:
            Dict[str, Any]: Dictionary containing parsed metadata
        """
        if not self.top_file or not os.path.exists(self.top_file):
            return {}

        metadata = {}
        try:
            with open(self.top_file, "r") as f:
                lines = f.readlines()

            # Parse topology file for useful information
            in_atoms = False
            atoms_info = []

            for line in lines:
                line = line.strip()
                if line.startswith(";"):
                    continue

                if "[ atoms ]" in line:
                    in_atoms = True
                    continue
                elif in_atoms and line.startswith("["):
                    in_atoms = False

                if in_atoms and line:
                    parts = line.split()
                    if len(parts) >= 4:  # Basic atom entry
                        atoms_info.append(
                            {
                                "type": parts[1],
                                "resname": parts[3],
                            }
                        )

            if atoms_info:
                metadata["elements"] = [info["type"] for info in atoms_info]
                metadata["resnames"] = [info["resname"] for info in atoms_info]

            return metadata

        except Exception as e:
            self.logger.warning(f"Could not parse topology file: {str(e)}")
            return {}

    def _get_metadata(self, universe: mda.Universe, frame_num: int) -> MoleculeMetadata:
        """
        Extract metadata from the universe for the current frame.

        Args:
            universe: MDAnalysis Universe object
            frame_num: Current frame number

        Returns:
            MoleculeMetadata object with frame information
        """
        if not universe:
            return MoleculeMetadata.create_default(frame_num)

        try:
            # Try to get residue information
            if hasattr(universe, "residues") and universe.residues:
                residue_sequence = []
                numbered_sequence = []
                for res in universe.residues:
                    try:
                        resname = getattr(res, "resname", "UNK")
                        resid = getattr(res, "resid", len(residue_sequence) + 1)
                        residue_sequence.append(resname)
                        numbered_sequence.append(f"{resname}{resid}")
                    except Exception:
                        residue_sequence.append("UNK")
                        numbered_sequence.append(f"UNK{len(residue_sequence)}")
            else:
                # Create a single residue for all atoms if no residue information
                residue_sequence = ["UNK"]
                numbered_sequence = ["UNK1"]

            # Get time and box information safely
            try:
                time = universe.trajectory.time
            except Exception:
                time = 0.0

            try:
                box_dimensions = universe.dimensions.tolist()
            except Exception:
                box_dimensions = [0.0, 0.0, 0.0, 90.0, 90.0, 90.0]

            return MoleculeMetadata(
                compound_name=os.path.basename(self.topology_file or "unknown"),
                frame_num=frame_num,
                residue_sequence=residue_sequence,
                numbered_sequence=numbered_sequence,
                time=time,
                box_dimensions=box_dimensions,
            )

        except Exception as e:
            self.logger.warning(f"Error extracting metadata: {str(e)}, using defaults")
            return MoleculeMetadata.create_default(frame_num)

    def _apply_metadata(self, universe: mda.Universe) -> None:
        """
        Apply metadata to the universe if provided.

        Args:
            universe: MDAnalysis Universe to modify
        """
        if not universe or not universe.atoms:
            return

        # First try to get metadata from topology file
        top_metadata = self._parse_top_file()

        # Combine with user-provided metadata, giving preference to user values
        combined_metadata = {**top_metadata, **self.metadata}

        # Separate residue and atom level attributes
        RESIDUE_ATTRS = {"resnames", "resids", "segids"}
        ATOM_ATTRS = {
            "names",
            "types",
            "elements",
            "chainIDs",
            "occupancies",
            "tempfactors",
            "altLocs",
        }

        # Apply residue-level metadata
        if universe.residues is not None:
            n_residues = len(universe.residues)
            for key in RESIDUE_ATTRS:
                if key in combined_metadata and hasattr(universe.residues, key):
                    try:
                        value = combined_metadata[key]
                        if isinstance(value, (list, tuple)):
                            if len(value) < n_residues:
                                value = list(value) * (n_residues // len(value) + 1)
                            value = value[:n_residues]
                        elif isinstance(value, str):
                            value = [value] * n_residues
                        setattr(universe.residues, key, value)
                    except Exception as e:
                        self.logger.debug(
                            f"Could not set residue attribute {key}: {str(e)}"
                        )

        # Apply atom-level metadata
        n_atoms = len(universe.atoms)
        for key in ATOM_ATTRS:
            if key in combined_metadata and hasattr(universe.atoms, key):
                try:
                    value = combined_metadata[key]
                    if isinstance(value, (list, tuple)):
                        if len(value) < n_atoms:
                            value = list(value) * (n_atoms // len(value) + 1)
                        value = value[:n_atoms]
                    elif isinstance(value, str):
                        value = [value] * n_atoms
                    setattr(universe.atoms, key, value)
                except Exception as e:
                    self.logger.debug(f"Could not set atom attribute {key}: {str(e)}")

    def xtc_to_multimodel_pdb(
        self,
        xtc_path: str,
        output_path: str,
        start: int = 0,
        end: Optional[int] = None,
        stride: int = 1,
    ) -> List[Path]:
        """
        Convert an XTC trajectory file to multiple PDB files, one per sequence.
        Each PDB file will contain all frames for its sequence as separate models.

        Args:
            xtc_path: Path to the input XTC file
            output_path: Base path where the output PDB files should be saved
            start: First frame to convert (0-based indexing)
            end: Last frame to convert (None means until the end)
            stride: Step size for frame selection

        Returns:
            List[Path]: Paths to the generated PDB files
        """
        if not os.path.exists(xtc_path):
            raise ValueError(f"XTC file not found: {xtc_path}")

        try:
            # Create universe from trajectory
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                universe = self._create_universe(xtc_path)

            if not universe or not universe.atoms:
                raise ValueError("Could not create universe from trajectory")

            # Apply any provided metadata
            self._apply_metadata(universe)

            # Calculate total frames to process
            trajectory_slice = universe.trajectory[start:end:stride]
            total_frames = len(trajectory_slice)

            # Get residue information and create sequence groups
            sequence_groups = {}  # Dict to store atom indices for each sequence
            if hasattr(universe, "residues") and universe.residues is not None:
                for residue in universe.residues:
                    resname = getattr(residue, "resname", "UNK")
                    resid = getattr(residue, "resid", 1)
                    sequence_key = f"{resname}{resid}"
                    sequence_groups[sequence_key] = {
                        "atoms": residue.atoms,
                        "resname": resname,
                        "resid": resid,
                    }
            else:
                sequence_groups["UNK1"] = {
                    "atoms": universe.atoms,
                    "resname": "UNK",
                    "resid": 1,
                }

            # Create output directory if needed
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Base name for output files
            base_name = os.path.splitext(output_path)[0]
            output_files = []

            # Process each sequence separately
            for seq_name, seq_info in sequence_groups.items():
                seq_atoms = seq_info["atoms"]
                seq_output = f"{base_name}_{seq_name}.pdb"

                # Create a temporary universe for this sequence
                temp_u = mda.Universe.empty(
                    len(seq_atoms),
                    n_residues=1,  # One residue per sequence
                    atom_resindex=[0] * len(seq_atoms),
                    trajectory=True,
                )

                # Set atom attributes
                names = (
                    seq_atoms.names
                    if hasattr(seq_atoms, "names")
                    else [f"A{i+1}" for i in range(len(seq_atoms))]
                )
                elements = (
                    seq_atoms.elements
                    if hasattr(seq_atoms, "elements")
                    else ["C"] * len(seq_atoms)
                )

                # Set topology attributes
                temp_u.add_TopologyAttr("names", names)
                temp_u.add_TopologyAttr("elements", elements)
                temp_u.add_TopologyAttr("resnames", [seq_info["resname"]])
                temp_u.add_TopologyAttr("resids", [seq_info["resid"]])
                temp_u.add_TopologyAttr("chainIDs", ["A"] * len(seq_atoms))
                temp_u.add_TopologyAttr("occupancies", [1.0] * len(seq_atoms))
                temp_u.add_TopologyAttr("tempfactors", [0.0] * len(seq_atoms))
                temp_u.add_TopologyAttr("altLocs", [" "] * len(seq_atoms))
                temp_u.add_TopologyAttr("segids", [""] * 1)  # One segment

                # Write trajectory for this sequence
                with PDBWriter(seq_output, multiframe=True) as pdb_writer:
                    # Write header information
                    remark_lines = [
                        f"REMARK   Generated by TrajectoryConverter",
                        f"REMARK   Source: {xtc_path}",
                        f"REMARK   Structure: {self.topology_file or 'inferred'}",
                        f"REMARK   Sequence: {seq_name}",
                        f"REMARK   Frames: {total_frames} (stride: {stride})",
                    ]

                    with open(seq_output, "w") as f:
                        f.write("\n".join(remark_lines) + "\n")

                    # Process frames
                    for frame_num, ts in enumerate(
                        tqdm(
                            trajectory_slice,
                            total=total_frames,
                            desc=f"Converting {seq_name}",
                            unit="frame",
                        )
                    ):
                        # Update coordinates for this sequence
                        temp_u.atoms.positions = seq_atoms.positions.copy()

                        # Write frame-specific remarks
                        frame_remarks = [
                            f"REMARK   Frame: {frame_num}",
                            f"REMARK   Time: {ts.time:.2f} ps",
                        ]
                        with open(seq_output, "a") as f:
                            f.write("\n".join(frame_remarks) + "\n")

                        # Write frame
                        pdb_writer.write(temp_u)

                output_files.append(Path(seq_output))
                self.logger.info(
                    f"Created PDB file for sequence {seq_name}: {seq_output}"
                )

            return output_files

        except Exception as e:
            raise ValueError(f"Failed to convert XTC to PDB: {str(e)}")

    def get_trajectory_info(self, xtc_path: str) -> dict:
        """
        Get information about a trajectory file.

        Args:
            xtc_path: Path to the XTC file

        Returns:
            dict: Dictionary containing trajectory information
        """
        if not os.path.exists(xtc_path):
            raise ValueError(f"XTC file not found: {xtc_path}")

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                universe = self._create_universe(xtc_path)

            if not universe or not universe.atoms:
                raise ValueError("Could not create universe from trajectory")

            # Try to parse topology file
            top_metadata = self._parse_top_file()

            # Get initial frame metadata
            metadata = self._get_metadata(universe, 0)

            return {
                "n_frames": len(universe.trajectory),
                "n_atoms": len(universe.atoms),
                "time_range": (
                    universe.trajectory[0].time,
                    universe.trajectory[-1].time,
                ),
                "dimensions": universe.dimensions.tolist(),
                "topology_source": self.topology_file or "inferred",
                "top_file": self.top_file,
                "residue_sequence": metadata.residue_sequence if metadata else [],
                "numbered_sequence": metadata.numbered_sequence if metadata else [],
                "metadata_status": {
                    "resnames": hasattr(universe.atoms, "resnames")
                    or bool(top_metadata.get("resnames")),
                    "chainIDs": hasattr(universe.atoms, "chainIDs"),
                    "resids": hasattr(universe.atoms, "resids"),
                    "names": hasattr(universe.atoms, "names"),
                    "elements": hasattr(universe.atoms, "elements")
                    or bool(top_metadata.get("elements")),
                },
            }
        except Exception as e:
            raise ValueError(f"Failed to get trajectory info: {str(e)}")
