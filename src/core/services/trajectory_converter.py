#!/usr/bin/env python3
# src/core/services/trajectory_converter.py

"""
Service for converting molecular dynamics trajectories between different formats.
Primarily focused on converting XTC trajectories to multi-model PDB files.
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

# Third party imports
import MDAnalysis as mda
from tqdm import tqdm
from MDAnalysis.coordinates.PDB import PDBWriter

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


class TrajectoryConverter:
    """Service class for converting between different trajectory formats."""

    def __init__(
        self,
        topology_file: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        top_file: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the TrajectoryConverter.

        Args:
            topology_file: Optional path to a topology file (e.g., GRO, PDB)
                         If not provided, will attempt to infer topology from trajectory
            metadata: Optional dictionary containing default metadata for PDB writing
                     Possible keys: resnames, chainIDs, resids, names, elements
            top_file: Optional path to a GROMACS topology (.top) file
            verbose: Whether to show detailed logging output
        """
        self.topology_file = topology_file
        self.metadata = metadata or {}
        self.top_file = top_file
        self.verbose = verbose
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the converter"""
        logger = logging.getLogger(__name__)

        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Set up file handler for all logs
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
                if self.verbose:
                    self.logger.info(f"Found structure file: {potential_top}")
                struct_file = potential_top
                break

        # Look for topology file
        top_file = os.path.join(base_dir, "topol.top")
        if os.path.exists(top_file):
            if self.verbose:
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
        Parse GROMACS topology file for additional metadata and bond information.

        Returns:
            Dict[str, Any]: Dictionary containing parsed metadata and bonds
        """
        if not self.top_file or not os.path.exists(self.top_file):
            return {}

        metadata = {}
        try:
            with open(self.top_file, "r") as f:
                lines = f.readlines()

            # Parse topology file for useful information
            in_atoms = False
            in_bonds = False
            atoms_info = []
            bonds = []
            residue_info = {}  # Store residue information by residue number

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for residue information in comments
                if line.startswith("; residue"):
                    try:
                        parts = line.split()
                        if len(parts) >= 4:
                            resid = int(parts[2])  # Get residue number
                            resname = parts[3]  # Get residue name
                            residue_info[resid] = resname
                    except (ValueError, IndexError):
                        continue
                    continue

                if line.startswith(";"):
                    continue

                if "[ atoms ]" in line:
                    in_atoms = True
                    in_bonds = False
                    continue
                elif "[ bonds ]" in line:
                    in_atoms = False
                    in_bonds = True
                    continue
                elif line.startswith("["):
                    in_atoms = False
                    in_bonds = False
                    continue

                if in_atoms and line:
                    parts = line.split()
                    if len(parts) >= 4:  # Basic atom entry
                        try:
                            atom_index = int(parts[0])  # 1-based index in topology
                            resid = int(parts[2])  # Residue number
                            atoms_info.append(
                                {
                                    "index": atom_index,
                                    "type": parts[1],
                                    "resid": resid,
                                    "resname": residue_info.get(resid, "UNK"),
                                }
                            )
                        except (ValueError, IndexError):
                            continue

                if in_bonds and line:
                    try:
                        parts = [
                            p for p in line.split() if p.strip()
                        ]  # Remove empty strings
                        if len(parts) >= 2:  # Basic bond entry (ai, aj, funct, ...)
                            # Convert to 0-based indexing for MDAnalysis
                            ai = int(parts[0].strip()) - 1  # First atom index
                            aj = int(parts[1].strip()) - 1  # Second atom index
                            # Log the bond being added
                            self.logger.debug(
                                f"Adding bond: {ai+1} - {aj+1} (0-based: {ai} - {aj})"
                            )
                            # Store the bond
                            bonds.append([ai, aj])
                            # Also store the reverse bond since GROMACS only lists each bond once
                            bonds.append([aj, ai])
                    except (ValueError, IndexError) as e:
                        self.logger.warning(
                            f"Could not parse bond line '{line}': {str(e)}"
                        )
                        continue

            if atoms_info:
                # Sort atoms by residue ID to ensure correct order
                atoms_info.sort(key=lambda x: x["resid"])

                # Extract unique residue sequence while preserving order
                seen_residues = set()
                residue_sequence = []
                for atom in atoms_info:
                    resid = atom["resid"]
                    if resid not in seen_residues:
                        seen_residues.add(resid)
                        residue_sequence.append(atom["resname"])

                metadata["elements"] = [info["type"] for info in atoms_info]
                metadata["resnames"] = [info["resname"] for info in atoms_info]
                metadata["atom_indices"] = [info["index"] - 1 for info in atoms_info]
                metadata["resids"] = [info["resid"] for info in atoms_info]
                metadata["residue_sequence"] = residue_sequence

            if bonds:
                self.logger.info(f"Found {len(bonds)} bonds in topology file")
                metadata["bonds"] = bonds
            else:
                self.logger.warning("No bonds found in topology file")

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

        try:
            n_atoms = len(universe.atoms)

            # First, set up residues if we have the sequence information
            if "residue_sequence" in combined_metadata:
                residue_sequence = combined_metadata["residue_sequence"]

                # Get atom-to-residue mapping from topology if available
                atom_residue_mapping = {}
                if "resids" in combined_metadata:
                    topology_resids = combined_metadata["resids"]
                    if len(topology_resids) == n_atoms:
                        # Use the mapping from topology
                        resids = topology_resids
                        resnames = combined_metadata.get("resnames", [])
                        if len(resnames) == n_atoms:
                            # We have complete residue information from topology
                            self.logger.info("Using residue mapping from topology file")
                            universe.add_TopologyAttr("resids", resids)
                            universe.add_TopologyAttr("resnames", resnames)
                        else:
                            # Map resnames based on resids
                            unique_resids = sorted(set(resids))
                            resid_to_resname = {
                                resid: residue_sequence[i % len(residue_sequence)]
                                for i, resid in enumerate(unique_resids)
                            }
                            resnames = [resid_to_resname[resid] for resid in resids]
                            universe.add_TopologyAttr("resids", resids)
                            universe.add_TopologyAttr("resnames", resnames)
                    else:
                        self.logger.warning(
                            f"Topology resids length ({len(topology_resids)}) "
                            f"doesn't match number of atoms ({n_atoms})"
                        )

                if "resids" not in combined_metadata or len(topology_resids) != n_atoms:
                    # Calculate atoms per residue based on total atoms and number of residues
                    n_residues = len(residue_sequence)
                    base_atoms_per_res = n_atoms // n_residues
                    remaining_atoms = n_atoms % n_residues

                    # Distribute atoms among residues
                    resids = []
                    resnames = []
                    current_resid = 1
                    atoms_assigned = 0

                    for i, resname in enumerate(residue_sequence):
                        # Calculate how many atoms belong to this residue
                        atoms_this_res = base_atoms_per_res
                        if i < remaining_atoms:
                            atoms_this_res += 1

                        # Assign residue ID and name to each atom in this residue
                        resids.extend([current_resid] * atoms_this_res)
                        resnames.extend([resname] * atoms_this_res)
                        current_resid += 1
                        atoms_assigned += atoms_this_res

                    # Verify we assigned all atoms
                    if atoms_assigned != n_atoms:
                        raise ValueError(
                            f"Atom assignment mismatch. Expected {n_atoms}, "
                            f"assigned {atoms_assigned}"
                        )

                    # Apply the residue attributes
                    universe.add_TopologyAttr("resids", resids)
                    universe.add_TopologyAttr("resnames", resnames)
                    self.logger.info(
                        f"Distributed {n_atoms} atoms across {n_residues} residues"
                    )

            else:
                # If no sequence information, create one residue for all atoms
                resids = [1] * n_atoms
                resnames = ["UNK"] * n_atoms
                universe.add_TopologyAttr("resids", resids)
                universe.add_TopologyAttr("resnames", resnames)

            # Create a single segment for the whole molecule
            universe.add_TopologyAttr("segids", ["A"] * n_atoms)

            # Add chain IDs (one chain for all residues)
            universe.add_TopologyAttr("chainIDs", ["A"] * n_atoms)

            # Apply atom level attributes
            ATOM_ATTRS = {
                "names",
                "types",
                "elements",
                "occupancies",
                "tempfactors",
                "altLocs",
            }

            for attr in ATOM_ATTRS:
                if attr in combined_metadata:
                    try:
                        values = combined_metadata[attr]
                        if len(values) >= len(universe.atoms):
                            universe.add_TopologyAttr(
                                attr, values[: len(universe.atoms)]
                            )
                    except Exception as e:
                        self.logger.warning(f"Could not apply {attr}: {str(e)}")

            # Apply bond information if available
            if "bonds" in combined_metadata:
                try:
                    bonds = combined_metadata["bonds"]
                    # Add bonds to the universe
                    universe.add_TopologyAttr("bonds", bonds)
                    self.logger.info(f"Added {len(bonds)} bonds from topology file")
                except Exception as e:
                    self.logger.warning(f"Could not apply bonds: {str(e)}")

        except Exception as e:
            self.logger.warning(f"Error applying metadata: {str(e)}")

    def _copy_to_local(self, file_path: str) -> Path:
        """Copy a file to a local temporary directory."""
        # Create temporary directory in user's home
        temp_dir = Path.home() / ".mdanalysis_temp"
        temp_dir.mkdir(exist_ok=True)

        # Create a unique subdirectory for this process
        process_dir = temp_dir / f"process_{os.getpid()}"
        process_dir.mkdir(exist_ok=True)

        # Copy file to temporary location
        src_path = Path(file_path)
        temp_path = process_dir / src_path.name

        with open(src_path, "rb") as src, open(temp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        return temp_path

    def get_trajectory_info(self, xtc_path: str) -> dict:
        """Get information about a trajectory file."""
        if not os.path.exists(xtc_path):
            raise ValueError(f"XTC file not found: {xtc_path}")

        try:
            # Copy files to local storage
            local_xtc = self._copy_to_local(xtc_path)
            local_top = (
                self._copy_to_local(self.topology_file) if self.topology_file else None
            )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                if local_top:
                    universe = mda.Universe(str(local_top), str(local_xtc))
                else:
                    universe = mda.Universe(str(local_xtc))

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
        finally:
            # Clean up temporary files
            if "local_xtc" in locals():
                try:
                    local_xtc.unlink()
                except:
                    pass
            if "local_top" in locals() and local_top:
                try:
                    local_top.unlink()
                except:
                    pass

    def xtc_to_multimodel_pdb(
        self,
        xtc_file: str,
        output_dir: str,
        start: int = 0,
        end: Optional[int] = None,
        stride: int = 1,
    ) -> None:
        """Convert XTC trajectory to multi-model PDB."""
        # Disable all MDAnalysis warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        try:
            # Copy files to local storage
            local_xtc = self._copy_to_local(xtc_file)
            local_top = self._copy_to_local(self.topology_file)

            # Create Universe using local copies
            u = mda.Universe(str(local_top), str(local_xtc))

            # Get the output path
            output_path = Path(output_dir) / "md_Ref.pdb"

            # Create a context manager for the progress bar
            total_frames = len(
                range(start, len(u.trajectory) if end is None else end, stride)
            )

            # Use tqdm only if this is not a child process and not quiet mode
            is_main_process = not bool(os.getenv("PARALLEL_WORKER"))
            is_quiet = not self.verbose or bool(os.getenv("QUIET"))

            with tqdm(
                total=total_frames,
                desc="Converting frames",
                disable=not is_main_process or is_quiet,
                leave=False,
                position=1,
                ncols=80,
            ) as pbar:
                # Process frames
                with PDBWriter(str(output_path)) as pdb:
                    for ts in u.trajectory[start:end:stride]:
                        pdb.write(u.atoms)
                        pbar.update(1)

        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error converting trajectory {xtc_file}: {str(e)}")
            raise
        finally:
            # Clean up temporary files
            if "local_xtc" in locals():
                try:
                    local_xtc.unlink()
                except:
                    pass
            if "local_top" in locals():
                try:
                    local_top.unlink()
                except:
                    pass
