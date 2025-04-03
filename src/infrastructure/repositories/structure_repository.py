# src/infrastructure/repositories/structure_repository.py
"""Repository implementation for molecular structures."""

from typing import List, Optional, Dict
import os
from Bio.PDB.PDBParser import PDBParser
import numpy as np

from ...core.interfaces.repository import Repository
from ...core.domain.models.molecular_graph import MolecularGraph
from ...core.domain.models.pdb_frame import PDBFrameCollection
from ...core.domain.models.compound_metadata import CompoundRegistry, CompoundMetadata
from ...core.domain.models.atom import Atom


class StructureRepository(Repository[MolecularGraph]):
    """Repository for handling molecular structure storage and retrieval."""

    def __init__(self, data_dir: str):
        """
        Initialize repository with data directory.

        Args:
            data_dir: Directory containing structure files
        """
        self._data_dir = data_dir
        self._parser = PDBParser(QUIET=True)
        self._cache: Dict[str, List[MolecularGraph]] = {}
        self._registry = CompoundRegistry()

    def get(
        self, id: str, chain_id: Optional[str] = None
    ) -> Optional[List[MolecularGraph]]:
        """
        Retrieve all frames for a structure by ID.

        Args:
            id: Structure identifier
            chain_id: Optional chain ID to filter atoms

        Returns:
            List of MolecularGraph objects, one per frame
        """
        if id in self._cache:
            return self._cache[id]

        file_path = os.path.join(self._data_dir, f"{id}.pdb")
        if not os.path.exists(file_path):
            return None

        frame_collection = PDBFrameCollection(file_path)

        # Register compound metadata
        if frame_collection.frames:
            first_frame = frame_collection.frames[0]
            self._registry.register_compound(
                compound_id=first_frame.compound_id,
                sequence=first_frame.sequence or "UNK",  # Use UNK if sequence is None
                source_file=file_path,
                num_frames=len(frame_collection.frames),
            )

        graphs = [
            self._create_molecular_graph(frame, chain_id)
            for frame in frame_collection.frames
        ]
        graphs = [g for g in graphs if g is not None]  # Filter out None results

        if graphs:
            self._cache[id] = graphs
            return graphs
        return None

    def list(self) -> Dict[str, List[MolecularGraph]]:
        """
        List all available molecular structures.

        Returns:
            Dictionary mapping structure IDs to lists of frames
        """
        structures = {}
        for file_name in os.listdir(self._data_dir):
            if file_name.endswith(".pdb"):
                id = os.path.splitext(file_name)[0]
                if frames := self.get(id):
                    structures[id] = frames
        return structures

    def create(self, entity: MolecularGraph) -> MolecularGraph:
        """Create a new molecular structure entry."""
        raise NotImplementedError("Creation not supported")

    def update(self, entity: MolecularGraph) -> MolecularGraph:
        """Update an existing molecular structure."""
        raise NotImplementedError("Updates not supported")

    def delete(self, id: str) -> None:
        """Delete a molecular structure."""
        raise NotImplementedError("Deletion not supported")

    def _create_molecular_graph(
        self, frame: "PDBFrame", chain_id: Optional[str] = None
    ) -> Optional[MolecularGraph]:
        """
        Convert PDBFrame to MolecularGraph.

        Args:
            frame: PDB frame containing atoms and connectivity
            chain_id: Optional chain ID to filter atoms

        Returns:
            MolecularGraph representation of the structure
        """
        # Filter atoms by chain if specified
        atoms = frame.atoms
        if chain_id:
            atoms = [atom for atom in atoms if atom["chain_id"] == chain_id]
            if not atoms:
                return None

        # Convert dictionary atoms to Atom objects
        atom_objects = []
        for atom in atoms:
            atom_objects.append(
                Atom(
                    atom_id=atom["atom_num"],
                    element=atom["element"],
                    coordinates=(atom["x"], atom["y"], atom["z"]),
                    residue_name=atom["residue_name"],
                    residue_id=atom["residue_num"],
                    chain_id=atom["chain_id"],
                    atom_name=atom["atom_name"],
                    serial=atom["atom_num"],
                )
            )

        # Create graph with atoms as nodes
        graph = MolecularGraph(atom_objects)

        # Add connectivity from CONECT records
        # Only add connections between atoms in the selected chain
        atom_indices = {atom.serial: i for i, atom in enumerate(atom_objects)}
        for connection in frame.connectivity:
            if len(connection) >= 2:
                # Convert PDB atom numbers to our indices
                if connection[0] in atom_indices and connection[1] in atom_indices:
                    idx1 = atom_indices[connection[0]]
                    idx2 = atom_indices[connection[1]]
                    graph.add_connection(idx1, idx2)

        return graph

    def get_metadata(self, id: str) -> Optional[CompoundMetadata]:
        """Get metadata for a compound."""
        return self._registry.get_compound(id)

    def list_metadata(self) -> List[CompoundMetadata]:
        """Get metadata for all compounds."""
        return self._registry.list_compounds()
