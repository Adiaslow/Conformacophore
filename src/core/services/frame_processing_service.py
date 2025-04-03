"""Service for processing multiple frames of molecular structures."""

from typing import List, Dict, Optional
from ..domain.models.molecular_graph import MolecularGraph
from ..domain.models.alignment_result import AlignmentResult
import os
import numpy as np
from ..domain.models.pdb_frame import PDBFrame
from .alignment_service import AlignmentService
from ...infrastructure.repositories.structure_repository import StructureRepository
from ..domain.models.atom import Atom


class FrameProcessingService:
    """Service for handling multi-frame molecular structure processing."""

    def __init__(
        self,
        alignment_service: AlignmentService,
        structure_repository: StructureRepository,
    ):
        """Initialize with required services."""
        self._alignment_service = alignment_service
        self._repository = structure_repository

    def process_frames(
        self,
        compound_id: str,
        reference: MolecularGraph,
        frames: List[MolecularGraph],
        clash_cutoff: float = 2.0,
    ) -> List[AlignmentResult]:
        """
        Process all frames against reference structure.

        Args:
            compound_id: Unique identifier for the compound
            reference: Reference structure to align to
            frames: List of frames to process
            clash_cutoff: Distance cutoff for clash detection

        Returns:
            List of alignment results, one per frame
        """
        # Update processing history
        parameters = {
            "reference_structure": reference,
            "clash_cutoff": clash_cutoff,
            "num_frames": len(frames),
        }

        results = []
        for frame in frames:
            result = self._alignment_service.align_structures(
                reference=reference,
                target=frame,
                clash_cutoff=clash_cutoff,
            )
            results.append(result)

        # Record processing results
        summary = {
            "num_processed": len(results),
            "avg_rmsd": sum(r.rmsd for r in results) / len(results),
            "num_clashes": sum(
                1 for r in results if r.clash_results and r.clash_results.has_clashes
            ),
        }

        self._repository._registry.update_processing(
            compound_id=compound_id,
            step_name="frame_alignment",
            parameters=parameters,
            results=summary,
        )

        return results

    def save_aligned_frame(
        self, frame: PDBFrame, alignment: AlignmentResult, output_path: str
    ) -> None:
        """
        Save aligned frame to PDB file.

        Args:
            frame: Frame to save
            alignment: Alignment result containing transformation
            output_path: Path to save aligned PDB
        """
        if alignment.transformation_matrix is None:
            return

        rotation, translation = alignment.transformation_matrix

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            # Write header
            f.write("TITLE     Aligned structure\n")
            f.write(f"REMARK    Original frame ID: {frame.model_num}\n")
            f.write(f"REMARK    RMSD: {alignment.rmsd:.3f}\n")
            f.write(f"REMARK    Matched atoms: {alignment.matched_atoms}\n")

            # Write transformed coordinates
            for i, atom in enumerate(frame.atoms):
                # Convert dictionary atom to Atom object
                atom_obj = Atom(
                    atom_id=atom["atom_num"],
                    element=atom["element"],
                    coordinates=(atom["x"], atom["y"], atom["z"]),
                    residue_name=atom["residue_name"],
                    residue_id=atom["residue_num"],
                    chain_id=atom["chain_id"],
                    atom_name=atom["atom_name"],
                    serial=atom["atom_num"],
                )

                # Apply transformation
                new_coords = np.dot(atom_obj.coordinates, rotation.T) + translation

                # Format PDB ATOM record
                f.write(
                    f"ATOM  {i+1:5d} {atom_obj.atom_name:^4s} {atom_obj.residue_name:3s} "
                    f"{atom_obj.chain_id}{atom_obj.residue_id:4d}    "
                    f"{new_coords[0]:8.3f}{new_coords[1]:8.3f}{new_coords[2]:8.3f}"
                    f"  1.00  0.00          {atom_obj.atom_name[0]:>2s}\n"
                )

            # Write connectivity if available
            for conn in frame.connectivity:
                if len(conn) >= 2:
                    f.write(f"CONECT{conn[0]:5d}{conn[1]:5d}\n")

            f.write("END\n")
