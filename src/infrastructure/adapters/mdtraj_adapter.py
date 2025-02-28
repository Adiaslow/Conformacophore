"""Adapter for MDTraj structure analysis."""

from typing import List, Tuple
import mdtraj as md
import numpy as np

from ...core.domain.models.molecular_graph import MolecularGraph


class MDTrajAdapter:
    """Adapter for MDTraj structure analysis functionality."""

    def convert_to_trajectory(self, graph: MolecularGraph) -> md.Trajectory:
        """
        Convert MolecularGraph to MDTraj Trajectory.

        Args:
            graph: Molecular structure as graph

        Returns:
            MDTraj Trajectory object
        """
        # Implementation depends on specific conversion logic
        raise NotImplementedError

    def calculate_rmsd(self, struct1: MolecularGraph, struct2: MolecularGraph) -> float:
        """
        Calculate RMSD between two structures using MDTraj.

        Args:
            struct1: First structure
            struct2: Second structure

        Returns:
            RMSD value
        """
        traj1 = self.convert_to_trajectory(struct1)
        traj2 = self.convert_to_trajectory(struct2)
        return float(md.rmsd(traj1, traj2)[0])
