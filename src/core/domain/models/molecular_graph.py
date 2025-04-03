#!/usr/bin/env python3
# src/core/domain/models/molecular_graph.py

"""
Domain model representing a molecular structure as a graph.
"""

from typing import List
import numpy as np
from .atom import Atom
from .bond import Bond


class MolecularGraph:
    """Graph representation of a molecular structure."""

    def __init__(self, atoms: List[Atom], bonds: List[Bond], model_num: int = 1):
        """
        Initialize a MolecularGraph.

        Args:
            atoms: List of Atom objects
            bonds: List of Bond objects
            model_num: Model number of the PDB file
        """
        self.atoms = atoms
        self.bonds = bonds
        self.model_num = model_num

    def get_coordinates(self) -> np.ndarray:
        """Get coordinates of all atoms in the graph.

        Returns:
            numpy array of shape (n_atoms, 3) containing xyz coordinates
        """
        return np.array([atom.coordinates for atom in self.atoms])
