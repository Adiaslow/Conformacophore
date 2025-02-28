#!/usr/bin/env python3
# src/core/domain/models/atom.py

"""
Domain model representing an atom in a molecular structure.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class Atom:
    """Represents an atom in a molecular structure."""

    atom_id: int
    element: str
    coordinates: Tuple[float, float, float]
    residue_name: str = ""
    residue_id: int = 0
    chain_id: str = "A"
    atom_name: str = ""
    b_factor: float = 0.0
    occupancy: float = 1.0
    alt_loc: str = ""
    charge: float = 0.0
    serial: Optional[int] = None
