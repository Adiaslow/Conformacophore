#!/usr/bin/env python3
# src/core/domain/models/bond.py

"""
Domain model representing a chemical bond between atoms.
"""

from dataclasses import dataclass
from enum import Enum, auto


class BondType(Enum):
    """Enumeration of possible bond types."""

    SINGLE = auto()
    DOUBLE = auto()
    TRIPLE = auto()
    AROMATIC = auto()
    UNKNOWN = auto()


@dataclass
class Bond:
    """Represents a chemical bond between two atoms."""

    atom1_id: int
    atom2_id: int
    bond_type: BondType = BondType.SINGLE
    bond_order: float = 1.0
