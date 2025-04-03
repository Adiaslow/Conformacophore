from dataclasses import dataclass


@dataclass
class Atom:
    """Represents an atom in the molecule."""

    index: int  # 0-based
    name: str
    residue_name: str
    residue_number: int
    element: str
    x: float
    y: float
    z: float
