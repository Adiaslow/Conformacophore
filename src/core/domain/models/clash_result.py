"""Domain model for clash detection results."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ClashResult:
    """Contains results from clash detection."""

    has_clashes: bool
    num_clashes: int
    clash_pairs: List[Tuple]
    min_distance: float
