from dataclasses import dataclass
from typing import List

@dataclass
class ClashResult:
    """Contains results from clash detection."""
    has_clashes: bool
    num_clashes: int
    clash_pairs: List[tuple]
    min_distance: float
