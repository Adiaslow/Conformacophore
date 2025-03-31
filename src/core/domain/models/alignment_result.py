"""Domain model for structure alignment results."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import numpy as np


@dataclass
class AlignmentResult:
    """Contains results from molecular structure alignment."""

    rmsd: float
    matched_atoms: int
    transformation_matrix: Optional[Tuple[np.ndarray, np.ndarray]]
    matched_pairs: List[Tuple[Any, Any]]
    clash_results: Optional["ClashResult"] = None
    isomorphic_match: bool = False
