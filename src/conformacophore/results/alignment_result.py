from dataclasses import dataclass
import numpy as np
from typing import Any, List, Optional, Tuple
from src.conformacophore.results.clash_result import ClashResult

@dataclass
class AlignmentResult:
    """Contains results from molecular structure alignment."""
    rmsd: float
    matched_atoms: int
    transformation_matrix: Optional[Tuple[np.ndarray, np.ndarray]]
    matched_pairs: List[Tuple[Any, Any]]
    clash_results: Optional[ClashResult] = None
