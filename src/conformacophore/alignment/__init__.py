from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from typing import List

from src.conformacophore.results.clash_result import ClashResult

class ClashDetector:
    """Handles detection of steric clashes between molecular chains."""

    def detect_clashes(
        self,
        structure: Structure,
        chain_to_check: Chain,
        target_chains: List[str],
        distance_cutoff: float = 2.0
    ) -> ClashResult:
        """
        Detect clashes between a chain and specified target chains.

        Args:
            structure: Complete structure containing all chains
            chain_to_check: Chain to check for clashes
            target_chains: List of chain IDs to check against
            distance_cutoff: Distance threshold for clash detection (Angstroms)

        Returns:
            ClashResult containing clash information
        """
        from Bio.PDB.NeighborSearch import NeighborSearch
        from Bio.PDB.Selection import unfold_entities

        # Get atoms from the chain we're checking
        chain_atoms = list(unfold_entities(chain_to_check, 'A'))

        # Get atoms from target chains
        target_atoms = []
        for chain_id in target_chains:
            if chain_id in structure[0]:
                target_atoms.extend(list(unfold_entities(structure[0][chain_id], 'A')))

        if not target_atoms:
            return ClashResult(
                has_clashes=False,
                num_clashes=0,
                clash_pairs=[],
                min_distance=float('inf')
            )

        # Create neighbor search for target atoms
        ns = NeighborSearch(target_atoms)

        # Check for clashes
        clash_pairs = []
        min_distance = float('inf')

        for atom in chain_atoms:
            if atom.element == 'H':  # Skip hydrogen atoms
                continue

            close_atoms = ns.search(atom.coord, distance_cutoff)

            for close_atom in close_atoms:
                if close_atom.element == 'H':  # Skip hydrogen atoms
                    continue

                distance = np.linalg.norm(atom.coord - close_atom.coord)
                min_distance = min(min_distance, distance)

                if distance < distance_cutoff:
                    clash_pairs.append((atom, close_atom))

        return ClashResult(
            has_clashes=len(clash_pairs) > 0,
            num_clashes=len(clash_pairs),
            clash_pairs=clash_pairs,
            min_distance=min_distance
        )
