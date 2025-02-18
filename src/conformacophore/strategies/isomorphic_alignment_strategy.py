from Bio.PDB.Superimposer import Superimposer
import networkx as nx
from typing import Any, Dict, List

from src.conformacophore.results.alignment_result import AlignmentResult
from src.conformacophore.strategies.alignment_strategy import AlignmentStrategy
from src.conformacophore.entities.molecular_graph import MolecularGraph


class IsomorphicAlignmentStrategy(AlignmentStrategy):
    """
    Implementation of structure alignment using graph isomorphism.

    This strategy aligns molecular structures by:
    1. Finding matching atoms using graph isomorphism
    2. Superimposing the matched atoms using SVD
    """

    def align(self, mol1: MolecularGraph, mol2: MolecularGraph) -> AlignmentResult:
        """
        Align two molecular structures using graph isomorphism.

        Args:
            mol1: First molecular graph
            mol2: Second molecular graph

        Returns:
            AlignmentResult containing RMSD and transformation
        """
        # Find graph matches
        matches = self._custom_graph_match(mol1.graph, mol2.graph)

        if not matches:
            return self._empty_result()

        # Convert matches to atom pairs
        matched_pairs = list(matches.items())
        matched_mol1_atoms = [pair[0] for pair in matched_pairs]
        matched_mol2_atoms = [pair[1] for pair in matched_pairs]

        # Perform superposition
        superimposer = Superimposer()
        superimposer.set_atoms(matched_mol1_atoms, matched_mol2_atoms)

        return AlignmentResult(
            rmsd=superimposer.rms,
            matched_atoms=len(matched_pairs),
            transformation_matrix=(superimposer.rotran[0], superimposer.rotran[1]),
            matched_pairs=matched_pairs
        )

    def _custom_graph_match(
        self,
        graph1: nx.Graph,
        graph2: nx.Graph
    ) -> Dict[Any, Any]:
        """
        Custom graph matching implementation optimized for molecular structures.

        Args:
            graph1: First molecular graph
            graph2: Second molecular graph

        Returns:
            Dictionary mapping nodes from graph1 to graph2
        """
        def find_terminal_nodes(graph: nx.Graph) -> List[Any]:
            """Find nodes with only one connection."""
            return [node for node in graph.nodes() if graph.degree(node) == 1]

        def node_matches(node1: Any, node2: Any) -> bool:
            """Check if two nodes represent the same type of atom."""
            return (graph1.nodes[node1]['element'] ==
                   graph2.nodes[node2]['element'])

        def explore_neighborhood(
            node1: Any,
            node2: Any,
            visited1: set = None,
            visited2: set = None
        ) -> Dict[Any, Any]:
            """
            Recursively explore and match node neighborhoods.

            Args:
                node1: Current node in first graph
                node2: Current node in second graph
                visited1: Set of visited nodes in first graph
                visited2: Set of visited nodes in second graph

            Returns:
                Dictionary of matched nodes
            """
            if visited1 is None:
                visited1 = set()
            if visited2 is None:
                visited2 = set()

            current_match = {node1: node2}
            visited1.add(node1)
            visited2.add(node2)

            neighbors1 = list(graph1.neighbors(node1))
            neighbors2 = list(graph2.neighbors(node2))

            for neighbor1 in neighbors1:
                if neighbor1 in visited1:
                    continue

                potential_matches = [
                    n2 for n2 in neighbors2
                    if node_matches(neighbor1, n2) and n2 not in visited2
                ]

                if not potential_matches:
                    return None

                for match2 in potential_matches:
                    neighbor_match = explore_neighborhood(
                        neighbor1,
                        match2,
                        visited1.copy(),
                        visited2.copy()
                    )

                    if neighbor_match:
                        current_match.update(neighbor_match)
                        break
                else:
                    return None

            return current_match

        # Start matching from terminal nodes
        matches = {}
        terminals1 = find_terminal_nodes(graph1)
        terminals2 = find_terminal_nodes(graph2)

        for start1 in terminals1:
            for start2 in terminals2:
                if node_matches(start1, start2):
                    match = explore_neighborhood(start1, start2)
                    if match and len(match) > len(matches):
                        matches = match

        return matches

    @staticmethod
    def _empty_result() -> AlignmentResult:
        """Create an empty alignment result."""
        return AlignmentResult(
            rmsd=float('inf'),
            matched_atoms=0,
            transformation_matrix=None,
            matched_pairs=[],
            clash_results=None
        )
