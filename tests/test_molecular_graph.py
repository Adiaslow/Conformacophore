import pytest
import networkx as nx
from Bio.PDB.PDBParser import PDBParser
from src.conformacophore.entities.molecular_graph import MolecularGraph
from src.conformacophore.visualizers.graph_visualizer import GraphVisualizer

def test_molecular_graph():
    # Path to the PDB file for testing
    pdb_file = 'tests/test_data/input/943.pdb'

    # Initialize PDB parser and read the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('test_structure', pdb_file)

    # Extract all atoms from the structure
    atoms = list(structure.get_atoms())

    # Create a molecular graph
    mol_graph = MolecularGraph(atoms)
    G = mol_graph.graph

    # Assertions to verify the graph structure
    assert isinstance(G, nx.Graph)
    assert len(G.nodes) == len(atoms)  # Ensure all atoms are nodes

    # Visualize the molecular graph
    visualizer = GraphVisualizer(G)
    visualizer.draw_graph()
