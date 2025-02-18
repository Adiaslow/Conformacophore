import networkx as nx
from typing import List, Any
from Bio.PDB import PDBParser

class MolecularGraph:
    """Represents a molecular structure as a graph."""

    def __init__(self, atoms: List[Any]):
        """Initialize molecular graph from list of atoms."""
        self.atoms = atoms
        self.graph = self._create_graph()

    def _create_graph(self) -> nx.Graph:
        """Create NetworkX graph from atoms using PDB connectivity."""
        G = nx.Graph()

        # Add nodes with attributes
        for atom in self.atoms:
            residue = atom.get_parent()
            node_attrs = {
                'name': atom.name,
                'element': atom.element,
                'coord': tuple(atom.coord),
                'residue_name': residue.resname,
                'residue_id': residue.id
            }
            G.add_node(atom.serial_number, **node_attrs)

        return G

    @staticmethod
    def from_pdb_file(pdb_file: str):
        """Create a MolecularGraph from a PDB file."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_file)
        atoms = list(structure.get_atoms())
        graph = MolecularGraph(atoms)

        # Read CONECT records from the PDB file
        with open(pdb_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("CONECT"):
                    parts = line.split()
                    atom_serial = int(parts[1])
                    bonded_atoms = [int(serial) for serial in parts[2:]]
                    for bonded_atom in bonded_atoms:
                        # Add edges based on CONECT records
                        if bonded_atom in graph.graph.nodes:
                            graph.graph.add_edge(atom_serial, bonded_atom)

        return graph
