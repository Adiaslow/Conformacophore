import matplotlib.pyplot as plt
import networkx as nx
from typing import Any

# CPK color scheme for elements
CPK_COLORS = {
    'H': 'white', 'C': 'black', 'N': 'blue', 'O': 'red', 'F': 'green', 'Cl': 'green',
    'Br': 'darkred', 'I': 'purple', 'P': 'orange', 'S': 'yellow', 'B': 'salmon',
    'Li': 'purple', 'Na': 'purple', 'K': 'purple', 'Rb': 'purple', 'Cs': 'purple',
    'Fr': 'purple', 'Be': 'darkgreen', 'Mg': 'darkgreen', 'Ca': 'darkgreen',
    'Sr': 'darkgreen', 'Ba': 'darkgreen', 'Ra': 'darkgreen', 'Ti': 'gray',
    'Fe': 'orange', 'default': 'gray'  # Default color for elements not in the list
}

class GraphVisualizer:
    """Class for visualizing molecular graphs."""

    def __init__(self, graph: nx.Graph):
        """Initialize the visualizer with a molecular graph."""
        self.graph = graph

    def get_cpk_color(self, element: str) -> str:
        """Return the CPK color for a given element."""
        return CPK_COLORS.get(element, CPK_COLORS['default'])

    def draw_graph(self, with_labels=True, node_size=500, edge_color='black', font_size=10):
        """Draw the molecular graph using matplotlib and networkx."""
        pos = nx.kamada_kawai_layout(self.graph)


        # Get node colors based on CPK color scheme
        node_colors = [self.get_cpk_color(data['element']) for _, data in self.graph.nodes(data=True)]

        # Get node labels as element symbols
        node_labels = {node: data['element'] for node, data in self.graph.nodes(data=True)}

        plt.figure(figsize=(10, 10))
        nx.draw(self.graph, pos, with_labels=with_labels, labels=node_labels, node_color=node_colors, node_size=node_size, edge_color=edge_color, font_size=font_size)

        plt.title("Molecular Graph")
        plt.show()

    def draw_graph_with_attributes(self):
        """Draw the molecular graph with atom attributes as labels."""
        labels = {node: f"{data['element']}-{data['residue_name']}{data['residue_id'][1]}" for node, data in self.graph.nodes(data=True)}

        self.draw_graph(with_labels=False)
        pos = nx.spring_layout(self.graph)

        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        plt.show()

    def draw_graph_3d(self):
        """Draw a 3D representation of the molecular graph."""
        pos = nx.spring_layout(self.graph, dim=3)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        for node, (x, y, z) in pos.items():
            ax.scatter(x, y, z, color=self.get_cpk_color(self.graph.nodes[node]['element']), s=500)
            ax.text(x, y, z, self.graph.nodes[node]['element'], fontsize=10)

        for (start, end) in self.graph.edges:
            x = [pos[start][0], pos[end][0]]
            y = [pos[start][1], pos[end][1]]
            z = [pos[start][2], pos[end][2]]
            ax.plot(x, y, z, color='black')

        plt.title("3D Molecular Graph")
        plt.show()
