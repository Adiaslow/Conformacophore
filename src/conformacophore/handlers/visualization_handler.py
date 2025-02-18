from src.conformacophore.visualizers.cluster_visualizer import ClusterVisualizer

class VisualizationHandler:
    def __init__(self):
        self.visualizer = ClusterVisualizer()

    def create_visualizations(self, rmsd_matrix, clusters, linkage_matrix, cutoff, compound_id, output_dir):
        self.visualizer.create_visualizations(rmsd_matrix, clusters, linkage_matrix, cutoff, compound_id, output_dir)
