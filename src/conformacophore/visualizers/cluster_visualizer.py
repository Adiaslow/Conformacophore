import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as plt_colors
from matplotlib import colormaps
from sklearn import manifold
from scipy.cluster import hierarchy
from typing import List, Dict

COLOR_LIST = ["red", "blue", "lime", "yellow", "darkorchid", "deepskyblue",
              "orange", "brown", "gray", "black", "darkgreen", "navy"]

class ClusterVisualizer:
    """Handles visualization of clustering results."""

    @staticmethod
    def get_cmap(num_clusters):
        """Get appropriate colormap based on number of clusters."""
        if num_clusters <= len(COLOR_LIST):
            return plt_colors.ListedColormap(COLOR_LIST[:num_clusters])
        return colormaps['rainbow']

    @staticmethod
    def plot_dendrogram(linkage, clusters, output_dir, compound_id, dpi=300):
        """
        Plot dendrogram with cluster coloring.

        Args:
            linkage: Hierarchical clustering linkage matrix
            clusters: List of cluster assignments
            output_dir: Directory to save plot (str)
            compound_id: Compound identifier
        """
        plt.figure(figsize=(10, 6))

        # Get colormap
        unique_clusters = len(np.unique(clusters))
        cmap = ClusterVisualizer.get_cmap(unique_clusters)
        colors = cmap(np.linspace(0, 1, unique_clusters))

        # Set color palette
        hierarchy.set_link_color_palette([plt_colors.rgb2hex(c) for c in colors])

        # Find the cutoff height that gives us the desired number of clusters
        cutoff_height = linkage[-(unique_clusters-1), 2]

        # Plot dendrogram with color threshold
        hierarchy.dendrogram(linkage, color_threshold=cutoff_height)

        plt.axhline(y=cutoff_height, color='gray', linestyle='--')
        plt.title(f'Clustering Dendrogram - Compound {compound_id}')
        plt.xlabel('Frame')
        plt.ylabel('Distance (Å)')

        # Ensure output_dir is a string
        output_dir_str = str(output_dir)
        plt.savefig(os.path.join(output_dir_str, f'compound_{compound_id}_dendrogram.png'),
                    dpi=dpi, bbox_inches='tight')
        plt.close()

        return colors

    @staticmethod
    def plot_linear_projection(clusters, frames, colors, output_dir, compound_id, dpi=300):
        """Plot linear projection of clusters."""
        plt.figure(figsize=(12, 1.5))

        # Create data matrix for imshow
        data = np.array(clusters).reshape(1, -1)

        # Create colormap
        cmap = plt_colors.ListedColormap([c for c in colors])

        plt.imshow(data, aspect='auto', interpolation='none', cmap=cmap)
        plt.yticks([])
        plt.xlabel('Frame')
        plt.title(f'Cluster Assignment - Compound {compound_id}')

        plt.savefig(os.path.join(output_dir, f'compound_{compound_id}_linear_projection'),
                    dpi=dpi, bbox_inches='tight')
        plt.close()
