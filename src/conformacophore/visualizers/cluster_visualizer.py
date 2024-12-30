import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as plt_colors
from matplotlib import colormaps
from scipy.cluster import hierarchy
from sklearn import manifold

class ClusterVisualizer:
    COLOR_LIST = ["red", "blue", "lime", "yellow", "darkorchid", "deepskyblue",
                  "orange", "brown", "gray", "black", "darkgreen", "navy"]

    def get_cmap(self, num_clusters):
        """Get appropriate colormap based on number of clusters."""
        if num_clusters <= len(self.COLOR_LIST):
            return plt_colors.ListedColormap(self.COLOR_LIST[:num_clusters])
        return colormaps['rainbow']

    def plot_dendrogram(self, linkage, clusters, output_dir, compound_id, dpi=300):
        plt.figure(figsize=(10, 6))

        unique_clusters = len(np.unique(clusters))
        cmap = self.get_cmap(unique_clusters)
        colors = cmap(np.linspace(0, 1, unique_clusters))

        hierarchy.set_link_color_palette([plt_colors.rgb2hex(c) for c in colors])
        cutoff_height = linkage[-(unique_clusters-1), 2]

        hierarchy.dendrogram(linkage, color_threshold=cutoff_height)
        plt.axhline(y=cutoff_height, color='gray', linestyle='--')
        plt.title(f'Clustering Dendrogram - Compound {compound_id}')
        plt.xlabel('Frame')
        plt.ylabel('Distance (Å)')
        plt.savefig(os.path.join(output_dir, f'compound_{compound_id}_dendrogram.png'),
                    dpi=dpi, bbox_inches='tight')
        plt.close()

        return colors

    def plot_linear_projection(self, clusters, frames, colors, output_dir, compound_id, dpi=300):
        plt.figure(figsize=(12, 1.5))
        data = np.array(clusters).reshape(1, -1)
        cmap = plt_colors.ListedColormap([c for c in colors])
        plt.imshow(data, aspect='auto', interpolation='none', cmap=cmap)
        plt.yticks([])
        plt.xlabel('Frame')
        plt.title(f'Cluster Assignment - Compound {compound_id}')
        plt.savefig(os.path.join(output_dir, f'compound_{compound_id}_linear_projection'),
                    dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_cluster_sizes(self, clusters, colors, output_dir, compound_id, dpi=300):
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        plt.figure(figsize=(8, 6))
        bars = plt.bar(unique_clusters, counts, color=colors)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                     ha='center', va='bottom')
        plt.title(f'Cluster Sizes - Compound {compound_id}')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Frames')
        plt.savefig(os.path.join(output_dir, f'compound_{compound_id}_cluster_sizes.png'),
                    dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_2d_projection(self, rmsd_matrix, clusters, colors, output_dir, compound_id, dpi=300):
        rmsd_norm = rmsd_matrix / np.max(rmsd_matrix)
        mds = manifold.MDS(n_components=2, dissimilarity="precomputed", normalized_stress="auto")
        coords = mds.fit_transform(rmsd_norm)

        unique_clusters = np.unique(clusters)
        cluster_spreads = {cluster_id: np.mean(rmsd_matrix[np.ix_(np.where(clusters == cluster_id)[0], np.where(clusters == cluster_id)[0])][np.nonzero(rmsd_matrix[np.ix_(np.where(clusters == cluster_id)[0], np.where(clusters == cluster_id)[0])])]) for cluster_id in unique_clusters}

        max_spread = max(cluster_spreads.values())
        normalized_spreads = {k: v/max_spread * 1000 for k, v in cluster_spreads.items()}

        plt.figure(figsize=(8, 8))

        for i, cluster_id in enumerate(unique_clusters):
            cluster_frames = np.where(clusters == cluster_id)[0]
            cluster_coords = coords[cluster_frames]
            centroid = np.mean(cluster_coords, axis=0)

            plt.scatter(centroid[0], centroid[1], s=normalized_spreads[cluster_id], c=[colors[i]], alpha=0.6, label=f'Cluster {cluster_id}')
            plt.annotate(f'{cluster_id}', (centroid[0], centroid[1]), xytext=(5,5), textcoords='offset points')

        plt.title(f'2D Distance Projection - Compound {compound_id}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('equal')

        nonzero_rmsd = rmsd_matrix[np.nonzero(rmsd_matrix)]
        if len(nonzero_rmsd) > 0:
            min_rmsd = np.min(nonzero_rmsd) * 10
            max_rmsd = np.max(rmsd_matrix) * 10
            plt.figtext(0.02, 0.02, f'RMSD range: {min_rmsd:.2f} - {max_rmsd:.2f} Å', bbox=dict(facecolor='white', alpha=0.8))

        plt.savefig(os.path.join(output_dir, f'compound_{compound_id}_2d_projection.png'), dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_distance_matrix(self, rmsd_matrix, output_dir, compound_id, dpi=300):
        plt.figure(figsize=(8, 8))
        im = plt.imshow(rmsd_matrix * 10, interpolation='nearest', cmap='viridis')
        plt.colorbar(im, label='RMSD (Å)')
        plt.title(f'RMSD Distance Matrix - Compound {compound_id}')
        plt.xlabel('Frame')
        plt.ylabel('Frame')
        plt.savefig(os.path.join(output_dir, f'compound_{compound_id}_distance_matrix.png'), dpi=dpi, bbox_inches='tight')
        plt.close()

    def create_visualizations(self, rmsd_matrix, clusters, linkage, cutoff, compound_id, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        colors = self.plot_dendrogram(linkage, clusters, output_dir, compound_id)
        self.plot_linear_projection(clusters, len(clusters), colors, output_dir, compound_id)
        self.plot_cluster_sizes(clusters, colors, output_dir, compound_id)
        self.plot_2d_projection(rmsd_matrix, clusters, colors, output_dir, compound_id)
        self.plot_distance_matrix(rmsd_matrix, output_dir, compound_id)
