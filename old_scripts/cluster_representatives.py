import os
import tempfile
import mdtraj as md
import numpy as np
import argparse
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import shutil
import pandas as pd
from matplotlib import colors as plt_colors
from matplotlib import colormaps
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


COLOR_LIST = ["red", "blue", "lime", "yellow", "darkorchid", "deepskyblue",
              "orange", "brown", "gray", "black", "darkgreen", "navy"]

class PDBHeaderHandler:
    """Handles reading and writing of PDB header information."""

    def __init__(self):
        self.headers = []
        self.model_headers = {}  # Headers specific to each model

    def read_headers(self, pdb_path: str):
        """Read header information from PDB file."""
        self.headers = []
        self.model_headers = {}

        # Header keywords to capture
        header_keywords = [
            'HEADER', 'TITLE', 'COMPND', 'SOURCE', 'KEYWDS',
            'EXPDTA', 'AUTHOR', 'REVDAT', 'REMARK', 'SEQRES'
        ]

        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        # First pass: capture global headers before any MODEL
        for line in lines:
            if any(line.startswith(keyword) for keyword in header_keywords):
                self.headers.append(line)

            if line.startswith('MODEL'):
                break

        # Second pass: capture model-specific headers
        current_model = -1
        reading_model = False

        for line in lines:
            if line.startswith('MODEL'):
                current_model += 1
                reading_model = True
                self.model_headers[current_model] = []

            elif line.startswith('ENDMDL'):
                reading_model = False

            elif reading_model:
                # Capture model-specific headers
                if any(line.startswith(keyword) for keyword in
                       ['COMPND', 'REMARK', 'SEQRES']):
                    self.model_headers[current_model].append(line)

        # If no models found, treat all headers as global
        if not self.model_headers:
            self.model_headers[0] = []

    def write_headers(self, file_handle):
        """Write header information to file."""
        # Write global headers
        for header in self.headers:
            file_handle.write(header)

    def write_model_headers(self, file_handle, model_num: int):
        """Write model-specific headers to file."""
        if model_num in self.model_headers:
            for header in self.model_headers[model_num]:
                file_handle.write(header)


def get_chain_atoms(traj, chain_letters):
    """
    Get atoms for specified chains using topology chain IDs.

    Args:
        traj: MDTraj trajectory
        chain_letters: List of chain letters to extract

    Returns:
        List of atom indices for the specified chains
    """
    chain_letters = [c.upper() for c in chain_letters]
    atoms = []

    # Create mapping of chain IDs to their atoms
    for chain in traj.topology.chains:
        if chain.chain_id.upper() in chain_letters:
            atoms.extend([atom.index for atom in chain.atoms])

    if not atoms:
        raise ValueError(f"No atoms found for specified chains {chain_letters}")

    return atoms

def extract_chains(pdb_file, chain_letters, topology=None):
    """
    Extract specific chains from a PDB file using chain letters.

    Args:
        pdb_file: Path to PDB file
        chain_letters: List of chain letters to extract
        topology: Optional topology file

    Returns:
        mdtraj.Trajectory object containing only the specified chains
    """
    traj = md.load(pdb_file, top=topology) if topology else md.load(pdb_file)

    try:
        chain_atoms = get_chain_atoms(traj, chain_letters)
        return traj.atom_slice(chain_atoms)
    except ValueError as e:
        # Print topology information for debugging
        print(f"\nAvailable chains in {os.path.basename(pdb_file)}:")
        for chain in traj.topology.chains:
            print(f"Chain ID: '{chain.chain_id}' with {len([atom for atom in chain.atoms])} atoms")
        raise e

def calculate_rmsd_matrix(pdbs, molecule_chain):
    """
    Calculate RMSD matrix between all PDB structures for a specific chain.

    Args:
        pdbs: List of PDB file paths
        molecule_chain: Chain letter to use for RMSD calculation
    """
    n_structs = len(pdbs)
    rmsd_matrix = np.zeros((n_structs, n_structs))

    # Load all structures and extract relevant chain
    trajs = []
    for pdb in pdbs:
        try:
            traj = extract_chains(pdb, [molecule_chain])
            trajs.append(traj)
        except ValueError as e:
            print(f"Warning: Error processing {pdb}")
            print(str(e))
            return None

    # Calculate RMSD matrix
    print("Calculating RMSD matrix...")
    for i in range(n_structs):
        for j in range(i+1, n_structs):
            # Align structures and calculate RMSD
            rmsd = md.rmsd(trajs[i], trajs[j])[0]  # Only one frame per trajectory
            rmsd_matrix[i,j] = rmsd_matrix[j,i] = rmsd

    return rmsd_matrix

def find_representative_structure(pdbs, rmsd_matrix):
    """Find the most representative structure using hierarchical clustering."""
    # Convert the RMSD matrix to a condensed form for hierarchy.linkage
    condensed_matrix = squareform(rmsd_matrix)

    # Perform hierarchical clustering
    Z = hierarchy.linkage(condensed_matrix, method='ward')

    # Get cluster assignments - we'll use 1 cluster to find the central structure
    clusters = hierarchy.fcluster(Z, t=1, criterion='maxclust')

    # Find the structure with minimum average RMSD to all other structures
    avg_rmsds = np.mean(rmsd_matrix, axis=1)
    representative_idx = np.argmin(avg_rmsds)

    return pdbs[representative_idx]

def combine_structures(pdb_file, target_chains, molecule_chain, output_path):
    """
    Save structure containing both target and molecule chains with headers preserved.

    Args:
        pdb_file: Path to PDB file containing all chains
        target_chains: List of chain letters for target
        molecule_chain: Chain letter for molecule
        output_path: Output path for output PDB
    """
    # Read headers
    header_handler = PDBHeaderHandler()
    header_handler.read_headers(pdb_file)

    # Load structure and verify all required chains are present
    traj = md.load(pdb_file)

    # Verify all chains are present
    present_chains = [chain.chain_id.upper() for chain in traj.topology.chains]
    required_chains = target_chains + [molecule_chain]
    missing_chains = [chain for chain in required_chains if chain not in present_chains]

    if missing_chains:
        raise ValueError(f"Missing required chains: {missing_chains}. Present chains: {present_chains}")

    # Save with headers preserved
    with open(output_path, 'w') as f:
        # Write global headers
        header_handler.write_headers(f)

        # Create a small function to save trajectory with file handle
        def save_traj_with_headers(trajectory, file_handle):
            # Use a temporary file to save the PDB
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as temp:
                trajectory.save_pdb(temp.name)

            # Read contents of temporary file after PDB save
            with open(temp.name, 'r') as temp_file:
                pdb_lines = temp_file.readlines()

            # Write PDB contents to actual file handle
            file_handle.writelines(pdb_lines)

            # Remove temporary file
            os.unlink(temp.name)

        # Save trajectory
        save_traj_with_headers(traj, f)

def validate_chain_input(chain):
    """Validate that chain input is a valid chain letter."""
    if not isinstance(chain, str) or len(chain) != 1:
        raise argparse.ArgumentTypeError("Chain must be a single letter (A-Z)")
    if not chain.isalpha():
        raise argparse.ArgumentTypeError("Chain must be a letter (A-Z)")
    return chain.upper()

def calculate_cluster_metrics(rmsd_matrix, max_clusters=10):
    """
    Calculate various clustering metrics for different numbers of clusters.
    """
    # Convert RMSD matrix to condensed form for hierarchical clustering
    condensed_matrix = squareform(rmsd_matrix)

    # Perform hierarchical clustering
    Z = hierarchy.linkage(condensed_matrix, method='ward')

    # Initialize metrics dictionaries
    metrics = {
        'elbow': [],
        'silhouette': [],
        'calinski': [],
        'davies': []
    }

    # Calculate metrics for different numbers of clusters
    for n_clusters in range(2, max_clusters + 1):
        # Get cluster labels
        labels = hierarchy.fcluster(Z, t=n_clusters, criterion='maxclust')

        # Calculate elbow metric (within-cluster sum of squares)
        within_ss = 0
        for i in range(1, n_clusters + 1):
            cluster_points = np.where(labels == i)[0]
            if len(cluster_points) > 1:
                cluster_rmsd = rmsd_matrix[np.ix_(cluster_points, cluster_points)]
                within_ss += np.sum(cluster_rmsd ** 2) / (2 * len(cluster_points))
        metrics['elbow'].append(within_ss)

        # Calculate silhouette score
        try:
            sil = silhouette_score(rmsd_matrix, labels, metric='precomputed')
            metrics['silhouette'].append(sil)
        except:
            metrics['silhouette'].append(np.nan)

        # Calculate Calinski-Harabasz score
        try:
            cal = calinski_harabasz_score(rmsd_matrix, labels)
            metrics['calinski'].append(cal)
        except:
            metrics['calinski'].append(np.nan)

        # Calculate Davies-Bouldin score
        try:
            dav = davies_bouldin_score(rmsd_matrix, labels)
            metrics['davies'].append(dav)
        except:
            metrics['davies'].append(np.nan)

    return metrics, Z

def plot_cluster_metrics(metrics, output_dir, compound_id, dpi=300):
    """Plot clustering metrics to help determine optimal number of clusters."""
    n_clusters = range(2, len(metrics['elbow']) + 2)

    # Create a 2x2 subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Elbow plot
    ax1.plot(n_clusters, metrics['elbow'], 'bo-')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Within-cluster Sum of Squares')

    # Silhouette plot
    ax2.plot(n_clusters, metrics['silhouette'], 'ro-')
    ax2.set_title('Silhouette Score\n(Higher is better)')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')

    # Calinski-Harabasz plot
    ax3.plot(n_clusters, metrics['calinski'], 'go-')
    ax3.set_title('Calinski-Harabasz Score\n(Higher is better)')
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('Calinski-Harabasz Score')

    # Davies-Bouldin plot
    ax4.plot(n_clusters, metrics['davies'], 'mo-')
    ax4.set_title('Davies-Bouldin Score\n(Lower is better)')
    ax4.set_xlabel('Number of Clusters')
    ax4.set_ylabel('Davies-Bouldin Score')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'compound_{compound_id}_cluster_metrics.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close()

def suggest_optimal_clusters(metrics):
    """Suggest optimal number of clusters based on various metrics."""
    suggestions = {}

    # Elbow method - find point of maximum curvature
    elbow = np.array(metrics['elbow'])
    diffs = np.diff(elbow, 2)  # Second derivative
    suggestions['elbow'] = np.argmax(np.abs(diffs)) + 2

    # Silhouette - maximum score
    silhouette = np.array(metrics['silhouette'])
    suggestions['silhouette'] = np.nanargmax(silhouette) + 2

    # Calinski-Harabasz - maximum score
    calinski = np.array(metrics['calinski'])
    suggestions['calinski'] = np.nanargmax(calinski) + 2

    # Davies-Bouldin - minimum score
    davies = np.array(metrics['davies'])
    suggestions['davies'] = np.nanargmin(davies) + 2

    return suggestions

def get_optimal_clusters(rmsd_matrix, max_clusters=10, output_dir=None, compound_id=None):
    """Calculate and visualize optimal number of clusters."""
    # Calculate metrics
    metrics, linkage_matrix = calculate_cluster_metrics(rmsd_matrix, max_clusters)

    # Plot metrics if output directory is provided
    if output_dir and compound_id:
        plot_cluster_metrics(metrics, output_dir, compound_id)

    # Get suggestions
    suggestions = suggest_optimal_clusters(metrics)

    return metrics, suggestions, linkage_matrix

def get_cmap(num_clusters):
    """Get appropriate colormap based on number of clusters."""
    if num_clusters <= len(COLOR_LIST):
        return plt_colors.ListedColormap(COLOR_LIST[:num_clusters])
    return colormaps['rainbow']

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
    cmap = get_cmap(unique_clusters)
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

def plot_cluster_sizes(clusters, colors, output_dir, compound_id, dpi=300):
    """Plot barplot of cluster sizes."""
    unique_clusters, counts = np.unique(clusters, return_counts=True)

    plt.figure(figsize=(8, 6))
    bars = plt.bar(unique_clusters, counts, color=colors)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')

    plt.title(f'Cluster Sizes - Compound {compound_id}')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Frames')

    plt.savefig(os.path.join(output_dir, f'compound_{compound_id}_cluster_sizes.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_2d_projection(rmsd_matrix, clusters, colors, output_dir, compound_id, dpi=300):
    """Plot 2D projection of cluster distances using MDS."""
    # Normalize RMSD matrix
    rmsd_norm = rmsd_matrix / np.max(rmsd_matrix)

    # Perform MDS
    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", normalized_stress="auto")
    coords = mds.fit_transform(rmsd_norm)

    # Calculate spreads for each cluster
    unique_clusters = np.unique(clusters)
    cluster_spreads = {}

    for cluster_id in unique_clusters:
        cluster_frames = np.where(clusters == cluster_id)[0]
        if len(cluster_frames) > 1:  # Need at least 2 frames to calculate spread
            cluster_rmsd = rmsd_matrix[np.ix_(cluster_frames, cluster_frames)]
            nonzero_rmsd = cluster_rmsd[np.nonzero(cluster_rmsd)]
            if len(nonzero_rmsd) > 0:
                spread = np.mean(nonzero_rmsd)
            else:
                spread = 0.1  # Default small spread if no non-zero RMSDs
        else:
            spread = 0.1  # Default small spread for single-frame clusters
        cluster_spreads[cluster_id] = spread

    # Calculate point sizes - normalize spreads for visualization
    max_spread = max(cluster_spreads.values())
    normalized_spreads = {k: v/max_spread * 1000 for k, v in cluster_spreads.items()}

    plt.figure(figsize=(8, 8))

    # Plot points by cluster
    for i, cluster_id in enumerate(unique_clusters):
        cluster_frames = np.where(clusters == cluster_id)[0]
        cluster_coords = coords[cluster_frames]

        # Calculate centroid for this cluster
        centroid = np.mean(cluster_coords, axis=0)

        plt.scatter(centroid[0], centroid[1],
                   s=normalized_spreads[cluster_id],
                   c=[colors[i]], alpha=0.6,
                   label=f'Cluster {cluster_id}')
        plt.annotate(f'{cluster_id}', (centroid[0], centroid[1]),
                    xytext=(5,5), textcoords='offset points')

    plt.title(f'2D Distance Projection - Compound {compound_id}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('equal')

    # Add RMSD range information
    nonzero_rmsd = rmsd_matrix[np.nonzero(rmsd_matrix)]
    if len(nonzero_rmsd) > 0:
        min_rmsd = np.min(nonzero_rmsd) * 10  # Convert to Angstroms
        max_rmsd = np.max(rmsd_matrix) * 10  # Convert to Angstroms
        plt.figtext(0.02, 0.02, f'RMSD range: {min_rmsd:.2f} - {max_rmsd:.2f} Å',
                    bbox=dict(facecolor='white', alpha=0.8))

    plt.savefig(os.path.join(output_dir, f'compound_{compound_id}_2d_projection.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_distance_matrix(rmsd_matrix, output_dir, compound_id, dpi=300):
    """Plot RMSD distance matrix."""
    plt.figure(figsize=(8, 8))

    im = plt.imshow(rmsd_matrix * 10,  # Convert to Angstroms
                    interpolation='nearest', cmap='viridis')
    plt.colorbar(im, label='RMSD (Å)')

    plt.title(f'RMSD Distance Matrix - Compound {compound_id}')
    plt.xlabel('Frame')
    plt.ylabel('Frame')

    plt.savefig(os.path.join(output_dir, f'compound_{compound_id}_distance_matrix.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close()

def create_visualizations(rmsd_matrix, clusters, linkage, cutoff, compound_id, output_dir):
    """
    Create all visualizations for a compound's clustering results.

    Args:
        rmsd_matrix: RMSD matrix between frames
        clusters: Cluster assignments for each frame
        linkage: Hierarchical clustering linkage matrix
        cutoff: Cutoff value used for clustering
        compound_id: Compound identifier
        output_dir: Directory to save visualizations
    """
    # Create visualization subdirectory
    os.makedirs(output_dir, exist_ok=True)

    # Create all visualizations
    colors = plot_dendrogram(linkage, clusters, output_dir, compound_id)
    plot_linear_projection(clusters, len(clusters), colors, output_dir, compound_id)
    plot_cluster_sizes(clusters, colors, output_dir, compound_id)
    plot_2d_projection(rmsd_matrix, clusters, colors, output_dir, compound_id)
    plot_distance_matrix(rmsd_matrix, output_dir, compound_id)

def find_lowest_rmsd_structure(pdbs, rmsd_matrix, clusters):
    """Find the structure with lowest RMSD in the most populous cluster.

    Returns:
        tuple: (selected_structure, dict with cluster information)
    """
    # Get cluster sizes
    unique_clusters, counts = np.unique(clusters, return_counts=True)

    # Find the largest cluster(s)
    max_size = np.max(counts)
    largest_clusters = unique_clusters[counts == max_size]

    # If multiple clusters have the same size, we'll compare their minimum RMSDs
    min_rmsd = float('inf')
    selected_structure = None
    selected_cluster_id = None

    for cluster_id in largest_clusters:
        # Get indices of structures in this cluster
        cluster_indices = np.where(clusters == cluster_id)[0]

        # Get RMSD submatrix for this cluster
        cluster_rmsd = rmsd_matrix[np.ix_(cluster_indices, cluster_indices)]

        # Find minimum non-zero RMSD in this cluster
        nonzero_mask = cluster_rmsd > 0
        if np.any(nonzero_mask):
            cluster_min_rmsd = np.min(cluster_rmsd[nonzero_mask])

            if cluster_min_rmsd < min_rmsd:
                min_rmsd = cluster_min_rmsd
                # Find the structure with the lowest average RMSD to other structures in the cluster
                nonzero_means = []
                for i in range(len(cluster_rmsd)):
                    row = cluster_rmsd[i]
                    nonzero_row = row[row > 0]
                    if len(nonzero_row) > 0:
                        nonzero_means.append(np.mean(nonzero_row))
                    else:
                        nonzero_means.append(float('inf'))

                if nonzero_means:
                    min_rmsd_idx = cluster_indices[np.argmin(nonzero_means)]
                    selected_structure = pdbs[min_rmsd_idx]
                    selected_cluster_id = cluster_id

    if selected_structure is None:
        # If we couldn't find a structure with valid RMSD, take the first one
        selected_structure = pdbs[0]
        selected_cluster_id = clusters[0]
        min_rmsd = 0.0

    print(f"\nSelected structure from cluster {selected_cluster_id}")
    print(f"Cluster size: {max_size}")
    print(f"Minimum RMSD in cluster: {min_rmsd * 10:.2f} Å")  # Convert to Angstroms

    cluster_info = {
        'cluster_id': selected_cluster_id,
        'size': max_size,
        'min_rmsd': min_rmsd * 10  # Convert to Angstroms
    }

    return selected_structure, cluster_info

def create_compound_summary(compound_id, metrics, suggestions, clusters, rmsd_matrix, selected_cluster_info):
    """Create summary dictionary for a single compound."""
    # Calculate basic statistics
    nonzero_rmsd = rmsd_matrix[np.nonzero(rmsd_matrix)]
    if len(nonzero_rmsd) > 0:
        min_rmsd = np.min(nonzero_rmsd) * 10  # Convert to Angstroms
        max_rmsd = np.max(rmsd_matrix) * 10
        avg_rmsd = np.mean(nonzero_rmsd) * 10
    else:
        min_rmsd = max_rmsd = avg_rmsd = 0.0

    # Get cluster statistics
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    cluster_stats = []
    for cluster_id, count in zip(unique_clusters, counts):
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_rmsd = rmsd_matrix[np.ix_(cluster_indices, cluster_indices)]

        # Calculate average RMSD only from non-zero values
        nonzero_cluster_rmsd = cluster_rmsd[cluster_rmsd > 0]
        if len(nonzero_cluster_rmsd) > 0:
            avg_internal_rmsd = np.mean(nonzero_cluster_rmsd) * 10
            max_internal_rmsd = np.max(cluster_rmsd) * 10
        else:
            avg_internal_rmsd = max_internal_rmsd = 0.0

        cluster_stats.append({
            'cluster_id': cluster_id,
            'size': count,
            'avg_internal_rmsd': avg_internal_rmsd,
            'max_internal_rmsd': max_internal_rmsd
        })

    # Create summary data dictionary
    summary_data = {
        'compound_id': compound_id,
        'total_structures': len(rmsd_matrix),
        'min_rmsd': min_rmsd,
        'max_rmsd': max_rmsd,
        'avg_rmsd': avg_rmsd,
        'optimal_clusters_elbow': suggestions['elbow'],
        'optimal_clusters_silhouette': suggestions['silhouette'],
        'optimal_clusters_calinski': suggestions['calinski'],
        'optimal_clusters_davies': suggestions['davies'],
        'selected_cluster': selected_cluster_info['cluster_id'],
        'selected_cluster_size': selected_cluster_info['size'],
        'selected_cluster_min_rmsd': selected_cluster_info['min_rmsd'],
        'num_clusters': len(unique_clusters)
    }

    # Add metrics for each possible number of clusters
    n_clusters = range(2, len(metrics['elbow']) + 2)
    for i, n in enumerate(n_clusters):
        summary_data.update({
            f'elbow_metric_{n}': metrics['elbow'][i],
            f'silhouette_metric_{n}': metrics['silhouette'][i],
            f'calinski_metric_{n}': metrics['calinski'][i],
            f'davies_metric_{n}': metrics['davies'][i]
        })

    # Add statistics for each cluster
    for stat in cluster_stats:
        cluster_id = stat['cluster_id']
        summary_data.update({
            f'cluster_{cluster_id}_size': stat['size'],
            f'cluster_{cluster_id}_avg_rmsd': stat['avg_internal_rmsd'],
            f'cluster_{cluster_id}_max_rmsd': stat['max_internal_rmsd']
        })

    return summary_data

def write_analysis_summary(output_dir):
    """Write the final comprehensive summary CSV for all compounds."""
    summary_path = os.path.join(output_dir, "analysis_summary.csv")

    if not hasattr(write_analysis_summary, 'summaries'):
        write_analysis_summary.summaries = []

    # Create DataFrame from all accumulated summaries
    df = pd.DataFrame(write_analysis_summary.summaries)

    # Sort by compound_id
    df = df.sort_values('compound_id')

    # Save to CSV
    df.to_csv(summary_path, index=False)
    print(f"\nWrote complete analysis summary to {summary_path}")
    return summary_path

def process_filtered_results(input_dir, output_dir, target_chains, molecule_chain):
    """
    Process filtered results to find representative structures.
    """
    # Initialize summaries list
    write_analysis_summary.summaries = []

    # Read summary statistics
    summary_file = os.path.join(input_dir, "summary_statistics.csv")
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary statistics file not found in {input_dir}")

    summary_df = pd.read_csv(summary_file)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each compound
    for _, row in summary_df.iterrows():
        compound_id = row['Compound']
        compound_dir = os.path.join(input_dir, str(compound_id))

        if not os.path.exists(compound_dir):
            print(f"Warning: Directory not found for compound {compound_id}")
            continue

        # Process compound and get summary
        try:
            pdb_files = [os.path.join(compound_dir, f) for f in os.listdir(compound_dir)
                        if f.endswith('.pdb')]

            if not pdb_files:
                print(f"Warning: No PDB files found for compound {compound_id}")
                continue

            print(f"\nProcessing compound {compound_id}")
            print(f"Found {len(pdb_files)} structures")

            # Calculate RMSD matrix and perform clustering
            rmsd_matrix = calculate_rmsd_matrix(pdb_files, molecule_chain)
            if rmsd_matrix is None:
                print(f"Skipping compound {compound_id} due to RMSD calculation error")
                continue

            # Perform clustering analysis
            metrics, suggestions, linkage_matrix = get_optimal_clusters(
                rmsd_matrix, max_clusters=10, output_dir=output_dir, compound_id=compound_id
            )

            # Get cluster assignments
            n_clusters = max(set(suggestions.values()), key=list(suggestions.values()).count)
            clusters = hierarchy.fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

            # Find representative structure
            representative, cluster_info = find_lowest_rmsd_structure(pdb_files, rmsd_matrix, clusters)

            # Create summary for this compound
            compound_summary = create_compound_summary(
                compound_id, metrics, suggestions, clusters, rmsd_matrix, cluster_info
            )
            write_analysis_summary.summaries.append(compound_summary)

            # Save representative structure
            output_path = os.path.join(output_dir, f"compound_{compound_id}_complex.pdb")
            combine_structures(representative, target_chains, molecule_chain, output_path)

        except Exception as e:
            print(f"Error processing compound {compound_id}: {str(e)}")
            continue

    # Write final summary CSV
    write_analysis_summary(output_dir)

def main():
    parser = argparse.ArgumentParser(description='Find representative structures from filtered superimposition results')
    parser.add_argument('input_dir', help='Directory containing filtered superimposition results')
    parser.add_argument('output_dir', help='Directory to store representative structures')
    parser.add_argument('--target-chains', type=validate_chain_input, nargs='+', required=True,
                      help='Chain letters for target protein (e.g., A B for chains A and B)')
    parser.add_argument('--molecule-chain', type=validate_chain_input, required=True,
                      help='Chain letter for molecules to cluster (e.g., X)')

    args = parser.parse_args()

    process_filtered_results(
        args.input_dir,
        args.output_dir,
        args.target_chains,
        args.molecule_chain
    )

if __name__ == '__main__':
    main()

"""
Example Usage:

python cluster_representatives.py /Users/Adam/Desktop/hexamers_water_data_filtered /Users/Adam/Desktop/hexamers_water_data_filtered_representatives --target-chains A B C --molecule-chain X

python cluster_representatives.py /Users/Adam/Desktop/hexamers_chc13_data_filtered /Users/Adam/Desktop/hexamers_chc13_data_filtered_representatives --target-chains A B C --molecule-chain X

python cluster_representatives.py /Users/Adam/Desktop/heptamers_water_data_filtered /Users/Adam/Desktop/heptamers_water_data_filtered_representatives --target-chains A B C --molecule-chain X

python cluster_representatives.py /Users/Adam/Desktop/heptamers_chc13_data_filtered /Users/Adam/Desktop/heptamers_chc13_data_filtered_representatives --target-chains A B C --molecule-chain X
"""
