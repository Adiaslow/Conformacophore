import pytest
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from src.conformacophore.handlers.clustering_handler import ClusteringHandler
from src.conformacophore.visualizers.cluster_visualizer import ClusterVisualizer
import time

def extract_frames_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structures = []

    with open(pdb_file, 'r') as file:
        lines = file.readlines()

    current_coords = []
    for line in lines:
        if line.startswith('MODEL'):
            current_coords = []
        elif line.startswith('ENDMDL'):
            structures.append(np.array(current_coords))
        elif line.startswith('ATOM'):
            parts = line.split()
            x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
            current_coords.append([x, y, z])

    return structures

def test_rmsd_clustering():
    start_time = time.time()

    pdb_file = 'tests/test_data/input/943.pdb'
    structures = extract_frames_from_pdb(pdb_file)

    print(f"Number of frames extracted: {len(structures)}")
    if len(structures) <= 1:
        raise ValueError("Insufficient frames were extracted from the PDB file for clustering")

    handler = ClusteringHandler()
    rmsd_matrix = handler.calculate_rmsd_matrix(structures)

    print(f"RMSD matrix shape: {rmsd_matrix.shape}")
    assert rmsd_matrix.shape == (len(structures), len(structures))

    metrics, suggestions, linkage_matrix = handler.get_optimal_clusters(rmsd_matrix, 5, 'tests/test_data/output', '943')

    assert len(metrics) > 0
    assert 'final' in suggestions

    clusters = handler.get_clusters(linkage_matrix, suggestions['final'])

    visualizer = ClusterVisualizer()
    visualizer.create_visualizations(rmsd_matrix, clusters, linkage_matrix, '943', 'tests/test_data/output', metrics, suggestions['final'])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Test RMSD clustering took {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    pytest.main()
