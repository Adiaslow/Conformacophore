import json
from pathlib import Path

# Load clustering results
results_path = Path("test_output/943/clustering_results.json")
with open(results_path, "r") as f:
    clustering_data = json.load(f)

# Print cluster sizes
print(f"Number of clusters: {clustering_data['num_clusters']}")
print(f"Cluster sizes: {clustering_data['cluster_sizes']}")

# Print representatives
representatives = clustering_data["cluster_indices"]
for i, cluster in enumerate(representatives):
    print(f"Cluster {i+1} representative: Frame {cluster[0]}")
