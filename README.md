# Conpharmacophore

A python-based tool for superimposing MD trajectories with protein-bound ligand models to find potential scaffolds, then optimizing scaffold binding affinity using a genetic algorithm.

## Environment Setup

To use the superimpose_trajectories.py script, you need to set up a Python environment with the required packages:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required packages
pip install biopython numpy networkx tqdm

# Optional: Install RDKit (required for some advanced features)
# Install via conda:
# conda install -c conda-forge rdkit
# Or via pip:
# pip install rdkit
```

## Running the Script

Once your environment is set up, you can run the script with:

```bash
# Basic usage
python src/scripts/superimpose_trajectories.py /path/to/input_dir /path/to/reference.pdb

# To save the first 5 superimposed structures
python src/scripts/superimpose_trajectories.py /path/to/input_dir /path/to/reference.pdb --save-structures

# To force reprocessing of already processed directories
python src/scripts/superimpose_trajectories.py /path/to/input_dir /path/to/reference.pdb --force

# To use multiple CPU cores
python src/scripts/superimpose_trajectories.py /path/to/input_dir /path/to/reference.pdb --num-processes 8

# Example:
python src/scripts/superimpose_trajectories.py /Volumes/LokeyLabShared/Adam/chads_library_conf/Hex/CHCl3/x_177 /Volumes/LokeyLabShared/Adam/chads_library_conf/ref/vhl1.pdb --num-processes 8 --force --save-structures
```

## Troubleshooting

If you encounter import errors, make sure you have activated the virtual environment:

```bash
source venv/bin/activate
```

## Viewing the Results

The script generates:

1. A CSV file in the parent directory with superimposition metrics
2. If requested with `--save-structures`, up to 5 PDB files in the output directory
   - These contain the protein target (chains A,B,C), reference ligand (chain D), and superimposed test model (chain E)
   - You can view these with PyMOL or similar molecular visualization software
