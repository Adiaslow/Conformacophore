import argparse
import os
import mdtraj as md
from typing import List

def validate_chain_input(chain: str) -> str:
    """Validate that chain input is a valid chain letter."""
    if not isinstance(chain, str) or len(chain) != 1:
        raise argparse.ArgumentTypeError("Chain must be a single letter (A-Z)")
    if not chain.isalpha():
        raise argparse.ArgumentTypeError("Chain must be a letter (A-Z)")
    return chain.upper()

def get_chain_atoms(traj: md.Trajectory, chain_letters: List[str]) -> List[int]:
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

    for chain in traj.topology.chains:
        if chain.chain_id.upper() in chain_letters:
            atoms.extend([atom.index for atom in chain.atoms])

    if not atoms:
        raise ValueError(f"No atoms found for specified chains {chain_letters}")

    return atoms

def extract_chains(pdb_file: str, chain_letters: List[str], topology: Optional[str] = None) -> md.Trajectory:
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
        print(f"\nAvailable chains in {os.path.basename(pdb_file)}:")
        for chain in traj.topology.chains:
            print(f"Chain ID: '{chain.chain_id}' with {len([atom for atom in chain.atoms])} atoms")
        raise e
