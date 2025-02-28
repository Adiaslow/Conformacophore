from rdkit import Chem
from rdkit.Chem import rdmolfiles
import os

def format_seqres(sequence, chain_id='A'):
    """Format a sequence string into PDB SEQRES format."""
    residues = sequence.split()
    return f"SEQRES   1 {chain_id}    {len(residues):2d}  " + " ".join(f"{res:3}" for res in residues)

def filter_pdb_block(pdb_block):
    """Remove COMPND lines from PDB block."""
    return "\n".join(line for line in pdb_block.split('\n')
                    if not line.startswith("COMPND"))

def write_pdb_model(f, mol, model_num, compound_id, frame, sequence):
    """Write a single MODEL block with the specified header format."""
    f.write(f"MODEL {model_num}\n")
    f.write(f"COMPND {compound_id}\n")
    f.write(f"REMARK Frame {frame}\n")
    f.write(f"{format_seqres(sequence)}\n")
    # Filter out COMPND lines from RDKit's output
    filtered_block = filter_pdb_block(rdmolfiles.MolToPDBBlock(mol))
    f.write(filtered_block)
    f.write("ENDMDL\n")

def sdf_to_pdb(input_sdf, output_dir):
    # Open the SDF file
    suppl = Chem.SDMolSupplier(input_sdf, removeHs=False, sanitize=False)
    if not suppl:
        print(f"Error reading SDF file: {input_sdf}")
        return

    compounds = {}
    for mol in suppl:
        if mol is None:
            continue

        try:
            Chem.SanitizeMol(mol)
        except:
            print("Sanitization failed for a molecule, skipping...")
            continue

        compound_id = mol.GetProp("Compound_ID") if mol.HasProp("Compound_ID") else f"compound_{mol.GetProp('_Name')}_{len(compounds)}"
        frame = mol.GetProp("Frame") if mol.HasProp("Frame") else str(len(compounds))
        sequence = mol.GetProp("ResidueSequence") if mol.HasProp("ResidueSequence") else ""

        if compound_id not in compounds:
            compounds[compound_id] = []
        compounds[compound_id].append((mol, frame, sequence))

    for compound_id, mol_data in compounds.items():
        output_pdb = os.path.join(output_dir, f"{compound_id}.pdb")
        with open(output_pdb, "w") as f:
            for i, (mol, frame, sequence) in enumerate(mol_data):
                chain_id = 'A'  # Use a single chain ID 'A' for all conformers

                # Set chain ID for all atoms
                for atom in mol.GetAtoms():
                    if atom.GetPDBResidueInfo() is None:
                        atom.SetMonomerInfo(Chem.AtomPDBResidueInfo())
                    atom.GetPDBResidueInfo().SetChainId(chain_id)

                write_pdb_model(f, mol, i+1, compound_id, frame, sequence)

        print(f"Saved {compound_id} with {len(mol_data)} conformers to {output_pdb}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_sdf> <output_dir>")
        sys.exit(1)

    input_sdf = sys.argv[1]
    output_dir = sys.argv[2]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sdf_to_pdb(input_sdf, output_dir)
