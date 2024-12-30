import argparse
from pipeline import Pipeline
from utils.helpers import validate_chain_input

def main():
    parser = argparse.ArgumentParser(description='Find representative structures from filtered superimposition results')
    parser.add_argument('input_dir', help='Directory containing filtered superimposition results')
    parser.add_argument('output_dir', help='Directory to store representative structures')
    parser.add_argument('--target-chains', type=validate_chain_input, nargs='+', required=True,
                      help='Chain letters for target protein (e.g., A B for chains A and B)')
    parser.add_argument('--ligand-chain', type=validate_chain_input, required=True,
                      help='Chain letter for ligand (e.g., L)')
    parser.add_argument('--molecule-chain', type=validate_chain_input, required=True,
                      help='Chain letter for molecules to cluster (e.g., X)')

    args = parser.parse_args()

    pipeline = Pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_chains=args.target_chains,
        ligand_chain=args.ligand_chain,
        molecule_chain=args.molecule_chain
    )
    pipeline.run()

if __name__ == '__main__':
    main()
