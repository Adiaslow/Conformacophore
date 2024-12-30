import os
import pytest
from main import process_filtered_results

@pytest.fixture
def setup_directories():
    input_dir = 'path/to/input'
    output_dir = 'path/to/output'

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    yield input_dir, output_dir

    # Cleanup after test
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
        os.rmdir(output_dir)

def test_pipeline(setup_directories):
    input_dir, output_dir = setup_directories
    target_chains = ['A', 'B']
    ligand_chain = 'L'
    molecule_chain = 'X'

    process_filtered_results(input_dir, output_dir, target_chains, ligand_chain, molecule_chain)

    assert os.path.exists(os.path.join(output_dir, "analysis_summary.csv"))
