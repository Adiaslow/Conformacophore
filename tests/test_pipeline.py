import os
import pytest
from src.conformacophore.pipeline import Pipeline

@pytest.fixture
def setup_directories():
    input_dir = 'tests/test_data/input'
    output_dir = 'tests/test_data/output'

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
    target_chains = ['A', 'B', 'C']
    ligand_chain = 'D'
    molecule_chain = 'A'

    pipeline = Pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        target_chains=target_chains,
        ligand_chain=ligand_chain,
        molecule_chain=molecule_chain
    )
    pipeline.run()

    assert os.path.exists(os.path.join(output_dir, "analysis_summary.csv"))
