import pytest
import os
import shutil
import pandas as pd
from src.conformacophore.pipeline import Pipeline

# Test data paths
TEST_DATA_DIR = os.path.join('tests', 'test_data')
INPUT_DIR = os.path.join(TEST_DATA_DIR, 'input')
OUTPUT_DIR = os.path.join(TEST_DATA_DIR, 'output')

@pytest.fixture
def test_directories(tmp_path):
    """Create the expected directory structure and files for testing."""
    # Create base directories
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    compound_dir = input_dir / "943"
    compound_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Copy test files
    source_pdb = os.path.join(INPUT_DIR, '943.pdb')
    target_pdb = os.path.join(INPUT_DIR, 'vhl1.pdb')
    shutil.copy2(source_pdb, str(compound_dir / '943.pdb'))
    shutil.copy2(target_pdb, str(input_dir / 'vhl1.pdb'))

    # Create summary statistics file with explicit dtypes
    summary_df = pd.DataFrame({
        'Compound': ['943'],
        'Frame': [1],
        'NumStructures': [1],
        'MinRMSD': [0.0],
        'MaxRMSD': [1.0],
        'MeanRMSD': [0.5]
    })
    # Ensure Compound is treated as string
    summary_df['Compound'] = summary_df['Compound'].astype(str)

    # Write to CSV with specific dtype for Compound column
    summary_df.to_csv(input_dir / 'summary_statistics.csv', index=False)

    return {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'reference_pdb': str(input_dir / 'vhl1.pdb')
    }

def test_pipeline_initialization(test_directories):
    """Test pipeline initialization with all components."""
    pipeline = Pipeline(
        input_dir=test_directories['input_dir'],
        output_dir=test_directories['output_dir'],
        reference_pdb=test_directories['reference_pdb'],
        target_chains=['A', 'B', 'C'],
        ligand_chain='D',
        molecule_chain='A'
    )

    assert pipeline.pdb_handler is not None
    assert pipeline.clustering_handler is not None
    assert pipeline.visualization_handler is not None
    assert pipeline.summary_handler is not None

def test_pipeline_execution(test_directories):
    """Test complete pipeline execution with actual test data."""
    pipeline = Pipeline(
        input_dir=test_directories['input_dir'],
        output_dir=test_directories['output_dir'],
        reference_pdb=test_directories['reference_pdb'],
        target_chains=['A', 'B', 'C'],
        ligand_chain='D',
        molecule_chain='A'
    )

    # Run pipeline
    pipeline.run()

    # Verify output files
    expected_files = [
        'compound_943_2d_projection.png',
        'compound_943_cluster_sizes.png',
        'compound_943_dendrogram.png',
        'compound_943_distance_matrix.png',
        'compound_943_linear_projection.png',
        'compound_943_complex.pdb',
        'analysis_summary.csv'
    ]

    output_files = os.listdir(test_directories['output_dir'])
    for expected_file in expected_files:
        assert any(f.endswith(expected_file) for f in output_files), \
            f"Missing expected output file: {expected_file}"

def test_pipeline_structure_validation(test_directories):
    """Test validation of input structures."""
    pipeline = Pipeline(
        input_dir=test_directories['input_dir'],
        output_dir=test_directories['output_dir'],
        reference_pdb=test_directories['reference_pdb'],
        target_chains=['A', 'B', 'C'],
        ligand_chain='D',
        molecule_chain='A'
    )

    pipeline.run()

    # Verify the representative structure was generated
    output_path = os.path.join(
        test_directories['output_dir'],
        'compound_943_complex.pdb'
    )
    assert os.path.exists(output_path)

    # Verify analysis summary was created
    summary_path = os.path.join(
        test_directories['output_dir'],
        'analysis_summary.csv'
    )
    assert os.path.exists(summary_path)

    # Check summary content
    summary_df = pd.read_csv(summary_path)
    assert 'Compound' in summary_df.columns
    assert len(summary_df) > 0

def test_pipeline_error_handling(test_directories):
    """Test pipeline error handling."""
    # Test with invalid chain IDs
    pipeline = Pipeline(
        input_dir=test_directories['input_dir'],
        output_dir=test_directories['output_dir'],
        reference_pdb=test_directories['reference_pdb'],
        target_chains=['X', 'Y', 'Z'],  # Invalid chains
        ligand_chain='D',
        molecule_chain='A'
    )

    # Should handle invalid chains gracefully
    pipeline.run()

    # Test with missing summary file
    os.remove(os.path.join(test_directories['input_dir'], 'summary_statistics.csv'))
    with pytest.raises(FileNotFoundError):
        pipeline.run()
