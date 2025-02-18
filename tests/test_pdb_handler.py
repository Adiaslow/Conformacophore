import pytest
import os
import tempfile
from pathlib import Path
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
import mdtraj as md

from src.conformacophore.handlers.pdb_handler import PDBHandler, EnhancedPDBIO
from src.conformacophore.handlers.pdb_header_handler import PDBHeaderHandler

MD_PDB = 'tests/test_data/input/943.pdb'

@pytest.fixture
def output_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

class TestMDTrajectory:
    """Tests for MD trajectory PDB file."""

    def test_read_md_structure(self):
        """Test reading MD structure."""
        header_handler = PDBHeaderHandler()
        header_handler.read_headers(MD_PDB)

        # Verify we can parse the file
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('md', MD_PDB)
        assert isinstance(structure, Structure)

        # Verify basic content
        model = structure[0]
        atoms = list(model.get_atoms())
        assert len(atoms) > 0, "Structure should contain atoms"

        # Check metadata
        assert any('Frame 0' in line for line in header_handler.model_metadata.get(0, {}).get('REMARK', [])), \
            "Should find Frame 0 REMARK"

    def test_extract_first_model(self):
        """Test extracting the first model."""
        handler = PDBHandler()
        structure = handler.get_structure_from_model(MD_PDB, model_num=0)

        assert isinstance(structure, Structure)
        assert len(structure) == 1

        # Check atom content
        model = structure[0]
        atoms = list(model.get_atoms())
        assert len(atoms) > 0

        # Verify first atom details
        first_atom = atoms[0]
        assert first_atom.name == "N1"

    def test_preserve_metadata(self, output_dir):
        """Test preservation of metadata during operations."""
        handler = PDBHandler()
        header_handler = PDBHeaderHandler()

        # Extract and save first model
        structure = handler.get_structure_from_model(MD_PDB, 0)
        output_path = os.path.join(output_dir, 'md_output.pdb')

        # Save structure
        writer = EnhancedPDBIO()
        writer.save(structure, output_path, model_num=0)

        # Verify output exists and can be read
        assert os.path.exists(output_path)

        # Read output file content
        with open(output_path) as f:
            content = f.read()
            # Check for essential elements
            assert 'MODEL' in content
            assert 'ATOM' in content
            assert 'ENDMDL' in content
            assert 'END' in content

    def test_conect_records(self, output_dir):
        """Test preservation of CONECT records."""
        header_handler = PDBHeaderHandler()
        header_handler.read_headers(MD_PDB)

        # Verify CONECT records were read
        assert len(header_handler.conect_records) > 0, "Should have CONECT records"

        # Verify format of CONECT records
        for record in header_handler.conect_records:
            assert record.startswith('CONECT'), "CONECT records should start with CONECT"

    def test_residue_handling(self):
        """Test proper handling of residue IDs."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('md', MD_PDB)

        # Get first residue
        model = structure[0]
        chain = next(model.get_chains())
        residue = next(chain.get_residues())

        # Verify residue information is preserved
        atoms = list(residue.get_atoms())
        assert len(atoms) > 0, "Residue should have atoms"

if __name__ == "__main__":
    pytest.main([__file__])
