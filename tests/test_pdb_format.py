#!/usr/bin/env python3
# tests/test_pdb_format.py

"""
Test script to validate PDB file format compliance.
Checks each model entry for proper formatting of all required records.
"""

import os
import re
from pathlib import Path
import unittest
from typing import List, Dict, Set, DefaultDict, Tuple
from collections import defaultdict


class TopologyParser:
    """Parser for GROMACS topology files."""

    @staticmethod
    def parse_bonds(topology_file: Path) -> Set[Tuple[int, int]]:
        """
        Parse bonds from a GROMACS topology file.

        Args:
            topology_file: Path to the topology file

        Returns:
            Set of tuples containing atom pairs that are bonded
        """
        bonds = set()
        in_bonds_section = False

        with open(topology_file, "r") as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith(";"):
                    continue

                # Check for bonds section
                if line == "[ bonds ]":
                    in_bonds_section = True
                    continue
                elif line.startswith("[") and in_bonds_section:
                    break

                # Parse bond entries
                if in_bonds_section and not line.startswith(";"):
                    parts = line.split()
                    if len(parts) >= 2:  # We only need the first two columns
                        atom1, atom2 = int(parts[0]), int(parts[1])
                        # Store bonds in both directions for easier comparison
                        bonds.add((min(atom1, atom2), max(atom1, atom2)))

        return bonds


class PDBFormatValidator:
    """Validator for PDB file format compliance."""

    def __init__(self, pdb_file: str):
        """Initialize validator with PDB file path."""
        self.pdb_file = Path(pdb_file)
        self.current_model = 0
        self.errors: Dict[int, List[str]] = {}
        self.atom_counts: Dict[int, int] = {}  # Track atoms per model
        self.conect_counts: Dict[int, int] = {}  # Track CONECT records per model
        self.bonds_per_model: DefaultDict[int, Set[Tuple[int, int]]] = defaultdict(set)
        self.topology_bonds: Set[Tuple[int, int]] = set()

        # Try to find and parse topology file
        self.topology_file = self.pdb_file.parent / "topol.top"
        if self.topology_file.exists():
            self.topology_bonds = TopologyParser.parse_bonds(self.topology_file)
        else:
            print(f"Warning: No topology file found at {self.topology_file}")

    def validate_model_record(self, line: str) -> bool:
        """Validate MODEL record format."""
        pattern = r"^MODEL\s+\d+\s*$"
        if not re.match(pattern, line):
            self._add_error("Invalid MODEL record format")
            return False
        return True

    def validate_compnd_record(self, line: str) -> bool:
        """Validate COMPND record format."""
        pattern = r"^COMPND\s+\d+\s*$"
        if not re.match(pattern, line):
            self._add_error("Invalid COMPND record format")
            return False
        return True

    def validate_remark_record(self, line: str) -> bool:
        """Validate REMARK record format."""
        pattern = r"^REMARK Frame \d+\s*$"
        if not re.match(pattern, line):
            self._add_error("Invalid REMARK record format")
            return False
        return True

    def validate_seqres_record(self, line: str) -> bool:
        """Validate SEQRES record format."""
        pattern = r"^SEQRES\s+\d+\s+[A-Z]\s+\d+\s+([A-Z]{3}\s*)+$"
        if not re.match(pattern, line):
            self._add_error("Invalid SEQRES record format")
            return False
        return True

    def validate_atom_record(self, line: str) -> bool:
        """Validate ATOM record format."""
        # Format: ATOM  {id:5d} {name:<4s} {resname:3s} X{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00      {element:>1s}
        pattern = r"^ATOM\s+(\d{1,5})\s+[A-Z0-9]+\s+[A-Z]{3}\sX\s*\d{1,4}\s+[-]?\d+\.\d{3}\s+[-]?\d+\.\d{3}\s+[-]?\d+\.\d{3}\s+\d+\.\d{2}\s+\d+\.\d{2}\s+[A-Z]\s*$"
        match = re.match(pattern, line)
        if not match:
            self._add_error("Invalid ATOM record format")
            return False

        # Track atom count for current model
        if self.current_model not in self.atom_counts:
            self.atom_counts[self.current_model] = 0
        self.atom_counts[self.current_model] += 1

        return True

    def validate_conect_record(self, line: str) -> bool:
        """Validate CONECT record format and track bonds."""
        pattern = r"^CONECT(\s+\d{1,5}){2,5}\s*$"  # Allows 1-4 connected atoms
        if not re.match(pattern, line):
            self._add_error("Invalid CONECT record format")
            return False

        # Extract atom numbers from CONECT record
        atoms = [int(x) for x in line.split()[1:]]
        if len(atoms) < 2:
            self._add_error("CONECT record must specify at least two atoms")
            return False

        # First atom is the central atom, others are connected to it
        central_atom = atoms[0]
        for connected_atom in atoms[1:]:
            # Store bonds in a consistent order (smaller index first)
            self.bonds_per_model[self.current_model].add(
                (min(central_atom, connected_atom), max(central_atom, connected_atom))
            )

        # Track CONECT record count
        if self.current_model not in self.conect_counts:
            self.conect_counts[self.current_model] = 0
        self.conect_counts[self.current_model] += 1

        return True

    def _add_error(self, error: str) -> None:
        """Add error message for current model."""
        if self.current_model not in self.errors:
            self.errors[self.current_model] = []
        self.errors[self.current_model].append(error)

    def validate_file(self) -> Dict[int, List[str]]:
        """
        Validate entire PDB file.

        Returns:
            Dict[int, List[str]]: Dictionary of errors by model number
        """
        required_records = {"MODEL", "COMPND", "REMARK", "ATOM"}
        if self.topology_bonds:
            required_records.add("CONECT")  # Only require CONECT if we have topology
        required_records.update({"END", "ENDMDL"})

        current_records: Set[str] = set()

        with open(self.pdb_file, "r") as f:
            for line in f:
                line = line.rstrip()
                record_type = line[:6].strip()

                if record_type == "MODEL":
                    if current_records:
                        self._check_required_records(current_records, required_records)
                        self._check_connectivity()
                    current_records = {record_type}
                    self.current_model += 1
                    self.validate_model_record(line)

                elif record_type == "COMPND":
                    current_records.add(record_type)
                    self.validate_compnd_record(line)

                elif record_type == "REMARK":
                    current_records.add(record_type)
                    self.validate_remark_record(line)

                elif record_type == "SEQRES":
                    current_records.add(record_type)
                    self.validate_seqres_record(line)

                elif record_type == "ATOM":
                    current_records.add(record_type)
                    self.validate_atom_record(line)

                elif record_type == "CONECT":
                    current_records.add(record_type)
                    self.validate_conect_record(line)

                elif record_type in ("END", "ENDMDL"):
                    current_records.add(record_type)

            # Check last model
            if current_records:
                self._check_required_records(current_records, required_records)
                self._check_connectivity()

        return self.errors

    def _check_required_records(self, current: Set[str], required: Set[str]) -> None:
        """Check if all required records are present in the model."""
        missing = required - current
        if missing:
            self._add_error(f"Missing required records: {', '.join(missing)}")

    def _check_connectivity(self) -> None:
        """Check bond connectivity for the current model."""
        if self.current_model not in self.atom_counts:
            return

        num_atoms = self.atom_counts[self.current_model]
        model_bonds = self.bonds_per_model[self.current_model]

        # If we have topology information, verify bonds match
        if self.topology_bonds:
            # Check for missing bonds (in topology but not in PDB)
            missing_bonds = self.topology_bonds - model_bonds
            if missing_bonds:
                self._add_error(
                    f"Missing {len(missing_bonds)} bonds that are present in topology file"
                )

            # Check for extra bonds (in PDB but not in topology)
            extra_bonds = model_bonds - self.topology_bonds
            if extra_bonds:
                self._add_error(
                    f"Found {len(extra_bonds)} bonds that are not in topology file"
                )

        # Check for atoms without any bonds
        atoms_with_bonds = {atom for bond in model_bonds for atom in bond}
        unbonded_atoms = set(range(1, num_atoms + 1)) - atoms_with_bonds
        if unbonded_atoms and self.topology_bonds:  # Only report if we have topology
            self._add_error(f"Found {len(unbonded_atoms)} atoms without any bonds")

    def get_statistics(self) -> Dict:
        """Get statistics about the PDB file."""
        stats = {
            "total_models": self.current_model,
            "atoms_per_model": self.atom_counts,
            "conect_records_per_model": self.conect_counts,
            "bonds_from_topology": (
                len(self.topology_bonds) if self.topology_bonds else 0
            ),
        }

        # Add per-model bond statistics
        if self.topology_bonds:
            stats.update(
                {
                    "matching_bonds_per_model": {
                        model: len(bonds & self.topology_bonds)
                        for model, bonds in self.bonds_per_model.items()
                    },
                    "missing_bonds_per_model": {
                        model: len(self.topology_bonds - bonds)
                        for model, bonds in self.bonds_per_model.items()
                    },
                    "extra_bonds_per_model": {
                        model: len(bonds - self.topology_bonds)
                        for model, bonds in self.bonds_per_model.items()
                    },
                }
            )

        return stats


class TestPDBFormat(unittest.TestCase):
    """Test case for PDB file format validation."""

    def setUp(self):
        """Set up test case."""
        self.test_file = Path("test_data/md_sample/md_Ref.pdb")

    def test_pdb_format(self):
        """Test PDB file format compliance."""
        # Check if file exists first
        if not self.test_file.exists():
            self.fail(f"Test file not found: {self.test_file}")

        validator = PDBFormatValidator(str(self.test_file))
        errors = validator.validate_file()

        # Get and print statistics
        stats = validator.get_statistics()
        print("\nPDB File Statistics:")
        print(f"Total models: {stats['total_models']}")
        print(f"Bonds from topology file: {stats['bonds_from_topology']}")

        print("\nAtoms per model:")
        for model, count in sorted(stats["atoms_per_model"].items()):
            print(f"  Model {model}: {count} atoms")

        if stats["bonds_from_topology"] > 0:
            print("\nBond statistics per model:")
            for model in sorted(stats["matching_bonds_per_model"].keys()):
                print(f"\nModel {model}:")
                print(f"  Matching bonds: {stats['matching_bonds_per_model'][model]}")
                print(f"  Missing bonds: {stats['missing_bonds_per_model'][model]}")
                print(f"  Extra bonds: {stats['extra_bonds_per_model'][model]}")

        # Print detailed error report
        if errors:
            error_report = ["\nPDB Format Validation Errors:"]
            for model_num, model_errors in sorted(errors.items()):
                error_report.append(f"\nModel {model_num}:")
                for error in model_errors:
                    error_report.append(f"  - {error}")

            self.fail("\n".join(error_report))
        else:
            print(f"\nSuccessfully validated PDB file: {self.test_file}")
            print("All models conform to the required format.")


if __name__ == "__main__":
    unittest.main()
