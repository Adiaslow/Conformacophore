"""Implementation of structure alignment using RDKit's MCS algorithm."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
from typing import List, Tuple, Optional, Dict, Any
from ..interfaces.structure_superimposer import StructureSuperimposer
from ..models.molecular_graph import MolecularGraph
from ..models.alignment_result import AlignmentResult
import logging


class RDKitMCSSuperimposer(StructureSuperimposer):
    """Superimposer that uses RDKit's Maximum Common Substructure algorithm."""

    def __init__(self, timeout: float = 60.0, match_valences: bool = False):
        """Initialize superimposer.

        Args:
            timeout: Maximum time in seconds to spend finding MCS
            match_valences: Whether to require matching valences in MCS
        """
        self.timeout = timeout
        self.match_valences = match_valences
        self._reference_mol: Optional[Chem.Mol] = None
        self._reference_molecular_graph: Optional[MolecularGraph] = None
        self.logger = logging.getLogger(__name__)

    def _create_rdkit_mol(self, graph: MolecularGraph) -> Chem.Mol:
        """Convert MolecularGraph to RDKit Mol.

        Args:
            graph: Molecular graph to convert

        Returns:
            RDKit Mol object

        Raises:
            ValueError: If failed to create RDKit molecule
        """
        # Create empty editable mol
        mol = Chem.EditableMol(Chem.Mol())

        # Add atoms
        atom_map = {}  # Map from our atom IDs to RDKit atom indices
        for i, atom in enumerate(graph.atoms):
            if atom.element.upper() == "H":  # Skip hydrogens
                continue
            rdatom = Chem.Atom(atom.element)
            # Set atom properties
            rdatom.SetFormalCharge(0)  # Default to neutral charge
            rdatom.SetNoImplicit(True)  # Don't use implicit Hs
            rdatom.SetNumExplicitHs(0)  # We're not including hydrogens
            idx = mol.AddAtom(rdatom)
            atom_map[atom.atom_id] = idx

        # Track added bonds to avoid duplicates
        added_bonds = set()

        # Add bonds
        for bond in graph.bonds:
            # Skip bonds involving hydrogens
            if any(
                graph.atoms[atom_id].element.upper() == "H"
                for atom_id in [bond.atom1_id, bond.atom2_id]
            ):
                continue

            # Skip if either atom was skipped
            if bond.atom1_id not in atom_map or bond.atom2_id not in atom_map:
                continue

            # Create unique bond identifier (always put smaller index first)
            bond_id = tuple(sorted([atom_map[bond.atom1_id], atom_map[bond.atom2_id]]))

            # Skip if bond already added
            if bond_id in added_bonds:
                continue

            # Add bond
            mol.AddBond(
                atom_map[bond.atom1_id],
                atom_map[bond.atom2_id],
                Chem.BondType.SINGLE,
            )
            added_bonds.add(bond_id)

        # Convert to non-editable molecule
        mol = mol.GetMol()

        # Update properties and sanitize
        for atom in mol.GetAtoms():
            # Calculate explicit valence based on actual bonds
            explicit_valence = sum(1 for bond in atom.GetBonds())
            atom.SetNumExplicitHs(0)  # We're not including hydrogens
            atom.SetNoImplicit(True)  # Don't use implicit Hs

        try:
            # Sanitize the molecule but skip kekulization
            Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to sanitize molecule: {str(e)}")
            raise ValueError(f"Failed to create valid RDKit molecule: {str(e)}")

        return mol

    def set_reference(self, reference: MolecularGraph) -> None:
        """Set a reference structure that will be cached for future alignments."""
        self._reference_molecular_graph = reference
        self._reference_mol = self._create_rdkit_mol(reference)

    def _calculate_transformation(
        self, ref_coords: np.ndarray, target_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate optimal rotation and translation."""
        # Center coordinates
        ref_center = np.mean(ref_coords, axis=0)
        target_center = np.mean(target_coords, axis=0)

        ref_centered = ref_coords - ref_center
        target_centered = target_coords - target_center

        # Calculate correlation matrix
        correlation_matrix = np.dot(target_centered.T, ref_centered)

        # SVD
        U, _, Vt = np.linalg.svd(correlation_matrix)

        # Calculate rotation matrix
        rotation = np.dot(U, Vt)

        # Ensure right-handed coordinate system
        if np.linalg.det(rotation) < 0:
            Vt[-1] *= -1
            rotation = np.dot(U, Vt)

        # Calculate translation
        translation = ref_center - np.dot(target_center, rotation.T)

        return rotation, translation

    def align(self, mol1: MolecularGraph, mol2: MolecularGraph) -> AlignmentResult:
        """Align target structure to reference using RDKit's MCS algorithm."""
        # Use cached reference if it's the same object
        if mol1 is self._reference_molecular_graph and self._reference_mol is not None:
            ref_mol = self._reference_mol
        else:
            # If a new reference is provided, update the cache
            self.set_reference(mol1)
            ref_mol = self._reference_mol

        if ref_mol is None:
            self.logger.error("Failed to create reference RDKit molecule")
            return AlignmentResult(
                rmsd=float("inf"),
                matched_atoms=0,
                transformation_matrix=None,
                matched_pairs=[],
                isomorphic_match=False,
            )

        # Convert target to RDKit mol
        target_mol = self._create_rdkit_mol(mol2)

        # Log molecule sizes
        self.logger.info(
            f"Reference molecule: {ref_mol.GetNumAtoms()} atoms, {ref_mol.GetNumBonds()} bonds"
        )
        self.logger.info(
            f"Target molecule: {target_mol.GetNumAtoms()} atoms, {target_mol.GetNumBonds()} bonds"
        )

        # Find MCS with relaxed constraints
        self.logger.info("Finding Maximum Common Substructure")
        mcs = rdFMCS.FindMCS(
            [ref_mol, target_mol],
            timeout=int(self.timeout),
            matchValences=self.match_valences,
            ringMatchesRingOnly=False,  # Relaxed
            completeRingsOnly=False,  # Relaxed
            atomCompare=rdFMCS.AtomCompare.CompareElements,  # Only compare elements
            bondCompare=rdFMCS.BondCompare.CompareOrder,  # Only compare bond orders
        )

        self.logger.info(
            f"MCS Results - Number of atoms: {mcs.numAtoms}, Number of bonds: {mcs.numBonds}"
        )
        self.logger.info(f"MCS SMARTS: {mcs.smartsString}")

        if mcs.numAtoms == 0:
            self.logger.warning("No common substructure found")
            return AlignmentResult(
                rmsd=float("inf"),
                matched_atoms=0,
                transformation_matrix=None,
                matched_pairs=[],
                isomorphic_match=False,
            )

        # Get atom mappings from MCS
        pattern = Chem.MolFromSmarts(mcs.smartsString)
        ref_match = ref_mol.GetSubstructMatch(pattern)
        target_match = target_mol.GetSubstructMatch(pattern)

        if not ref_match or not target_match:
            self.logger.warning(
                f"Failed to map MCS to molecules. Ref match: {bool(ref_match)}, Target match: {bool(target_match)}"
            )
            return AlignmentResult(
                rmsd=float("inf"),
                matched_atoms=0,
                transformation_matrix=None,
                matched_pairs=[],
                isomorphic_match=False,
            )

        # Create matched pairs from MCS mapping
        matched_pairs = list(zip(ref_match, target_match))
        self.logger.info(f"Found {len(matched_pairs)} matched atom pairs")

        # Get coordinates for matched atoms
        ref_coords = np.array([mol1.atoms[i].coordinates for i in ref_match])
        target_coords = np.array([mol2.atoms[j].coordinates for j in target_match])

        # Calculate optimal transformation
        rotation, translation = self._calculate_transformation(
            ref_coords, target_coords
        )

        # Apply transformation and calculate RMSD
        aligned_coords = np.dot(target_coords, rotation.T) + translation
        rmsd = np.sqrt(np.mean(np.sum((ref_coords - aligned_coords) ** 2, axis=1)))

        self.logger.info(f"Found MCS with {len(matched_pairs)} atoms, RMSD: {rmsd:.3f}")

        return AlignmentResult(
            rmsd=rmsd,
            matched_atoms=len(matched_pairs),
            transformation_matrix=(rotation, translation),
            matched_pairs=matched_pairs,
            isomorphic_match=True,
        )
