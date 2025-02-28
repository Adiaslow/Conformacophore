"""Model for tracking compound metadata throughout processing pipeline."""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class CompoundMetadata:
    """Metadata for a chemical compound and its processing history."""

    compound_id: str
    sequence: str
    num_frames: int
    source_file: str
    creation_date: datetime = datetime.now()
    processing_history: List[Dict] = None
    properties: Dict = None

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.processing_history is None:
            self.processing_history = []
        if self.properties is None:
            self.properties = {}

    def add_processing_step(
        self, step_name: str, parameters: Dict, results: Optional[Dict] = None
    ) -> None:
        """
        Add a processing step to the history.

        Args:
            step_name: Name of the processing step
            parameters: Parameters used in the step
            results: Optional results or metrics from the step
        """
        step = {
            "step": step_name,
            "timestamp": datetime.now(),
            "parameters": parameters,
            "results": results or {},
        }
        self.processing_history.append(step)

    def update_properties(self, properties: Dict) -> None:
        """
        Update compound properties.

        Args:
            properties: Dictionary of properties to update/add
        """
        self.properties.update(properties)


class CompoundRegistry:
    """Registry for tracking all compounds in the pipeline."""

    def __init__(self):
        """Initialize empty registry."""
        self._compounds: Dict[str, CompoundMetadata] = {}

    def register_compound(
        self, compound_id: str, sequence: str, source_file: str, num_frames: int
    ) -> CompoundMetadata:
        """
        Register a new compound or update existing one.

        Args:
            compound_id: Unique identifier for compound
            sequence: Peptide sequence
            source_file: Source file path
            num_frames: Number of frames/conformers

        Returns:
            CompoundMetadata instance
        """
        metadata = CompoundMetadata(
            compound_id=compound_id,
            sequence=sequence,
            source_file=source_file,
            num_frames=num_frames,
        )
        self._compounds[compound_id] = metadata
        return metadata

    def get_compound(self, compound_id: str) -> Optional[CompoundMetadata]:
        """Get compound metadata by ID."""
        return self._compounds.get(compound_id)

    def list_compounds(self) -> List[CompoundMetadata]:
        """Get list of all registered compounds."""
        return list(self._compounds.values())

    def update_processing(
        self,
        compound_id: str,
        step_name: str,
        parameters: Dict,
        results: Optional[Dict] = None,
    ) -> None:
        """
        Update processing history for a compound.

        Args:
            compound_id: Compound identifier
            step_name: Name of processing step
            parameters: Processing parameters
            results: Optional processing results
        """
        if compound := self.get_compound(compound_id):
            compound.add_processing_step(step_name, parameters, results)
