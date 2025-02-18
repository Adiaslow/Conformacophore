from typing import Dict, List

class PDBHeaderHandler:
    """Handles reading and writing of PDB header and metadata information."""

    def __init__(self):
        self.global_headers: List[str] = []
        self.model_headers: Dict[int, List[str]] = {}
        self.model_metadata: Dict[int, Dict[str, List[str]]] = {}
        self.conect_records: List[str] = []

    def read_headers(self, pdb_path: str):
        """Read all header and metadata information from PDB file."""
        self.global_headers = []
        self.model_headers = {}
        self.model_metadata = {}
        self.conect_records = []

        current_model = None
        in_model = False

        with open(pdb_path, 'r') as f:
            for line in f:
                record_type = line[:6].strip()

                if record_type == 'MODEL':
                    # Extract model number, handling both "MODEL 1" and "MODEL    1" formats
                    model_str = line[6:].strip()
                    try:
                        current_model = int(model_str) - 1  # Convert to 0-based indexing
                    except ValueError:
                        current_model = 0
                    in_model = True
                    if current_model not in self.model_metadata:
                        self.model_metadata[current_model] = {}
                    continue

                if record_type == 'ENDMDL':
                    in_model = False
                    continue

                if record_type == 'END':
                    break

                if record_type == 'CONECT':
                    self.conect_records.append(line)
                    continue

                # Handle headers and metadata
                if record_type in ['HEADER', 'TITLE']:
                    self.global_headers.append(line)
                elif record_type in ['COMPND', 'REMARK', 'SEQRES']:
                    if in_model and current_model is not None:
                        if record_type not in self.model_metadata[current_model]:
                            self.model_metadata[current_model][record_type] = []
                        self.model_metadata[current_model][record_type].append(line)
                    else:
                        self.global_headers.append(line)

    def write_model_information(self, file_handle, model_num: int):
        """Write all information relevant to a specific model."""
        # Write global headers
        for header in self.global_headers:
            file_handle.write(header)

        # Write model-specific metadata
        if model_num in self.model_metadata:
            for record_type, lines in self.model_metadata[model_num].items():
                for line in lines:
                    file_handle.write(line)
