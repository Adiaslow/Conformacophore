class PDBHeaderHandler:
    """Handles reading and writing of PDB header information."""

    def __init__(self):
        self.headers = []
        self.model_headers = {}  # Headers specific to each model

    def read_headers(self, pdb_path: str):
        """Read header information from PDB file."""
        self.headers = []
        self.model_headers = {}

        # Header keywords to capture
        header_keywords = [
            'HEADER', 'TITLE', 'COMPND', 'SOURCE', 'KEYWDS',
            'EXPDTA', 'AUTHOR', 'REVDAT', 'REMARK', 'SEQRES'
        ]

        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        # First pass: capture global headers before any MODEL
        for line in lines:
            if any(line.startswith(keyword) for keyword in header_keywords):
                self.headers.append(line)

            if line.startswith('MODEL'):
                break

        # Second pass: capture model-specific headers
        current_model = -1
        reading_model = False

        for line in lines:
            if line.startswith('MODEL'):
                current_model += 1
                reading_model = True
                self.model_headers[current_model] = []

            elif line.startswith('ENDMDL'):
                reading_model = False

            elif reading_model:
                # Capture model-specific headers
                if any(line.startswith(keyword) for keyword in
                       ['COMPND', 'REMARK', 'SEQRES']):
                    self.model_headers[current_model].append(line)

        # If no models found, treat all headers as global
        if not self.model_headers:
            self.model_headers[0] = []

    def write_headers(self, file_handle):
        """Write header information to file."""
        # Write global headers
        for header in self.headers:
            file_handle.write(header)

    def write_model_headers(self, file_handle, model_num: int):
        """Write model-specific headers to file."""
        if model_num in self.model_headers:
            for header in self.model_headers[model_num]:
                file_handle.write(header)
