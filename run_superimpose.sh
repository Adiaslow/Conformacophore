#!/bin/bash

# Create Python virtual environment if it doesnt exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install biopython numpy networkx tqdm
