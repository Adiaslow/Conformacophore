#!/usr/bin/env python3

"""Setup script for molecular structure superimposition package."""

from setuptools import setup, find_packages

setup(
    name="conformacophore",
    version="0.1.0",
    description="Molecular structure superimposition and analysis",
    author="Adam",
    author_email="adam@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "networkx>=2.6.0",
        "biopython>=1.79",
        "rdkit>=2022.3.1",
        "tqdm>=4.62.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "superimpose-trajectories=scripts.superimpose_trajectories:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
