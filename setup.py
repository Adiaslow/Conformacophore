from setuptools import setup, find_packages

setup(
    name="conformacophore",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "MDAnalysis>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=2.0.0",
        "networkx>=3.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.0.0",
        "mdtraj>=1.9.0",
        "tqdm>=4.65.0",
        "biopython>=1.81",
    ],
    entry_points={
        "console_scripts": [
            "align-compound=src.scripts.align_compound:main",
        ],
    },
)
