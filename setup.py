from setuptools import setup, find_packages

setup(
    name="conformacophore",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "MDAnalysis>=2.8.0",
        "tqdm>=4.43.0",
    ],
)
