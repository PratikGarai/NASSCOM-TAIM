# setup.py
from pathlib import Path
from setuptools import find_namespace_packages, setuptools

BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as f:
    required_installs = [ln.strip() for ln in f.readlines()]


setup(
    name="weatherai",
    version="0.0.3",
    author=[
        "Aneesh Aparajit G", 
        "Pratik Garai", 
        "Neeraj J Manurkar",
        "Aravind P"
    ], 
    install_requires=[required_installs],
    packages=find_namespace_packages(),
    description="""This is a package which has Pix2Pix models built on PyTorch for predicting the weather of states based on visual interpretation of the tabular data.""",
    keywords=["weather", "ai", "pytorch", "modelling", "hackathon", "nasscom"]
)
