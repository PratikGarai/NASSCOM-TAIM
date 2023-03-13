from pathlib import Path
from setuptools import find_namespace_packages, setup


BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as f:
    required_installs = [ln.strip() for ln in f.readlines()]


setup(
    name="aqiai",
    version="1.0.0",
    author= ['aneesh-aparajit', 'PratikGarai', 'ARTHURFLECK1828', 'neerajjr11'],
    install_requires=[required_installs],
    packages=find_namespace_packages(),
    description="A module to predict AQI for Telangana",
    keywords="aqi nasscom-taim telangana ai ml dl climax",
    url=" https://github.com/PratikGarai/NASSCOM-TAIM"
)