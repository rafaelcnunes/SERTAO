from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"

long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="SERTAO",
    version="0.1.0",
    author="Rafael Nunes",
    author_email="rafadcnunes@gmail.com",
    description="SERTAO: System for Estimation of cosmological paRameTers and Analysis of Observables",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "getdist",
        "dynesty",
        "mpi4py",
        "cython",
        "numba",
        "corner",
        "h5py",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    include_package_data=True,
    zip_safe=False,
)


