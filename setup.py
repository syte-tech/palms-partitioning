import sys
import os
import subprocess
from setuptools import (
    setup, 
    Extension, 
    find_packages
)
import pybind11


sources = [
    "src/cpp/bindings.cpp",
    "src/cpp/ArmadilloConverter.cpp",
    "src/cpp/Compute1rErrors.cpp",
    "src/cpp/Extract1Dstripes.cpp",
    "src/cpp/FindBest1DPartition.cpp",
    "src/cpp/GenerateSystemMatrices.cpp",
    "src/cpp/GetIndexes.cpp",
    "src/cpp/Interval.cpp",
    "src/cpp/LinewisePartitioning.cpp",
    "src/cpp/ReconstructionFromPartition.cpp",
    "src/cpp/Stripe.cpp"
]

extra_compile_args = ['-std=c++14', '-fopenmp', '-O3']
extra_link_args = ['-fopenmp', '-larmadillo']
include_dirs = [pybind11.get_include(), "src/cpp"]
library_dirs = []

if sys.platform == 'darwin':
    extra_compile_args = ['-std=c++14', '-Xpreprocessor', '-fopenmp', '-O3']
    extra_link_args = ['-lomp', '-larmadillo']
    
    def get_brew_path(package):
        try:
            return subprocess.check_output(['brew', '--prefix', package]).decode().strip()
        except Exception:
            return f'/opt/homebrew/opt/{package}'

    omp_prefix = get_brew_path('libomp')
    include_dirs.append(os.path.join(omp_prefix, 'include'))
    library_dirs.append(os.path.join(omp_prefix, 'lib'))

    arma_prefix = get_brew_path('armadillo')
    include_dirs.append(os.path.join(arma_prefix, 'include'))
    library_dirs.append(os.path.join(arma_prefix, 'lib'))

ext_modules = [
    Extension(
        "palms_cpp",
        sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    ),
]

setup(
    name="palms",
    version="1.0",
    author="Lukas Kiefer (Wrapper by Julian Rasch)",
    description="PALMS Image and Point Cloud Partitioning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    
    # Dependencies
    install_requires=[
        "numpy",
        "scipy",
    ],
    
    zip_safe=False,
)