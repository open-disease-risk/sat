"""SAT setup script"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from read_version import read_version
from setuptools import find_namespace_packages, setup

with open("README.md", "r") as fh:
    LONG_DESC = fh.read()
    setup(
        name="sat",
        version=read_version("sat", "__init__.py"),
        author="Dominik Dahlem",
        author_email="mail@dominik-dahlem.com",
        description="Survival Analysis with Transformers",
        long_description=LONG_DESC,
        long_description_content_type="text/markdown",
        url="",
        # packages=find_packages(),
        zip_safe=False,
        packages=find_namespace_packages(include=["sat.*"]),
        classifiers=[
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
            # In which environments to test this plugin
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Operating System :: OS Independent",
        ],
        install_requires=[
            "hydra-core",
            "read_version",
            "pytorch",
            "pytorch-cuda==11.8",
            "cudatoolkit==11.8",
            "nvidia-ml-py3",
            "transformers[torch]",
            "tokenizers",
            "datasets",
            "evaluate",
            "h5py",
            "pandas",
            "pytables",
            "scikit-learn",
            "hydra-core",
            "git",
            "numba",
            "numpy",
            "nvidia-cuda-nvrtc-cu11==11.8.89",
            "nvidia-cuda-runtime-cu11==11.8.89",
            "nvidia-cudnn-cu11==8.9.0.131",
            "logdecorator",
            "hydra-colorlog",
        ],
        # If this plugin is providing configuration files, be sure to include them in the package.
        # See MANIFEST.in.
        # For configurations to be discoverable at runtime, they should also be added to the search path.
        include_package_data=True,
    )
