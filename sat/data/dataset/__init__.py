"""Dataset module."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from sat.data.dataset.parse_hsa_synthetic import hsa as hsa_synthetic
from sat.data.dataset.parse_metabric import metabric
from sat.data.dataset.parse_metabric_numerics import metabric as metabric_numeric
from sat.data.dataset.parse_omop import omop
from sat.data.dataset.parse_seer import seer
from sat.data.dataset.parse_synthetic import synthetic
from sat.data.dataset.parse_synthetic_numerics import synthetic as synthetic_numerics

__all__ = [
    "metabric",
    "metabric_numeric",
    "synthetic",
    "synthetic_numerics",
    "seer",
    "hsa_synthetic",
    "omop",
]
