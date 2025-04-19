"""FEMR extensions for MEDS data processing in SAT.

This package contains custom extensions of FEMR labelers and featurizers
for use in the SAT package for survival analysis.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"


from sat.data.dataset.femr_extensions.labelers import (
    CompetingRiskLabeler,
    SurvivalLabeler,
    CustomEventLabeler,
)
from sat.data.dataset.femr_extensions.adapter import FEMRAdapter
from sat.data.dataset.femr_extensions.registry import (
    LabelerRegistry,
    register_labeler,
)

__all__ = [
    # Labelers
    "CompetingRiskLabeler",
    "SurvivalLabeler",
    "CustomEventLabeler",
    # Adapter
    "FEMRAdapter",
    # Registry
    "LabelerRegistry",
    "register_labeler",
]
