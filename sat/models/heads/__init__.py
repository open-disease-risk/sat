"""Task heads for survival analysis

This package contains the various task heads used for survival analysis, multi-task learning,
and event prediction tasks.
"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

# Base classes and utilities
from .output import SAOutput, TaskOutput

# Base and config classes
from .base import (
    BaseConfig,
    SentenceEmbedding,
    TokenEmbedding,
    SatBertConfig,
)
from .survival import SurvivalConfig
from .dsm import DSMConfig
from .classification import EventClassificationTaskConfig
from .regression import EventDurationTaskConfig
from .mtl import MTLConfig

# Task heads
from .survival import SurvivalTaskHead
from .dsm import DSMTaskHead
from .classification import EventClassificationTaskHead
from .regression import EventDurationTaskHead
from .mtl import MTLForSurvival

__all__ = [
    # Output classes
    "SAOutput",
    "TaskOutput",
    # Base and Configuration classes
    "BaseConfig",
    "TokenEmbedding",
    "SentenceEmbedding",
    "SurvivalConfig",
    "DSMConfig",
    "EventClassificationTaskConfig",
    "EventDurationTaskConfig",
    "MTLConfig",
    "SatBertConfig",
    # Task heads
    "SurvivalTaskHead",
    "DSMTaskHead",
    "EventClassificationTaskHead",
    "EventDurationTaskHead",
    "MTLForSurvival",
]
