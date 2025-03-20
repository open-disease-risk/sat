"""Task heads for survival analysis

This package contains the various task heads used for survival analysis, multi-task learning,
and event prediction tasks.
"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

# Base classes and utilities
from .output import SAOutput, TaskOutput

# Configuration classes
from .config import (
    BaseConfig,
    SurvivalConfig,
    EventClassificationTaskConfig,
    EventDurationTaskConfig,
    MTLConfig,
    SatBertConfig,
)

# Task heads
from .survival import SurvivalTaskHead
from .classification import EventClassificationTaskHead
from .regression import EventDurationTaskHead
from .mtl import MTLForSurvival

__all__ = [
    # Output classes
    "SAOutput",
    "TaskOutput",
    # Configuration classes
    "BaseConfig",
    "TokenEmbedding",
    "SentenceEmbedding",
    "SurvivalConfig",
    "EventClassificationTaskConfig",
    "EventDurationTaskConfig",
    "MTLConfig",
    "SatBertConfig",
    # Task heads
    "SurvivalTaskHead",
    "EventClassificationTaskHead",
    "EventDurationTaskHead",
    "MTLForSurvival",
]
