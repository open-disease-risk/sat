"""Task heads for survival analysis

This package contains the various task heads used for survival analysis, multi-task learning,
and event prediction tasks.
"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

# Base classes and utilities
from .base import SurvivalPreTrainedModel, SurvivalTask
from .output import SAOutput, TaskOutput

# Configuration classes
from sat.models.tasks.config import (
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
    # Base classes
    "SurvivalPreTrainedModel",
    "SurvivalTask",
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
