"""Task heads for survival analysis

This package contains the various task heads used for survival analysis, multi-task learning,
and event prediction tasks.
"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

# Import SatBertConfig from transformers
from transformers.models.bert.configuration_bert import BertConfig

# Task heads
from .base import BaseConfig
from .classification import EventClassificationTaskConfig, EventClassificationTaskHead
from .embeddings import SentenceEmbedding, TokenEmbedding
from .mtl import MTLConfig, MTLForSurvival
from .mensa import MENSAConfig, MENSATaskHead
from .dsm import DSMConfig, DSMTaskHead

# Base classes and utilities
from .output import SAOutput, TaskOutput
from .regression import EventDurationTaskConfig, EventDurationTaskHead
from .survival import SurvivalConfig, SurvivalTaskHead


# Create SatBertConfig class
class SatBertConfig(BertConfig):
    model_type = "sat-bert"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


__all__ = [
    # Output classes
    "SAOutput",
    "TaskOutput",
    # Embedding enums
    "TokenEmbedding",
    "SentenceEmbedding",
    # Configuration classes
    "BaseConfig",
    "SurvivalConfig",
    "EventClassificationTaskConfig",
    "EventDurationTaskConfig",
    "MTLConfig",
    "SatBertConfig",
    "MENSAConfig",
    "DSMConfig",
    # Task heads
    "SurvivalTaskHead",
    "EventClassificationTaskHead",
    "EventDurationTaskHead",
    "MTLForSurvival",
    "MENSATaskHead",
    "DSMTaskHead",
]
