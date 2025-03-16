"""Task heads for survival analysis

This package contains the various task heads used for survival analysis, multi-task learning,
and event prediction tasks.
"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

# Base classes and utilities
from .base import SurvivalPreTrainedModel, SurvivalTask, pad_col
from .outputs import SAOutput, TaskOutput, DistributionOutput

# Configuration classes
from .config import (
    BaseConfig,
    TokenEmbedding,
    SentenceEmbedding,
    SurvivalConfig,
    EventClassificationTaskConfig,
    EventDurationTaskConfig,
    DistributionTaskConfig,
    MTLConfig,
    SatBertConfig,
)

# Task heads
from .survival_head import SurvivalTaskHead
from .classification_head import EventClassificationTaskHead
from .duration_head import EventDurationTaskHead
from .distribution_head import DistributionHead
from .mtl import MTLForSurvival

# Embeddings and neural network components
from .embeddings import (
    BaseEmbeddingPooler,
    TokenEmbeddingPooler,
    SentenceEmbeddingPooler,
    StableEmbeddingProcessor,
)

from .nets import (
    SimpleMLP,
    StableMLPSequence,
    CauseSpecificNet,
    CauseSpecificNetCompRisk,
)

# Re-export everything from current distribution head
from .distribution_head import *

__all__ = [
    # Output classes
    "SAOutput",
    "TaskOutput",
    "DistributionOutput",
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
    "DistributionTaskConfig",
    "MTLConfig",
    "SatBertConfig",
    # Task heads
    "SurvivalTaskHead",
    "EventClassificationTaskHead",
    "EventDurationTaskHead",
    "DistributionHead",
    "MTLForSurvival",
    # Embeddings
    "BaseEmbeddingPooler",
    "TokenEmbeddingPooler",
    "SentenceEmbeddingPooler",
    "StableEmbeddingProcessor",
    # Neural nets
    "SimpleMLP",
    "StableMLPSequence",
    "CauseSpecificNet",
    "CauseSpecificNetCompRisk",
    # Utilities
    "pad_col",
]
