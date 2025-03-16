"""Output classes for survival analysis models"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

from dataclasses import dataclass
from typing import Optional, Any

import torch
from transformers.utils import ModelOutput


@dataclass
class SAOutput(ModelOutput):
    """
    Base class for outputs of survival models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification or survival regression loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Raw model outputs used for predictions and loss calculation.
        predictions (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`, *optional*):
            Processed predictions, which might differ from logits for some model types.
        hazard (`torch.FloatTensor` of shape `(batch_size, num_time_points)`, *optional*):
            Hazard function values evaluated at specific time points, representing the instantaneous risk of an event.
        risk (`torch.FloatTensor` of shape `(batch_size, num_time_points)`, *optional*):
            Cumulative risk values (1 - survival) at specific time points.
        survival (`torch.FloatTensor` of shape `(batch_size, num_time_points)`, *optional*):
            Survival function values at specific time points, representing probability of surviving beyond each time.
        time_to_event (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
            Predicted time to event for each sample.
        event (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
            Binary indicator of whether an event occurred (1) or was censored (0) for each sample.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hazard: Optional[torch.FloatTensor] = None
    risk: Optional[torch.FloatTensor] = None
    survival: Optional[torch.FloatTensor] = None
    time_to_event: Optional[torch.FloatTensor] = None
    event: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None


@dataclass
class TaskOutput(ModelOutput):
    """
    Base class for outputs of task heads.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels=1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels=1) scores (before SoftMax).
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    predictions: Optional[torch.FloatTensor] = None


@dataclass
class DistributionOutput(ModelOutput):
    """
    Output class for distribution-based survival models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Loss value for the distribution model.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Mean prediction values from the distribution (expected survival times).
        hazard (`torch.FloatTensor`, *optional*):
            Hazard function values evaluated on a time grid.
        surv (`torch.FloatTensor`, *optional*):
            Survival function values evaluated on a time grid.
        quantiles (`torch.FloatTensor`, *optional*):
            Quantile predictions at standard levels.
        params (`torch.FloatTensor`, *optional*):
            Distribution parameters (e.g., shape and scale for Weibull).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states from the model.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention values from the model.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hazard: Optional[torch.FloatTensor] = None
    surv: Optional[torch.FloatTensor] = None
    quantiles: Optional[torch.FloatTensor] = None
    params: Optional[torch.FloatTensor] = None
    dist: Optional[Any] = None  # Distribution object
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    predictions: Optional[torch.FloatTensor] = None
