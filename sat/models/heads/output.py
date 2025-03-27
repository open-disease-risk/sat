"""Output classes for the heads for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch

from dataclasses import dataclass

from transformers.utils import ModelOutput
from typing import Optional, Tuple

from sat.utils import logging


logger = logging.get_default_logger()


@dataclass
class SAOutput(ModelOutput):
    """
    Base class for outputs of survival analysis models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hazard (`torch.FloatTensor`, *optional*):
            Hazard function values computed by the model.
        risk (`torch.FloatTensor`, *optional*):
            Cumulative risk function values computed by the model.
        survival (`torch.FloatTensor`, *optional*):
            Survival function values computed by the model.
        time_to_event (`torch.FloatTensor`, *optional*):
            Time-to-event predictions from a regression task head.
        event (`torch.FloatTensor`, *optional*):
            Event probability predictions from a classification task head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        shape (`torch.FloatTensor`, *optional*):
            Shape parameters for DSM distribution mixtures.
        scale (`torch.FloatTensor`, *optional*):
            Scale parameters for DSM distribution mixtures.
        logits_g (`torch.FloatTensor`, *optional*):
            Logits for DSM mixture component weights.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hazard: Optional[torch.FloatTensor] = None
    risk: Optional[torch.FloatTensor] = None
    survival: Optional[torch.FloatTensor] = None
    time_to_event: Optional[torch.FloatTensor] = None
    event: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # DSM-specific fields
    shape: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None
    logits_g: Optional[torch.FloatTensor] = None


@dataclass
class TaskOutput(ModelOutput):
    """
    Base class for outputs of survival analysis models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        predictions (`torch.FloatTensor` of shape `(batch_size, )`):
            Predictions of the task.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    predictions: torch.FloatTensor = None
