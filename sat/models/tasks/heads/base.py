"""Base classes and utilities for survival analysis task heads"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import abc
import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from .config import BaseConfig
from sat.utils import logging

logger = logging.get_default_logger()


def pad_col(input, val=0, where="end"):
    """Adds a column of `val` at the start or end of `input` with optimized memory usage.

    This optimized version avoids unnecessary tensor allocations.
    """
    if len(input.shape) != 3:
        raise ValueError(f"Only works for `phi` tensor that is 3-D.")

    # Get shape for new tensor
    batch_size, num_events, seq_len = input.shape
    new_shape = (batch_size, num_events, seq_len + 1)

    # Create output tensor directly with correct size - pre-allocated with zeros
    # Helps avoid fragmentation and improves memory locality
    result = torch.zeros(new_shape, dtype=input.dtype, device=input.device)

    # Fill with data efficiently using direct copy operation
    if where == "end":
        result[:, :, :seq_len].copy_(input)  # Faster than assignment
        if val != 0:
            result[:, :, -1].fill_(val)  # Inplace fill is more efficient
    elif where == "start":
        result[:, :, 1:].copy_(input)  # Direct copy
        if val != 0:
            result[:, :, 0].fill_(val)  # Inplace fill
    else:
        raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

    return result


class SurvivalPreTrainedModel(PreTrainedModel):
    """Base class for all survival pretrained models.

    This class extends the Hugging Face PreTrainedModel class with specific
    functionality needed for survival analysis models.
    """

    base_model_prefix = "survival"
    config_class = BaseConfig
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.BatchNorm1d):
            module.bias.data.zero_()
            module.weight.data.fill_(
                0.7
            )  # Significantly less than 1.0 for better stability


class SurvivalTask(abc.ABC, SurvivalPreTrainedModel):
    """Abstract base class for all survival task models."""

    def __init__(self, config: BaseConfig):
        super().__init__(config)
