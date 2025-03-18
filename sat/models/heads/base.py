"""Base classes for the heads for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import abc
import torch

from torch import nn
from transformers.modeling_utils import PreTrainedModel

from sat.models.tasks.config import BaseConfig
from sat.utils import logging

logger = logging.get_default_logger()


class SurvivalPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            logger.debug("Initialize linear weights...")

            if self.config.initializer == "normal":
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif self.config.initializer == "xavier_uniform":
                torch.nn.init.xavier_uniform_(
                    module.weight, gain=torch.nn.init.calculate_gain("leaky_relu")
                )
            elif self.config.initializer == "xavier_normal":
                torch.nn.init.xavier_normal_(
                    module.weight, gain=torch.nn.init.calculate_gain("leaky_relu")
                )
            elif self.config.initializer == "kaiming_normal":
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif self.config.initializer == "kaiming_uniform":
                nn.init.kaiming_uniform_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            else:
                raise ValueError(
                    f"Initializer {self.config.initializer} not supported!"
                )

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            logger.debug("Initialize layer norm weights...")
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SurvivalTask(abc.ABC, SurvivalPreTrainedModel):
    def __init__(self, config: BaseConfig):
        super().__init__(config)
