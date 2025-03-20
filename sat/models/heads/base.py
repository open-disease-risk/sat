"""Base classes for the heads for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import abc
import torch

from torch import nn
from transformers.modeling_utils import PreTrainedModel

from sat.utils import logging

from .config import BaseConfig

logger = logging.get_default_logger()


class BaseTask(PreTrainedModel):
    """Base class for all task models."""

    def _init_weights(self, module):
        """Initialize the weights - generic initialization"""
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


class SurvivalTask(abc.ABC, BaseTask):
    """Base class for survival analysis tasks."""
    supports_gradient_checkpointing = True

    def __init__(self, config: BaseConfig):
        super().__init__(config)

    def _init_weights(self, module):
        """Survival-specific weight initialization"""
        if isinstance(module, nn.Linear):
            # For survival tasks, we want to initialize the final layer differently
            # to ensure proper hazard function behavior
            if (
                hasattr(self, "nets")
                and hasattr(self.nets, "event_nets")
                and hasattr(module, "weight")
                and module is self.nets.event_nets[-1].net[-1]
            ):
                # Initialize final layer to produce small initial hazard values
                logger.debug(
                    "Initializing final survival layer with special weights..."
                )
                nn.init.constant_(module.weight, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, -2.0)  # Start with low hazard
            else:
                # Use standard initialization for other layers
                super()._init_weights(module)
        else:
            super()._init_weights(module)


class RegressionTask(abc.ABC, BaseTask):
    """Base class for regression tasks."""
    supports_gradient_checkpointing = True

    def __init__(self, config: BaseConfig):
        super().__init__(config)

    def _init_weights(self, module):
        """Regression-specific weight initialization"""
        if isinstance(module, nn.Linear):
            # For regression tasks, we want different initialization
            if (
                hasattr(self, "nets")
                and hasattr(self.nets, "event_nets")
                and hasattr(module, "weight")
                and module is self.nets.event_nets[-1].net[-1]
            ):
                # Output layer initialization for regression
                logger.debug(
                    "Initializing final regression layer with special weights..."
                )
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    # Initialize bias to small positive value for ReLU activations
                    nn.init.constant_(module.bias, 0.1)
            else:
                super()._init_weights(module)
        else:
            super()._init_weights(module)


class ClassificationTask(abc.ABC, BaseTask):
    """Base class for classification tasks."""
    supports_gradient_checkpointing = True

    def __init__(self, config: BaseConfig):
        super().__init__(config)

    def _init_weights(self, module):
        """Classification-specific weight initialization"""
        if isinstance(module, nn.Linear):
            # For classification tasks, we want different initialization
            if (
                hasattr(self, "nets")
                and hasattr(self.nets, "event_nets")
                and hasattr(module, "weight")
                and module is self.nets.event_nets[-1].net[-1]
            ):
                # Initialize output layer for sigmoid activation
                logger.debug(
                    "Initializing final classification layer with special weights..."
                )
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                super()._init_weights(module)
        else:
            super()._init_weights(module)


class MTLTask(abc.ABC, BaseTask):
    """Base class for multi-task learning tasks."""
    supports_gradient_checkpointing = True
    
    def __init__(self, config: BaseConfig):
        super().__init__(config)

    def _init_weights(self, module):
        """MTL-specific weight initialization
        
        This should ONLY be applied to the shared network layers owned directly
        by the MTL task, not to the sub-task modules which will handle their
        own initialization.
        """
        if isinstance(module, nn.Linear):
            # For MTL shared layers
            if hasattr(self, "net") and hasattr(module, "weight"):
                # Initialize shared network for balanced task learning
                logger.debug(f"Initializing MTL shared layer: {module}")
                nn.init.kaiming_uniform_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                super()._init_weights(module)
        else:
            super()._init_weights(module)
