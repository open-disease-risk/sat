"""Distribution Parameter Head for parametric survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DistributionTaskConfig
from .base import SurvivalPreTrainedModel
from .outputs import DistributionOutput
from .nets import StableMLPSequence
from sat.models.tasks.distributions.base import (
    WeibullDistribution,
    LogNormalDistribution,
)
from sat.utils import logging

logger = logging.get_default_logger()


class DistributionHead(SurvivalPreTrainedModel):
    """Head that outputs parameters for a parametric survival distribution.

    This head transforms the encoder output into parameters for a parametric
    distribution, enabling more flexible modeling of survival outcomes than
    traditional hazard-based approaches.
    """

    config_class = DistributionTaskConfig

    def __init__(self, config):
        """Initialize distribution parameter head.

        Args:
            config: Configuration object for the distribution head
        """
        super().__init__(config)
        self.distribution_type = config.distribution_type
        self.num_events = config.num_events

        # Each distribution has 2 parameters (e.g., shape and scale for Weibull)
        self.param_dim = 2
        self.output_dim = self.num_events * self.param_dim

        # Network to predict distribution parameters
        self.network = StableMLPSequence(
            input_size=config.num_inputs,
            hidden_size=config.intermediate_size,
            output_size=self.output_dim,
            num_hidden_layers=config.num_hidden_layers,
            hidden_dropout_prob=config.hidden_dropout_prob,
            batch_norm=config.batch_norm,
            bias=config.bias,
        )

        # Initialize weights
        self.init_weights()

    def forward(
        self,
        encoder_output=None,
        last_hidden_state=None,
        shared_embedding=None,
        **kwargs,
    ):
        """Forward pass of the distribution head.

        Args:
            encoder_output: Output from encoder (unused in this implementation)
            last_hidden_state: Last hidden state from encoder (unused in this implementation)
            shared_embedding: Shared embeddings from the MTL model
            **kwargs: Additional keyword arguments

        Returns:
            SAOutput object containing distribution object and other attributes
        """
        # Use shared embedding as input features
        x = shared_embedding
        batch_size = x.size(0)

        # Get raw parameters from network
        params = self.network(x)

        # Reshape parameters for multiple events
        if self.num_events > 1:
            params = params.view(batch_size, self.num_events, self.param_dim)

        # Apply transformations to ensure parameters are valid
        if self.distribution_type == "Weibull":
            # Ensure positive values for shape and scale using softplus
            params = F.softplus(params)

            # Create distribution object
            dist = WeibullDistribution(params)
        elif self.distribution_type == "LogNormal":
            # Scale parameter must be positive
            if self.num_events > 1:
                params[:, :, 1] = F.softplus(params[:, :, 1])
            else:
                params[:, 1] = F.softplus(params[:, 1])

            # Create distribution object
            dist = LogNormalDistribution(params)
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")

        # Create time grid for hazard and survival function evaluation
        # This matches the expected output format from other heads
        time_grid = torch.linspace(0, 1, 100, device=x.device)

        # Compute hazard and survival on time grid
        hazard = dist.hazard(time_grid)
        surv = dist.survival(time_grid)

        # For quantiles, use standard levels
        quantile_levels = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=x.device)
        quantiles = dist.quantile(quantile_levels)

        # Get mean prediction (expected survival time)
        mean = dist.mean()

        # Create output with all attributes using DistributionOutput
        mean_values = mean.unsqueeze(-1) if mean.dim() == 1 else mean
        output = DistributionOutput(
            logits=mean_values,  # Mean prediction as the primary output
            predictions=mean_values,  # Also provide as predictions
            hazard=hazard,  # Hazard function
            surv=surv,  # Survival function
            quantiles=quantiles,  # Quantile predictions
            params=params,  # Distribution parameters
            dist=dist,  # Distribution object
        )

        return output
