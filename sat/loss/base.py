"""Base classes and utilities for loss functions"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch
import torch.nn.functional as F
from torch import nn
from enum import Enum, auto
from typing import Dict, Optional, Union, List

from sat.utils import logging
from sat.models.tasks.heads import SAOutput, TaskOutput

logger = logging.get_default_logger()


class LossBalanceStrategy(Enum):
    """Enum for different loss balancing strategies."""

    NONE = auto()  # No balancing, just use raw weights
    SCALE_NORMALIZATION = auto()  # Normalize losses by their scales
    GRAD_NORMALIZATION = auto()  # Normalize gradients
    HOMOSCEDASTIC = auto()  # Uncertainty-based weighting (Kendall et al.)
    ADAPTIVE = auto()  # Adaptive weights learned during training


class Loss(nn.Module):
    """Base class for losses.

    Provides utilities for loss computation and loss component balancing.
    """

    def __init__(
        self,
        num_events: int = 1,
        balance_strategy: Union[str, LossBalanceStrategy] = LossBalanceStrategy.NONE,
        ema_momentum: float = 0.9,
    ):
        super(Loss, self).__init__()
        self.num_events = num_events

        # Loss balancing configuration
        self.ema_momentum = ema_momentum
        if isinstance(balance_strategy, str):
            try:
                self.balance_strategy = LossBalanceStrategy[balance_strategy.upper()]
            except KeyError:
                logger.warning(
                    f"Unknown balance strategy '{balance_strategy}', using NONE instead"
                )
                self.balance_strategy = LossBalanceStrategy.NONE
        else:
            self.balance_strategy = balance_strategy

    def durations(self, references: torch.Tensor):
        """Extract durations from references tensor, preserving gradient flow.

        Args:
            references: Target tensor containing event indicators and durations.
                Shape: [batch_size, 4*num_events]

        Returns:
            Durations tensor with gradient connections preserved
        """
        # Efficient slice operation with pre-defined start and end indices
        start_idx = 3 * self.num_events
        end_idx = start_idx + self.num_events
        return references[:, start_idx:end_idx].float()

    def duration_percentiles(self, references: torch.Tensor):
        """Extract duration percentiles from references tensor, preserving gradient flow.

        Args:
            references: Target tensor containing event indicators and durations.
                Shape: [batch_size, 4*num_events]

        Returns:
            Duration percentiles tensor with gradient connections preserved
        """
        # Use direct indexing for better efficiency
        return references[:, : self.num_events].long()

    def events(self, references: torch.Tensor):
        """Extract event indicators from references tensor, preserving gradient flow.

        Args:
            references: Target tensor containing event indicators and durations.
                Shape: [batch_size, 4*num_events]

        Returns:
            Event indicators tensor with gradient connections preserved
        """
        # Efficient slice operation with pre-defined start and end indices
        start_idx = self.num_events
        end_idx = start_idx + self.num_events
        return references[:, start_idx:end_idx].long()

    def fraction_with_quantile(self, references: torch.Tensor):
        """Extract fraction with quantile from references tensor, preserving gradient flow.

        Args:
            references: Target tensor containing event indicators and durations.
                Shape: [batch_size, 4*num_events]

        Returns:
            Fraction with quantile tensor with gradient connections preserved
        """
        # Efficient slice operation with pre-defined start and end indices
        start_idx = 2 * self.num_events
        end_idx = start_idx + self.num_events
        return references[:, start_idx:end_idx].float()

    def survivals(self, predictions):
        # Pre-allocate list with known size for better memory efficiency
        surv = [None] * len(predictions)

        # Process each prediction efficiently
        for i, logits in enumerate(predictions):
            # Apply operations in-place where possible
            hazard = F.softplus(logits)
            # Use inplace operations for better memory efficiency
            surv[i] = hazard.cumsum(dim=1).neg_().exp_()[:, :-1]
        return surv

    def _init_loss_params(self, loss_components: List[str]):
        """Initialize parameters for loss balancing based on selected strategy.

        Args:
            loss_components: List of names for different loss components
        """
        if self.balance_strategy == LossBalanceStrategy.NONE:
            return

        if self.balance_strategy == LossBalanceStrategy.SCALE_NORMALIZATION:
            # Register buffers for tracking loss scales
            for name in loss_components:
                self.register_buffer(f"{name}_scale", torch.tensor(1.0))

        elif self.balance_strategy == LossBalanceStrategy.HOMOSCEDASTIC:
            # Register log variance parameters for each loss
            for name in loss_components:
                self.register_parameter(f"log_var_{name}", nn.Parameter(torch.zeros(1)))

        elif self.balance_strategy == LossBalanceStrategy.ADAPTIVE:
            # Register log weights for each loss component
            for name in loss_components:
                self.register_parameter(
                    f"log_weight_{name}", nn.Parameter(torch.zeros(1))
                )

    def balance_losses(
        self, loss_dict: Dict[str, torch.Tensor], weights: Dict[str, float] = None
    ) -> torch.Tensor:
        """Balance multiple loss components according to the configured strategy.

        Args:
            loss_dict: Dictionary mapping loss component names to tensor values
            weights: Optional dictionary of initial weights for each component

        Returns:
            Balanced total loss
        """
        if weights is None:
            weights = {name: 1.0 for name in loss_dict.keys()}

        # Simple weighted sum without balancing
        if self.balance_strategy == LossBalanceStrategy.NONE:
            return sum(weights[name] * loss for name, loss in loss_dict.items())

        # Scale normalization with EMA tracking
        elif self.balance_strategy == LossBalanceStrategy.SCALE_NORMALIZATION:
            total_loss = 0.0

            for name, loss in loss_dict.items():
                # Update scale using exponential moving average
                if self.training:
                    scale_attr = f"{name}_scale"
                    if hasattr(self, scale_attr):
                        scale = getattr(self, scale_attr)
                        scale.data = (
                            self.ema_momentum * scale
                            + (1 - self.ema_momentum) * loss.detach()
                        )
                    else:
                        self.register_buffer(scale_attr, loss.detach().clone())
                        scale = getattr(self, scale_attr)

                # Normalize loss by its scale
                normalized_loss = loss / (
                    getattr(self, f"{name}_scale", torch.tensor(1.0)) + 1e-8
                )
                total_loss = total_loss + weights[name] * normalized_loss

            return total_loss

        # Homoscedastic uncertainty weighting
        elif self.balance_strategy == LossBalanceStrategy.HOMOSCEDASTIC:
            total_loss = 0.0

            for name, loss in loss_dict.items():
                # Get precision (1/variance) from learned log variance
                log_var = getattr(self, f"log_var_{name}")
                precision = torch.exp(-log_var)

                # Apply precision-weighted loss with regularization
                total_loss = total_loss + precision * loss + log_var

            return total_loss / len(loss_dict)

        # Adaptive weighting with learned weights
        elif self.balance_strategy == LossBalanceStrategy.ADAPTIVE:
            # Get all log weights
            log_weights = [
                getattr(self, f"log_weight_{name}") for name in loss_dict.keys()
            ]

            # Compute softmax weights (ensure they sum to 1)
            weights = torch.softmax(torch.cat(log_weights), dim=0)

            # Compute weighted loss
            total_loss = 0.0
            for i, (name, loss) in enumerate(loss_dict.items()):
                total_loss = total_loss + weights[i] * loss

            return total_loss

        # Gradient normalization
        elif self.balance_strategy == LossBalanceStrategy.GRAD_NORMALIZATION:
            total_loss = 0.0

            for name, loss in loss_dict.items():
                with torch.no_grad():
                    scale = loss.abs().detach() + 1e-8

                # Scale gradients but maintain original loss magnitude
                balanced_loss = (loss / scale).detach() * loss
                total_loss = total_loss + weights[name] * balanced_loss

            return total_loss

        return sum(weights[name] * loss for name, loss in loss_dict.items())
