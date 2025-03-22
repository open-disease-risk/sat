"""Meta Loss class for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch
from typing import List, Optional, Dict, Union

from torch import nn
from transformers.utils import ModelOutput

from sat.utils import logging
from .balancing import BalancingStrategy
from .base import Loss

logger = logging.get_default_logger()


class MetaLoss(Loss):
    """A loss class that combines multiple loss components using flexible balancing strategies."""

    def __init__(
        self,
        losses: List[Loss],
        coeffs: Optional[List[float]] = None,
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
        num_events: int = 1,
    ):
        """
        Initialize MetaLoss with multiple loss components and balancing strategy.

        Args:
            losses: List of loss components
            coeffs: Initial loss coefficients (for fixed weighting)
            balance_strategy: Strategy for balancing loss components
            balance_params: Additional parameters for the balancing strategy
            num_events: Number of events
        """
        super(MetaLoss, self).__init__(
            num_events=num_events,
            balance_strategy=balance_strategy,
            balance_params=balance_params,
        )

        self.losses = nn.ModuleList(losses)
        self.iteration = 0

        # Initialize coefficients
        if coeffs is None:
            coeffs = [1.0] * len(losses)

        # Initialize balancer
        self.balancer = self.get_balancer(num_losses=len(losses), coeffs=coeffs)

        # Register coefficients for state_dict
        self.register_buffer("coeffs", torch.tensor(coeffs).to(torch.float32))

    def forward(
        self,
        predictions: ModelOutput,
        references: torch.Tensor,
        iteration: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute balanced combined loss.

        Args:
            predictions: Model predictions
            references: Ground truth references
            iteration: Current training iteration (for adaptive methods)

        Returns:
            Balanced total loss
        """
        # Compute individual losses
        individual_losses = []
        for loss in self.losses:
            individual_losses.append(loss(predictions, references))

        # Use provided iteration or increment internal counter
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1

        # Balance losses using the selected strategy
        total_loss = self.balancer(individual_losses, self.iteration)

        # Update coeffs buffer for tracking
        with torch.no_grad():
            self.coeffs = torch.tensor(
                self.balancer.get_weights(), device=self.coeffs.device
            )

        return total_loss

    def get_loss_weights(self) -> List[float]:
        """
        Get current loss component weights.

        Returns:
            List of weights for each loss component
        """
        return self.balancer.get_weights()
