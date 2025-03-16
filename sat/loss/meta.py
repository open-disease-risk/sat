"""Meta loss that combines multiple loss functions"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch
from torch import nn
from transformers.utils import ModelOutput

from sat.loss.base import Loss


class MetaLoss(Loss):
    """A loss class that linearly combines multiple loss components."""

    def __init__(
        self, losses: list, coeffs: list, balance_strategy="none", ema_momentum=0.9
    ):
        # Pass balance_strategy and ema_momentum to the base Loss class
        super(MetaLoss, self).__init__(
            1, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )

        self.losses = nn.ModuleList(losses)
        self.register_buffer("coeffs", torch.tensor(coeffs).to(torch.float32))

    def forward(
        self, predictions: ModelOutput, references: torch.Tensor
    ) -> torch.Tensor:
        # Create dictionaries to collect individual loss components and their coefficients
        loss_dict = {}
        coeff_dict = {}

        for i, loss in enumerate(self.losses):
            # Get the loss name to use as dictionary key
            loss_name = type(loss).__name__

            # Store the individual loss and its coefficient
            current_loss = loss(predictions, references)
            loss_dict[loss_name] = current_loss
            coeff_dict[loss_name] = self.coeffs[i]

        # Pass all individual components to balance_losses for proper balancing
        return self.balance_losses(loss_dict, coeff_dict)
