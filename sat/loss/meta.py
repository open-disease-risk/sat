"""Meta Loss class for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch

from torch import nn
from transformers.utils import ModelOutput

from sat.utils import logging

from .base import Loss

logger = logging.get_default_logger()


class MetaLoss(Loss):
    """A loss class that linearly combines multiple loss components."""

    def __init__(self, losses: list, coeffs: list):
        super(MetaLoss, self).__init__()

        self.losses = nn.ModuleList(losses)
        self.register_buffer("coeffs", torch.tensor(coeffs).to(torch.float32))

    def forward(
        self, predictions: ModelOutput, references: torch.Tensor
    ) -> torch.Tensor:
        l = 0.0
        for i, loss in enumerate(self.losses):
            l += self.coeffs[i] * loss(predictions, references)

        return l
