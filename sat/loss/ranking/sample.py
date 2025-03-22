"""Loss functions for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch

from sat.models.heads import SAOutput
from sat.utils import logging

from ..base import RankingLoss

logger = logging.get_default_logger()


class SampleRankingLoss(RankingLoss):
    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        sigma: float = 1.0,
        num_events: int = 1,
    ):
        super(SampleRankingLoss, self).__init__(
            duration_cuts, importance_sample_weights, sigma, num_events
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute the deephit ranking loss.

        Parameters:
            predictions (SAOutput: Predictions of the model (SAOutput: 5 x events x batch size x cuts)
            references (torch.Tensor): Reference values. (dims: batch size x 4)

        Returns:
            torch.Tensor: The loss value.
        """
        events = self.events(references).permute(1, 0)
        e = events.shape[1]

        # This calculation already returns a tensor
        eta = self.ranking_loss(
            events,
            self.durations(references).permute(1, 0),
            predictions.survival.permute(1, 0, 2),
            predictions.hazard.permute(1, 0, 2),
            self.weights[1:].unsqueeze(1).unsqueeze(2).repeat(1, e, e),
        )

        # The ensure_tensor is still kept as a fallback
        return self.ensure_tensor(eta, device=references.device)
