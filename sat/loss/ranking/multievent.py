"""Loss functions for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch
from sat.utils import logging
from sat.models.heads import SAOutput
from ..base import RankingLoss

logger = logging.get_default_logger()


class MultiEventRankingLoss(RankingLoss):
    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        sigma: float = 1.0,
        num_events: int = 1,
    ):
        super(MultiEventRankingLoss, self).__init__(
            duration_cuts, importance_sample_weights, sigma, num_events
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute the deephit ranking loss.

        Parameters:
            predictions (SAOutput: Predictions of the model (SAOutput: 5 x events x batch size x cuts)
            references (torch.Tensor): Reference values. (dims: batch size x 4)

        Returns:
            float: The loss value.
        """
        events = self.events(references)
        n = events.shape[0]
        e = events.shape[1]
        eta = self.ranking_loss(
            events,
            self.durations(references),
            predictions.survival,
            predictions.hazard,
            self.weights[1:]
            .unsqueeze(0)
            .unsqueeze(2)
            .repeat(1, 1, e)
            .expand(n, -1, -1),
        )
        return eta
