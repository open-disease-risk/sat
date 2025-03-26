"""Loss functions for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch

from sat.models.heads import SAOutput
from sat.utils import logging

from ..base import RankingLoss

logger = logging.get_default_logger()


class SampleRankingLoss(RankingLoss):
    """
    Computes ranking loss by comparing survival probabilities between observations.

    This is a highly efficient implementation of the DeepHit ranking loss that uses
    tensor permutation and the base RankingLoss's ranking_loss method. It ensures
    subjects with earlier events have appropriately higher risk than those with
    later events.

    Performance characteristics:
    - More efficient than DeepHitRankingLoss, especially for larger batch sizes and event counts
    - Scales linearly with both batch size and number of events
    - Preserves the same ranking behavior as DeepHitRankingLoss

    Note: This implementation produces lower absolute values than DeepHitRankingLoss
    but maintains identical relative ranking behavior.
    """

    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        sigma: float = 1.0,
        num_events: int = 1,
        margin: float = 0.0,
    ):
        """
        Initialize SampleRankingLoss.

        Args:
            duration_cuts: Path to CSV file containing duration cut points
            importance_sample_weights: Optional path to CSV file with importance weights
            sigma: Scaling factor for loss (smaller values create sharper differences)
            num_events: Number of competing events
            margin: Minimum margin required between survival probabilities (default: 0.0)
                   When > 0, enforces a minimum difference between survival values
        """
        super(SampleRankingLoss, self).__init__(
            duration_cuts, importance_sample_weights, sigma, num_events, margin
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the ranking loss by permuting tensors to compare observations.

        Parameters:
            predictions (SAOutput): Predictions of the model with survival probabilities
            references (torch.Tensor): Reference values (dims: batch size x 4 * num_events)

        Returns:
            torch.Tensor: The loss value
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
