"""Loss functions for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch
from sat.utils import logging
from sat.models.heads import SAOutput
from ..base import RankingLoss

logger = logging.get_default_logger()


class MultiEventRankingLoss(RankingLoss):
    """
    Computes ranking loss between different event types for the same observation.

    This loss is specifically designed for competing risks scenarios, where multiple
    event types can occur for the same subject. It enforces a ranking between different
    event types based on their risk (hazard/survival) for each observation.

    Performance characteristics:
    - Operates on tensor dimensions [batch, events]
    - Complementary to SampleRankingLoss, which ranks observations within each event type
    - Important for datasets with competing risks like HSA synthetic
    - Uses vectorized ranking implementation from parent class for better performance
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
        Initialize MultiEventRankingLoss.

        Args:
            duration_cuts: Path to CSV file containing duration cut points
            importance_sample_weights: Optional path to CSV file with importance weights
            sigma: Scaling factor for exponential term (smaller values create sharper differences)
            num_events: Number of competing events
            margin: Minimum margin required between correctly ranked samples (default: 0.0)
        """
        super(MultiEventRankingLoss, self).__init__(
            duration_cuts, importance_sample_weights, sigma, num_events, margin
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-event ranking loss for competing risks.

        This compares the survival of different event types for the same observation,
        effectively enforcing a ranking between event types based on risk.

        Parameters:
            predictions (SAOutput): Predictions from model (dims: batch size x events x cuts)
            references (torch.Tensor): Reference values (dims: batch size x 4*num_events)

        Returns:
            torch.Tensor: The loss value
        """
        # Extract events and set up dimensions - no permutation needed
        events = self.events(references)
        n = events.shape[0]  # Batch size
        e = events.shape[1]  # Number of events
        device = references.device

        # For efficiency, only calculate weights expansion once and store
        # Weights shape: [num_events] -> [n, num_events-1, e]
        # This avoids repeated expansion and broadcasting in ranking_loss
        weights_expanded = None
        if self.weights is not None:
            # Skip the first weight (censoring) and use only event weights
            weights_expanded = self.weights[1:].to(device)
            # Expand to match the expected dimensions
            weights_expanded = (
                weights_expanded.unsqueeze(0)
                .unsqueeze(2)
                .repeat(1, 1, e)
                .expand(n, -1, -1)
            )

        # Use the vectorized ranking loss from the parent class
        eta = self.ranking_loss(
            events,
            self.durations(references),
            predictions.survival,
            predictions.hazard,
            weights_expanded,
        )

        return eta
