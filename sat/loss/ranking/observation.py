"""Loss functions for ranking events by risk for individual observations"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch

from sat.models.heads import SAOutput
from sat.utils import logging

from ..base import RankingLoss

logger = logging.get_default_logger()


class ObservationEventRankingLoss(RankingLoss):
    """
    Ranking loss that ranks competing events by risk for each individual observation.

    Unlike DeepHitRankingLoss which ranks observations within each event type,
    this loss pivots on individual observations and ranks the competing event types
    by their risk, encouraging the model to correctly order different event risks
    for the same individual.
    """

    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        sigma: float = 0.1,
        num_events: int = 2,  # Requires at least 2 event types
        balance_strategy: str = "fixed",
        balance_params: dict = None,
    ):
        super(ObservationEventRankingLoss, self).__init__(
            duration_cuts, importance_sample_weights, sigma, num_events
        )

        if num_events < 2:
            raise ValueError(
                "ObservationEventRankingLoss requires at least 2 event types"
            )

        # Initialize loss balancer if needed
        self.balance_strategy = balance_strategy
        self.balance_params = balance_params or {}
        self._balancer = None

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the observation-event ranking loss.

        For each individual observation, ranks the competing events by risk,
        penalizing when the risk ordering doesn't match the observed event.

        Parameters:
            predictions (SAOutput): Predictions of the model (SAOutput with risk estimates)
            references (torch.Tensor): Reference values (dims: batch size x 4)

        Returns:
            torch.Tensor: The loss value.
        """
        batch_size = predictions.logits.shape[0]
        device = references.device

        # Extract event information
        events = self.events(references)  # [batch_size, num_events]
        durations = self.durations(references)  # [batch_size, num_events]

        # Get risk values from predictions (survival is 1-risk)
        risk = predictions.risk  # [batch_size, num_events, num_time_bins+1]

        # Compute ranking loss
        ranking_loss = torch.zeros(1, device=device)

        # Count observations with actual events for normalization
        observed_event_count = 0

        for i in range(batch_size):
            # Check if any event occurred for this observation
            event_occurred = torch.any(events[i] == 1)

            if not event_occurred:
                continue  # Skip censored observations

            observed_event_count += 1

            # Get the event that actually occurred (first one if multiple)
            actual_event_idx = torch.where(events[i] == 1)[0][0].item()
            actual_event_time = durations[i, actual_event_idx]

            # Find closest time bin index for the actual event time
            time_bin_idx = torch.searchsorted(self.duration_cuts, actual_event_time)
            # Ensure index is within bounds
            time_bin_idx = torch.clamp(time_bin_idx, max=self.num_time_bins - 1)

            # Get risk at that time for all event types for this observation
            # Add 1 to time_bin_idx because risk includes time 0, but ensure it's within bounds
            risk_idx = torch.min(
                time_bin_idx + 1, torch.tensor(risk.size(2) - 1, device=device)
            )
            risks_at_t = risk[i, :, risk_idx]

            # Actual event's risk
            actual_event_risk = risks_at_t[actual_event_idx]

            # Compare with all other event types
            for j in range(self.num_events):
                if j == actual_event_idx:
                    continue  # Skip comparison with itself

                # Risk difference: actual_event_risk should be higher than others
                risk_diff = risks_at_t[j] - actual_event_risk

                # Apply exponential scaling with sigma (smaller sigma = sharper differences)
                # We want the actual event to have higher risk, so we penalize when other events
                # have higher risk (risk_diff > 0)
                exp_diff = torch.exp(risk_diff / self.sigma)

                # Apply weight for this event pair if provided
                if self.weights is not None:
                    # Use both event weights
                    weight = (
                        self.weights[actual_event_idx + 1] + self.weights[j + 1]
                    ) / 2
                    exp_diff = exp_diff * weight

                ranking_loss += exp_diff

        # Normalize by number of observations with events
        if observed_event_count > 0:
            ranking_loss = ranking_loss / observed_event_count
        else:
            ranking_loss = torch.zeros(1, device=device)

        # The ensure_tensor is kept as a fallback
        return self.ensure_tensor(ranking_loss, device=device)
