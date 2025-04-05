"""Event-based RankNet implementation for survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch

from sat.models.heads import SAOutput
from sat.utils import logging

from .ranknet import RankNetLoss

logger = logging.get_default_logger()


class EventRankNetLoss(RankNetLoss):
    """
    Event-based RankNet implementation for survival analysis.

    This implementation focuses on ranking different event types for the same sample.
    Similar to MultiEventRankingLoss, it compares risk scores between different events,
    but uses the RankNet probabilistic framework for learning to rank.

    Key differences from base RankNetLoss:
    - Focus on comparing different event types for the same sample
    - Designed for competing risks scenarios
    - More effective when samples have multiple event types
    """

    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        num_events: int = 1,
        sigma: float = 1.0,
        sampling_ratio: float = 1.0,  # Default to use all event pairs (usually small number)
        use_adaptive_sampling: bool = True,
    ):
        """
        Initialize Event RankNet loss.

        Args:
            duration_cuts: Path to CSV file containing duration cut points
            importance_sample_weights: Optional path to CSV file with importance weights
            num_events: Number of competing events
            sigma: Controls the steepness of the sigmoid (temperature parameter)
            sampling_ratio: Ratio of all possible pairs to sample (0.0-1.0)
            use_adaptive_sampling: If True, adaptively sample more pairs from difficult regions
        """
        super(EventRankNetLoss, self).__init__(
            duration_cuts,
            importance_sample_weights,
            num_events,
            sigma,
            sampling_ratio,
            use_adaptive_sampling,
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the event-based RankNet loss.

        Parameters:
            predictions (SAOutput): Predictions of the model with survival probabilities
            references (torch.Tensor): Reference values (dims: batch size x 4 * num_events)

        Returns:
            torch.Tensor: The loss value
        """
        # Extract events and durations
        events = self.events(references)  # [batch_size, num_events]
        durations = self.durations(references)  # [batch_size, num_events]

        batch_size = events.shape[0]
        num_events = events.shape[1]
        device = references.device

        # Skip if we only have one event type
        if num_events <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Create weights tensor if needed
        weights = None
        if self.weights is not None:
            # Skip the first weight (censoring) and use only event weights
            weights = self.weights[1:].to(device)

        # Initialize loss accumulator
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_sample_count = 0

        # Process each sample separately
        for sample_idx in range(batch_size):
            # Get events and durations for this sample
            sample_events = events[sample_idx]  # [num_events]
            sample_durations = durations[sample_idx]  # [num_events]

            # Find event types that occurred for this sample
            event_indices = torch.where(sample_events == 1)[0]

            # Skip if sample has fewer than 2 events
            if len(event_indices) < 2:
                continue

            # For each sample, we compare all pairs of events that occurred
            num_sample_events = len(event_indices)
            max_event_pairs = num_sample_events * (num_sample_events - 1) // 2

            # Generate all event pairs for this sample
            event_i_indices = []
            event_j_indices = []

            # Generate all unique pairs of events
            for i in range(num_sample_events):
                for j in range(i + 1, num_sample_events):
                    event_i_indices.append(event_indices[i].item())
                    event_j_indices.append(event_indices[j].item())

            # Convert to tensor
            event_i_indices = torch.tensor(event_i_indices, device=device)
            event_j_indices = torch.tensor(event_j_indices, device=device)

            # Get durations for these event pairs
            i_durations = sample_durations[event_i_indices]
            j_durations = sample_durations[event_j_indices]

            # Calculate targets: P_{ij} = 1 if t_i < t_j (earlier event should have higher risk)
            targets = (i_durations < j_durations).float()

            # Handle ties (same duration)
            ties = i_durations == j_durations
            # Set target probability to 0.5 for ties
            targets[ties] = 0.5

            # Get survival predictions for this sample
            sample_survival = predictions.survival[
                sample_idx
            ]  # [num_events, num_time_bins+1]

            # Extract survival for specific event types
            survival_i = sample_survival[event_i_indices]
            survival_j = sample_survival[event_j_indices]

            # Interpolate survival at event times
            interp_i = self.interpolate_survival_batch(survival_i, i_durations, device)
            interp_j = self.interpolate_survival_batch(survival_j, j_durations, device)

            # Convert to risk scores (1 - survival)
            risk_i = 1.0 - interp_i
            risk_j = 1.0 - interp_j

            # Skip if no valid pairs
            if len(risk_i) == 0:
                continue

            # Compute RankNet loss for this sample
            sample_loss = self.compute_ranknet_loss(risk_i, risk_j, targets)

            # Apply event-specific weights if provided
            if weights is not None:
                # Average the weights of the compared events
                event_weights = torch.stack(
                    [weights[event_i_indices], weights[event_j_indices]]
                )
                avg_weights = event_weights.mean(dim=0)
                sample_loss = sample_loss * avg_weights.mean()

            # Add to total loss
            total_loss = total_loss + sample_loss
            valid_sample_count += 1

        # Return average loss across samples
        if valid_sample_count > 0:
            return total_loss / valid_sample_count
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
