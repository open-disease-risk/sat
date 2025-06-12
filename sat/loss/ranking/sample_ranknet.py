"""Sample-based RankNet implementation for survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch

from sat.models.heads import SAOutput
from sat.utils import logging

from .ranknet import RankNetLoss

logger = logging.get_default_logger()


class SampleRankNetLoss(RankNetLoss):
    """
    Sample-based RankNet implementation for survival analysis.

    This implementation focuses on ranking different samples with the same event type.
    Similar to SampleRankingLoss, it compares risk scores between samples, but uses
    the RankNet probabilistic framework for learning to rank.

    Key differences from base RankNetLoss:
    - Focus on comparing different samples for the same event type
    - Designed for sample-level ranking (as opposed to event-level)
    - More effective when many samples share the same event type
    """

    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        num_events: int = 1,
        sigma: float = 1.0,
        sampling_ratio: float = 0.3,
        use_adaptive_sampling: bool = True,
    ):
        """
        Initialize Sample RankNet loss.

        Args:
            duration_cuts: Path to CSV file containing duration cut points
            importance_sample_weights: Optional path to CSV file with importance weights
            num_events: Number of competing events
            sigma: Controls the steepness of the sigmoid (temperature parameter)
            sampling_ratio: Ratio of all possible pairs to sample (0.0-1.0)
            use_adaptive_sampling: If True, adaptively sample more pairs from difficult regions
        """
        super(SampleRankNetLoss, self).__init__(
            duration_cuts,
            importance_sample_weights,
            num_events,
            sigma,
            sampling_ratio,
            use_adaptive_sampling,
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the sample-based RankNet loss.

        Parameters:
            predictions (SAOutput): Predictions of the model with survival probabilities
            references (torch.Tensor): Reference values (dims: batch size x 4 * num_events)

        Returns:
            torch.Tensor: The loss value
        """
        # Permute dimensions to change from [batch, events] to [events, batch]
        # This allows comparing different observations with the same event type
        events = self.events(references).permute(1, 0)
        durations = self.durations(references).permute(1, 0)

        num_events = events.shape[0]
        # batch_size = events.shape[1]  # Used for debugging/logging if needed
        device = references.device

        # Create weights tensor if needed
        weights = None
        if self.weights is not None:
            # Skip the first weight (censoring) and use only event weights
            weights = self.weights[1:].to(device)

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_valid_events = 0

        # Process each event type separately
        for event_idx in range(num_events):
            # Get event indicators and durations for this event type
            event_indicators = events[event_idx]  # [batch_size]
            event_durations = durations[event_idx]  # [batch_size]

            # Skip if no events of this type
            event_count = event_indicators.sum()
            if event_count < 2:  # Need at least 2 events to form pairs
                continue

            # Find indices of samples with this event
            event_indices = torch.where(event_indicators == 1)[0]

            # Sample pairs for this event type
            num_event_samples = len(event_indices)
            max_event_pairs = num_event_samples * (num_event_samples - 1) // 2

            # Determine how many pairs to sample (balanced across events)
            if self.sampling_ratio >= 1.0:
                # Use all possible pairs
                event_pairs = max_event_pairs
            else:
                # Sample a subset of pairs
                event_pairs = int(max_event_pairs * self.sampling_ratio)
                event_pairs = max(1, min(event_pairs, max_event_pairs))

            # Create pairs of samples with this event
            if num_event_samples <= 128 or event_pairs >= max_event_pairs // 2:
                # For smaller sample sets, generate all pairs and sample
                row_idx, col_idx = torch.triu_indices(
                    num_event_samples, num_event_samples, offset=1, device=device
                )

                # Convert to original sample indices
                i_indices = event_indices[row_idx]
                j_indices = event_indices[col_idx]

                # Random selection if we're not using all pairs
                if event_pairs < len(row_idx):
                    select_idx = torch.randperm(len(row_idx), device=device)[
                        :event_pairs
                    ]
                    i_indices = i_indices[select_idx]
                    j_indices = j_indices[select_idx]
            else:
                # For larger sample sets, directly sample random pairs
                i_indices = torch.zeros(event_pairs, dtype=torch.long, device=device)
                j_indices = torch.zeros(event_pairs, dtype=torch.long, device=device)

                for idx in range(event_pairs):
                    # Sample two different indices from event_indices
                    while True:
                        i, j = torch.randperm(num_event_samples, device=device)[:2]
                        if i != j:
                            i_indices[idx] = event_indices[i]
                            j_indices[idx] = event_indices[j]
                            break

            # Get durations for selected pairs
            i_durations = event_durations[i_indices]
            j_durations = event_durations[j_indices]

            # Calculate targets: P_{ij} = 1 if t_i < t_j (earlier event should have higher risk)
            # In survival analysis, earlier events should have higher risk (lower survival)
            targets = (i_durations < j_durations).float()

            # Handle ties (same duration)
            ties = i_durations == j_durations
            # Set target probability to 0.5 for ties
            targets[ties] = 0.5

            # Extract survival predictions for this event type
            survival_i = predictions.survival[
                i_indices, event_idx, :
            ]  # [num_pairs, num_time_bins+1]
            survival_j = predictions.survival[
                j_indices, event_idx, :
            ]  # [num_pairs, num_time_bins+1]

            # Interpolate survival at the event time
            interp_i = self.interpolate_survival_batch(survival_i, i_durations, device)
            interp_j = self.interpolate_survival_batch(survival_j, j_durations, device)

            # Convert to risk scores (1 - survival)
            risk_i = 1.0 - interp_i
            risk_j = 1.0 - interp_j

            # Compute RankNet loss for this event type
            event_loss = self.compute_ranknet_loss(risk_i, risk_j, targets)

            # Apply event weighting if provided
            if weights is not None:
                event_weight = weights[event_idx]
                event_loss = event_loss * event_weight

            # Add to total loss
            total_loss = total_loss + event_loss
            total_valid_events += 1

        # Return average loss across event types
        if total_valid_events > 0:
            return total_loss / total_valid_events
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
