"""Event ranking using ListMLE loss for survival analysis"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch

from sat.models.heads import SAOutput
from sat.utils import logging
from .list_mle import ListMLELoss

logger = logging.get_default_logger()


class EventListMLELoss(ListMLELoss):
    """
    Computes ListMLE loss between different event types for the same observation.

    This loss is specifically designed for competing risks scenarios, where multiple
    event types can occur for the same subject. It enforces a ranking between different
    event types based on their risk (hazard/survival) for each observation using a
    listwise approach.

    Performance characteristics:
    - Operates on tensor dimensions [batch, events]
    - Complementary to SampleListMLELoss, which ranks observations for each event type
    - Important for datasets with competing risks
    - Uses vectorized listwise ranking implementation for better performance
    """

    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        num_events: int = 1,
        epsilon: float = 1e-10,
        temperature: float = 1.0,
    ):
        """
        Initialize EventListMLELoss.

        Args:
            duration_cuts: Path to CSV file containing duration cut points
            importance_sample_weights: Optional path to CSV file with importance weights
            num_events: Number of competing events
            epsilon: Small value for numerical stability
            temperature: Controls the sharpness of the probability distribution
        """
        super(EventListMLELoss, self).__init__(
            duration_cuts, importance_sample_weights, num_events, epsilon, temperature
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the ListMLE loss across event types for each observation.

        Parameters:
            predictions (SAOutput): Predictions from model (dims: batch size x events x cuts)
            references (torch.Tensor): Reference values (dims: batch size x 4*num_events)

        Returns:
            torch.Tensor: The loss value
        """
        # Extract events and durations
        events = self.events(references)  # [batch_size, num_events]
        durations = self.durations(references)  # [batch_size, num_events]

        batch_size = events.shape[0]
        num_events = events.shape[1]
        device = references.device

        # Skip if only one event type
        if num_events <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Create weights tensor if needed
        weights = None
        if self.weights is not None:
            # Skip the first weight (censoring) and use only event weights
            weights = self.weights[1:].to(device)

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_observations = 0

        # Process each observation separately
        for i in range(batch_size):
            # Get event indicators for this observation
            event_indicators = events[i]  # Shape: [num_events]

            # Check if this observation has any events
            has_events = (event_indicators == 1).any()

            if has_events:
                # Get durations for this observation
                obs_durations = durations[i]  # Shape: [num_events]

                # Get survival probabilities at specific durations for each event type
                surv_probs = []

                for j in range(num_events):
                    # Only process if this is an actual event
                    if event_indicators[j] == 1:
                        duration = obs_durations[j]

                        # Find the index in duration_cuts similar to ranking_loss
                        cuts = self.duration_cuts.to(device)
                        index_smaller = cuts <= duration
                        t0_index = torch.sum(index_smaller) - 1
                        t0_index = max(0, t0_index)
                        t1_index = min(t0_index + 1, len(cuts) - 1)

                        # Get time points and survival values for all event types at these indices
                        t0 = cuts[t0_index]
                        t1 = cuts[t1_index]

                        # Get survival for all event types at these indices
                        # Shape: [num_events]
                        s_t0 = predictions.survival[i, :, t0_index]
                        s_t1 = predictions.survival[i, :, t1_index]

                        # Interpolate
                        dt = t1 - t0
                        if dt > 0:
                            # Get hazard rate for interpolation
                            epsilon = 1e-6
                            log_s_t0 = torch.log(s_t0 + epsilon)
                            log_s_t1 = torch.log(s_t1 + epsilon)
                            h_star = (log_s_t0 - log_s_t1) / dt

                            # Interpolate survival at exact time for all event types
                            s_at_duration = s_t0 * torch.exp(-(duration - t0) * h_star)
                        else:
                            s_at_duration = s_t0

                        # Store these probabilities
                        surv_probs.append(s_at_duration)

                # Stack survival probabilities for all events that this observation experienced
                # Shape: [num_observed_events, num_events]
                surv_probs = torch.stack(surv_probs)

                # For each event that occurred, rank all possible event types
                event_indices = torch.where(event_indicators == 1)[0]
                valid_event_count = len(event_indices)

                obs_loss = torch.tensor(0.0, device=device)

                for idx, event_idx in enumerate(event_indices):
                    # For ranking event types:
                    # - The event that actually occurred should have higher risk
                    # - So we use (1 - survival) as score for each event type
                    scores = 1.0 - surv_probs[idx]  # Shape: [num_events]

                    # Create ground truth rankings
                    # The event that actually occurred should be ranked highest
                    rankings = torch.zeros(num_events, device=device)
                    rankings[event_idx] = 1.0  # Mark the actual event with highest rank

                    # Create mask for valid events (events with indicators)
                    event_mask = event_indicators >= 0  # All are valid here

                    # Compute ListMLE loss for this event occurrence
                    # Reshape tensors to ensure they're 2D for compute_list_mle_loss
                    # The error occurs because scores/rankings are 1D but compute_list_mle_loss expects 2D
                    event_loss = self.compute_list_mle_loss(
                        scores=scores.unsqueeze(1),
                        rankings=rankings.unsqueeze(1),
                        mask=event_mask.unsqueeze(1),
                    )

                    # Apply event-specific weight if available
                    if weights is not None:
                        event_weight = weights[event_idx]
                        event_loss = event_weight * event_loss

                    obs_loss = obs_loss + event_loss

                # Average loss for this observation
                if valid_event_count > 0:
                    obs_loss = obs_loss / valid_event_count
                    total_loss = total_loss + obs_loss
                    valid_observations += 1

        # Return average loss across observations
        if valid_observations > 0:
            return total_loss / valid_observations
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
