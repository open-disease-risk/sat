"""Sample ranking using ListMLE loss for survival analysis"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch

from sat.models.heads import SAOutput
from sat.utils import logging

from .list_mle import ListMLELoss

logger = logging.get_default_logger()


class SampleListMLELoss(ListMLELoss):
    """
    Computes ListMLE loss by comparing survival probabilities between observations.

    This implementation ranks observations for a fixed event type, similar to
    SampleRankingLoss, but using a listwise approach rather than pairwise.

    Performance characteristics:
    - Permutes dimensions to [events, batch] for efficient observation comparison
    - Uses vectorized listwise ranking implementation
    - Better scaling than pairwise ranking with large datasets
    - Complements EventListMLELoss which ranks different event types for the same observation
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
        Initialize SampleListMLELoss.

        Args:
            duration_cuts: Path to CSV file containing duration cut points
            importance_sample_weights: Optional path to CSV file with importance weights
            num_events: Number of competing events
            epsilon: Small value for numerical stability
            temperature: Controls the sharpness of the probability distribution
        """
        super(SampleListMLELoss, self).__init__(
            duration_cuts, importance_sample_weights, num_events, epsilon, temperature
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the ListMLE loss by permuting tensors to compare observations.

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
        batch_size = events.shape[1]
        device = references.device

        # Create weights tensor if needed
        weights = None
        if self.weights is not None:
            # Skip the first weight (censoring) and use only event weights
            weights = self.weights[1:].to(device)

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_valid_events = 0

        # Process each event type separately
        for i in range(num_events):
            # Get event indicators and durations for this event type
            event_indicators = events[i]  # Shape: [batch_size]
            event_durations = durations[i]  # Shape: [batch_size]

            # Filter for observations with this event (event_indicators == 1)
            event_mask = event_indicators == 1

            # Only proceed if we have events of this type
            if event_mask.sum() > 0:
                # For each time point, get survival probability at that time
                # Extract survival probabilities for this event type
                surv_probs = predictions.survival[
                    :, i, :
                ]  # [batch_size, num_time_bins]

                # Interpolate survival probability at exact durations
                # This requires mapping durations to the correct time bin

                # Find the index of each duration in duration_cuts
                # Similar to the interpolation logic in ranking_loss
                durations_expanded = event_durations.unsqueeze(1)  # [batch_size, 1]
                cuts_expanded = self.duration_cuts.to(device).unsqueeze(
                    0
                )  # [1, num_cuts]

                # Get indices for interpolation
                index_smaller = (
                    cuts_expanded <= durations_expanded
                )  # [batch_size, num_cuts]
                t0_index = torch.sum(index_smaller, dim=1) - 1  # [batch_size]
                t0_index = torch.clamp(t0_index, min=0)
                t1_index = torch.clamp(t0_index + 1, max=len(self.duration_cuts) - 1)

                # Get time points and survival values
                t0 = torch.gather(
                    self.duration_cuts.to(device), 0, t0_index
                )  # [batch_size]
                t1 = torch.gather(
                    self.duration_cuts.to(device), 0, t1_index
                )  # [batch_size]

                # Build indices for gathering from survival probabilities
                gather_indices_t0 = t0_index.unsqueeze(1)  # [batch_size, 1]
                gather_indices_t1 = t1_index.unsqueeze(1)  # [batch_size, 1]

                # Get survival probabilities at t0 and t1
                s_t0 = torch.gather(surv_probs, 1, gather_indices_t0).squeeze(
                    1
                )  # [batch_size]
                s_t1 = torch.gather(surv_probs, 1, gather_indices_t1).squeeze(
                    1
                )  # [batch_size]

                # Interpolate survival at exact durations
                dt = t1 - t0
                interp_mask = dt > 0
                s_at_durations = s_t0.clone()

                if interp_mask.any():
                    # Get hazard rate for interpolation
                    h_star = torch.zeros_like(s_t0)
                    epsilon = 1e-6
                    log_s_t0 = torch.log(s_t0[interp_mask] + epsilon)
                    log_s_t1 = torch.log(s_t1[interp_mask] + epsilon)
                    h_star[interp_mask] = (log_s_t0 - log_s_t1) / dt[interp_mask]

                    # Interpolate survival at exact time
                    s_at_durations[interp_mask] = s_t0[interp_mask] * torch.exp(
                        -(event_durations[interp_mask] - t0[interp_mask])
                        * h_star[interp_mask]
                    )

                # For ranking observations within an event type:
                # - Earlier events should have higher risk (1 - survival probability)
                # - So we use (1 - survival) as score
                scores = 1.0 - s_at_durations  # Shape: [batch_size]

                # Ground truth ranking should be based on durations
                # Shorter durations should be ranked higher (more urgent)
                # We use negative durations as rankings so smaller durations get higher rank
                rankings = -event_durations  # Shape: [batch_size]

                # Compute ListMLE loss for this event type
                event_weight = 1.0
                if weights is not None:
                    event_weight = weights[i]

                # Reshape tensors to ensure they're 2D for compute_list_mle_loss
                # The error occurs because scores/rankings are 1D but compute_list_mle_loss expects 2D
                event_loss = self.compute_list_mle_loss(
                    scores=scores.unsqueeze(1),
                    rankings=rankings.unsqueeze(1),
                    mask=event_mask.unsqueeze(1),
                )

                # Apply event-specific weight
                total_loss = total_loss + event_weight * event_loss
                total_valid_events += 1

        # Return average loss across event types
        if total_valid_events > 0:
            return total_loss / total_valid_events
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
