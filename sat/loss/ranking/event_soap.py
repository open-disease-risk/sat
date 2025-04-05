"""Event-based SOAP loss for survival analysis"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch

from sat.models.heads import SAOutput
from sat.utils import logging

from .soap import SOAPLoss

logger = logging.get_default_logger()


class EventSOAPLoss(SOAPLoss):
    """
    Event-based SOAP loss implementation for survival analysis.

    This implementation applies the Statistically Optimal Accelerated Pairwise (SOAP)
    loss approach to ranking different event types for the same observation.

    Similar to MultiEventRankingLoss, it focuses on comparing different event types
    for the same observation, but uses optimized pair sampling to significantly
    reduce computational overhead for datasets with many event types.

    Performance characteristics:
    - Scales with O(e log e) for event types rather than O(e²)
    - Complements SampleSOAPLoss which ranks observations within each event type
    - Particularly important for competing risks scenarios with many event types
    - Optimized for datasets like HSA synthetic with multiple competing risks
    """

    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        num_events: int = 1,
        margin: float = 0.1,
        sigma: float = 1.0,
        num_pairs: int = None,
        sampling_strategy: str = "uniform",
        adaptive_margin: bool = False,
    ):
        """
        Initialize EventSOAPLoss.

        Args:
            duration_cuts: Path to CSV file containing duration cut points
            importance_sample_weights: Optional path to CSV file with importance weights
            num_events: Number of competing events
            margin: Minimum margin between correctly ranked samples
            sigma: Scaling factor for loss (smaller values create sharper differences)
            num_pairs: Number of pairs to sample per anchor (None = auto-calculate)
            sampling_strategy: Strategy for pair sampling ("uniform", "importance", "hard")
            adaptive_margin: Whether to adapt margin based on duration differences
        """
        super(EventSOAPLoss, self).__init__(
            duration_cuts,
            importance_sample_weights,
            num_events,
            margin,
            sigma,
            num_pairs,
            sampling_strategy,
            adaptive_margin,
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the event-based SOAP loss.

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

        # Only compute loss if we have multiple event types
        if num_events <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Initialize for results
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_valid_samples = 0

        # Create weights if needed
        weights = None
        if self.weights is not None:
            # Skip the first weight (censoring) and use only event weights
            weights = self.weights[1:].to(device)

        # Process each observation separately
        for i in range(batch_size):
            # Get events and durations for this observation
            sample_events = events[i]  # [num_events]
            sample_durations = durations[i]  # [num_events]

            # Check if this sample has any events
            if sample_events.sum() < 2:
                # Need at least 2 events to compare different event types
                continue

            # For event-based ranking, we need to reshape to work with compute_soap_loss
            # Reshape to [num_events, 1] for consistency with compute_soap_loss
            sample_events = sample_events.unsqueeze(1)
            sample_durations = sample_durations.unsqueeze(1)

            # Extract survival for this observation across all event types
            # Reshape from [batch_size, num_events, num_time_bins+1] to [num_events, 1, num_time_bins+1]
            sample_survival = predictions.survival[i].unsqueeze(1)
            sample_hazard = predictions.hazard[i].unsqueeze(1)

            # Sample event pairs for efficient computation
            # For event-based, we can usually do full pairwise because num_events is small
            # But use sampling for very large number of event types
            pairs = self.sample_pairs(
                sample_events,
                sample_durations,
                None,  # Auto-calculate, likely full pairwise for reasonable num_events
            )

            # Skip if no valid pairs
            if pairs.shape[0] == 0:
                continue

            # Compute SOAP loss for this observation
            sample_loss = self.compute_soap_loss(
                sample_events,
                sample_durations,
                sample_survival,
                sample_hazard,
                pairs,
                weights,  # Pass event-specific weights if available
            )

            # Add to total loss
            total_loss = total_loss + sample_loss
            total_valid_samples += 1

        # Return average loss
        if total_valid_samples > 0:
            return total_loss / total_valid_samples
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
