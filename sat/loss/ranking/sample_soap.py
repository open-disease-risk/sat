"""Sample-based SOAP loss for survival analysis"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch

from sat.models.heads import SAOutput
from sat.utils import logging

from .soap import SOAPLoss

logger = logging.get_default_logger()


class SampleSOAPLoss(SOAPLoss):
    """
    Sample-based SOAP loss implementation for survival analysis.

    This implementation applies the Statistically Optimal Accelerated Pairwise (SOAP)
    loss approach to ranking observations (samples) for a specific event type.

    Similar to SampleRankingLoss, it focuses on comparing different observations
    with the same event type, but uses an optimized pair sampling strategy to
    significantly reduce computational overhead for large batch sizes.

    Performance characteristics:
    - Scales with O(n log n) instead of O(n²) of traditional pairwise ranking
    - Permutes dimensions for efficient observation comparison
    - Particularly effective for large batch sizes
    - Complements EventSOAPLoss which ranks different event types for the same observation
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
        Initialize SampleSOAPLoss.

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
        super(SampleSOAPLoss, self).__init__(
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
        Compute the sample-based SOAP loss.

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
        for event_idx in range(num_events):
            # Get event indicators and durations for this event type
            event_indicators = events[event_idx]  # Shape: [batch_size]
            event_durations = durations[event_idx]  # Shape: [batch_size]

            # Only proceed if we have at least one event of this type
            if event_indicators.sum() > 0:
                # For sample-based ranking, we need to reshape the events and durations
                # to work with the compute_soap_loss function
                # Shape: [batch_size, 1] for use with compute_soap_loss
                reshaped_events = event_indicators.unsqueeze(1)
                reshaped_durations = event_durations.unsqueeze(1)

                # Extract survival probabilities for this event type
                event_survival = predictions.survival[:, event_idx, :].unsqueeze(1)
                event_hazard = predictions.hazard[:, event_idx, :].unsqueeze(1)

                # Sample pairs for efficient computation
                pairs = self.sample_pairs(
                    reshaped_events,
                    reshaped_durations,
                    None,  # Let the function determine optimal number of pairs
                )

                # Skip if no valid pairs
                if pairs.shape[0] == 0:
                    continue

                # Compute SOAP loss for this event type
                event_weight = weights[event_idx] if weights is not None else None
                event_loss = self.compute_soap_loss(
                    reshaped_events,
                    reshaped_durations,
                    event_survival,
                    event_hazard,
                    pairs,
                    event_weight.unsqueeze(0) if event_weight is not None else None,
                )

                total_loss = total_loss + event_loss
                total_valid_events += 1

        # Return average loss across event types
        if total_valid_events > 0:
            return total_loss / total_valid_events
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
