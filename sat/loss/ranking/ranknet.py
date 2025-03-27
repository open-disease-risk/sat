"""RankNet implementation for survival analysis ranking."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torch.nn.functional as F

from sat.models.heads import SAOutput
from sat.utils import logging
from ..base import RankingLoss

logger = logging.get_default_logger()


class RankNetLoss(RankingLoss):
    """
    RankNet implementation for survival analysis.

    RankNet is a pairwise learning-to-rank algorithm that uses a probabilistic framework
    to model the relative order of pairs. It was originally proposed by Burges et al. (2005)
    in "Learning to Rank using Gradient Descent".

    This implementation adapts RankNet for survival analysis by:
    1. Converting survival predictions to risk scores
    2. Comparing risk scores between pairs
    3. Using a probabilistic loss based on sigmoid cross-entropy

    Key Advantages:
    - Smooth differentiable loss function
    - Directly minimizes pairwise ordering errors
    - Probabilistic interpretation of ranking performance
    - Efficient implementation for large datasets
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
        Initialize RankNet loss.

        Args:
            duration_cuts: Path to CSV file containing duration cut points
            importance_sample_weights: Optional path to CSV file with importance weights
            num_events: Number of competing events
            sigma: Controls the steepness of the sigmoid (temperature parameter)
            sampling_ratio: Ratio of all possible pairs to sample (0.0-1.0)
            use_adaptive_sampling: If True, adaptively sample more pairs from difficult regions
        """
        # Note: margin parameter is not used in RankNet, but parent class requires it
        super(RankNetLoss, self).__init__(
            duration_cuts, importance_sample_weights, sigma, num_events, margin=0.0
        )
        self.sampling_ratio = sampling_ratio
        self.use_adaptive_sampling = use_adaptive_sampling

    def sample_pairs(self, events: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Sample pairs for efficient RankNet computation.

        Args:
            events: Event indicators [batch_size, num_events]
            batch_size: Number of samples

        Returns:
            torch.Tensor: Sampled pair indices [num_pairs, 2]
        """
        device = events.device

        # Calculate total possible pairs
        max_possible_pairs = batch_size * (batch_size - 1) // 2

        # Determine number of pairs to sample
        num_pairs = int(max_possible_pairs * self.sampling_ratio)
        num_pairs = max(1, min(num_pairs, max_possible_pairs))

        # Generate all possible unique pairs efficiently
        if batch_size <= 128 or num_pairs >= max_possible_pairs // 2:
            # For smaller batches, generate all pairs and sample
            row_idx, col_idx = torch.triu_indices(
                batch_size, batch_size, offset=1, device=device
            )
            pairs = torch.stack([row_idx, col_idx], dim=1)

            # Random selection if we're not using all pairs
            if num_pairs < len(pairs):
                indices = torch.randperm(pairs.shape[0], device=device)[:num_pairs]
                pairs = pairs[indices]
        else:
            # For larger batches, direct random generation is more memory-efficient
            pairs = torch.zeros((num_pairs, 2), dtype=torch.long, device=device)

            for i in range(num_pairs):
                # Generate random pair ensuring i != j
                while True:
                    i_idx = torch.randint(0, batch_size, (1,), device=device)
                    j_idx = torch.randint(0, batch_size, (1,), device=device)
                    if i_idx != j_idx:
                        pairs[i, 0] = i_idx
                        pairs[i, 1] = j_idx
                        break

        return pairs

    def compute_ranknet_loss(
        self,
        risk_i: torch.Tensor,  # Risk scores for samples i
        risk_j: torch.Tensor,  # Risk scores for samples j
        targets: torch.Tensor,  # Target probabilities P(i > j) (1 if i should rank higher)
    ) -> torch.Tensor:
        """
        Compute the RankNet loss for given pairs.

        Args:
            risk_i: Risk scores for first items in pairs
            risk_j: Risk scores for second items in pairs
            targets: Target probabilities (1 if risk_i should be > risk_j, 0 otherwise)

        Returns:
            torch.Tensor: RankNet loss value
        """
        # Compute probability that i ranks higher than j using the logistic function
        # P(i > j) = sigma(s_i - s_j) where sigma is the sigmoid function
        diff = (risk_i - risk_j) * self.sigma
        pred_prob = torch.sigmoid(diff)

        # Binary cross-entropy loss between predicted probability and target
        # This is equivalent to the original RankNet formulation:
        # -targets * log(pred_prob) - (1-targets) * log(1-pred_prob)
        loss = F.binary_cross_entropy(pred_prob, targets)

        return loss

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the RankNet loss for survival prediction.

        Parameters:
            predictions (SAOutput): Predictions of the model with survival probabilities
            references (torch.Tensor): Reference values (dims: batch size x 4 * num_events)

        Returns:
            torch.Tensor: The loss value
        """
        events = self.events(references)  # [batch_size, num_events]
        durations = self.durations(references)  # [batch_size, num_events]

        batch_size = events.shape[0]
        device = references.device

        # Create weights tensor if needed
        weights = None
        if self.weights is not None:
            # Skip the first weight (censoring) and use only event weights
            weights = self.weights[1:].to(device)

        # Sample pairs for efficient computation
        pairs = self.sample_pairs(events, batch_size)

        if pairs.shape[0] == 0:  # No pairs to process
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Extract pair indices
        i_indices = pairs[:, 0]
        j_indices = pairs[:, 1]

        # Initialize total loss
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_valid_events = 0

        # Process each event type separately
        for event_idx in range(self.num_events):
            # Skip if no events of this type
            if not torch.any(events[:, event_idx]):
                continue

            # Create mask for valid pairs (both i and j have this event)
            i_has_event = events[i_indices, event_idx]
            j_has_event = events[j_indices, event_idx]

            both_have_events = i_has_event & j_has_event

            # Skip if no valid pairs
            if not torch.any(both_have_events):
                continue

            # Filter pairs where both have this event
            valid_pairs = torch.where(both_have_events)[0]
            valid_i = i_indices[valid_pairs]
            valid_j = j_indices[valid_pairs]

            # Get durations for these pairs
            i_durations = durations[valid_i, event_idx]
            j_durations = durations[valid_j, event_idx]

            # Calculate targets: P_{ij} = 1 if t_i < t_j (earlier event should have higher risk)
            # In survival analysis, earlier events should have higher risk
            targets = (i_durations < j_durations).float()

            # Handle ties (same duration)
            ties = i_durations == j_durations
            # Set target probability to 0.5 for ties
            targets[ties] = 0.5

            # Extract survival predictions for this event
            survival_i = predictions.survival[valid_i, event_idx, :]
            survival_j = predictions.survival[valid_j, event_idx, :]

            # Calculate point-wise risk scores (use survival at event time for ranking)
            # Interpolate survival at the event time
            interp_i = self.interpolate_survival_batch(survival_i, i_durations, device)
            interp_j = self.interpolate_survival_batch(survival_j, j_durations, device)

            # Convert to risk scores (1 - survival)
            risk_i = 1.0 - interp_i
            risk_j = 1.0 - interp_j

            # Compute RankNet loss for this event type
            event_loss = self.compute_ranknet_loss(risk_i, risk_j, targets)

            # Apply importance weighting if provided
            if weights is not None:
                event_loss = event_loss * weights[event_idx]

            # Add to total loss
            total_loss = total_loss + event_loss
            total_valid_events += 1

        # Return average loss across event types
        if total_valid_events > 0:
            return total_loss / total_valid_events
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

    def interpolate_survival_batch(
        self,
        survival_curves: torch.Tensor,  # [batch_size, num_time_bins+1]
        durations: torch.Tensor,  # [batch_size]
        device: torch.device,
    ) -> torch.Tensor:
        """
        Efficient batch interpolation of survival curves at specified times.

        Args:
            survival_curves: Survival curves for multiple samples
            durations: Times at which to evaluate the survival for each sample
            device: Device for tensor computation

        Returns:
            torch.Tensor: Interpolated survival values [batch_size]
        """
        batch_size = survival_curves.shape[0]

        # Create tensor version of duration cuts
        cuts = self.duration_cuts.to(device)

        # Find interval indices for each duration
        # For each duration, find the largest index i where cuts[i] <= duration
        dur_expanded = durations.unsqueeze(1)  # [batch_size, 1]
        cuts_expanded = cuts.unsqueeze(0)  # [1, num_cuts]

        # Create mask of valid cut points
        is_valid_cut = cuts_expanded <= dur_expanded  # [batch_size, num_cuts]

        # Get rightmost index where cut <= duration
        indices = torch.sum(is_valid_cut, dim=1) - 1  # [batch_size]
        indices = torch.clamp(indices, min=0, max=len(cuts) - 2)  # Ensure valid indices

        # Get left and right cut points
        t0_indices = indices
        t1_indices = torch.clamp(indices + 1, max=len(cuts) - 1)

        # Get times for each interval
        t0 = cuts[t0_indices]  # [batch_size]
        t1 = cuts[t1_indices]  # [batch_size]

        # Get survival values at interval endpoints
        s0 = torch.gather(survival_curves, 1, t0_indices.unsqueeze(1)).squeeze(
            1
        )  # [batch_size]
        s1 = torch.gather(survival_curves, 1, t1_indices.unsqueeze(1)).squeeze(
            1
        )  # [batch_size]

        # Linear interpolation weights
        dt = t1 - t0  # [batch_size]

        # Handle edge case of identical cut points
        interp_weights = torch.zeros_like(dt)
        valid_dt = dt > 0

        if valid_dt.any():
            interp_weights[valid_dt] = (durations[valid_dt] - t0[valid_dt]) / dt[
                valid_dt
            ]

        # Linear interpolation
        interp_survival = s0 + interp_weights * (s1 - s0)

        return interp_survival
