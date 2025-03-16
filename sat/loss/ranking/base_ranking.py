"""Base Ranking Loss classes for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd
import torch
import torch.nn.functional as F

from sat.models.tasks.heads import SAOutput
from sat.loss.base import Loss


class RankingLoss(Loss):
    """Base class for ranking losses."""

    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        sigma: float = 1.0,
        num_events: int = 1,
    ):
        super(RankingLoss, self).__init__(num_events)

        self.sigma = sigma

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

        df = pd.read_csv(duration_cuts, header=None, names=["cuts"])
        self.register_buffer(
            "duration_cuts", torch.Tensor(df.cuts.values)
        )  # tn duration cut points

    def ranking_loss(
        self, events, durations, survival, hazard, weights
    ) -> torch.Tensor:
        device = events.device
        n = events.shape[0]
        e = events.shape[1]

        # Convert to bool directly during creation for better efficiency
        I = events.bool()
        I_censored = (
            ~I
        )  # censored indicator (n x e) - more efficient than events.to(bool) and then negating

        # Reuse duration_cuts with expand instead of repeating for better memory efficiency
        T = self.duration_cuts.expand(n, e, -1)  # duration cut points (n x e x tn)

        # Compute indices more efficiently by reusing the unsqueezed durations
        durations_unsqueezed = durations.unsqueeze(2)
        indexSmaller = self.duration_cuts.view(1, 1, -1) <= durations_unsqueezed

        # Optimize summation with explicit dimension
        t0Index = (
            torch.sum(indexSmaller, dim=2, keepdim=False) - 1
        )  # left boundary of time interval (n x e)

        # Use unsqueeze+expand instead of repeat for better memory efficiency
        t0Index_expanded = t0Index.unsqueeze(1).expand(n, e, e)
        t1Index = t0Index_expanded + 1  # right boundary of time interval (n x e x e)

        # Handle out-of-bounds indices efficiently with inplace operation
        duration_cuts_len = len(self.duration_cuts)
        fixOOB = t1Index == duration_cuts_len
        # Use inplace indexing for better performance
        t1Index[fixOOB] = duration_cuts_len - 1

        # Gather operations for boundary values - reuse where possible
        T0 = torch.gather(T, 2, t0Index_expanded)  # left boundary times (n x e x e)
        T1 = torch.gather(T, 2, t1Index)  # right boundary times (n x e x e)

        # Compute survivals and hazards at boundaries
        SatT0 = torch.gather(
            survival, 2, t0Index_expanded
        )  # survival at T0 (n x e x e)
        SatT1 = torch.gather(survival, 2, t1Index)  # survival at T1 (n x e x e)
        hstar = torch.gather(hazard, 2, t0Index_expanded)  # hazard at T0 (n x e x e)

        # Calculate time differences once
        dT = T1 - T0

        # Optimize interpolation with inplace operations where possible
        # Adding small epsilon for numerical stability
        positive_mask = dT > 0.0  # More efficient than torch.gt
        if positive_mask.any():
            eps = 1e-6  # Use a constant instead of magic number
            log_diff = torch.log(SatT0[positive_mask] + eps) - torch.log(
                SatT1[positive_mask] + eps
            )
            hstar[positive_mask] = (
                log_diff / dT[positive_mask]
            )  # solve for hazard given the survival at T0 and T1 (n x e x e)

        SatT = SatT0 * torch.exp(
            -(durations.unsqueeze(1).repeat(1, e, 1) - T0) * hstar
        )  # solve for survival at time t (n x e x e)

        # Compute epsilon time more efficiently
        # Pre-compute epsilon for t-epsilon calculation
        t_epsilon = (
            self.duration_cuts[-1] - self.duration_cuts[0]
        ) / self.duration_cuts[-1]

        # Use expand instead of repeat for better memory efficiency
        durations_expanded = durations.unsqueeze(1).expand(n, e, e)
        TMinus = F.relu(durations_expanded - t_epsilon)

        # Calculate SatTMinus efficiently
        SatTMinus = SatT0 * torch.exp(
            -(TMinus - T0) * hstar
        )  # solve for survival at time t-epsilon (n x e x e)

        # Extract diagonals and optimize with expand instead of repeat
        diag_S = torch.diagonal(SatT, dim1=-2, dim2=-1).unsqueeze(2).expand(n, e, e)
        diag_S2 = (
            torch.diagonal(SatTMinus, dim1=-2, dim2=-1).unsqueeze(2).expand(n, e, e)
        )

        # Calculate differences more efficiently by reusing transposed tensors
        # Compute the transposes once and reuse
        SatT_T = SatT.transpose(1, 2)
        diag_S2_T = diag_S2.transpose(1, 2)

        # Calculate the survival differences
        dS1 = diag_S - SatT_T  # dS_{ij} = S_{i}(T_{i}) - S_{j}(T_{i}) (n x e x e)
        dS2 = (
            SatTMinus - diag_S2_T
        )  # dS_{ij} = S_{i}(T_{j}-1) - S_{j}(T_{j}-1) (n x e x e)
        dS3 = SatT - diag_S.transpose(
            1, 2
        )  # dS_{ij} = S_{i}(T_{j}) - S_{j}(T_{j}) (n x e x e)

        # Expand the indicators for better memory efficiency
        I_expanded_i = I.unsqueeze(2).expand(n, e, e).float()
        I_expanded_j = I.unsqueeze(1).expand(n, e, e).float()
        I_censored_expanded_j = I_censored.unsqueeze(1).expand(n, e, e).float()

        # More efficient duration comparison by reusing pre-expanded tensors
        # Calculate this once instead of twice
        dur_diff = F.relu(
            torch.sign(durations_expanded - durations.unsqueeze(2).expand(n, e, e))
        )

        # Calculate A matrices with optimized operations
        A1 = I_expanded_i * dur_diff
        A2 = A1 * I_expanded_j  # when event occurred for subject j
        A3 = A1 * I_censored_expanded_j  # when subject j is censored

        # Pre-compute scaling factor for exponentials
        inv_sigma = 1.0 / self.sigma

        # Calculate exponentials more efficiently
        exp_dS1 = torch.exp(dS1 * inv_sigma)
        exp_dS2 = torch.exp(dS2 * inv_sigma)
        exp_dS3 = torch.exp(dS3 * inv_sigma)

        # Compute weighted sum efficiently
        weighted_sum = A1 * exp_dS1 + A2 * exp_dS2 + A3 * exp_dS3

        # Calculate mean with explicit weights
        eta = torch.mean(weights * weighted_sum)

        return eta  # (1 x 1)


class SampleRankingLoss(RankingLoss):
    """Sample Ranking Loss for survival analysis."""

    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        sigma: float = 1.0,
        num_events: int = 1,
    ):
        super(SampleRankingLoss, self).__init__(
            duration_cuts, importance_sample_weights, sigma, num_events
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute the deephit ranking loss.

        Parameters:
            predictions (SAOutput: Predictions of the model (SAOutput: 5 x events x batch size x cuts)
            references (torch.Tensor): Reference values. (dims: batch size x 4)

        Returns:
            float: The loss value.
        """
        # Compute events and permute once, storing result for reuse
        events_permuted = self.events(references).permute(1, 0)

        # Extract shape information once
        n = events_permuted.shape[0]
        e = events_permuted.shape[1]

        # Compute durations and permute once
        durations_permuted = self.durations(references).permute(1, 0)

        # Cache permuted prediction attributes
        # Using contiguous() improves memory layout after permute operations
        survival_permuted = predictions.survival.permute(1, 0, 2).contiguous()
        hazard_permuted = predictions.hazard.permute(1, 0, 2).contiguous()

        # Get weights slice only once
        weight_slice = self.weights[1:]

        # Create expanded weights more efficiently
        # This approach avoids the repeat operation when possible
        if weight_slice.shape[0] == 1:
            # For single event, use expand which is more memory efficient
            weights_expanded = weight_slice.view(1, 1, 1).expand(1, e, e)
        else:
            # For multiple events, size the weight matrix correctly
            weights_expanded = weight_slice.view(-1, 1, 1).expand(-1, e, e)

        # Call ranking loss with prepared inputs
        eta = self.ranking_loss(
            events_permuted,
            durations_permuted,
            survival_permuted,
            hazard_permuted,
            weights_expanded,
        )

        return eta


class MultiEventRankingLoss(RankingLoss):
    """Multiple Event Ranking Loss for survival analysis."""

    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        sigma: float = 1.0,
        num_events: int = 1,
    ):
        super(MultiEventRankingLoss, self).__init__(
            duration_cuts, importance_sample_weights, sigma, num_events
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute the deephit ranking loss.

        Parameters:
            predictions (SAOutput: Predictions of the model (SAOutput: 5 x events x batch size x cuts)
            references (torch.Tensor): Reference values. (dims: batch size x 4)

        Returns:
            float: The loss value.
        """
        # Extract shape information
        batch_size = references.shape[0]
        n_events = self.num_events

        # Pre-compute common tensors
        events = self.events(references)
        durations = self.durations(references)

        # Extract prediction components
        survival = predictions.survival
        hazard = predictions.hazard

        # Initialize loss tensor
        loss = torch.tensor(0.0, device=references.device)

        # Process each event
        for i in range(n_events):
            # Extract event-specific data
            event_i = events[:, i].unsqueeze(1)
            duration_i = durations[:, i].unsqueeze(1)

            # Skip events with no positive samples
            if not event_i.any():
                continue

            # Extract survival and hazard for this event
            survival_i = survival[:, i, :].unsqueeze(1)
            hazard_i = hazard[:, i, :].unsqueeze(1)

            # Create weights tensor for this event
            weights = torch.ones(
                (batch_size, batch_size, batch_size), device=references.device
            )

            # Compute loss for this event
            event_loss = self.ranking_loss(
                event_i, duration_i, survival_i, hazard_i, weights
            )

            # Add to total loss
            loss += event_loss * self.weights[i + 1]

        return loss
