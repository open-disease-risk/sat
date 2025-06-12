"""Loss base class for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

from typing import Dict, List, Optional, Union

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from sat.utils import logging

from .balancing import BalancingStrategy, LossBalancer

logger = logging.get_default_logger()


class Loss(nn.Module):
    """Base class for losses."""

    def __init__(
        self,
        num_events: int = 1,
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
    ):
        super(Loss, self).__init__()
        self.num_events = num_events
        self.balance_strategy = balance_strategy
        self.balance_params = balance_params or {}
        self._balancer = None  # Will be initialized when needed

    def get_balancer(
        self, num_losses: int = 1, coeffs: Optional[List[float]] = None
    ) -> LossBalancer:
        """
        Get or initialize a loss balancer.

        Args:
            num_losses: Number of losses to balance
            coeffs: Initial coefficients for fixed weighting strategy

        Returns:
            Configured loss balancer
        """
        if self._balancer is None:
            self._balancer = LossBalancer.create(
                strategy=self.balance_strategy,
                num_losses=num_losses,
                coeffs=coeffs,
                **self.balance_params,
            )
        return self._balancer

    def get_loss_weights(self) -> List[float]:
        """
        Get current loss weights if a balancer is active.

        Returns:
            List of current loss weights or [1.0] if no balancer is active
        """
        if self._balancer is not None:
            return self._balancer.get_weights()
        return [1.0]

    def durations(self, references: torch.Tensor):
        return references[
            :, (3 * self.num_events) : (3 * self.num_events + self.num_events)
        ].float()

    def duration_percentiles(self, references: torch.Tensor):
        return references[:, 0 : self.num_events].long()

    def events(self, references: torch.Tensor):
        return references[
            :, (1 * self.num_events) : (1 * self.num_events + self.num_events)
        ].long()

    def fraction_with_quantile(self, references: torch.Tensor):
        return references[
            :, (2 * self.num_events) : (2 * self.num_events + self.num_events)
        ].float()

    def survivals(self, predictions):
        surv = []
        for logits in predictions:
            hazard = F.softplus(logits)
            surv.append(hazard.cumsum(1).mul(-1).exp()[:, :-1])
        return surv

    def ensure_tensor(
        self, value: Union[float, int, torch.Tensor], device=None
    ) -> torch.Tensor:
        """
        Ensures that a loss value is a proper PyTorch tensor.

        Args:
            value: Loss value that might be a Python scalar or tensor
            device: Optional device to place the tensor on

        Returns:
            PyTorch tensor representation of the value
        """
        if isinstance(value, torch.Tensor):
            return value

        # Get the device from the class parameters if not specified
        if device is None:
            # Try to find a registered buffer or parameter to get its device
            for _, buffer in self._buffers.items():
                if buffer is not None and isinstance(buffer, torch.Tensor):
                    device = buffer.device
                    break

            if device is None:
                for _, param in self._parameters.items():
                    if param is not None:
                        device = param.device
                        break

        # Convert scalar to tensor
        if device is not None:
            return torch.tensor(float(value), device=device)
        return torch.tensor(float(value))


class RankingLoss(Loss):
    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        sigma: float = 1.0,
        num_events: int = 1,
        margin: float = 0.0,  # Added margin parameter with default of 0 (no margin)
    ):
        super(RankingLoss, self).__init__(num_events)

        self.sigma = sigma
        self.margin = margin  # Store margin value

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
        self.num_time_bins = len(df.cuts)

    def ranking_loss(
        self, events, durations, survival, hazard, weights
    ) -> torch.Tensor:
        """
        Efficient implementation of ranking loss with vectorized operations.

        Args:
            events: Event indicators (dims: batch_size x num_events)
            durations: Event times (dims: batch_size x num_events)
            survival: Survival probabilities (dims: batch_size x num_events x num_time_bins+1)
            hazard: Hazard values (dims: batch_size x num_events x num_time_bins)
            weights: Importance weights (dims: batch_size x num_events-1 x num_events or None)

        Returns:
            torch.Tensor: The computed loss value
        """
        device = events.device
        n = events.shape[0]  # Batch size
        e = events.shape[1]  # Number of events

        # Create event masks once
        i_event = events.to(bool)  # Event indicator
        I_censored = ~i_event  # censored indicator (n x e)

        # Initialize duration cut points tensor efficiently
        T = self.duration_cuts.to(device).expand(n, e, -1)  # (n x e x tn)

        # Compute indices for time intervals - done once and reused
        durations_expanded = durations.unsqueeze(2)  # (n x e x 1)
        cuts_expanded = self.duration_cuts.to(device).view(1, 1, -1)  # (1 x 1 x tn)
        indexSmaller = cuts_expanded <= durations_expanded  # (n x e x tn)

        # Calculate left and right boundary indices
        t0Index = torch.sum(indexSmaller, dim=2) - 1  # (n x e)

        # Fix negative indices (for durations smaller than all cuts)
        t0Index = torch.clamp(t0Index, min=0)

        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)  # (n x e x e)
        t1Index = t0Index + 1  # (n x e x e)

        # Fix out of bounds indices
        max_idx = len(self.duration_cuts) - 1
        fixOOB = t1Index >= len(self.duration_cuts)
        t1Index[fixOOB] = max_idx

        # Gather time values efficiently
        T0 = torch.gather(T, 2, t0Index)  # (n x e x e)
        T1 = torch.gather(T, 2, t1Index)  # (n x e x e)

        # Gather survival and hazard values efficiently
        SatT0 = torch.gather(survival, 2, t0Index)  # (n x e x e)
        SatT1 = torch.gather(survival, 2, t1Index)  # (n x e x e)
        hstar = torch.gather(hazard, 2, t0Index)  # (n x e x e)

        # Calculate time differences
        dT = T1 - T0  # (n x e x e)

        # Handle interpolation for hazard
        positive_mask = torch.gt(dT, 0.0)

        # Use masked operations to avoid unnecessary calculations
        if positive_mask.any():
            # Add small epsilon for numerical stability
            epsilon = 1e-6
            log_SatT0 = torch.log(SatT0[positive_mask] + epsilon)
            log_SatT1 = torch.log(SatT1[positive_mask] + epsilon)
            hstar[positive_mask] = (log_SatT0 - log_SatT1) / dT[positive_mask]

        # Calculate survival at specific times
        durations_tiled = durations.unsqueeze(1).repeat(1, e, 1)  # (n x e x e)
        SatT = SatT0 * torch.exp(-(durations_tiled - T0) * hstar)  # (n x e x e)

        # Calculate epsilon time for survival computation
        t_epsilon = (
            self.duration_cuts[-1] - self.duration_cuts[0]
        ) / self.duration_cuts[-1]
        TMinus = torch.nn.functional.relu(durations_tiled - t_epsilon)  # (n x e x e)

        # Calculate survival at t-epsilon
        SatTMinus = SatT0 * torch.exp(-(TMinus - T0) * hstar)  # (n x e x e)

        # Extract diagonals efficiently
        diag_S = (
            torch.diagonal(SatT, dim1=1, dim2=2).unsqueeze(2).repeat(1, 1, e)
        )  # (n x e x e)
        diag_S2 = (
            torch.diagonal(SatTMinus, dim1=1, dim2=2).unsqueeze(2).repeat(1, 1, e)
        )  # (n x e x e)

        # Calculate survival differences
        SatT_T = torch.transpose(SatT, 1, 2)  # (n x e x e)
        diag_S2_T = torch.transpose(diag_S2, 1, 2)  # (n x e x e)
        diag_S_T = torch.transpose(diag_S, 1, 2)  # (n x e x e)

        dS1 = diag_S - SatT_T  # (n x e x e)
        dS2 = SatTMinus - diag_S2_T  # (n x e x e)
        dS3 = SatT - diag_S_T  # (n x e x e)

        # Create comparison masks efficiently
        durations_i = durations.unsqueeze(1).repeat(1, e, 1)  # (n x e x e)
        durations_j = durations.unsqueeze(2).repeat(1, 1, e)  # (n x e x e)
        comp = torch.sign(durations_i - durations_j)  # (n x e x e)

        # Apply ReLU to keep only positive values
        comp_pos = torch.nn.functional.relu(comp)  # (n x e x e)

        # Create event masks efficiently
        I_expanded = i_event.unsqueeze(2).repeat(1, 1, e).float()  # (n x e x e)
        I_T = i_event.unsqueeze(1).repeat(1, e, 1).float()  # (n x e x e)
        I_censored_T = I_censored.unsqueeze(1).repeat(1, e, 1).float()  # (n x e x e)

        # Create ranking pair masks efficiently
        A1 = I_expanded * comp_pos  # (n x e x e)
        A2 = A1 * I_T  # (n x e x e)
        A3 = A1 * I_censored_T  # (n x e x e)

        # Apply margin-based ranking penalties if margin > 0
        if hasattr(self, "margin") and self.margin > 0:
            # Calculate margin penalties where applicable
            margin_dS1 = torch.clamp(self.margin - dS1, min=0.0) * A1
            margin_dS2 = torch.clamp(self.margin - dS2, min=0.0) * A2
            margin_dS3 = torch.clamp(self.margin - dS3, min=0.0) * A3

            # Combine exponential and margin components
            loss_dS1 = torch.exp(dS1 / self.sigma) + margin_dS1
            loss_dS2 = torch.exp(dS2 / self.sigma) + margin_dS2
            loss_dS3 = torch.exp(dS3 / self.sigma) + margin_dS3
        else:
            # Traditional loss using only exponential scaling
            loss_dS1 = torch.exp(dS1 / self.sigma)
            loss_dS2 = torch.exp(dS2 / self.sigma)
            loss_dS3 = torch.exp(dS3 / self.sigma)

        # Apply weights if provided
        if weights is not None:
            # Ensure weights have appropriate device
            weights = weights.to(device)
            loss_term = weights * (A1 * loss_dS1 + A2 * loss_dS2 + A3 * loss_dS3)
        else:
            loss_term = A1 * loss_dS1 + A2 * loss_dS2 + A3 * loss_dS3

        # Calculate mean efficiently
        # Count number of non-zero elements for proper normalization
        num_valid = torch.sum((A1 + A2 + A3) > 0)

        if num_valid > 0:
            eta = torch.sum(loss_term) / num_valid
        else:
            # Return zero tensor with gradient if no valid comparisons
            eta = torch.tensor(0.0, device=device, requires_grad=True)

        return eta
