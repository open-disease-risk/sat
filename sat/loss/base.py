"""Loss base class for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd
from typing import Optional, Dict, Union, List

import torch
import torch.nn.functional as F

from torch import nn

from sat.utils import logging
from .balancing import LossBalancer, BalancingStrategy

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
            for buffer_name, buffer in self._buffers.items():
                if buffer is not None and isinstance(buffer, torch.Tensor):
                    device = buffer.device
                    break

            if device is None:
                for param_name, param in self._parameters.items():
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
        device = events.device
        n = events.shape[0]
        e = events.shape[1]

        I = events.to(bool)
        I_censored = ~I  # censored indicator (n x e)

        T = self.duration_cuts.expand(n, e, -1)  # duration cut points (n x e x tn)
        indexSmaller = self.duration_cuts.view(1, 1, -1) <= durations.unsqueeze(2)
        t0Index = (
            torch.sum(indexSmaller, dim=2) - 1
        )  # left boundary of time interval (n x e)
        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)
        t1Index = t0Index + 1  # right boundary of time interval (n x e)

        # if we run out of bounds, we match t0Index this means that dT will be
        # zero and causes NaNs in hstar, which we need to fix
        fixOOB = t1Index == len(self.duration_cuts)
        t1Index[fixOOB] = len(self.duration_cuts) - 1

        T0 = torch.gather(
            T, 2, t0Index
        )  # left boundary of time interval for all events i and time constraints j (n x e x e)
        T1 = torch.gather(
            T, 2, t1Index
        )  # right boundary of time interval for all events i and time constraints j (n x e)

        SatT0 = torch.gather(survival, 2, t0Index)  # survival at T0 (n x e x e)
        SatT1 = torch.gather(survival, 2, t1Index)  # survival at T1 (n x e x e)
        hstar = torch.gather(hazard, 2, t0Index)  # hazard at T0 (n x e x e)

        dT = T1 - T0

        # when dT is zero or negative we know that the time duration for an
        # observation is greater or equal to the maximum duration cut. So, we
        # need to use the corresponding hazard rather than interpolating. Since
        # we initialized hstar with the hazard at T0, we only need to take care
        # of the valid interpolations below:
        positive_mask = torch.gt(dT, 0.0)
        hstar[positive_mask] = torch.div(
            torch.log(0.000001 + SatT0[positive_mask])
            - torch.log(0.000001 + SatT1[positive_mask]),
            (dT[positive_mask]),
        )  # solve for hazard given the survival at T0 and T1 (n x e x e)

        SatT = SatT0 * torch.exp(
            -(durations.unsqueeze(1).repeat(1, e, 1) - T0) * hstar
        )  # solve for survival at time t (n x e x e)

        # compute an epsilon time to be subtracted from t in order to compute
        # the survival at t-epsilon for when the event occurred for sample i and
        # j
        t_epsilon = (
            self.duration_cuts[-1] - self.duration_cuts[0]
        ) / self.duration_cuts[-1]
        TMinus = torch.nn.functional.relu(
            durations.unsqueeze(1).repeat(1, e, 1) - t_epsilon
        )
        SatTMinus = SatT0 * torch.exp(
            -(TMinus - T0) * hstar
        )  # solve for survival at time t-epsilon (n x e x e)

        # get the n inner diagonals of e x e and repeat column-wise
        diag_S = torch.diagonal(SatT, dim1=-2, dim2=-1).unsqueeze(2).repeat(1, 1, e)
        diag_S2 = (
            torch.diagonal(SatTMinus, dim1=-2, dim2=-1).unsqueeze(2).repeat(1, 1, e)
        )

        dS1 = diag_S - torch.transpose(
            SatT, 1, 2
        )  # dS_{ij} = S_{i}(T_{i}) - S_{j}(T_{i}) (n x e x e)
        dS2 = SatTMinus - torch.transpose(
            diag_S2, 1, 2
        )  # dS_{ij} = S_{i}(T_{j}-1) - S_{j}(T_{j}-1) (n x e x e)
        dS3 = SatT - torch.transpose(
            diag_S, 1, 2
        )  # dS_{ij} = S_{i}(T_{j}) - S_{j}(T_{j}) (n x e x e)

        # A_{nij}=1 if t_i < t_j and A_{ij}=0 if t_i >= t_j
        #              and A_{ij}=1 when event occured for subject i (n x e x e)
        A1 = I.unsqueeze(2).repeat(1, 1, e).float() * torch.nn.functional.relu(
            torch.sign(
                durations.unsqueeze(1).repeat(1, e, 1)
                - durations.unsqueeze(2).repeat(1, 1, e)
            )
        )
        A2 = (
            A1 * I.unsqueeze(1).repeat(1, e, 1).float()
        )  # and A_{ij}=1 when event occured for subject j (n x e x e)
        A3 = (
            A1 * I_censored.unsqueeze(1).repeat(1, e, 1).float()
        )  # and A_{ij}=1 when subject j is censored (n x e x e)

        # Apply margin-based loss component if margin > 0
        if hasattr(self, "margin") and self.margin > 0:
            # Apply margin to enforce minimum difference (only for valid ranking pairs)
            margin_dS1 = torch.clamp(self.margin - dS1, min=0.0) * A1
            margin_dS2 = torch.clamp(self.margin - dS2, min=0.0) * A2
            margin_dS3 = torch.clamp(self.margin - dS3, min=0.0) * A3

            # Combine exponential and margin components
            loss_dS1 = torch.exp(dS1 / self.sigma) + margin_dS1
            loss_dS2 = torch.exp(dS2 / self.sigma) + margin_dS2
            loss_dS3 = torch.exp(dS3 / self.sigma) + margin_dS3
        else:
            # Traditional DeepHit loss using only exponential scaling
            loss_dS1 = torch.exp(dS1 / self.sigma)
            loss_dS2 = torch.exp(dS2 / self.sigma)
            loss_dS3 = torch.exp(dS3 / self.sigma)

        eta = torch.mean(
            weights * (A1 * loss_dS1 + A2 * loss_dS2 + A3 * loss_dS3),
        )

        return eta  # (1 x 1)
