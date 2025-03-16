"""Ranking-based loss functions for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd
import torch
import torch.nn.functional as F

from sat.models.tasks.heads import SAOutput
from sat.loss.base import Loss


class RankingLoss(Loss):
    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        sigma: float = 1.0,
        num_events: int = 1,
        balance_strategy: str = "none",
        ema_momentum: float = 0.9,
    ):
        super(RankingLoss, self).__init__(
            num_events, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )

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
            )  # hazard rate for time interval (n x e)

        # Pre-compute some values used in multiple places
        durations_expanded = durations.view(n, e, 1).expand(n, e, e)
        mask_i = durations_expanded.unsqueeze(3).expand(n, e, e, e)
        mask_j = durations.view(n, 1, e, 1).expand(n, e, e, e)

        # Main computation - handle different cases with optimized operations
        eta_i = survival.unsqueeze(2).expand(n, e, e, -1)  # (n x e x e x tn)
        eta_j = survival.unsqueeze(1).expand(n, e, e, -1)  # (n x e x e x tn)

        # Calculate the maximum between 1 and eta_i / eta_j
        # Use element-wise operations for better efficiency
        ratio = eta_i / (eta_j + 1e-6)  # Adding epsilon for numerical stability
        risk_ratio = torch.maximum(ratio, torch.ones_like(ratio))  # (n x e x e x tn)

        # Transform the mask variables for optimizations
        I_matrix = I.unsqueeze(2).expand(n, e, e)  # (n x e x e)
        I_censored_matrix = I_censored.unsqueeze(2).expand(n, e, e)  # (n x e x e)

        # Create more efficient masks with precomputed durations
        censored_i_smaller = (
            I_censored_matrix
            & (durations_expanded <= durations.unsqueeze(1).expand(n, e, e))
            & (durations.unsqueeze(1).expand(n, e, e) > 0.0)
        )  # (n x e x e)

        events_i_smaller = I_matrix & (
            durations_expanded < durations.unsqueeze(1).expand(n, e, e)
        )  # (n x e x e)

        events_equal = I_matrix & (
            durations_expanded == durations.unsqueeze(1).expand(n, e, e)
        )  # (n x e x e)

        # Handle different cases efficiently
        # Case 1: i censored and j censored - no information, skip
        # Case 2: i censored and j observed, with tau_i < tau_j
        loss_case2 = torch.zeros(n, e, e, device=device)
        if censored_i_smaller.any():
            for i_event in range(e):
                for j_event in range(e):
                    idx = censored_i_smaller[:, i_event, j_event]
                    if idx.any():
                        sampled_idx = t0Index[idx, i_event]
                        loss_case2[idx, i_event, j_event] = torch.sum(
                            torch.log(
                                torch.gather(
                                    eta_i[idx, i_event, j_event],
                                    1,
                                    sampled_idx.unsqueeze(1),
                                )
                                / (
                                    torch.gather(
                                        eta_j[idx, i_event, j_event],
                                        1,
                                        sampled_idx.unsqueeze(1),
                                    )
                                    + 1e-6
                                )
                            )
                            * weights[idx, i_event]
                        )

        # Case 3: i observed and j observed/censored with tau_i < tau_j
        loss_case3 = torch.zeros(n, e, e, device=device)
        if events_i_smaller.any():
            for i_event in range(e):
                for j_event in range(e):
                    idx = events_i_smaller[:, i_event, j_event]
                    if idx.any():
                        sampled_idx = t0Index[idx, i_event]
                        loss_case3[idx, i_event, j_event] = torch.sum(
                            -hstar[idx, i_event, j_event].squeeze(-1)
                            + torch.log(
                                risk_ratio[idx, i_event, j_event].gather(
                                    1, sampled_idx.unsqueeze(1)
                                )
                            ).squeeze()
                            * weights[idx, i_event]
                        )

        # Case 4: i observed and j observed with tau_i = tau_j
        loss_case4 = torch.zeros(n, e, e, device=device)
        if events_equal.any():
            for i_event in range(e):
                for j_event in range(e):
                    if i_event == j_event:
                        # Same events - skip
                        continue

                    idx = events_equal[:, i_event, j_event]
                    if idx.any():
                        sampled_idx = t0Index[idx, i_event]
                        loss_case4[idx, i_event, j_event] = torch.sum(
                            torch.log(
                                1
                                / (
                                    1
                                    + torch.exp(
                                        self.sigma
                                        * (
                                            hstar[idx, i_event, j_event].squeeze(-1)
                                            - hstar[idx, j_event, i_event].squeeze(-1)
                                        )
                                    )
                                )
                            )
                            * weights[idx, i_event]
                        )

        # Combine all losses efficiently
        total_loss = -(loss_case2.sum() + loss_case3.sum() + loss_case4.sum())
        # Count valid pairs for normalization
        num_valid_pairs = (
            censored_i_smaller.sum() + events_i_smaller.sum() + events_equal.sum()
        )

        # Normalize if we have valid pairs
        if num_valid_pairs > 0:
            total_loss = total_loss / num_valid_pairs

        return total_loss

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute a loss.

        Parameters:
            predictions (SAOutput: Predictions of the model (SAOutput: 5 x events x batch size x cuts)
            references (torch.Tensor): Reference values. (dims: batch size x 4)

        Returns:
            float: The loss value.
        """
        # variables x batch x events x duration cuts
        logits = predictions.logits
        device = logits.device
        batch_size = logits.shape[0]

        # apply softplus to convert the logits into positive hazard values
        hazard = F.softplus(logits)

        # compute the survival function for each entry in the batch, each event
        # the survival function is a function of time (from the cuts in the discretised
        # piecewise constant hazard function, so we are computing the probabilities
        # for each of these time points)
        # survial time is determined as S(t) = exp(-\sum_{i=0}^t h(t_i))
        survival = torch.exp(-torch.cumsum(hazard, dim=2))  # (batch x events x cuts)
        # We also need S(t=0) = 1, so prepend this to the survival curve
        survival = torch.cat(
            [torch.ones(batch_size, self.num_events, 1, device=device), survival], dim=2
        )  # (batch x events x (cuts + 1))

        # flatten the first two dimensions to create (batch * events) for the ranking loss
        events = (
            self.events(references)
            .view(batch_size, self.num_events)
            .unsqueeze(1)
            .expand(batch_size, self.num_events, self.num_events)
        )  # (batch x events x events)
        durations = (
            self.durations(references)
            .view(batch_size, self.num_events)
            .unsqueeze(1)
            .expand(batch_size, self.num_events, self.num_events)
        )  # (batch x events x events)

        # Now some events may not be included, if that's the case (event == -1), let's
        # create a weight matrix to ignore those positions and the associated durations.
        weights = (events >= 0).float()  # (batch x events x events)

        loss = self.ranking_loss(events, durations, survival, hazard, weights)
        return loss


class SampleRankingLoss(RankingLoss):
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
        device = predictions.logits.device

        # create samples as per the approach in Deep Hit
        # we will create batch_size / 2 permutations for the ranking loss
        bsz = predictions.logits.shape[0]
        idx1 = torch.randperm(bsz, device=device)

        # create a second random permutation for the pairing
        idx2 = torch.randperm(bsz, device=device)
        # extract events and durations from the reference array
        events1 = self.events(references[idx1]).to(device).view(bsz, self.num_events)
        events2 = self.events(references[idx2]).to(device).view(bsz, self.num_events)
        durations1 = (
            self.durations(references[idx1]).to(device).view(bsz, self.num_events)
        )
        durations2 = (
            self.durations(references[idx2]).to(device).view(bsz, self.num_events)
        )

        batch1 = torch.arange(0, bsz, device=device).long()

        # We're going to compute a loss for each event
        loss = 0.0
        for i in range(self.num_events):
            e1 = events1[:, i].view(bsz, 1).expand(bsz, self.num_events)
            e2 = events2[:, i].view(bsz, 1).expand(bsz, self.num_events)
            d1 = durations1[:, i].view(bsz, 1).expand(bsz, self.num_events)
            d2 = durations2[:, i].view(bsz, 1).expand(bsz, self.num_events)

            # apply softplus to convert the logits into positive hazard values
            hazard = F.softplus(predictions.logits)  # (batch x events x cuts)

            # compute the survival function for each entry in the batch, each event
            # the survival function is a function of time (from the cuts in the discretised
            # piecewise constant hazard function, so we are computing the probabilities
            # for each of these time points)
            # survial time is determined as S(t) = exp(-\sum_{i=0}^t h(t_i))
            survival = torch.exp(
                -torch.cumsum(hazard, dim=2)
            )  # (batch x events x cuts)
            # We also need S(t=0) = 1, so prepend this to the survival curve
            survival = torch.cat(
                [torch.ones(bsz, self.num_events, 1, device=device), survival], dim=2
            )  # (batch x events x (cuts + 1))

            # create all pairs as (e1, e2, ...), where we will use i for pairing of events
            e = torch.cat([e1, e2], dim=1)  # (batch x (e1, e2, ...))
            d = torch.cat([d1, d2], dim=1)  # (batch x (e1, e2, ...))

            s1 = survival[batch1, i]  # (batch, cuts + 1)
            s2 = survival[batch1, i]  # (batch, cuts + 1)
            h1 = hazard[batch1, i]  # (batch, cuts)
            h2 = hazard[batch1, i]  # (batch, cuts)

            # (idx1, idx2) and (idx2, idx1) will both be considered
            # but we can share the work by reusing the arrays
            weights = (e >= 0).float()  # Ignore events when event == -1
            loss += self.ranking_loss(
                e, d, survival[:, i : i + 1], hazard[:, i : i + 1], weights
            )

        return loss / self.num_events


class MultiEventRankingLoss(RankingLoss):
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
        """Compute a loss.

        Parameters:
            predictions (SAOutput: Predictions of the model (SAOutput: 5 x events x batch size x cuts)
            references (torch.Tensor): Reference values. (dims: batch size x 4)

        Returns:
            float: The loss value.
        """
        # variables x batch x events x duration cuts
        logits = predictions.logits
        device = logits.device
        batch_size = logits.shape[0]

        # apply softplus to convert the logits into positive hazard values
        hazard = F.softplus(logits)

        # compute the survival function for each entry in the batch, each event
        # the survival function is a function of time (from the cuts in the discretised
        # piecewise constant hazard function, so we are computing the probabilities
        # for each of these time points)
        # survial time is determined as S(t) = exp(-\sum_{i=0}^t h(t_i))
        survival = torch.exp(-torch.cumsum(hazard, dim=2))  # (batch x events x cuts)
        # We also need S(t=0) = 1, so prepend this to the survival curve
        survival = torch.cat(
            [torch.ones(batch_size, self.num_events, 1, device=device), survival], dim=2
        )  # (batch x events x (cuts + 1))

        # extract events and durations from the reference array
        events = self.events(references).to(device).view(batch_size, self.num_events)
        durations = (
            self.durations(references).to(device).view(batch_size, self.num_events)
        )

        # We're going to compute a loss for each event
        loss = 0.0
        count = 0
        for i in range(self.num_events):
            for j in range(i + 1, self.num_events):
                # Skip when we don't have enough valid pairs
                if torch.sum(events[:, i] * events[:, j]) == 0:
                    continue

                # create all pairs as (ei, ej), where we will use i for pairing of events
                e = torch.stack([events[:, i], events[:, j]], dim=1)  # (batch x 2)
                d = torch.stack(
                    [durations[:, i], durations[:, j]], dim=1
                )  # (batch x 2)

                # Expand events and durations for ranking loss
                e = e.unsqueeze(1).expand(batch_size, 2, 2)
                d = d.unsqueeze(1).expand(batch_size, 2, 2)

                # Combine survival and hazard for events i and j
                s = torch.stack([survival[:, i], survival[:, j]], dim=1)
                h = torch.stack([hazard[:, i], hazard[:, j]], dim=1)

                # Create weights to ignore events when event == -1
                weights = (e >= 0).float()

                loss += self.ranking_loss(e, d, s, h, weights)
                count += 1

        if count > 0:
            return loss / count
        else:
            return torch.tensor(0.0, device=device)
