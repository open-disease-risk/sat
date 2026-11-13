"""Loss functions for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd

import torch
import torch.nn.functional as F

from torch import nn
from transformers.utils import ModelOutput
from typing import List

from sat.pycox.models.loss import NLLPCHazardLoss
from sat.utils import logging
from sat.models.tasks.heads import SAOutput, TaskOutput
from sat.utils.km import KaplanMeierArea

logger = logging.get_default_logger()


class Loss(nn.Module):
    """Base class for losses."""

    def __init__(self, num_events: int = 1):
        super(Loss, self).__init__()
        self.num_events = num_events

    def durations(self, references: torch.Tensor):
        # Efficient slice operation with pre-defined start and end indices
        start_idx = 3 * self.num_events
        end_idx = start_idx + self.num_events
        return references[:, start_idx:end_idx].float()

    def duration_percentiles(self, references: torch.Tensor):
        # Use direct indexing for better efficiency
        return references[:, : self.num_events].long()

    def events(self, references: torch.Tensor):
        # Efficient slice operation with pre-defined start and end indices
        start_idx = self.num_events
        end_idx = start_idx + self.num_events
        return references[:, start_idx:end_idx].long()

    def fraction_with_quantile(self, references: torch.Tensor):
        # Efficient slice operation with pre-defined start and end indices
        start_idx = 2 * self.num_events
        end_idx = start_idx + self.num_events
        return references[:, start_idx:end_idx].float()

    def survivals(self, predictions):
        # Pre-allocate list with known size for better memory efficiency
        surv = [None] * len(predictions)

        # Process each prediction efficiently
        for i, logits in enumerate(predictions):
            # Apply operations in-place where possible
            hazard = F.softplus(logits)
            # Use inplace operations for better memory efficiency
            surv[i] = hazard.cumsum(dim=1).neg_().exp_()[:, :-1]
        return surv


class MetaLoss(Loss):
    """A loss class that linearly combines multiple loss components."""

    def __init__(self, losses: list, coeffs: list):
        super(MetaLoss, self).__init__()

        self.losses = nn.ModuleList(losses)
        self.register_buffer("coeffs", torch.tensor(coeffs).to(torch.float32))

    def forward(
        self, predictions: ModelOutput, references: torch.Tensor
    ) -> torch.Tensor:
        l = 0.0
        for i, loss in enumerate(self.losses):
            l += self.coeffs[i] * loss(predictions, references)

        return l


class SATNLLPCHazardLoss(Loss):
    def __init__(
        self,
        importance_sample_weights: str = None,
        num_events: int = 1,
    ):
        super(SATNLLPCHazardLoss, self).__init__(num_events)

        self.loss_fct = NLLPCHazardLoss(reduction="none")

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

    def nllp_hazard_loss(self, predictions, references, event_type) -> torch.Tensor:
        # Directly get events as bool to avoid unnecessary conversion later
        events = self.events(references)[:, event_type].bool()
        duration_percentiles = self.duration_percentiles(references)[:, event_type]
        fraction_duration = self.fraction_with_quantile(references)[:, event_type]
        predictions = predictions[:, event_type]

        # Apply loss and weight in one efficient operation
        event_loss = self.loss_fct(
            predictions,
            duration_percentiles,
            events,
            fraction_duration,
        )

        # Use a single operation with the weight factor
        return event_loss.mean() * self.weights[event_type + 1]

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

        # Pre-calculate common tensors that we'll need for each event
        # For single event case, avoid the loop
        if self.num_events == 1:
            return self.nllp_hazard_loss(logits, references, 0)

        # For multiple events, use accumulation
        loss = 0.0
        for i in range(self.num_events):
            loss += self.nllp_hazard_loss(logits, references, i)

        return loss


class RankingLoss(Loss):
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

        # Compute indicator matrices more efficiently using expand instead of repeat
        # and reusing pre-computed tensors

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
        # Extract events once to avoid repeated calls
        events = self.events(references)

        # Get batch size and number of events
        n = events.shape[0]
        e = events.shape[1]

        # Cache durations to avoid recomputation
        durations = self.durations(references)

        # Efficient weight expansion:
        # 1. Extract weights slice only once
        weights_slice = self.weights[1:]

        # 2. Pre-allocate and expand tensor in one step when possible
        # This is more efficient than sequential operations
        weights_expanded = weights_slice.expand(n, e, e).transpose(1, 2)

        # Call ranking loss with the prepared inputs
        eta = self.ranking_loss(
            events,
            durations,
            predictions.survival,
            predictions.hazard,
            weights_expanded,
        )
        return eta


class MSELoss(Loss):
    """MSE Loss."""

    def __init__(
        self,
        training_set: str,
        importance_sample_weights: str = None,
        l2_type: str = "uncensored",
        num_events: int = 1,
    ):
        super(MSELoss, self).__init__(num_events)

        self.l2_type = l2_type
        self.kms: List[KaplanMeierArea] = []

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

        if l2_type == "margin":
            if training_set is None:
                raise ValueError(
                    "If 'margin' is chosen, training set values must be included."
                )

            logger.debug("Train the Kaplan Meier Curves")
            # read training data into pandas dataframe with given columns
            df = pd.read_csv(training_set, header=0)
            for event in range(self.num_events):
                training_event_times = df["duration"]
                training_event_indicators = df["event"] == event + 1
                self.kms.append(
                    KaplanMeierArea(training_event_times, training_event_indicators)
                )

    def mse(
        self,
        predictions,
        references,
        event_type: int,
    ) -> torch.Tensor:
        device = references.device
        # Convert directly to bool instead of using to(bool) for better efficiency
        event_indicators = self.events(references)[:, event_type].bool()
        durations = self.durations(references)[:, event_type]
        predictions = predictions[:, event_type]

        if self.l2_type == "uncensored":
            # Use tensor operations directly without creating intermediate tensors
            if event_indicators.any():
                scores = durations[event_indicators] - predictions[event_indicators]
                # Use square() method for better inplace operation
                loss = scores.square().mean()
            else:
                # Handle empty case efficiently
                loss = torch.tensor(0.0, device=device)
        elif self.l2_type == "margin":
            # Pre-compute once to avoid repeated calculations
            not_event_indicators = ~event_indicators

            # Process censored samples more efficiently
            if not_event_indicators.any():
                censor_times = durations[not_event_indicators]
                # Minimize CPU-GPU transfer by batching operations
                censor_np = censor_times.detach().cpu().numpy()
                weights = torch.tensor(
                    1.0 - self.kms[event_type].predict(censor_np), device=device
                )
                best_guesses = torch.tensor(
                    self.kms[event_type].best_guess(censor_np), device=device
                )

                # Pre-allocate tensor only once with proper size and device
                scores = torch.empty_like(predictions)

                # Check if we have any events before indexing
                if event_indicators.any():
                    scores[event_indicators] = (
                        durations[event_indicators] - predictions[event_indicators]
                    )

                # Compute for non-events
                scores[not_event_indicators] = weights * (
                    best_guesses - predictions[not_event_indicators]
                )

                # More efficient sum calculations
                event_count = event_indicators.sum()
                weight_sum = weights.sum()

                # Compute weighted multiplier directly on device
                total_weight = event_count + weight_sum
                if total_weight > 0:
                    weighted_multiplier = 1.0 / total_weight
                    # Avoid unnecessary indexing with [0]
                    loss = weighted_multiplier * scores.square().mean()
                else:
                    loss = torch.tensor(0.0, device=device)
            else:
                # All events case
                scores = durations - predictions
                loss = scores.square().mean()
        else:
            raise ValueError("L2 type must be either 'uncensored' or 'margin'.")

        # Apply weight directly
        return self.weights[event_type + 1] * loss

    def forward(
        self, predictions: TaskOutput, references: torch.Tensor
    ) -> torch.Tensor:
        # Extract predictions once to avoid multiple attribute lookups
        pred_values = predictions.predictions

        # For single event case, avoid the loop for better performance
        if self.num_events == 1:
            return self.mse(pred_values, references, 0)

        # For multiple events, use efficient accumulation
        loss = 0.0
        for i in range(self.num_events):
            loss += self.mse(pred_values, references, i)

        return loss


class QuantileLoss(Loss):
    """Quantile Loss for Quantile Regression"""

    def __init__(
        self,
        quantiles: List[float],
        training_set: str,
        num_events: int,
        importance_sample_weights: str = None,
        l_type: str = "uncensored",
    ):
        super(QuantileLoss, self).__init__(num_events)

        self.l_type = l_type
        self.quantiles = quantiles
        self.num_events = num_events
        self.kms: List[KaplanMeierArea] = []

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

        if self.l_type == "margin":
            if training_set is None:
                raise ValueError(
                    "If 'margin' is chosen, training set values must be included."
                )

            logger.debug("Train the Kaplan Meier Curves")
            # read training data into pandas dataframe with given columns
            df = pd.read_csv(training_set, header=0)
            for event in range(self.num_events):
                training_event_times = df["duration"]
                training_event_indicators = df["event"] == event + 1
                self.kms.append(
                    KaplanMeierArea(training_event_times, training_event_indicators)
                )

    def quantile_loss(self, predictions, references, event_type) -> torch.Tensor:
        device = references.device

        # Extract event indicators and standardize variable name
        event_indicators = self.events(references)[:, event_type].bool()
        durations = self.durations(references)[:, event_type]
        predictions = predictions[:, event_type]

        if self.l_type == "uncensored":
            # Only process if there are events
            if event_indicators.any():
                # Pre-allocate scores tensor with correct size
                event_durations = durations[event_indicators]
                event_predictions = predictions[event_indicators]
                scores = torch.zeros_like(event_durations)

                # Process each quantile efficiently
                for i, q in enumerate(self.quantiles):
                    # Compute errors directly
                    errors = event_durations - event_predictions[:, i]
                    # Use torch.maximum for more efficient element-wise max
                    q_minus_1 = q - 1.0
                    scores += torch.maximum(q_minus_1 * errors, q * errors)

                # Calculate mean once at the end
                loss = scores.mean()
            else:
                # Handle empty case
                loss = torch.tensor(0.0, device=device)
        elif self.l_type == "margin":
            # Pre-compute masks
            non_event_indicators = ~event_indicators

            # Only perform expensive CPU operations if we have censored events
            if non_event_indicators.any():
                censor_times = durations[non_event_indicators]
                # Batch CPU operations by computing them once
                censor_np = censor_times.detach().cpu().numpy()

                # Create tensors directly on the correct device
                weights = torch.tensor(
                    1.0 - self.kms[event_type].predict(censor_np), device=device
                )
                best_guesses = torch.tensor(
                    self.kms[event_type].best_guess(censor_np), device=device
                )

                # Pre-allocate tensors once
                scores = torch.zeros_like(durations)
                errors = torch.empty_like(durations)

                # Process each quantile efficiently
                for i, q in enumerate(self.quantiles):
                    # Only compute for event indicators if we have any
                    if event_indicators.any():
                        errors[event_indicators] = (
                            durations[event_indicators]
                            - predictions[event_indicators][:, i]
                        )

                    # Compute for non-events
                    errors[non_event_indicators] = weights * (
                        best_guesses - predictions[non_event_indicators][:, i]
                    )

                    # Use torch.maximum for more efficient element-wise max
                    q_minus_1 = q - 1.0
                    scores += torch.maximum(q_minus_1 * errors, q * errors)

                # Calculate weights efficiently
                event_count = event_indicators.sum()
                weights_sum = weights.sum()
                total_weight = event_count + weights_sum

                if total_weight > 0:
                    weighted_multiplier = 1.0 / total_weight
                    # Avoid creating an unnecessary tensor and indexing with [0]
                    loss = weighted_multiplier * scores.mean()
                else:
                    loss = torch.tensor(0.0, device=device)
            else:
                # All events are event_indicators = True
                scores = torch.zeros_like(durations)

                for i, q in enumerate(self.quantiles):
                    errors = durations - predictions[:, i]
                    q_minus_1 = q - 1.0
                    scores += torch.maximum(q_minus_1 * errors, q * errors)

                loss = scores.mean()
        else:
            raise ValueError("L type must be either 'uncensored' or 'margin'.")

        # Apply weight directly
        return self.weights[event_type + 1] * loss

    def forward(
        self, predictions: TaskOutput, references: torch.Tensor
    ) -> torch.Tensor:
        # Extract predictions once to avoid repeated attribute lookups
        pred_values = predictions.predictions

        # For single event case, avoid the loop overhead
        if self.num_events == 1:
            return self.quantile_loss(pred_values, references, 0)

        # For multiple events, use efficient accumulation
        loss = 0.0
        for i in range(self.num_events):
            loss += self.quantile_loss(pred_values, references, i)

        return loss


class CrossEntropyLoss(Loss):
    """Cross Entropy Loss."""

    def __init__(
        self,
        event_time_thr: float,
        training_set: str,
        num_events: int,
        l_type: str = "margin",
        importance_sample_weights: str = None,
    ):
        super(CrossEntropyLoss, self).__init__(num_events)

        self.loss_func = nn.BCELoss(reduction="none")
        self.num_events = num_events
        self.event_time_thr = event_time_thr
        self.l_type = l_type
        self.kms: List[KaplanMeierArea] = []

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

        if l_type == "margin":
            if training_set is None:
                raise ValueError(
                    "If 'margin' is chosen, training set values must be included."
                )

            logger.debug("Train the Kaplan Meier Curves")
            # read training data into pandas dataframe with given columns
            df = pd.read_csv(training_set, header=0)
            for event in range(self.num_events):
                training_event_times = df["duration"]
                training_event_indicators = df["event"] == event + 1
                self.kms.append(
                    KaplanMeierArea(training_event_times, training_event_indicators)
                )

    def ce(self, predictions, references, event_type) -> torch.Tensor:
        device = references.device
        durations = self.durations(references)[:, event_type]
        # Use bool() directly instead of to(bool) for better efficiency
        event_indicator = self.events(references)[:, event_type].bool()

        if self.l_type == "uncensored":
            # survival times less than T - compute once
            event_occurred = durations <= self.event_time_thr

            # Efficiently compute targets - convert to float once at the end
            targets = (event_occurred & event_indicator).float()

            # More efficient computation using & operator instead of logical_or
            # for bitwise operations on boolean tensors
            relevant_for_classification = ~event_occurred | event_indicator

            # Only compute if there are relevant samples
            if relevant_for_classification.any():
                # Extract indices once for better indexing performance
                relevant_indices = relevant_for_classification.nonzero(as_tuple=True)[0]

                # Use direct indexing for better performance
                preds = (
                    predictions[:, event_type]
                    .index_select(0, relevant_indices)
                    .squeeze(-1)
                )
                targets = targets.index_select(0, relevant_indices)

                # Compute loss directly with weight factor
                loss = (
                    self.loss_func(preds, targets).mean() * self.weights[event_type + 1]
                )
            else:
                # Handle empty case
                loss = torch.zeros(1, device=device)

        elif self.l_type == "margin":
            # Pre-compute not_event_indicator once
            not_event_indicator = ~event_indicator

            # Only do expensive CPU operations if we have censored samples
            if not_event_indicator.any():
                censor_times = durations[not_event_indicator]
                # Perform CPU operations in batch
                censor_np = censor_times.detach().cpu().numpy()
                # Create tensors directly on the device
                weights = torch.tensor(
                    1.0 - self.kms[event_type].predict(censor_np), device=device
                )
                best_guesses = torch.tensor(
                    self.kms[event_type].best_guess(censor_np), device=device
                )

                # More efficient event_occurred calculation - preallocate with correct type
                event_occurred = torch.zeros_like(durations, dtype=torch.bool)

                # Use masks with indexing for better performance
                if event_indicator.any():
                    event_occurred[event_indicator] = (
                        durations[event_indicator] <= self.event_time_thr
                    )

                # Only process if we have censored events
                if not_event_indicator.any():
                    event_occurred[not_event_indicator] = (
                        best_guesses <= self.event_time_thr
                    )

                # Compute targets efficiently
                targets = (event_occurred & event_indicator).float()

                # Compute relevant classification indices more efficiently
                relevant_for_classification = ~event_occurred | event_indicator

                # Only proceed if there are relevant samples
                if relevant_for_classification.any():
                    # Use direct indexing with nonzero indices
                    relevant_indices = relevant_for_classification.nonzero(
                        as_tuple=True
                    )[0]
                    preds = (
                        predictions[:, event_type]
                        .index_select(0, relevant_indices)
                        .squeeze(-1)
                    )
                    targets = targets.index_select(0, relevant_indices)

                    # Pre-compute loss once
                    losses = self.loss_func(preds, targets)

                    # Apply weights to censored samples more efficiently
                    # Find which indices in relevant_indices correspond to not_event_indicator
                    not_event_mask = not_event_indicator.index_select(
                        0, relevant_indices
                    )
                    if not_event_mask.any():
                        losses[not_event_mask] *= weights

                    # More efficient weight computation
                    event_count = event_indicator.sum()
                    weight_sum = weights.sum()
                    total_weight = event_count + weight_sum

                    if total_weight > 0:
                        weighted_multiplier = 1.0 / total_weight
                        loss = (
                            self.weights[event_type + 1]
                            * weighted_multiplier
                            * losses.mean()
                        )
                    else:
                        loss = torch.zeros(1, device=device)
                else:
                    loss = torch.zeros(1, device=device)
            else:
                # All events are event_indicator = True
                event_occurred = durations <= self.event_time_thr
                targets = (event_occurred & event_indicator).float()
                relevant_for_classification = ~event_occurred | event_indicator

                if relevant_for_classification.any():
                    relevant_indices = relevant_for_classification.nonzero(
                        as_tuple=True
                    )[0]
                    preds = (
                        predictions[:, event_type]
                        .index_select(0, relevant_indices)
                        .squeeze(-1)
                    )
                    targets = targets.index_select(0, relevant_indices)
                    loss = (
                        self.loss_func(preds, targets).mean()
                        * self.weights[event_type + 1]
                    )
                else:
                    loss = torch.zeros(1, device=device)
        else:
            raise ValueError(f"Unknown l_type: {self.l_type}")

        return loss

    def forward(
        self, predictions: TaskOutput, references: torch.Tensor
    ) -> torch.Tensor:
        # Extract predictions once to avoid repeated attribute access
        pred_values = predictions.predictions

        # Special case for single event for better performance
        if self.num_events == 1:
            return self.ce(pred_values, references, 0)

        # For multiple events, use efficient accumulation
        loss = 0.0
        for event in range(self.num_events):
            loss += self.ce(pred_values, references, event)

        return loss


class L1Loss(Loss):
    """L1 loss"""

    def __init__(
        self,
        training_set: str,
        importance_sample_weights: str = None,
        l1_type: str = "hinge",
        num_events: int = 1,
    ):
        super(L1Loss, self).__init__(num_events)

        self.l1_type = l1_type
        self.kms: List[KaplanMeierArea] = []

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

        if l1_type == "margin":
            if training_set is None:
                raise ValueError(
                    "If 'margin' is chosen, training set values must be included."
                )

            logger.debug("Train the Kaplan Meier Curves")
            # read training data into pandas dataframe with given columns
            df = pd.read_csv(training_set, header=0)
            for event in range(self.num_events):
                training_event_times = df["duration"]
                training_event_indicators = df["event"] == event + 1
                self.kms.append(
                    KaplanMeierArea(training_event_times, training_event_indicators)
                )

    def l1(
        self,
        predictions,
        references,
        event_type: int,
    ) -> torch.Tensor:
        device = references.device

        # Use bool() directly instead of to(bool) for better performance
        event_indicators = self.events(references)[:, event_type].bool()
        durations = self.durations(references)[:, event_type]
        predictions = predictions[:, event_type]

        if self.l1_type == "uncensored":
            # Only process if there are events
            if event_indicators.any():
                scores = durations[event_indicators] - predictions[event_indicators]
                # Use abs() method directly for better performance
                loss = scores.abs().mean()
            else:
                loss = torch.zeros(1, device=device)
        elif self.l1_type == "hinge":
            # Pre-compute not_event_indicators once
            not_event_indicators = ~event_indicators

            # Compute scores more efficiently
            scores = durations - predictions

            # Use F.relu for more efficient max(0, x) operation if we have any non-events
            if not_event_indicators.any():
                scores[not_event_indicators] = F.relu(scores[not_event_indicators])

            # Use tensor method instead of torch function for better performance
            loss = scores.abs().mean()
        elif self.l1_type == "margin":
            # Pre-compute not_event_indicators once
            not_event_indicators = ~event_indicators

            # Only perform CPU operations if there are censored events
            if not_event_indicators.any():
                censor_times = durations[not_event_indicators]
                # Batch CPU operations to minimize GPU-CPU transfers
                censor_np = censor_times.detach().cpu().numpy()

                # Create tensors directly on the device
                weights = torch.tensor(
                    1.0 - self.kms[event_type].predict(censor_np), device=device
                )
                best_guesses = torch.tensor(
                    self.kms[event_type].best_guess(censor_np), device=device
                )

                # Allocate scores tensor only once
                scores = torch.empty_like(predictions)

                # Only compute for event indicators if we have any
                if event_indicators.any():
                    scores[event_indicators] = (
                        durations[event_indicators] - predictions[event_indicators]
                    )

                # Compute for non-events
                scores[not_event_indicators] = weights * (
                    best_guesses - predictions[not_event_indicators]
                )

                # More efficient sum calculations
                event_count = event_indicators.sum()
                weight_sum = weights.sum()

                # Calculate weighted multiplier directly
                total_weight = event_count + weight_sum
                if total_weight > 0:
                    weighted_multiplier = 1.0 / total_weight
                    # Use method chaining for better performance
                    loss = weighted_multiplier * scores.abs().sum()
                else:
                    loss = torch.zeros(1, device=device)
            else:
                # All events are event_indicator = True
                scores = durations - predictions
                loss = scores.abs().mean()
        else:
            raise ValueError("L1 type must be either 'hinge' or 'margin'.")

        # Apply weight in a single operation
        return self.weights[event_type + 1] * loss

    def forward(
        self, predictions: TaskOutput, references: torch.Tensor
    ) -> torch.Tensor:
        # Extract predictions once to avoid multiple attribute lookups
        pred_values = predictions.predictions

        # Optimize single event case to avoid loop overhead
        if self.num_events == 1:
            return self.l1(pred_values, references, 0)

        # For multiple events, use efficient accumulation
        loss = 0.0
        for event in range(self.num_events):
            loss += self.l1(pred_values, references, event)

        return loss


# not good practice -- I know -- but this is going to be refactored
# class ImprovedSurvtraceLoss(Loss):
#     """Improved Survtrace loss"""

#     def __init__(
#         self,
#         importance_sample_weights: str,
#     ):
#         device_str, device = get_device()
#         # load the importance sampling weights
#         df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
#         self.importance_sample_weights = (
#             torch.tensor(df.weights.values).to(torch.float32).to(device)
#         )

#         self.loss_fct = NLLPCHazardLoss(reduction="none")

#     def loss(self, predictions: SAOutput, references: torch.Tensor) -> float:
#         """Compute the NLLPC hazard loss.

#         This function implements a loss for both competing and single event
#         cases. It iterates through the events and masks every other event as
#         censored in turn.

#         Parameters:
#             predictions (SAOutput: Predictions of the model (SAOutput: 5 x events x batch size x cuts)
#             references (torch.Tensor): Reference values. (dims: batch size x 4)

#         Returns:
#             float: The loss value.

#         """
#         logits = predictions.logits
#         device_str, device = get_device()
#         num_events = len(logits)
#         events = self.events(references)

#         loss = 0.0
#         for i in range(num_events):
#             cond = torch.logical_or((events == (i + 1)), (events == 0))

#             loss += (
#                 self.loss_fct(
#                     logits[i] * cond[:, None],
#                     self.duration_percentiles(references) * cond,
#                     (events == (i + 1)),
#                     self.fraction_with_quantile(references) * cond,
#                 )
#                 * 1.0
#                 / (
#                     self.importance_sample_weights[events]
#                     * (
#                         num_events * (events == 0).to(torch.int)
#                         + (events != 0).to(torch.int)
#                     )
#                 )
#             ).mean()

#         return loss


# TODO: This needs to be refactored into the current loss template
class MismatchLoss(Loss):
    """Mismatch Loss Class."""

    def __init__(
        self,
        duration_cuts: torch.Tensor,
        max_time: float,
    ):
        super(MismatchLoss, self).__init__()
        self.duration_cuts = duration_cuts
        self.max_time = torch.tensor(max_time)

    def mean_lifetime(self, predictions, references) -> torch.Tensor:
        device = references.device
        num_events = predictions.shape[1]
        # The next two lines compute the expected event time, we consider that the
        # surv probability goes to zero at max_time
        time_intervals = torch.cat(
            (self.duration_cuts, self.max_time.unsqueeze(0)), 0
        ) - torch.cat((torch.tensor(0).to(device).unsqueeze(0), self.duration_cuts), 0)

        surv = self.survivals(predictions)
        mean_lifetimes = torch.zeros(num_events, surv[0].shape[0]).to(device)
        dummy = self.duration_cuts.expand(surv[0].shape).to(device)

        for i in range(num_events):
            mean_lifetimes[i, :] = (
                torch.sum(
                    time_intervals
                    * (
                        torch.cat(
                            (
                                surv[i],
                                torch.tensor(0)
                                .to(device)
                                .expand(dummy.shape[0])
                                .view(-1, 1),
                            ),
                            1,
                        )
                        + torch.cat(
                            (
                                torch.tensor(1)
                                .to(device)
                                .expand(dummy.shape[0])
                                .view(-1, 1),
                                surv[i],
                            ),
                            1,
                        )
                    ),
                    dim=1,
                )
                / 2
            )

            return mean_lifetimes

    def mismatch_loss(self, references, mean_lifetimes) -> torch.Tensor:
        device = references.device
        duration = self.durations(references)
        events = self.events(references)

        # Finding the first event, and cases where we have mismatch
        est_event = torch.argmin(mean_lifetimes, dim=0) + 1
        mismatch = (est_event != events) & (events != 0)

        # The following variables are defined to help mismatch loss computation
        mean_life_temp = mean_lifetimes[:, mismatch]
        est_event_temp = est_event[mismatch]
        event_temp = events[mismatch]
        mean_life_event = torch.zeros(event_temp.shape[0]).to(device)
        mean_life_est = torch.zeros(event_temp.shape[0]).to(device)

        for i in range(event_temp.shape[0]):
            mean_life_event[i] = mean_life_temp[
                event_temp[i] - 1, i
            ]  # Estimated time of actual event
            mean_life_est[i] = mean_life_temp[
                est_event_temp[i] - 1, i
            ]  # Estimated event time of wrong estimated event

        mismatch_loss = torch.mean(
            nn.ReLU()(duration[mismatch] - mean_life_est)
            + nn.ReLU()(mean_life_event - duration[mismatch])
            + mean_life_event
            - mean_life_est
        )

        return mismatch_loss

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute the mismatch loss.

        This function implements SurvTrace loss for both competing and single event cases
        Parameters:
            predictions (SAOutput: Predictions of the model (SAOutput: 5 x events x batch size x cuts)
            references (torch.Tensor): Reference values. (dims: batch size x 4)

        Returns:
            float: The loss value.
        """
        logits = predictions.logits
        mean_lifetimes = self.mean_lifetime(logits, references)
        logger.debug(f"Mean lifetimes {mean_lifetimes}")
        loss = self.mismatch_loss(references, mean_lifetimes)

        return loss
