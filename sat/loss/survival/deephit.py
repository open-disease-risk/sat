"""DeepHit loss components for survival analysis with competing risks."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import pandas as pd
import torch
from typing import Dict, Optional, Union, List

from ..balancing import BalancingStrategy
from sat.models.heads import SAOutput
from sat.utils import logging

from ..base import Loss

logger = logging.get_default_logger()


class DeepHitLikelihoodLoss(Loss):
    """
    Likelihood loss component of DeepHit.

    Computes the negative log-likelihood for competing risks survival data,
    considering both event and censored observations.
    """

    def __init__(
        self,
        num_events: int = 1,
        importance_sample_weights: Optional[str] = None,
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
    ):
        """
        Initialize DeepHitLikelihoodLoss.

        Args:
            num_events: Number of competing events
            importance_sample_weights: Optional path to CSV file with importance weights
            balance_strategy: Strategy for balancing loss components
            balance_params: Additional parameters for the balancing strategy
        """
        super(DeepHitLikelihoodLoss, self).__init__(
            num_events=num_events,
            balance_strategy=balance_strategy,
            balance_params=balance_params,
        )

        # Load importance sampling weights if provided
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the negative log-likelihood loss component with optimized tensor operations.

        Args:
            predictions: Model predictions (SAOutput with survival probabilities)
            references: Ground truth references

        Returns:
            Negative log-likelihood loss
        """
        batch_size = predictions.logits.shape[0]
        device = references.device

        # Extract event information once
        events = self.events(references)  # [batch_size, num_events]
        duration_idx = self.duration_percentiles(references)  # [batch_size, num_events]

        # Create all event masks at once (more efficient than in-loop creation)
        event_masks = [(events[:, i] == 1) for i in range(self.num_events)]

        # Get hazard and survival values once
        hazard = predictions.hazard  # [batch_size, num_events, num_time_bins]
        survival = predictions.survival  # [batch_size, num_events, num_time_bins+1]

        # Compute negative log-likelihood for uncensored subjects
        uncensored_loss = torch.zeros(1, device=device)
        num_uncensored = 0

        for i in range(self.num_events):
            # Get mask for subjects with event type i
            mask = event_masks[i]
            mask_sum = mask.sum().item()  # Cache sum to avoid recomputation

            if mask_sum > 0:
                # Get indices of event times for subjects with event type i
                time_idx = duration_idx[mask, i]
                time_idx_unsqueezed = time_idx.unsqueeze(1)  # Reuse this tensor

                # Get hazard at event time for event type i
                event_hazards = hazard[mask, i, :]
                event_hazard_at_t = torch.gather(
                    event_hazards, 1, time_idx_unsqueezed
                ).squeeze(1)

                # Get survival up to event time for all event types (including i)
                event_survival = survival[mask, :, :]

                # For the specific event type i, get survival right before event
                event_survival_before_t = torch.gather(
                    event_survival[:, i, :], 1, time_idx_unsqueezed
                ).squeeze(1)

                # For all other event types, get survival at event time
                # Use in-place operations to reduce memory allocations
                time_indices = (time_idx + 1).unsqueeze(
                    1
                )  # +1 because survival includes time 0
                other_events_survival = torch.ones(mask_sum, device=device)

                for j in range(self.num_events):
                    if j != i:
                        other_event_survival_at_t = torch.gather(
                            event_survival[:, j, :], 1, time_indices
                        ).squeeze(1)
                        other_events_survival.mul_(
                            other_event_survival_at_t
                        )  # In-place multiplication

                # Calculate probability with vectorized operations
                prob_i_t = (
                    event_hazard_at_t * event_survival_before_t * other_events_survival
                )
                prob_i_t = torch.clamp(prob_i_t, min=1e-7)  # Numerical stability

                # Apply negative log and weights efficiently
                event_nll = -torch.log(prob_i_t)
                if self.weights is not None:
                    weight = self.weights[i + 1].to(
                        device
                    )  # Ensure weight is on correct device
                    event_nll.mul_(weight)  # In-place multiplication

                # Aggregate loss
                uncensored_loss += event_nll.sum()
                num_uncensored += mask_sum

        # Compute negative log-likelihood for censored subjects
        censored_mask = torch.all(events == 0, dim=1)
        censored_loss = torch.zeros(1, device=device)

        mask_sum = censored_mask.sum().item()  # Cache this computation
        if mask_sum > 0:
            # For censored subjects, get the last observed time
            censored_times = torch.max(duration_idx[censored_mask], dim=1)[0]
            censored_times_unsqueezed = (censored_times + 1).unsqueeze(
                1
            )  # Reuse this tensor

            # Get survival values for all censored subjects
            censored_survival = survival[censored_mask]

            # Use in-place operations to calculate overall survival
            overall_survival = torch.ones(mask_sum, device=device)
            for i in range(self.num_events):
                # Get survival at censoring time (+1 for indexing since survival includes time 0)
                surv_i_at_censor = torch.gather(
                    censored_survival[:, i, :], 1, censored_times_unsqueezed
                ).squeeze(1)
                overall_survival.mul_(surv_i_at_censor)  # In-place multiplication

            # Numerical stability with clamp
            overall_survival = torch.clamp(overall_survival, min=1e-7)

            # Calculate negative log-likelihood
            censor_nll = -torch.log(overall_survival)

            # Apply weight for censored observations with in-place operation
            if self.weights is not None:
                weight = self.weights[0].to(
                    device
                )  # Ensure weight is on correct device
                censor_nll.mul_(weight)  # In-place multiplication

            censored_loss = censor_nll.sum()

        # Combine losses and normalize efficiently
        # Using scalar division is more efficient than creating new tensors
        total_loss = (uncensored_loss + censored_loss) / batch_size

        # Return properly formed tensor without extra conversion if possible
        if isinstance(total_loss, torch.Tensor):
            return total_loss

        # Fallback for scalar result
        return self.ensure_tensor(total_loss, device=device)


class DeepHitRankingLoss(Loss):
    """
    Ranking loss component of DeepHit.

    Computes the ranking loss to ensure proper ordering of survival probabilities,
    penalizing when subjects with earlier events have higher survival probabilities
    than those with later events.
    """

    def __init__(
        self,
        duration_cuts: str,
        sigma: float = 0.1,
        num_events: int = 1,
        importance_sample_weights: Optional[str] = None,
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
        margin: float = 0.05,  # Margin for the ranking loss
    ):
        """
        Initialize DeepHitRankingLoss with enhanced discriminative capability.

        Args:
            duration_cuts: Path to CSV file containing duration cut points for discretization
            sigma: Scaling factor for ranking loss (smaller values = sharper differences)
            num_events: Number of competing events
            importance_sample_weights: Optional path to CSV file with importance weights
            balance_strategy: Strategy for balancing loss components
            balance_params: Additional parameters for the balancing strategy
            margin: Minimum margin required between survival probabilities
        """
        super(DeepHitRankingLoss, self).__init__(
            num_events=num_events,
            balance_strategy=balance_strategy,
            balance_params=balance_params,
        )

        self.sigma = sigma
        self.margin = (
            margin  # Margin for the ranking loss - enforces minimum difference
        )

        # Load duration cut points
        df = pd.read_csv(duration_cuts, header=None, names=["cuts"])
        self.register_buffer(
            "duration_cuts", torch.tensor(df.cuts.values, dtype=torch.float32)
        )
        self.num_time_bins = len(df.cuts)

        # Load importance sampling weights if provided
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the enhanced ranking loss component with margin and improved stability.

        Args:
            predictions: Model predictions (SAOutput with survival probabilities)
            references: Ground truth references

        Returns:
            Ranking loss component
        """
        batch_size = predictions.logits.shape[0]
        survival = predictions.survival  # [batch_size, num_events, num_time_bins+1]
        device = references.device

        # Extract event information
        events = self.events(references)  # [batch_size, num_events]
        durations = self.durations(references)  # [batch_size, num_events]

        # Pre-compute device-specific constants
        duration_cuts = self.duration_cuts.to(device)
        max_time_bin = self.num_time_bins - 1
        max_survival_idx = survival.size(2) - 1

        # Initialize loss accumulator
        rank_loss = torch.zeros(1, device=device)
        pair_count = 0  # Track number of valid pairs

        # Process each event type with vectorized operations where possible
        for event_type in range(self.num_events):
            # Get mask for this event type
            event_occurred = events[:, event_type] == 1
            event_count = event_occurred.sum().item()

            # Skip if no events of this type
            if event_count < 2:  # Need at least 2 events to rank
                continue

            # Get indices and times for subjects with this event
            event_indices = torch.where(event_occurred)[0]
            event_times = durations[event_occurred, event_type]

            # More efficient pair-based calculation using matrix operations
            # Calculate each event sample's contribution in batch
            for i, (i_idx, time_i) in enumerate(zip(event_indices, event_times)):
                # Create risk indicator using vectorized comparison
                # 1 if subject j's time > subject i's time (j should be lower risk)
                risk_indicator = (durations[:, event_type] > time_i).float()

                if risk_indicator.sum() == 0:
                    continue  # No valid comparisons

                # Find time bin index efficiently
                time_bin_idx = torch.searchsorted(duration_cuts, time_i)
                time_bin_idx = torch.clamp(time_bin_idx, max=max_time_bin)

                # Get survival index with bounds checking
                survival_idx = torch.min(
                    time_bin_idx + 1, torch.tensor(max_survival_idx, device=device)
                )

                # Get survival values for all subjects at this time
                all_survival_at_t = survival[:, event_type, survival_idx]

                # Get reference subject's survival
                i_survival_at_t = all_survival_at_t[i_idx]

                # Vectorized computation of survival differences
                survival_diff = i_survival_at_t - all_survival_at_t

                # Apply margin to enforce minimum difference
                # Only penalize when the difference is smaller than the margin
                margin_diff = torch.clamp(self.margin - survival_diff, min=0.0)

                # Combine with exponential scaling for traditional DeepHit loss
                exp_diff = torch.exp(survival_diff / self.sigma)

                # Combine margin and exponential components
                # This creates a hybrid loss that enforces both margin and exponential ranking
                combined_diff = exp_diff + margin_diff

                # Apply risk indicator to only consider valid pairs
                valid_comparisons = risk_indicator * combined_diff

                # Apply weight if needed
                if self.weights is not None:
                    # Get weight for this event type directly on device
                    weight = self.weights[event_type + 1].to(device)
                    valid_comparisons.mul_(weight)  # In-place multiplication

                # Accumulate loss and count pairs
                rank_loss += valid_comparisons.sum()
                pair_count += risk_indicator.sum().item()

        # Normalize by number of pairs instead of batch size for more stable gradients
        if pair_count > 0:
            rank_loss = rank_loss / pair_count
        else:
            # No valid pairs for ranking
            rank_loss = torch.zeros(1, device=device)
            logger.warning("No valid pairs found for ranking loss")

        return rank_loss


class DeepHitCalibrationLoss(Loss):
    """
    Calibration loss component of DeepHit.

    Computes the mean squared error between predicted event probabilities
    and actual event indicators at specific time points.
    """

    def __init__(
        self,
        duration_cuts: str,
        eval_times: Optional[List[float]] = None,
        num_events: int = 1,
        importance_sample_weights: Optional[str] = None,
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
    ):
        """
        Initialize DeepHitCalibrationLoss.

        Args:
            duration_cuts: Path to CSV file containing duration cut points for discretization
            eval_times: Optional list of specific times to evaluate calibration
            num_events: Number of competing events
            importance_sample_weights: Optional path to CSV file with importance weights
            balance_strategy: Strategy for balancing loss components
            balance_params: Additional parameters for the balancing strategy
        """
        super(DeepHitCalibrationLoss, self).__init__(
            num_events=num_events,
            balance_strategy=balance_strategy,
            balance_params=balance_params,
        )

        # Load duration cut points
        df = pd.read_csv(duration_cuts, header=None, names=["cuts"])
        self.register_buffer(
            "duration_cuts", torch.tensor(df.cuts.values, dtype=torch.float32)
        )
        self.num_time_bins = len(df.cuts)

        # Store evaluation times if provided
        if eval_times is not None:
            self.register_buffer(
                "eval_times", torch.tensor(eval_times, dtype=torch.float32)
            )
            self.eval_time_indices = []
            for t in eval_times:
                idx = torch.searchsorted(self.duration_cuts, t).item()
                if idx >= self.num_time_bins:
                    idx = self.num_time_bins - 1
                self.eval_time_indices.append(idx)
            self.eval_time_indices = torch.tensor(self.eval_time_indices)
        else:
            self.eval_times = None
            self.eval_time_indices = None

        # Load importance sampling weights if provided
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the calibration loss component.

        Args:
            predictions: Model predictions (SAOutput with survival probabilities)
            references: Ground truth references

        Returns:
            Calibration loss component
        """
        batch_size = predictions.logits.shape[0]
        survival = predictions.survival  # [batch_size, num_events, num_time_bins+1]

        # If no evaluation times provided, use all duration cut points
        if self.eval_time_indices is None:
            eval_time_indices = torch.arange(self.num_time_bins, device=survival.device)
        else:
            eval_time_indices = self.eval_time_indices.to(survival.device)

        # Extract event information
        events = self.events(references)  # [batch_size, num_events]
        durations = self.durations(references)  # [batch_size, num_events]

        device = references.device
        calibration_loss = torch.zeros(1, device=device)
        num_comparisons = 0

        for event_type in range(self.num_events):
            for t_idx in eval_time_indices:
                # Get time value (ensure t_idx is within bounds)
                t_idx_safe = min(t_idx.item(), len(self.duration_cuts) - 1)
                t = self.duration_cuts[t_idx_safe]

                # Create binary indicator: 1 if subject had event of this type before time t
                event_before_t = (
                    (events[:, event_type] == 1) & (durations[:, event_type] <= t)
                ).float()

                # Get predicted probability of event before time t
                # 1 - S(t) for this event type
                # +1 because survival includes time 0, but ensure it's within bounds
                survival_idx = min(t_idx_safe + 1, survival.size(2) - 1)
                pred_prob = 1.0 - survival[:, event_type, survival_idx]

                # Square difference between actual and predicted
                squared_diff = (event_before_t - pred_prob) ** 2

                # Apply weight for this event type
                if self.weights is not None:
                    squared_diff = squared_diff * self.weights[event_type + 1]

                calibration_loss += squared_diff.sum()
                num_comparisons += batch_size

        # Normalize by number of comparisons
        if num_comparisons > 0:
            calibration_loss = calibration_loss / num_comparisons
        else:
            calibration_loss = torch.zeros(1, device=references.device)

        # The ensure_tensor is still kept as a fallback
        return self.ensure_tensor(calibration_loss, device=references.device)
