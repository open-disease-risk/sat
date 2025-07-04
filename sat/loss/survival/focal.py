"""Focal Loss for survival analysis tasks."""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import collections.abc
from typing import Dict, List, Optional, Union

import pandas as pd
import torch

from sat.models.heads import SAOutput
from sat.utils import logging

from ..balancing import BalancingStrategy
from ..base import Loss

logger = logging.get_default_logger()


class SurvivalFocalLoss(Loss):
    """
    Focal Loss implementation for survival analysis tasks.

    Focal Loss addresses imbalance by down-weighting well-classified examples,
    focusing the model's attention on harder samples. In survival analysis context,
    this focuses on improving survival predictions that are harder to predict.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where p_t is the model's estimated probability for the target class.

    This implementation supports both a single global gamma value or different
    gamma values for each event type (multi-focal parameters).
    """

    def __init__(
        self,
        gamma: Union[float, List[float], torch.Tensor] = 2.0,
        importance_sample_weights: Optional[str] = None,
        reduction: str = "mean",
        num_events: int = 1,
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
    ):
        """
        Initialize SurvivalFocalLoss.

        Args:
            gamma: Focusing parameter(s) (γ) that adjusts the rate at which easy examples
                  are down-weighted (higher gamma = more down-weighting).
                  Can be a single value for all events or a list/tensor with one value per event.
            importance_sample_weights: Path to a CSV file containing importance weights.
                                      These weights are used for addressing class imbalance.
            reduction: Reduction method ('mean', 'sum', or 'none').
            num_events: Number of competing events.
            balance_strategy: Strategy for balancing loss components.
            balance_params: Additional parameters for the balancing strategy.
        """
        super(SurvivalFocalLoss, self).__init__(
            num_events=num_events,
            balance_strategy=balance_strategy,
            balance_params=balance_params,
        )

        self.reduction = reduction

        # Check if gamma might be a Hydra ListConfig object
        is_sequence = isinstance(gamma, collections.abc.Sequence) and not isinstance(
            gamma, str
        )

        # Process gamma parameter - check if we have multiple gamma values
        # multi_focal = False  # Set in _set_multi_focal method instead

        # If it's a sequence (including ListConfig)
        if is_sequence:
            # Always convert sequence to list of floats to handle ListConfig and other sequence types
            gamma_list = [float(g) for g in gamma]

            # Create a tensor from the gamma list
            gamma = torch.tensor(gamma_list, dtype=torch.float32)

            # Special handling for tests that expect a single-element list to be treated as multi-focal
            # This matches the test expectations where any list (even with one element) is treated as multi-focal
            if len(gamma_list) == 1 and num_events > 1:
                # Repeat the single gamma value for all event types
                last_value = gamma[0].item()
                padded_gamma = torch.ones(num_events, dtype=torch.float32) * last_value
                gamma = padded_gamma
                self.register_buffer("gamma", gamma)
                self.multi_focal = True

            # If there are multiple gamma values
            elif len(gamma_list) > 1:
                # Validate that we have the right number of gamma values
                if gamma.numel() != num_events:
                    logger.warning(
                        f"Number of gamma values ({gamma.numel()}) doesn't match number of events ({num_events}). "
                        f"Using the first {min(gamma.numel(), num_events)} values or padding with the last value."
                    )
                    # Take the first num_events values or pad with the last value
                    if gamma.numel() < num_events:
                        last_value = gamma[-1].item()
                        padded_gamma = (
                            torch.ones(num_events, dtype=torch.float32) * last_value
                        )
                        padded_gamma[: gamma.numel()] = gamma
                        gamma = padded_gamma
                    else:
                        gamma = gamma[:num_events]

                self.register_buffer("gamma", gamma)
                self.multi_focal = True
            else:
                # Single value in a sequence and num_events = 1
                gamma_value = float(gamma_list[0])
                self.register_buffer(
                    "gamma", torch.tensor(gamma_value, dtype=torch.float32)
                )
                self.multi_focal = False

        # If it's a tensor
        elif isinstance(gamma, torch.Tensor):
            if gamma.numel() > 1:
                # Multi-focal with tensor input
                if gamma.numel() != num_events:
                    logger.warning(
                        f"Number of gamma values ({gamma.numel()}) doesn't match number of events ({num_events}). "
                        f"Using the first {min(gamma.numel(), num_events)} values or padding with the last value."
                    )
                    # Take the first num_events values or pad with the last value
                    if gamma.numel() < num_events:
                        last_value = gamma[-1].item()
                        padded_gamma = (
                            torch.ones(num_events, dtype=torch.float32) * last_value
                        )
                        padded_gamma[: gamma.numel()] = gamma
                        gamma = padded_gamma
                    else:
                        gamma = gamma[:num_events]

                self.register_buffer("gamma", gamma)
                self.multi_focal = True
            else:
                # Single value as tensor
                gamma_value = gamma.item()
                self.register_buffer(
                    "gamma", torch.tensor(float(gamma_value), dtype=torch.float32)
                )
                self.multi_focal = False

        # Otherwise (scalar value)
        else:
            # Handle scalar values (float, int)
            self.register_buffer(
                "gamma", torch.tensor(float(gamma), dtype=torch.float32)
            )
            self.multi_focal = False

        # Load the importance sampling weights if provided
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)

            # Check that we have the right number of weights
            if (
                weights.numel() != num_events + 1
            ):  # +1 for the background/no-event class
                logger.warning(
                    f"Number of importance weights ({weights.numel()}) doesn't match expected value ({num_events + 1}). "
                    f"Using default weights instead."
                )
                weights = torch.ones(num_events + 1)
        else:
            # Default: equal weighting
            weights = torch.ones(num_events + 1)

        self.register_buffer("weights", weights)

    def focal_loss_function(
        self, predictions: torch.Tensor, targets: torch.Tensor, event_type: int
    ) -> torch.Tensor:
        """
        Compute the focal loss for a specific event type's survival function.

        Args:
            predictions: Model predictions (survival probabilities) [batch_size]
            targets: Ground truth labels (binary for each event type) [batch_size]
            event_type: Current event type index

        Returns:
            torch.Tensor: Focal loss value

        Note:
            This function expects predictions and targets to have the same shape.
            Shape transformation should be handled before calling this function.
        """
        # Ensure targets are float for calculations
        targets = targets.float()

        # Clamp predictions for numerical stability (in-place is faster)
        epsilon = 1e-7
        predictions = torch.clamp(predictions, epsilon, 1.0 - epsilon)

        # Compute p_t directly
        p_t = torch.where(targets > 0.5, predictions, 1 - predictions)

        # Compute log(p_t) directly
        log_pt = torch.where(
            targets > 0.5, torch.log(predictions), torch.log(1 - predictions)
        )

        # Get the appropriate gamma for this event type
        gamma_value = self.gamma[event_type] if self.multi_focal else self.gamma

        # Compute (1 - p_t)^gamma * -log(p_t)
        focal_term = ((1 - p_t) ** gamma_value) * (-log_pt)

        # Apply importance weights (vectorized)
        # Weight is event_type + 1 for positive examples, 0 for negative
        weight_t = torch.where(
            targets > 0.5, self.weights[event_type + 1], self.weights[0]
        )
        focal_loss = weight_t * focal_term

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the focal loss across all event types' survival functions.

        Args:
            predictions: Model predictions (SAOutput with survival)
                        survival should be a list of [batch_size, time_bins] tensors,
                        one per event type
            references: Ground truth references with shape [batch_size, 4*num_events]
                       includes duration_idx, events, fractions, durations

        Returns:
            torch.Tensor: Focal loss value

        Note:
            This method automatically handles dimension mismatches by extracting
            the appropriate time points from the survival curves and reshaping
            tensors to ensure compatibility.
        """
        # For survival analysis, we use the survival function
        survival = predictions.survival
        if survival is None:
            raise ValueError(
                "SurvivalFocalLoss requires predictions.survival to be provided"
            )

        device = references.device

        # Get all event indicators at once
        all_events = self.events(references)

        # Initialize loss on correct device
        loss = torch.zeros(1, device=device)

        # Process all event types
        for event_type in range(self.num_events):
            # Get event indicators for this event type
            event_indicators = all_events[:, event_type]

            # Get survival predictions for this event type
            event_survival = survival[event_type]

            # Efficiently handle shape compatibility
            if event_survival.dim() > 1:
                # Use the last time point for survival probability if multi-dimensional
                if event_survival.shape[1] > 0:
                    event_survival = event_survival[:, -1]
                else:
                    event_survival = event_survival.squeeze(1)

            # Ensure consistent shapes without creating unnecessary copies
            batch_size = event_survival.size(0)
            if event_indicators.size(0) != batch_size:
                # This should rarely happen, but handle it just in case
                event_indicators = event_indicators[:batch_size]

            # Compute loss for this event type
            event_loss = self.focal_loss_function(
                event_survival, event_indicators, event_type
            )

            loss += event_loss

        return loss
