"""Quantile Regression Loss for predicting distribution quantiles"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd
import torch
from typing import List

from sat.models.tasks.heads import TaskOutput
from sat.loss.base import Loss
from sat.utils.km import KaplanMeierArea
from sat.utils import logging

logger = logging.get_default_logger()


class QuantileLoss(Loss):
    """Quantile Loss for Quantile Regression"""

    def __init__(
        self,
        quantiles: List[float],
        training_set: str,
        num_events: int,
        importance_sample_weights: str = None,
        l_type: str = "uncensored",
        balance_strategy: str = "none",
        ema_momentum: float = 0.9,
    ):
        super(QuantileLoss, self).__init__(
            num_events, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )

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
            # Read training data with the new descriptive headers
            df = pd.read_csv(training_set, header=0)

            for event in range(self.num_events):
                # Get duration and event indicators for this event
                duration_col = f"duration_event{event+1}"
                event_col = f"event{event+1}"

                training_event_times = df[duration_col]
                training_event_indicators = df[event_col] == 1
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
                # Handle empty case, but keep gradient flow
                loss = predictions.sum() * 0
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
                    1.0 - self.kms[event_type].predict(censor_np),
                    device=device,
                    dtype=torch.float32,
                )
                best_guesses = torch.tensor(
                    self.kms[event_type].best_guess(censor_np),
                    device=device,
                    dtype=torch.float32,
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
                    # Handle empty case, but keep gradient flow
                    loss = predictions.sum() * 0
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
        device = pred_values.device

        # For single event case, avoid the loop overhead
        if self.num_events == 1:
            return self.quantile_loss(pred_values, references, 0)

        # For multiple events, pre-allocate losses and use tensor operations
        losses = []
        for i in range(self.num_events):
            event_loss = self.quantile_loss(pred_values, references, i)
            losses.append(event_loss)

        # Stack losses and sum - more efficient than scalar accumulation
        if losses:  # Check if we have any losses
            stacked_losses = torch.stack(losses)
            return stacked_losses.sum()
        else:
            # Ensure gradient flow even if no losses
            return pred_values.sum() * 0
