"""L1 (Mean Absolute Error) loss for regression tasks"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch
import torch.nn.functional as F
import pandas as pd

from typing import List
from sat.models.tasks.heads import TaskOutput
from sat.loss.base import Loss
from sat.utils.km import KaplanMeierArea
from sat.utils import logging

logger = logging.get_default_logger()


class L1Loss(Loss):
    """Mean Absolute Error (L1) Loss with support for censored data.

    This loss computes the mean absolute error between predicted values and target values,
    with special handling for censored data points. Three modes are available:

    1. Uncensored Mode (l1_type="uncensored"):
       Only uncensored data points (where the event was observed) contribute to the loss.
       Loss = |true_time - predicted_time| for event=1 samples only.

    2. Hinge Mode (l1_type="hinge"):
       For censored data points, a penalty is applied only if the model predicts the event
       will occur before the censoring time (which we know is incorrect).
       - For event=1: Loss = |true_time - predicted_time|
       - For event=0: Loss = max(0, censoring_time - predicted_time)

    3. Margin Mode (l1_type="margin"):
       Uses Kaplan-Meier estimates from the training data to create a weighted loss for
       censored data points. For each censored point, a "best guess" is computed based on
       population statistics, and the loss is weighted by the probability the event would
       have occurred.
       - For event=1: Loss = |true_time - predicted_time|
       - For event=0: Loss = w * |best_guess - predicted_time|
         where w = 1 - S(t) is the probability the event would occur before censoring time t,
         and best_guess is the expected event time given survival past censoring time.
    """

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
                # Use predictions to create a zero tensor that maintains gradient connection
                loss = predictions.new_zeros(1).squeeze()
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
                    1.0 - self.kms[event_type].predict(censor_np),
                    device=device,
                    dtype=torch.float32,
                )
                best_guesses = torch.tensor(
                    self.kms[event_type].best_guess(censor_np),
                    device=device,
                    dtype=torch.float32,
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
                    # Use predictions to create a zero tensor that maintains gradient connection
                    loss = predictions.new_zeros(1).squeeze()
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
        device = references.device

        # Optimize single event case to avoid loop overhead
        if self.num_events == 1:
            return self.l1(pred_values, references, 0)

        # For multiple events, use efficient accumulation
        # Initialize with tensor from first event to ensure proper gradient flow
        loss = self.l1(pred_values, references, 0)

        # Add remaining events
        for event in range(1, self.num_events):
            loss += self.l1(pred_values, references, event)

        return loss
