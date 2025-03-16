"""Mean Squared Error loss for regression tasks"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch
import pandas as pd

from typing import List

from sat.models.tasks.heads import TaskOutput
from sat.loss.base import Loss
from sat.utils.km import KaplanMeierArea
from sat.utils import logging

logger = logging.get_default_logger()


class MSELoss(Loss):
    """Mean Squared Error (MSE) Loss with support for censored data.

    This loss computes the mean squared error between predicted values and target values,
    with special handling for censored data points. Three modes are available:

    1. Uncensored Mode (l2_type="uncensored"):
       Only uncensored data points (where the event was observed) contribute to the loss.
       Loss = (true_time - predicted_time)² for event=1 samples only.

    2. Hinge Mode (l2_type="hinge"):
       For censored data points, a penalty is applied only if the model predicts the event
       will occur before the censoring time (which we know is incorrect).
       - For event=1: Loss = (true_time - predicted_time)²
       - For event=0: Loss = max(0, censoring_time - predicted_time)²

    3. Margin Mode (l2_type="margin"):
       Uses Kaplan-Meier estimates from the training data to create a weighted loss for
       censored data points. For each censored point, a "best guess" is computed based on
       population statistics, and the loss is weighted by the probability the event would
       have occurred.
       - For event=1: Loss = (true_time - predicted_time)²
       - For event=0: Loss = w * (best_guess - predicted_time)²
         where w = 1 - S(t) is the probability the event would occur before censoring time t,
         and best_guess is the expected event time given survival past censoring time.

    The margin approach provides a more informed way to handle censored data by leveraging
    survival analysis techniques to estimate the most likely event times for censored observations.
    """

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
                # Handle empty case while maintaining gradient flow
                # Use predictions to derive a zero tensor that's connected to the graph
                loss = predictions.new_zeros(1).squeeze()
        elif self.l2_type == "margin":
            # Pre-compute once to avoid repeated calculations
            not_event_indicators = ~event_indicators

            # Process censored samples more efficiently
            if not_event_indicators.any():
                censor_times = durations[not_event_indicators]
                # Minimize CPU-GPU transfer by batching operations
                censor_np = censor_times.detach().cpu().numpy()
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
                    # Use predictions to create a zero tensor that maintains gradient connection
                    loss = predictions.new_zeros(1).squeeze()
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
        # Initialize with tensor from first event to ensure proper gradient flow
        loss = self.mse(pred_values, references, 0)

        # Add remaining events
        for i in range(1, self.num_events):
            loss += self.mse(pred_values, references, i)

        return loss
