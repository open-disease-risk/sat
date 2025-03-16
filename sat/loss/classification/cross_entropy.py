"""Cross-entropy loss for classification tasks"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch
import torch.nn as nn
import pandas as pd

from typing import List

from sat.models.tasks.heads import TaskOutput
from sat.loss.base import Loss
from sat.utils.km import KaplanMeierArea
from sat.utils import logging

logger = logging.get_default_logger()


class CrossEntropyLoss(Loss):
    """Cross Entropy Loss with support for censored data.

    This loss computes the cross-entropy between predicted probabilities and target classes,
    with special handling for censored data points. Three modes are available:

    1. Uncensored Mode (ce_type="uncensored"):
       Only uncensored data points (where the event was observed) contribute to the loss.
       Loss = -log(p_true) for event=1 samples only.

    2. Hinge Mode (ce_type="hinge"):
       For censored data points, a penalty is applied only if the model predicts higher
       probability for events occurring before the censoring time (which we know is incorrect).
       - For event=1: Standard cross-entropy loss
       - For event=0: Penalizes only predictions with high probability for events before censoring time

    3. Margin Mode (ce_type="margin"):
       Uses Kaplan-Meier estimates from the training data to create a weighted loss for
       censored data points. For each censored point, probabilities are weighted by
       survival analysis-based estimates.
       - For event=1: Standard cross-entropy loss
       - For event=0: Loss is weighted by a function of S(t), the survival probability at time t

    The margin approach provides a more informed way to handle censored data by incorporating
    survival probabilities into the loss calculation, giving appropriate weight to different
    time intervals based on population-level statistics.
    """

    def __init__(
        self,
        event_time_thr: float,
        training_set: str,
        num_events: int,
        l_type: str = "margin",
        importance_sample_weights: str = None,
        balance_strategy: str = "none",
        ema_momentum: float = 0.9,
    ):
        super(CrossEntropyLoss, self).__init__(
            num_events, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )

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

                # Compute loss directly - weights will be applied in forward
                loss = self.loss_func(preds, targets).mean()
            else:
                # Handle empty case, but keep gradient flow
                loss = predictions.sum() * 0

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
                    1.0 - self.kms[event_type].predict(censor_np),
                    device=device,
                    dtype=torch.float32,
                )
                best_guesses = torch.tensor(
                    self.kms[event_type].best_guess(censor_np),
                    device=device,
                    dtype=torch.float32,
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
                        loss = weighted_multiplier * losses.mean()
                    else:
                        loss = predictions.sum() * 0
                else:
                    loss = predictions.sum() * 0
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
                    loss = self.loss_func(preds, targets).mean()
                else:
                    loss = predictions.sum() * 0
        else:
            raise ValueError(f"Unknown l_type: {self.l_type}")

        return loss

    def forward(
        self, predictions: TaskOutput, references: torch.Tensor
    ) -> torch.Tensor:
        # Extract predictions once to avoid repeated attribute access
        pred_values = predictions.predictions

        # For multiple events, collect losses for each event type
        losses = []
        for event in range(self.num_events):
            event_loss = self.weights[event + 1].item() * self.ce(
                pred_values, references, event
            )
            losses.append(event_loss)

        # Stack losses and sum - more efficient than using sum() directly
        if losses:  # Check if we have any losses
            stacked_losses = torch.stack(losses)
            return stacked_losses.sum()
        else:
            # Ensure gradient flow even if no losses
            return pred_values.sum() * 0
