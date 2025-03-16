"""Focal loss for handling class imbalance in classification tasks with censoring awareness"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd
import torch
import torch.nn.functional as F

from typing import List

from sat.models.tasks.heads import TaskOutput
from sat.loss.base import Loss
from sat.utils.km import KaplanMeierArea
from sat.utils import logging

logger = logging.get_default_logger()


class FocalLoss(Loss):
    """Focal loss for handling class imbalance in classification tasks with censoring awareness.

    Focal loss is designed to address the class imbalance problem by down-weighting easy examples
    and focusing on hard negatives. It adds a modulating factor to the standard cross-entropy
    criterion that places more emphasis on hard, misclassified examples.

    Mathematical formulation:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the target class.

    This implementation supports three modes for handling censored data:

    1. Uncensored Mode (focal_type="uncensored"):
       Only uncensored data points (where the event was observed) contribute to the loss.
       Focal loss is calculated only for event=1 samples, ignoring censored data points.

    2. Hinge Mode (focal_type="hinge"):
       For censored data points, a penalty is applied only if the model assigns high probability
       to events occurring before the censoring time (which we know is incorrect).
       - For event=1: Standard focal loss
       - For event=0: Penalizes predictions with high probability for events before censoring time

    3. Margin Mode (focal_type="margin"):
       Uses Kaplan-Meier estimates from the training data to create a weighted loss for
       censored data points. For each censored point, probabilities are weighted by
       survival analysis-based estimates.
       - For event=1: Standard focal loss
       - For event=0: Loss is weighted by a function of S(t), the survival probability at time t

    The margin approach provides the most sophisticated handling of censored data by incorporating
    survival probabilities from the population into the loss calculation, giving appropriate
    weight to different time intervals based on statistical estimates.

    Parameters:
    - p_t is the predicted probability of the target class
    - alpha_t is a weighting factor for class t
    - gamma is the focusing parameter that adjusts the rate at which easy examples are down-weighted

    References:
    - Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection."
      In Proceedings of the IEEE International Conference on Computer Vision (pp. 2980-2988).
    """

    def __init__(
        self,
        event_time_thr: float,
        training_set: str,
        num_events: int,
        l_type: str = "margin",
        alpha: float = None,
        gamma: float = 2.0,
        importance_sample_weights: str = None,
        balance_strategy: str = "none",
        ema_momentum: float = 0.9,
    ):
        """Initialize FocalLoss with censoring awareness.

        Args:
            event_time_thr (float): Time threshold for event occurrence
            training_set (str): Path to training set data for KaplanMeier estimation
            num_events (int): Number of event types
            l_type (str): Loss type, either 'margin' or 'uncensored'
            alpha (float, optional): Weighting factor for the rare class in focal loss
            gamma (float): Focusing parameter for focal loss
            importance_sample_weights (str, optional): Path to importance sampling weights
            balance_strategy (str): Strategy for balancing loss components
            ema_momentum (float): Momentum for exponential moving average
        """
        super(FocalLoss, self).__init__(
            num_events, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )

        self.alpha = alpha
        self.gamma = gamma
        self.num_events = num_events
        self.event_time_thr = event_time_thr
        self.l_type = l_type
        self.kms: List[KaplanMeierArea] = []

        # Load importance sampling weights if provided
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

        # Initialize KaplanMeier models if using margin loss type
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

    def _apply_focal_weighting(self, bce_loss, preds, targets):
        """Apply focal loss weighting to binary cross entropy loss.

        Args:
            bce_loss (torch.Tensor): Binary cross entropy loss
            preds (torch.Tensor): Predicted probabilities after sigmoid
            targets (torch.Tensor): Target binary labels

        Returns:
            torch.Tensor: Focal-weighted loss
        """
        # Compute pt (predicted probability for the target class)
        pt = targets * preds + (1 - targets) * (1 - preds)

        # Compute focal weight: (1-pt)^gamma
        focal_weight = (1 - pt).pow(self.gamma)

        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        return focal_weight * bce_loss

    def focal(self, predictions, references, event_type) -> torch.Tensor:
        """Compute focal loss for a single event type with censoring awareness.

        Args:
            predictions (torch.Tensor): Model predictions
            references (torch.Tensor): Target references
            event_type (int): The event type index

        Returns:
            torch.Tensor: Computed focal loss
        """
        device = references.device
        durations = self.durations(references)[:, event_type]
        # Use bool() directly for better efficiency
        event_indicator = self.events(references)[:, event_type].bool()

        if self.l_type == "uncensored":
            # Survival times less than T - compute once
            event_occurred = durations <= self.event_time_thr

            # Efficiently compute targets - convert to float once at the end
            targets = (event_occurred & event_indicator).float()

            # Focus on relevant samples
            relevant_for_classification = ~event_occurred | event_indicator

            # Only compute if there are relevant samples
            if relevant_for_classification.any():
                # Extract indices once for better indexing performance
                relevant_indices = relevant_for_classification.nonzero(as_tuple=True)[0]

                # Get predictions and targets for relevant samples
                preds = (
                    predictions[:, event_type]
                    .index_select(0, relevant_indices)
                    .squeeze(-1)
                )
                targets = targets.index_select(0, relevant_indices)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(preds)

                # Compute binary cross entropy loss
                bce_loss = F.binary_cross_entropy_with_logits(
                    preds, targets, reduction="none"
                )

                # Apply focal weighting
                loss = self._apply_focal_weighting(bce_loss, probs, targets).mean()
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

                # Efficient event_occurred calculation
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

                # Compute relevant classification indices
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

                    # Apply sigmoid to get probabilities
                    probs = torch.sigmoid(preds)

                    # Compute binary cross entropy loss
                    bce_losses = F.binary_cross_entropy_with_logits(
                        preds, targets, reduction="none"
                    )

                    # Apply focal weighting
                    losses = self._apply_focal_weighting(bce_losses, probs, targets)

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

                    # Apply sigmoid to get probabilities
                    probs = torch.sigmoid(preds)

                    # Compute binary cross entropy loss
                    bce_loss = F.binary_cross_entropy_with_logits(
                        preds, targets, reduction="none"
                    )

                    # Apply focal weighting
                    loss = self._apply_focal_weighting(bce_loss, probs, targets).mean()
                else:
                    loss = predictions.sum() * 0
        else:
            raise ValueError(f"Unknown l_type: {self.l_type}")

        return loss

    def forward(
        self, predictions: TaskOutput, references: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss with censoring awareness.

        Args:
            predictions (TaskOutput): Model predictions
            references (torch.Tensor): Target references

        Returns:
            torch.Tensor: Computed focal loss
        """
        # Extract predictions once to avoid repeated attribute access
        pred_values = predictions.predictions

        # For multiple events, collect losses for each event type
        losses = []
        for event in range(self.num_events):
            event_loss = self.weights[event + 1].item() * self.focal(
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
