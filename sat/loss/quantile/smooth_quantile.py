"""Smooth Quantile Regression Loss for predicting distribution quantiles with better optimization properties"""

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


class SmoothQuantileLoss(Loss):
    """Smooth Quantile Regression Loss for predicting specific quantiles with improved optimization.

    This is a smooth approximation of the standard quantile loss, providing continuous gradients
    throughout the optimization space. It addresses the non-differentiability issue of the standard
    quantile loss at the origin, which can cause challenges in optimization.

    Mathematical formulation:
    The smooth Huber-style quantile loss is defined as:

    For u > 0:
        L_q(u) = q * u - (α/2) * u² if u ≤ q/α
        L_q(u) = q²/(2α) if u > q/α

    For u ≤ 0:
        L_q(u) = (q-1) * u - (α/2) * u² if u ≤ (q-1)/α
        L_q(u) = (q-1)²/(2α) if u > (q-1)/α

    where u = y - ŷ is the prediction error, q is the quantile level, and α is the smoothing parameter.

    For multiple quantiles, the loss is computed for each quantile level and then averaged.

    This implementation supports three modes for handling censored data:

    1. Uncensored Mode (sq_type="uncensored"):
       Only uncensored data points (where the event was observed) contribute to the loss.
       Smooth quantile loss is calculated only for event=1 samples.

    2. Margin Mode (sq_type="margin"):
       Uses Kaplan-Meier estimates from the training data to create a weighted loss for
       censored data points. For each censored point, a "best guess" is computed based on
       population statistics, and the loss is weighted by the probability the event would
       have occurred.
       - For event=1: Standard smooth quantile loss
       - For event=0: Loss is weighted using survival probability estimates
    """

    def __init__(
        self,
        quantiles=[0.5],
        alpha=0.5,
        reduction="mean",
        num_events=1,
        training_set=None,
        importance_sample_weights=None,
        sq_type="uncensored",
        balance_strategy="none",
        ema_momentum=0.9,
    ):
        """Initialize SmoothQuantileLoss.

        Args:
            quantiles (list): List of quantile levels between 0 and 1. Default: [0.5] (median)
            alpha (float): Smoothing parameter that controls the degree of smoothing.
                           Smaller values result in stronger smoothing. Default: 0.5
            reduction (str): Specifies the reduction applied to the output: 'none' | 'mean' | 'sum'.
                'none': no reduction will be applied,
                'mean': the weighted mean of the output is taken,
                'sum': the output will be summed. Default: 'mean'
            num_events (int): Number of event types. Default: 1
            training_set (str, optional): Path to training data CSV file, required for margin mode.
            importance_sample_weights (str, optional): Path to importance sampling weights.
            sq_type (str): Censoring handling type. Either 'uncensored' or 'margin'. Default: 'uncensored'
            balance_strategy (str): Strategy for balancing loss components. Default: 'none'
            ema_momentum (float): Momentum for exponential moving average in adaptive balancing. Default: 0.9
        """
        super().__init__(
            num_events, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )
        self.quantiles = quantiles
        self.alpha = alpha
        self.reduction = reduction
        self.sq_type = sq_type
        self.kms: List[KaplanMeierArea] = []

        # Load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

        # Initialize Kaplan-Meier estimators for margin mode
        if self.sq_type == "margin":
            if training_set is None:
                raise ValueError(
                    "If 'margin' is chosen, training set values must be included."
                )

            logger.debug("Train the Kaplan Meier Curves")
            # Read training data with descriptive headers
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

    def _smooth_quantile_loss(self, preds, targets, quantiles):
        """Compute the smooth quantile loss.

        Args:
            preds: Predicted values. Shape: [batch_size] or [batch_size, num_quantiles]
            targets: Target values. Shape: [batch_size]
            quantiles: Quantile levels. Shape: [num_quantiles]

        Returns:
            loss: The computed smooth quantile loss
        """
        # If predictions have the shape [batch_size, num_quantiles]
        if len(preds.shape) > 1:
            num_quantiles = preds.size(1)

            # Make sure we have as many quantiles in preds as in quantiles
            if num_quantiles != len(quantiles):
                raise ValueError(
                    f"Number of quantiles in predictions ({num_quantiles}) "
                    f"doesn't match number of quantiles provided ({len(quantiles)})"
                )

            # Expand targets to match predictions shape
            expanded_targets = targets.unsqueeze(1).expand(-1, num_quantiles)

            # Calculate errors
            errors = expanded_targets - preds

            # Calculate smooth quantile loss for each quantile
            q_losses = []
            for i, q in enumerate(quantiles):
                q_error = errors[:, i]

                # Smooth quantile loss computation
                q_loss = self._compute_smooth_quantile_loss_single(
                    q_error, q, self.alpha
                )
                q_losses.append(q_loss)

            # Stack losses for all quantiles
            losses = torch.stack(q_losses, dim=1)  # [batch_size, num_quantiles]

            # Apply reduction
            if self.reduction == "none":
                return losses
            elif self.reduction == "mean":
                return losses.mean()
            elif self.reduction == "sum":
                return losses.sum()
        else:
            # Single quantile prediction
            if len(quantiles) > 1:
                raise ValueError(
                    "Multiple quantiles provided but predictions only have one output dimension"
                )

            q = quantiles[0]
            errors = targets - preds

            # Calculate smooth quantile loss
            losses = self._compute_smooth_quantile_loss_single(errors, q, self.alpha)

            # Apply reduction
            if self.reduction == "none":
                return losses
            elif self.reduction == "mean":
                return losses.mean()
            elif self.reduction == "sum":
                return losses.sum()

    def _compute_smooth_quantile_loss_single(self, errors, q, alpha):
        """Compute the smooth quantile loss for a single quantile.

        Args:
            errors: Error values (targets - predictions). Shape: [batch_size]
            q: Quantile level (scalar between 0 and 1)
            alpha: Smoothing parameter

        Returns:
            loss: The computed smooth quantile loss
        """
        # Compute thresholds
        pos_threshold = q / alpha
        neg_threshold = (q - 1) / alpha

        # Positive error case (u > 0)
        pos_mask = errors > 0
        pos_error = errors[pos_mask]
        pos_small_error_mask = pos_error <= pos_threshold

        pos_loss = torch.zeros_like(errors)
        if pos_error.numel() > 0:
            # Case: 0 < u <= q/alpha
            if pos_small_error_mask.any():
                small_err = pos_error[pos_small_error_mask]
                pos_loss[pos_mask][pos_small_error_mask] = q * small_err - (
                    alpha / 2
                ) * small_err.pow(2)

            # Case: u > q/alpha
            if (~pos_small_error_mask).any():
                pos_loss[pos_mask][~pos_small_error_mask] = q**2 / (2 * alpha)

        # Negative error case (u <= 0)
        neg_mask = errors <= 0
        neg_error = errors[neg_mask]
        # Note: for negative errors, we compare with neg_threshold which is (q-1)/alpha (negative)
        # The condition becomes: error >= neg_threshold for "small" errors
        neg_small_error_mask = neg_error >= neg_threshold

        neg_loss = torch.zeros_like(errors)
        if neg_error.numel() > 0:
            # Case: (q-1)/alpha <= u <= 0
            if neg_small_error_mask.any():
                small_err = neg_error[neg_small_error_mask]
                neg_loss[neg_mask][neg_small_error_mask] = (q - 1) * small_err - (
                    alpha / 2
                ) * small_err.pow(2)

            # Case: u < (q-1)/alpha
            if (~neg_small_error_mask).any():
                neg_loss[neg_mask][~neg_small_error_mask] = (q - 1) ** 2 / (2 * alpha)

        # Combine losses
        return pos_loss + neg_loss

    def smooth_quantile_loss_with_censoring(self, predictions, references, event_type):
        """Compute the smooth quantile loss with handling for censored data.

        Args:
            predictions: Predictions for one event type. Shape: [batch_size, num_quantiles]
            references: Full reference tensor. Shape: [batch_size, 4*num_events]
            event_type: Event type index

        Returns:
            loss: The computed loss value with censoring handled
        """
        device = references.device

        # Extract event indicators and durations
        event_indicators = self.events(references)[:, event_type].bool()
        durations = self.durations(references)[:, event_type]

        if self.sq_type == "uncensored":
            # Only process if there are events
            if event_indicators.any():
                # Get only the event samples
                event_durations = durations[event_indicators]
                event_predictions = predictions[event_indicators]

                # Calculate smooth quantile loss for actual events only
                loss = self._smooth_quantile_loss(
                    event_predictions, event_durations, self.quantile_tensor
                )
            else:
                # Handle empty case while maintaining gradient flow
                loss = predictions.new_zeros(1).squeeze()

        elif self.sq_type == "margin":
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

                # Initialize combined loss
                losses = []

                # Process event samples if they exist
                if event_indicators.any():
                    event_durations = durations[event_indicators]
                    event_predictions = predictions[event_indicators]
                    event_loss = self._smooth_quantile_loss(
                        event_predictions, event_durations, self.quantile_tensor
                    )
                    event_count = event_indicators.sum()
                    losses.append((event_loss, event_count))

                # Process censored samples
                non_event_predictions = predictions[non_event_indicators]
                non_event_loss = self._smooth_quantile_loss(
                    non_event_predictions, best_guesses, self.quantile_tensor
                )
                # Weight the loss by survival probability factors
                weights_sum = weights.sum()
                if weights_sum > 0:
                    losses.append((non_event_loss * weights.mean(), weights_sum))

                # Combine losses with proper weighting
                if losses:
                    total_weight = sum(weight for _, weight in losses)
                    if total_weight > 0:
                        weighted_loss = sum(
                            loss * (weight / total_weight) for loss, weight in losses
                        )
                        loss = weighted_loss
                    else:
                        loss = predictions.new_zeros(1).squeeze()
                else:
                    loss = predictions.new_zeros(1).squeeze()
            else:
                # All samples are event indicators = True
                loss = self._smooth_quantile_loss(
                    predictions, durations, self.quantile_tensor
                )

        else:
            raise ValueError("sq_type must be either 'uncensored' or 'margin'.")

        # Apply weight
        return self.weights[event_type + 1] * loss

    def forward(self, predictions: TaskOutput, references: torch.Tensor):
        """Compute the smooth quantile loss.

        Args:
            predictions: Model output containing predicted quantile values.
                Shape: [batch_size, num_quantiles] or [batch_size, num_events, num_quantiles]
            references: Target tensor containing true values.
                Shape: [batch_size] or [batch_size, 4*num_events]

        Returns:
            loss: The computed loss value
        """
        # For regression tasks, primarily use time_to_event field from SAOutput
        if predictions.predictions is not None:
            pred_values = predictions.predictions
        else:
            raise ValueError(
                "Unable to determine prediction values from SAOutput object"
            )

        # If predictions is [batch_size, num_events, num_quantiles], we need to handle multiple events
        if len(pred_values.shape) == 3:
            # Initialize with first event tensor to maintain gradient flow
            if self.num_events == 1:
                loss = self.smooth_quantile_loss_with_censoring(
                    pred_values[:, 0], references, 0
                )
                return self.balance_losses({f"smooth_q": loss}, {f"smooth_q": 1.0})

            # Initialize loss components dictionary for balancing
            loss = self.smooth_quantile_loss_with_censoring(
                pred_values[:, 0], references, 0
            )

            # Compute loss for each event type
            for event_idx in range(1, self.num_events):
                event_preds = pred_values[:, event_idx]  # [batch_size, num_quantiles]

                # Compute smooth quantile loss with censoring for this event
                loss += self.smooth_quantile_loss_with_censoring(
                    event_preds, references, event_idx
                )
            return loss
        else:
            # Simple case: single regression task with quantile outputs
            targets = references
            if len(references.shape) > 1 and references.shape[1] == 4 * self.num_events:
                # If references is [batch_size, 4*num_events], use durations as regression targets
                targets = self.durations(references)
                # Assuming single event, take the first column
                targets = targets[:, 0]

            loss = self._smooth_quantile_loss(
                pred_values, targets, self.quantile_tensor
            )
            return loss
