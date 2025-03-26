"""Loss functions for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd

import torch

from typing import List, Dict, Optional, Union

from ..balancing import BalancingStrategy

from sat.utils import logging
from sat.models.heads import TaskOutput
from sat.utils.km import KaplanMeierArea
from ..base import Loss

logger = logging.get_default_logger()


class L1Loss(Loss):
    """L1 loss"""

    def __init__(
        self,
        training_set: str,
        importance_sample_weights: str = None,
        l1_type: str = "hinge",
        num_events: int = 1,
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
    ):
        super(L1Loss, self).__init__(
            num_events=num_events,
            balance_strategy=balance_strategy,
            balance_params=balance_params,
        )

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

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Train the Kaplan Meier Curves")

            # read training data into pandas dataframe with given columns
            df = pd.read_csv(training_set, header=0)
            for event in range(self.num_events):
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
        """
        Optimized L1 loss computation with vectorized operations.

        Args:
            predictions: Model predictions
            references: Ground truth references
            event_type: Index of the event type to compute loss for

        Returns:
            Weighted L1 loss for this event type
        """
        device = references.device

        # Extract tensors once to avoid redundant indexing
        event_indicators = self.events(references)[:, event_type].to(bool)
        event_mask = event_indicators
        non_event_mask = ~event_indicators
        durations = self.durations(references)[:, event_type]
        preds = predictions[:, event_type]

        # Pre-compute common expressions
        diff = durations - preds

        if self.l1_type == "uncensored":
            # Use masked operations for better efficiency
            if torch.any(event_mask):
                # Only compute loss for uncensored events
                event_diffs = diff[event_mask]
                loss = torch.mean(torch.abs(event_diffs))
            else:
                # No events, return zero loss
                return torch.tensor(0.0, device=device)

        elif self.l1_type == "hinge":
            # Create a copy to avoid modifying the original tensor
            scores = diff.clone()

            # Apply hinge constraint only to censored samples
            if torch.any(non_event_mask):
                censored_scores = scores[non_event_mask]
                censored_zeros = torch.zeros_like(censored_scores)
                scores[non_event_mask] = torch.maximum(censored_scores, censored_zeros)

            # Calculate absolute mean
            loss = torch.mean(torch.abs(scores))

        elif self.l1_type == "margin":
            # Process only if we have both event and non-event samples
            if not torch.any(event_mask) and not torch.any(non_event_mask):
                return torch.tensor(0.0, device=device)

            # Initialize scores tensor
            scores = torch.zeros_like(preds)

            # Handle event samples
            if torch.any(event_mask):
                scores[event_mask] = diff[event_mask]

            # Handle censored samples - moving CPU operations to batch processing
            if torch.any(non_event_mask):
                censor_times = durations[non_event_mask]

                # Move to CPU only once for KM calculations
                censor_times_cpu = censor_times.detach().cpu().numpy()
                weights_np = 1.0 - self.kms[event_type].predict(censor_times_cpu)
                best_guesses_np = self.kms[event_type].best_guess(censor_times_cpu)

                # Move back to device in one operation
                weights = torch.tensor(weights_np, device=device)
                best_guesses = torch.tensor(best_guesses_np, device=device)

                # Calculate censored scores
                censored_preds = preds[non_event_mask]
                scores[non_event_mask] = weights * (best_guesses - censored_preds)

            # Compute weighted normalization factor
            weight_sum = (
                torch.sum(event_mask.float()) + torch.sum(weights)
                if torch.any(non_event_mask)
                else torch.sum(event_mask.float())
            )

            # Avoid division by zero
            if weight_sum > 0:
                weighted_multiplier = 1.0 / weight_sum
                loss = weighted_multiplier * torch.sum(torch.abs(scores))
            else:
                loss = torch.tensor(0.0, device=device)

        else:
            raise ValueError("L1 type must be 'uncensored', 'hinge', or 'margin'.")

        # Apply importance weight
        return self.weights[event_type + 1] * loss

    def forward(
        self, predictions: TaskOutput, references: torch.Tensor
    ) -> torch.Tensor:
        predictions = predictions.predictions
        device = references.device

        # Initialize loss as tensor
        loss = torch.zeros(1, device=device)
        for event in range(self.num_events):
            loss += self.l1(predictions, references, event)

        # The ensure_tensor is still kept as a fallback
        return self.ensure_tensor(loss, device=device)
