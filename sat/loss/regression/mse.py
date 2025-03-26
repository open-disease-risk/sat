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


class MSELoss(Loss):
    """MSE Loss."""

    def __init__(
        self,
        training_set: str,
        importance_sample_weights: str = None,
        l2_type: str = "uncensored",
        num_events: int = 1,
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
    ):
        super(MSELoss, self).__init__(
            num_events=num_events,
            balance_strategy=balance_strategy,
            balance_params=balance_params,
        )

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

    def mse(
        self,
        predictions,
        references,
        event_type: int,
    ) -> torch.Tensor:
        device = references.device
        event_indicators = self.events(references)[:, event_type].to(bool)
        durations = self.durations(references)[:, event_type]
        predictions = predictions[:, event_type]

        if self.l2_type == "uncensored":
            # Only compute loss for samples where events occurred
            if torch.sum(event_indicators) > 0:
                scores = durations[event_indicators] - predictions[event_indicators]
                loss = torch.mean(torch.square(scores))
            else:
                loss = torch.zeros(1, device=device)
        elif self.l2_type == "margin":
            # Handle censored samples
            censor_mask = ~event_indicators
            censor_count = torch.sum(censor_mask)
            event_count = torch.sum(event_indicators)

            # Initialize scores tensor
            scores = torch.zeros_like(predictions)

            # Process event samples
            if event_count > 0:
                scores[event_indicators] = (
                    durations[event_indicators] - predictions[event_indicators]
                )

            # Process censored samples
            if censor_count > 0:
                censor_times = durations[censor_mask]

                # Convert to numpy for KM functions, then back to tensor on device
                censor_times_np = censor_times.detach().cpu().numpy()
                weights = torch.tensor(
                    1.0 - self.kms[event_type].predict(censor_times_np),
                    device=device,
                    dtype=torch.float32,
                )
                best_guesses = torch.tensor(
                    self.kms[event_type].best_guess(censor_times_np),
                    device=device,
                    dtype=torch.float32,
                )

                scores[censor_mask] = weights * (
                    best_guesses - predictions[censor_mask]
                )

                # Calculate weighted multiplier
                weighted_multiplier = 1.0 / (event_count + torch.sum(weights))
                loss = (
                    weighted_multiplier
                    * torch.sum(torch.square(scores))
                    / predictions.size(0)
                )
            else:
                # If no censored samples, just use event samples
                loss = (
                    torch.mean(torch.square(scores[event_indicators]))
                    if event_count > 0
                    else torch.zeros(1, device=device)
                )
        else:
            raise ValueError("L2 type must be either 'uncensored' or 'margin'.")

        return self.weights[event_type + 1] * loss

    def forward(
        self, predictions: TaskOutput, references: torch.Tensor
    ) -> torch.Tensor:
        predictions = predictions.predictions
        device = references.device

        # Initialize loss as tensor on the correct device
        loss = torch.zeros(1, device=device)

        # Process all events
        if self.num_events > 0:
            for i in range(self.num_events):
                loss += self.mse(predictions, references, i)

        return loss
