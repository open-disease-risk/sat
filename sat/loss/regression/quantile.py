"""Loss functions for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd

import torch
from typing import List

from sat.utils import logging
from sat.models.heads import TaskOutput
from sat.utils.km import KaplanMeierArea
from ..base import Loss

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
    ):
        super(QuantileLoss, self).__init__(num_events)

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
            # read training data into pandas dataframe with given columns
            df = pd.read_csv(training_set, header=0)
            for event in range(self.num_events):
                training_event_times = df["duration"]
                training_event_indicators = df["event"] == event + 1
                self.kms.append(
                    KaplanMeierArea(training_event_times, training_event_indicators)
                )

    def quantile_loss(self, predictions, references, event_type) -> torch.Tensor:
        device = references.device

        event_indicators = self.events(references)[:, event_type].to(bool)
        durations = self.durations(references)[:, event_type]
        predictions = predictions[:, event_type]

        if self.l_type == "uncensored":
            scores = torch.zeros_like(durations[event_indicators])
            for i, q in enumerate(self.quantiles):
                errors = (
                    durations[event_indicators] - predictions[event_indicators][:, i]
                )
                scores += torch.max((q - 1) * errors, q * errors)

            loss = torch.mean(scores)
        elif self.l_type == "margin":
            censor_times = durations[~event_indicators]
            weights = torch.Tensor(
                1.0 - self.kms[event_type].predict(censor_times.detach().cpu().numpy())
            ).to(device)
            best_guesses = torch.Tensor(
                self.kms[event_type].best_guess(censor_times.detach().cpu().numpy())
            ).to(device)

            scores = torch.zeros_like(durations)
            errors = torch.empty_like(durations)
            for i, q in enumerate(self.quantiles):
                errors[event_indicators] = (
                    durations[event_indicators] - predictions[event_indicators][:, i]
                )
                errors[~event_indicators] = weights * (
                    best_guesses - predictions[~event_indicators][:, i]
                )
                scores += torch.max((q - 1) * errors, q * errors)

            weighted_multiplier = torch.ones(1).to(device) / (
                torch.sum(event_indicators) + torch.sum(weights)
            )
            loss = (weighted_multiplier * torch.mean(scores))[0]
        else:
            raise ValueError("L type must be either 'uncensored' or 'margin'.")

        return self.weights[event_type + 1] * loss

    def forward(
        self, predictions: TaskOutput, references: torch.Tensor
    ) -> torch.Tensor:
        predictions = predictions.predictions

        loss = 0.0
        for i in range(self.num_events):
            loss += self.quantile_loss(predictions, references, i)

        return loss
