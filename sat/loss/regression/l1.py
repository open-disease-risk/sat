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
        device = references.device

        event_indicators = self.events(references)[:, event_type].to(bool)
        durations = self.durations(references)[:, event_type]
        predictions = predictions[:, event_type]

        if self.l1_type == "uncensored":
            scores = durations[event_indicators] - predictions[event_indicators]
            loss = torch.mean(torch.abs(scores))
        elif self.l1_type == "hinge":
            scores = durations - predictions
            scores[~event_indicators] = torch.maximum(
                scores[~event_indicators], torch.zeros_like(scores[~event_indicators])
            )
            loss = torch.mean(torch.abs(scores))
        elif self.l1_type == "margin":
            censor_times = durations[~event_indicators]
            weights = torch.Tensor(
                1.0 - self.kms[event_type].predict(censor_times.detach().cpu().numpy())
            ).to(device)
            best_guesses = torch.Tensor(
                self.kms[event_type].best_guess(censor_times.detach().cpu().numpy())
            ).to(device)

            scores = torch.empty_like(predictions)
            scores[event_indicators] = (
                durations[event_indicators] - predictions[event_indicators]
            )
            scores[~event_indicators] = weights * (
                best_guesses - predictions[~event_indicators]
            )
            weighted_multiplier = torch.ones(1).to(device) / (
                torch.sum(event_indicators) + torch.sum(weights)
            )
            loss = (weighted_multiplier * torch.sum(torch.abs(scores)))[0]
        else:
            raise ValueError("L1 type must be either 'hinge' or 'margin'.")

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
