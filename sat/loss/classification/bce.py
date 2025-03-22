"""Loss functions for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd

import torch

from torch import nn
from typing import List, Dict, Optional, Union

from ..balancing import BalancingStrategy

from sat.utils import logging
from sat.models.heads import TaskOutput
from sat.utils.km import KaplanMeierArea

from ..base import Loss

logger = logging.get_default_logger()


class CrossEntropyLoss(Loss):
    """Cross Entropy Loss."""

    def __init__(
        self,
        event_time_thr: float,
        training_set: str,
        num_events: int,
        l_type: str = "margin",
        importance_sample_weights: str = None,
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
    ):
        super(CrossEntropyLoss, self).__init__(
            num_events=num_events,
            balance_strategy=balance_strategy,
            balance_params=balance_params,
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
            # read training data into pandas dataframe with given columns
            df = pd.read_csv(training_set, header=0)
            for event in range(self.num_events):
                training_event_times = df["duration"]
                training_event_indicators = df["event"] == event + 1
                self.kms.append(
                    KaplanMeierArea(training_event_times, training_event_indicators)
                )

    def ce(self, predictions, references, event_type) -> torch.Tensor:
        device = references.device
        durations = self.durations(references)[:, event_type]
        event_indicator = self.events(references)[:, event_type].to(bool)

        if self.l_type == "uncensored":
            # survival times less than T
            event_occurred = durations <= self.event_time_thr

            # The classification target is 1 if the event occurred and where
            # survival times are < T, Otherwise 0.
            targets = torch.logical_and(event_occurred, event_indicator).float()

            # for classification we only care about observations with the
            # event or when survival times are > T
            relevant_for_classification = torch.logical_or(
                ~event_occurred, event_indicator
            )

            # subset predictions and targets to what is relevant for
            # classification
            preds = predictions[:, event_type][relevant_for_classification].squeeze()
            targets = targets[relevant_for_classification]

            logger.debug(
                f"Compute loss between predictions {preds.shape} and targets {targets.shape}"
            )

            loss = self.weights[event_type + 1] * torch.mean(
                self.loss_func(preds, targets)
            )
        elif self.l_type == "margin":
            censor_times = durations[~event_indicator]
            weights = torch.Tensor(
                1.0 - self.kms[event_type].predict(censor_times.detach().cpu().numpy())
            ).to(device)
            best_guesses = torch.Tensor(
                self.kms[event_type].best_guess(censor_times.detach().cpu().numpy())
            ).to(device)

            # survival times less than T accounting also for best guesses
            event_occurred = torch.empty_like(durations).bool()
            event_occurred[event_indicator] = (
                durations[event_indicator] <= self.event_time_thr
            )
            event_occurred[~event_indicator] = best_guesses <= self.event_time_thr

            # The classification target is 1 if the event occurred and where
            # survival times are < T, Otherwise 0.
            targets = torch.logical_and(event_occurred, event_indicator).float()

            # for classification we only care about observations with the
            # event or when survival times are > T
            relevant_for_classification = torch.logical_or(
                ~event_occurred, event_indicator
            )

            # subset predictions and targets to what is relevant for
            # classification
            preds = predictions[:, event_type][relevant_for_classification].squeeze()
            targets = targets[relevant_for_classification]

            logger.debug(
                f"Compute loss between predictions {preds.shape} and targets {targets.shape}"
            )

            losses = self.loss_func(preds, targets)
            losses[~event_indicator] = weights * losses[~event_indicator]

            weighted_multiplier = torch.tensor(1.0).to(device) / (
                torch.sum(event_indicator) + torch.sum(weights)
            )

            loss = (
                self.weights[event_type + 1] * weighted_multiplier * torch.mean(losses)
            )

            return loss

    def forward(
        self, predictions: TaskOutput, references: torch.Tensor
    ) -> torch.Tensor:
        predictions = predictions.predictions
        device = references.device

        # Initialize loss as tensor
        loss = torch.zeros(1, device=device)
        for event in range(self.num_events):
            loss += self.ce(predictions, references, event)

        # The ensure_tensor is still kept as a fallback
        return self.ensure_tensor(loss, device=device)
