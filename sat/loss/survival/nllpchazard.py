"""Negative log likelihood piece wise constant hazard loss for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd

import torch
from typing import Dict, Optional, Union

from ..balancing import BalancingStrategy

from sat.pycox.models.loss import NLLPCHazardLoss
from sat.models.heads import SAOutput
from sat.utils import logging

from ..base import Loss

logger = logging.get_default_logger()


class SATNLLPCHazardLoss(Loss):
    def __init__(
        self,
        importance_sample_weights: str = None,
        num_events: int = 1,
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
    ):
        super(SATNLLPCHazardLoss, self).__init__(
            num_events=num_events,
            balance_strategy=balance_strategy,
            balance_params=balance_params,
        )

        self.loss_fct = NLLPCHazardLoss(reduction="none")

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

    def nllp_hazard_loss(self, predictions, references, event_type) -> torch.Tensor:
        events = self.events(references)[:, event_type].to(bool)
        duration_percentiles = self.duration_percentiles(references)[:, event_type]
        fraction_duration = self.fraction_with_quantile(references)[:, event_type]
        predictions = predictions[:, event_type]

        return (
            self.loss_fct(
                predictions,
                duration_percentiles,
                events,
                fraction_duration,
            ).mean()
            * self.weights[event_type + 1]
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute a loss.

        Parameters:
            predictions (SAOutput: Predictions of the model (SAOutput: 5 x events x batch size x cuts)
            references (torch.Tensor): Reference values. (dims: batch size x 4)

        Returns:
            float: The loss value.
        """
        # variables x batch x events x duration cuts
        logits = predictions.logits

        loss = 0.0
        for i in range(self.num_events):
            loss += self.nllp_hazard_loss(logits, references, i)

        return loss
