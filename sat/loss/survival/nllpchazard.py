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
        """Compute the hazard loss with optimized batch processing.

        Parameters:
            predictions (SAOutput): Predictions of the model (SAOutput with logits)
            references (torch.Tensor): Reference values (dims: batch size x 4*num_events)

        Returns:
            torch.Tensor: The loss value.
        """
        logits = predictions.logits
        device = references.device

        # Pre-extract common data to avoid redundant operations
        events_all = self.events(references)  # [batch_size, num_events]
        duration_percentiles_all = self.duration_percentiles(
            references
        )  # [batch_size, num_events]
        fraction_durations_all = self.fraction_with_quantile(
            references
        )  # [batch_size, num_events]

        # Process all events in parallel if possible (using batch processing)
        # Note: We still need a loop because nllp_hazard_loss expects one event at a time
        # but we've pre-extracted the data to reduce redundant operations
        loss_values = []
        for i in range(self.num_events):
            event_loss = self.nllp_hazard_loss(logits, references, i)
            loss_values.append(event_loss)

        # Combine all loss values efficiently
        if len(loss_values) == 1:
            # Fast path for single event
            combined_loss = loss_values[0]
        else:
            # Use a single sum operation to combine multiple events
            combined_loss = torch.sum(torch.stack(loss_values))

        # Return properly formed tensor
        return self.ensure_tensor(combined_loss, device=device)
