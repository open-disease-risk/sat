"""NLLPC Hazard loss for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd
import torch
from transformers.utils import ModelOutput

from sat.pycox.models.loss import NLLPCHazardLoss
from sat.models.tasks.heads import SAOutput
from sat.loss.base import Loss


class SATNLLPCHazardLoss(Loss):
    def __init__(
        self,
        importance_sample_weights: str = None,
        num_events: int = 1,
        balance_strategy: str = "none",
        ema_momentum: float = 0.9,
    ):
        super(SATNLLPCHazardLoss, self).__init__(
            num_events, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )

        # Initialize loss balancing parameters
        self._init_loss_params(["nllpc"])

        self.loss_fct = NLLPCHazardLoss(reduction="none")

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

    def nllp_hazard_loss(self, predictions, references, event_type) -> torch.Tensor:
        # Directly get events as bool to avoid unnecessary conversion later
        events = self.events(references)[:, event_type].bool()
        duration_percentiles = self.duration_percentiles(references)[:, event_type]
        fraction_duration = self.fraction_with_quantile(references)[:, event_type]
        predictions = predictions[:, event_type]

        # Apply loss and weight in one efficient operation
        event_loss = self.loss_fct(
            predictions,
            duration_percentiles,
            events,
            fraction_duration,
        )

        # Use a single operation with the weight factor
        return event_loss.mean() * self.weights[event_type + 1]

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute a loss.

        Parameters:
            predictions (SAOutput): Predictions of the model containing logits
            references (torch.Tensor): Reference values (dims: batch size x 4) containing event types and durations

        Returns:
            float: The loss value.
        """
        import logging

        logger = logging.getLogger("sat")

        # Debug logging to understand what we're getting
        logger.debug(f"NLLPC Loss - Prediction type: {type(predictions).__name__}")
        logger.debug(
            f"NLLPC Loss - Available attributes: {[attr for attr in dir(predictions) if not attr.startswith('_')]}"
        )

        # Extract logits directly from SAOutput
        logits = predictions.logits
        logger.debug(f"NLLPC Loss - Logits shape: {logits.shape}")

        # Pre-calculate common tensors that we'll need for each event
        # For single event case, avoid the loop
        if self.num_events == 1:
            logger.debug(f"NLLPC Loss - Computing loss for single event")
            loss = self.nllp_hazard_loss(logits, references, 0)
            balanced_loss = self.balance_losses({"nllpc": loss}, {"nllpc": 1.0})
            logger.debug(
                f"NLLPC Loss - Raw loss: {loss}, Balanced loss: {balanced_loss}"
            )
            return balanced_loss

        # For multiple events, use accumulation
        loss = 0.0
        for i in range(self.num_events):
            loss += self.nllp_hazard_loss(logits, references, i)

        balanced_loss = self.balance_losses({"nllpc": loss}, {"nllpc": 1.0})
        logger.debug(f"NLLPC Loss - Raw loss: {loss}, Balanced loss: {balanced_loss}")
        return balanced_loss
