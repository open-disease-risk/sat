"""Mismatch Loss for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch
import torch.nn.functional as F

from sat.utils import logging
from sat.models.tasks.heads import SAOutput
from sat.loss.base import Loss

logger = logging.get_default_logger()


class MismatchLoss(Loss):
    """Mismatch Loss Class."""

    def __init__(
        self,
        duration_cuts: torch.Tensor,
        max_time: float,
        balance_strategy: str = "none",
        ema_momentum: float = 0.9,
        num_events: int = 1,
    ):
        super(MismatchLoss, self).__init__(
            num_events, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )

        # Initialize loss balancing parameters
        self._init_loss_params(["mismatch"])
        self.duration_cuts = duration_cuts
        self.max_time = torch.tensor(max_time)

    def mean_lifetime(self, predictions, references):
        num_events = predictions.survival.shape[1]
        device = references.device

        surv_t = predictions.survival  # (batch_size, events, time)
        intervals = surv_t.new_ones(surv_t.shape[0], surv_t.shape[1])

        # optimize: compute differences at once instead of in a loop
        diff = self.duration_cuts[1:] - self.duration_cuts[:-1]
        # Use broadcasting for better memory efficiency
        diff_expanded = (
            diff.unsqueeze(0).unsqueeze(0).expand(surv_t.shape[0], surv_t.shape[1], -1)
        )

        # optimize: compute survival averages efficiently
        surv_avg = 0.5 * (surv_t[..., 1:] + surv_t[..., :-1])

        # optimize: compute partial integrals all at once
        partial_integrals = surv_avg * diff_expanded

        # Sum along the time dimension
        result = torch.sum(partial_integrals, dim=2)

        # Scale the results
        mean_lifetime = result * (self.max_time / self.duration_cuts[-1])

        # Return the result
        return mean_lifetime

    def mismatch_loss(self, references, mean_lifetimes):
        """Returns the loss value."""
        n_events = mean_lifetimes.shape[1]
        batch_size = mean_lifetimes.shape[0]
        device = references.device

        events = self.events(references)  # batch_size, events
        durations = self.durations(references)  # batch_size, events

        max_lifetime = mean_lifetimes.max()
        loss = torch.tensor(0.0, device=device)

        # optimize: create a mask for all events at once
        event_mask = events.bool()
        observed_count = event_mask.sum()

        if observed_count > 0:
            # For observed events use MSE with true durations
            observed_durations = durations[event_mask]
            observed_mean_lifetimes = mean_lifetimes.view(-1)[event_mask.view(-1)]
            observed_loss = F.mse_loss(
                observed_mean_lifetimes, observed_durations, reduction="mean"
            )
            loss = loss + observed_loss

        # optimize: handle censored events more efficiently
        censored_mask = ~event_mask
        censored_count = censored_mask.sum()

        if censored_count > 0:
            # For censored events, use hinge loss: penalty if mean_lifetime < duration
            censored_durations = durations[censored_mask]
            censored_mean_lifetimes = mean_lifetimes.view(-1)[censored_mask.view(-1)]

            # Calculate how much the model is "wrong" for each censored event
            # (when predicted lifetime is less than observed censoring time)
            diff = censored_durations - censored_mean_lifetimes

            # Apply hinge loss: only penalize when mean_lifetime < duration
            hinge_loss = F.relu(diff).mean()

            loss = loss + hinge_loss

        # Average the observed and censored losses
        num_loss_terms = 1 if observed_count == 0 or censored_count == 0 else 2
        loss = loss / num_loss_terms

        return loss

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute the mismatch loss.

        This function implements SurvTrace loss for both competing and single event cases
        Parameters:
            predictions (SAOutput): Predictions of the model containing survival function values
            references (torch.Tensor): Reference values (dims: batch size x 4) containing event types and durations

        Returns:
            float: The loss value.
        """
        # Calculate mean lifetimes first (reused computation)
        mean_lifetimes = self.mean_lifetime(predictions, references)

        # Calculate mismatch loss using the mean lifetimes
        loss = self.mismatch_loss(references, mean_lifetimes)

        return self.balance_losses({"mismatch": loss}, {"mismatch": 1.0})
