"""Loss functions for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch

from torch import nn

from sat.utils import logging
from sat.models.heads import SAOutput
from ..base import Loss

logger = logging.get_default_logger()


# TODO: This needs to be refactored into the current loss template
class MismatchLoss(Loss):
    """Mismatch Loss Class."""

    def __init__(
        self,
        duration_cuts: torch.Tensor,
        max_time: float,
    ):
        super(MismatchLoss, self).__init__()
        self.duration_cuts = duration_cuts
        self.max_time = torch.tensor(max_time)

    def mean_lifetime(self, predictions, references) -> torch.Tensor:
        device = references.device
        num_events = predictions.shape[1]
        # The next two lines compute the expected event time, we consider that the
        # surv probability goes to zero at max_time
        time_intervals = torch.cat(
            (self.duration_cuts, self.max_time.unsqueeze(0)), 0
        ) - torch.cat((torch.tensor(0).to(device).unsqueeze(0), self.duration_cuts), 0)

        surv = self.survivals(predictions)
        mean_lifetimes = torch.zeros(num_events, surv[0].shape[0]).to(device)
        dummy = self.duration_cuts.expand(surv[0].shape).to(device)

        for i in range(num_events):
            mean_lifetimes[i, :] = (
                torch.sum(
                    time_intervals
                    * (
                        torch.cat(
                            (
                                surv[i],
                                torch.tensor(0)
                                .to(device)
                                .expand(dummy.shape[0])
                                .view(-1, 1),
                            ),
                            1,
                        )
                        + torch.cat(
                            (
                                torch.tensor(1)
                                .to(device)
                                .expand(dummy.shape[0])
                                .view(-1, 1),
                                surv[i],
                            ),
                            1,
                        )
                    ),
                    dim=1,
                )
                / 2
            )

            return mean_lifetimes

    def mismatch_loss(self, references, mean_lifetimes) -> torch.Tensor:
        device = references.device
        duration = self.durations(references)
        events = self.events(references)

        # Finding the first event, and cases where we have mismatch
        est_event = torch.argmin(mean_lifetimes, dim=0) + 1
        mismatch = (est_event != events) & (events != 0)

        # The following variables are defined to help mismatch loss computation
        mean_life_temp = mean_lifetimes[:, mismatch]
        est_event_temp = est_event[mismatch]
        event_temp = events[mismatch]
        mean_life_event = torch.zeros(event_temp.shape[0]).to(device)
        mean_life_est = torch.zeros(event_temp.shape[0]).to(device)

        for i in range(event_temp.shape[0]):
            mean_life_event[i] = mean_life_temp[
                event_temp[i] - 1, i
            ]  # Estimated time of actual event
            mean_life_est[i] = mean_life_temp[
                est_event_temp[i] - 1, i
            ]  # Estimated event time of wrong estimated event

        mismatch_loss = torch.mean(
            nn.ReLU()(duration[mismatch] - mean_life_est)
            + nn.ReLU()(mean_life_event - duration[mismatch])
            + mean_life_event
            - mean_life_est
        )

        return mismatch_loss

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute the mismatch loss.

        This function implements SurvTrace loss for both competing and single event cases
        Parameters:
            predictions (SAOutput: Predictions of the model (SAOutput: 5 x events x batch size x cuts)
            references (torch.Tensor): Reference values. (dims: batch size x 4)

        Returns:
            torch.Tensor: The loss value.
        """
        logits = predictions.logits
        mean_lifetimes = self.mean_lifetime(logits, references)
        logger.debug(f"Mean lifetimes {mean_lifetimes}")

        # Compute loss (already a tensor from mismatch_loss)
        loss = self.mismatch_loss(references, mean_lifetimes)

        # The ensure_tensor is still kept as a fallback
        return self.ensure_tensor(loss, device=references.device)
