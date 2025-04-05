"""Loss functions for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch

from sat.models.heads import SAOutput
from sat.utils import logging

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
        """
        Vectorized implementation to calculate mean lifetime with reduced memory allocations.
        """
        device = references.device
        num_events = predictions.shape[1]

        # Compute expected event time with vectorized operations
        # We consider that the survival probability goes to zero at max_time
        time_intervals = torch.cat(
            (self.duration_cuts, self.max_time.unsqueeze(0)), 0
        ) - torch.cat((torch.tensor(0).to(device).unsqueeze(0), self.duration_cuts), 0)

        surv = self.survivals(predictions)
        batch_size = surv[0].shape[0]
        mean_lifetimes = torch.zeros(num_events, batch_size, device=device)

        # Create common tensors once to reuse across all events
        zeros = torch.zeros(batch_size, 1, device=device)
        ones = torch.ones(batch_size, 1, device=device)

        # Process all event types in a loop, but with vectorized operations within each iteration
        for i in range(num_events):
            # Efficiently concatenate tensors
            surv_with_zero = torch.cat((surv[i], zeros), dim=1)
            one_with_surv = torch.cat((ones, surv[i]), dim=1)

            # Sum the survival values and compute weighted average
            sum_survival = surv_with_zero + one_with_surv
            mean_lifetimes[i] = torch.sum(time_intervals * sum_survival, dim=1) / 2

        return mean_lifetimes

    def mismatch_loss(self, references, mean_lifetimes) -> torch.Tensor:
        """
        Vectorized implementation of the mismatch loss with reduced loop operations.
        """
        device = references.device
        duration = self.durations(references)
        events = self.events(references)

        # Finding the first event, and cases where we have mismatch
        est_event = torch.argmin(mean_lifetimes, dim=0) + 1

        # Use vectorized boolean operations for mask creation
        mismatch = (est_event != events) & (events != 0)

        # If no mismatches, return zero loss
        if not torch.any(mismatch):
            return torch.tensor(0.0, device=device)

        # Get mismatched values
        mean_life_temp = mean_lifetimes[:, mismatch]
        est_event_temp = est_event[mismatch]
        event_temp = events[mismatch]
        duration_mismatch = duration[mismatch]

        # Create index tensors for efficient gathering
        batch_indices = torch.arange(mean_life_temp.size(1), device=device)

        # Adjust event indices (subtract 1 for zero-based indexing)
        actual_event_indices = event_temp - 1
        est_event_indices = est_event_temp - 1

        # Vectorized selection of mean lifetimes using advanced indexing
        # Get estimated time of actual event
        mean_life_event = mean_life_temp[actual_event_indices, batch_indices]

        # Get estimated event time of wrong estimated event
        mean_life_est = mean_life_temp[est_event_indices, batch_indices]

        # Compute loss components with vectorized operations
        time_diff1 = torch.nn.functional.relu(duration_mismatch - mean_life_est)
        time_diff2 = torch.nn.functional.relu(mean_life_event - duration_mismatch)
        event_diff = mean_life_event - mean_life_est

        # Combine loss components and take mean
        mismatch_loss = torch.mean(time_diff1 + time_diff2 + event_diff)

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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Mean lifetimes {mean_lifetimes}")

        # Compute loss (already a tensor from mismatch_loss)
        loss = self.mismatch_loss(references, mean_lifetimes)

        # The ensure_tensor is still kept as a fallback
        return self.ensure_tensor(loss, device=references.device)
