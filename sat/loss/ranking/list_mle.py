"""ListMLE loss implementation for ranking in survival analysis"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torch.nn.functional as F

from sat.models.heads import SAOutput
from sat.utils import logging
from ..base import RankingLoss

logger = logging.get_default_logger()


class ListMLELoss(RankingLoss):
    """
    Base class for ListMLE loss implementation.

    ListMLE (List Maximum Likelihood Estimator) is a listwise ranking approach
    that directly optimizes the likelihood of the correctly ordered list.
    This provides an alternative to pairwise ranking losses.

    The loss is based on Plackett-Luce model which defines a probability
    distribution over all possible permutations of a list.

    Performance characteristics:
    - Scales better with data size than pairwise methods O(n log n) vs O(n²)
    - Provides listwise context rather than focusing on individual pairs
    - More directly optimizes the ranking metric than pairwise approaches
    """

    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        num_events: int = 1,
        epsilon: float = 1e-10,
        temperature: float = 1.0,
    ):
        """
        Initialize ListMLE loss.

        Args:
            duration_cuts: Path to CSV file containing duration cut points
            importance_sample_weights: Optional path to CSV file with importance weights
            num_events: Number of competing events
            epsilon: Small value for numerical stability
            temperature: Controls the sharpness of the probability distribution
                         Lower values make the distribution more peaked
        """
        super(ListMLELoss, self).__init__(
            duration_cuts, importance_sample_weights, 1.0, num_events, 0.0
        )
        self.epsilon = epsilon
        self.temperature = temperature

    def compute_list_mle_loss(
        self, scores: torch.Tensor, rankings: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute ListMLE loss for a batch of scores and rankings.

        Args:
            scores: Predicted scores/probabilities to be ranked
                   Shape: [batch_size, list_length]
            rankings: Ground truth rankings or relevance scores
                     Higher values indicate higher rank/relevance
                     Shape: [batch_size, list_length]
            mask: Optional binary mask to filter out invalid entries
                  Shape: [batch_size, list_length]

        Returns:
            torch.Tensor: The ListMLE loss value
        """
        batch_size, list_length = scores.size()
        device = scores.device

        # Get sorting indices based on ground truth rankings (descending)
        # This gives us the "correct" ordering
        _, indices = rankings.sort(dim=1, descending=True)

        # Reorder scores according to ground truth ranking
        ordered_scores = torch.gather(scores, 1, indices)

        # Apply temperature scaling to control sharpness
        scaled_scores = ordered_scores / self.temperature

        # Apply mask if provided
        if mask is not None:
            # Reorder mask to match the new ordering
            ordered_mask = torch.gather(mask, 1, indices)
            # Set masked scores to a large negative value to effectively remove them
            scaled_scores = scaled_scores.masked_fill(~ordered_mask, -1e9)
            # Count valid elements for each item in batch
            valid_lengths = ordered_mask.sum(dim=1)
        else:
            valid_lengths = torch.full((batch_size,), list_length, device=device)

        # Compute the log-likelihood for Plackett-Luce model
        # For each position i, we compute softmax over remaining positions [i:end]
        loss = torch.zeros(batch_size, device=device)

        for i in range(list_length - 1):
            # Get scores for remaining positions
            remaining_scores = scaled_scores[:, i:]

            # Calculate log softmax of first position over all remaining positions
            log_softmax_scores = F.log_softmax(remaining_scores, dim=1)

            # The likelihood is maximized when the first position has highest score
            position_loss = log_softmax_scores[:, 0]

            # Only add loss for positions within valid length
            # valid_position should be batch_size in dimension to match position_loss
            valid_position = i < valid_lengths
            loss = loss - position_loss * valid_position

        # Average over batch
        if valid_lengths.sum() > 0:
            return loss.sum() / valid_lengths.sum()
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Base implementation that should be overridden by subclasses.

        Parameters:
            predictions (SAOutput): Predictions of the model with survival probabilities
            references (torch.Tensor): Reference values (dims: batch size x 4 * num_events)

        Returns:
            torch.Tensor: The loss value
        """
        raise NotImplementedError("Subclasses must implement forward method")
