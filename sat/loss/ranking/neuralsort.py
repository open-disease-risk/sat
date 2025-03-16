"""NeuralSort Loss for differentiable ranking in survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd
import torch

from sat.models.tasks.heads import SAOutput
from sat.loss.base import Loss


class NeuralSortLoss(Loss):
    """NeuralSort Loss for survival analysis ranking.

    This implementation is based on the NeuralSort algorithm proposed by Grover et al. (2019)
    and adapted for survival analysis. NeuralSort provides a differentiable surrogate for the
    non-differentiable sorting operation, enabling end-to-end training with ranking metrics.

    Mathematical formulation:
    The NeuralSort operator produces a unimodal row-stochastic matrix P where P_ij represents
    the probability that element i is mapped to position j in the sorted permutation.

    P = softmax(-(A - A^T) / tau)

    where A_ij = ||s_i - s_j||^2 and tau is a temperature parameter.

    For survival analysis, we want items with earlier event times to have higher risk scores,
    and we can use the NeuralSort operator to learn this ranking in a differentiable way.

    References:
    - Grover, A. et al. (2019). Stochastic Optimization of Sorting Networks via Continuous Relaxations.
      In International Conference on Learning Representations (ICLR).
    - Adams, R. & Zemel, R. (2011). Ranking via Sinkhorn Propagation.
      arXiv preprint arXiv:1106.1925.
    """

    def __init__(
        self,
        tau=1.0,
        eps=1e-10,
        importance_sample_weights=None,
        num_events=1,
        balance_strategy="none",
        ema_momentum=0.9,
    ):
        """Initialize NeuralSort loss.

        Args:
            tau (float): Temperature parameter controlling the discreteness of the sorting.
                         Lower values make the sort closer to a hard sort. Default: 1.0
            eps (float): Small constant for numerical stability in computations. Default: 1e-10
            importance_sample_weights (str): Path to importance sampling weights. Default: None
            num_events (int): Number of event types. Default: 1
        """
        super().__init__(
            num_events, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )
        self.tau = tau
        self.eps = eps

        # Initialize the loss components for loss balancing
        self._init_loss_params(["intra", "inter"])

        # Load importance sampling weights if provided
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

    def forward(self, predictions: SAOutput, references: torch.Tensor):
        """Compute the NeuralSort loss.

        Args:
            predictions: Model output containing survival probabilities.
                Shape: [batch_size, num_events, num_time_points]
            references: Target tensor containing event indicators and durations.
                Shape: [batch_size, 4*num_events]

        Returns:
            loss: The computed loss value
        """
        # Extract events and durations from references
        events = self.events(references)  # [batch_size, num_events]
        durations = self.durations(references)  # [batch_size, num_events]

        # Get survival probabilities from predictions
        survival = predictions.survival  # [batch_size, num_events, num_time_points]

        if len(survival.shape) < 3:
            raise ValueError(
                f"Expected survival to have shape [batch_size, num_events, num_time_points], got {survival.shape}"
            )

        # Compute both intra-subject and inter-subject ranking losses
        intra_subject_loss = self._compute_intra_subject_ranking(
            events, durations, survival
        )
        inter_subject_loss = self._compute_inter_subject_ranking(
            events, durations, survival
        )

        # Create a loss dictionary for balancing between intra and inter subject components
        loss_dict = {"intra": intra_subject_loss, "inter": inter_subject_loss}

        # Use the loss balancing API to balance and combine the losses
        return self.balance_losses(loss_dict)

    def _neural_sort(self, scores, tau=1.0):
        """Compute NeuralSort permutation matrix.

        Args:
            scores: Tensor of shape [n] or [batch_size, n] containing the scores to sort
            tau: Temperature parameter for scaling the pairwise distances

        Returns:
            P: Tensor of shape [n, n] or [batch_size, n, n] representing the soft permutation matrix
        """
        n = scores.size(-1)
        batch_size = scores.size(0) if scores.dim() > 1 else 1

        # Handle both batched and non-batched inputs
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)  # [1, n]

        # Compute pairwise distances between scores
        # For each element, create a matrix of distance to all other elements
        s_i = scores.unsqueeze(2).expand(batch_size, n, n)  # [batch_size, n, n]
        s_j = scores.unsqueeze(1).expand(batch_size, n, n)  # [batch_size, n, n]

        # A_ij = ||s_i - s_j||^2
        A = torch.abs(s_i - s_j)  # Using L1 norm for better numerical stability

        # Scale distances by temperature
        scaled_A = -A / tau

        # Compute soft permutation matrix using a variant of Sinkhorn operation
        # This creates a stochastic matrix that approximates a permutation
        P = torch.softmax(scaled_A, dim=-1)  # [batch_size, n, n]

        # If input was not batched, squeeze the batch dimension
        if batch_size == 1:
            P = P.squeeze(0)

        return P

    def _neural_sort_loss(self, scores, target_ranks):
        """Compute the NeuralSort loss given scores and target ranks.

        Args:
            scores: Tensor of shape [n] or [batch_size, n] containing model scores
            target_ranks: Tensor of shape [n] or [batch_size, n] containing target ranks (0 = highest)

        Returns:
            loss: The loss value comparing the neural sort permutation with the target
        """
        device = scores.device
        n = scores.size(-1)
        batch_size = scores.size(0) if scores.dim() > 1 else 1

        # Handle both batched and non-batched inputs
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)  # [1, n]

        if target_ranks.dim() == 1:
            target_ranks = target_ranks.unsqueeze(0)  # [1, n]

        # Compute the NeuralSort permutation matrix
        P = self._neural_sort(scores, self.tau)  # [batch_size, n, n]

        # Convert target ranks to target permutation matrix
        # A rank of 0 means the element should be first in sorted order
        target_perm = torch.zeros(batch_size, n, n, device=device)
        for b in range(batch_size):
            for i in range(n):
                rank = target_ranks[b, i].item()
                if 0 <= rank < n:  # Ensure rank is valid
                    target_perm[b, i, rank] = 1.0

        # Compute KL divergence between the predicted and target permutations
        # KL(P || target_perm) = sum_ij P_ij * log(P_ij / target_perm_ij)
        # We add a small epsilon to target_perm to avoid log(0)
        loss = torch.sum(
            P * torch.log(P / (target_perm + self.eps) + self.eps), dim=[1, 2]
        )

        # Return mean loss across batch
        return loss.mean()

    def _compute_intra_subject_ranking(self, events, durations, survival_probs):
        """Compute the NeuralSort loss for ranking events within the same subject.

        For each subject with multiple events, we learn a soft permutation of the events
        based on their occurrence times using the NeuralSort operator.

        Args:
            events: Binary indicators of events. Shape [batch_size, num_events]
            durations: Time to events. Shape [batch_size, num_events]
            survival_probs: Survival probabilities. Shape [batch_size, num_events, num_time_points]

        Returns:
            loss: The computed intra-subject ranking loss
        """
        batch_size, num_events, num_time_points = survival_probs.shape
        device = durations.device

        # Skip if we only have one event type
        if num_events <= 1:
            return torch.tensor(0.0, device=device)

        # Find subjects with multiple events
        event_counts = torch.sum(events > 0, dim=1)  # [batch_size]
        multi_event_subjects = torch.where(event_counts > 1)[
            0
        ]  # indices of subjects with multiple events

        if multi_event_subjects.numel() == 0:
            # No subjects with multiple events
            # Return a small value connected to the computation graph
            return survival_probs.sum() * 0 + torch.tensor(
                1e-8, device=device, requires_grad=True
            )

        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        valid_subjects = 0

        # Compute risk scores as the negative mean survival probability
        # Higher risk means lower survival probability
        risk_scores = -torch.mean(survival_probs, dim=2)  # [batch_size, num_events]

        # For each subject with multiple events
        for subject_idx in multi_event_subjects:
            subject_events = events[subject_idx]  # [num_events]
            subject_durations = durations[subject_idx]  # [num_events]
            subject_risks = risk_scores[subject_idx]  # [num_events]

            # Get indices of observed events
            observed_mask = subject_events > 0
            if torch.sum(observed_mask) <= 1:  # Need at least 2 events for comparison
                continue

            observed_indices = torch.nonzero(observed_mask).squeeze(-1)
            observed_durations = subject_durations[observed_mask]
            observed_risks = subject_risks[observed_mask]

            # Skip if any durations are non-positive
            if torch.any(observed_durations <= 0):
                continue

            # Target ranks should put earlier events first (lower rank)
            # argsort returns indices that would sort the tensor in ascending order
            # so we want to sort by duration and assign ranks accordingly
            target_ranks = torch.argsort(observed_durations)

            # Compute NeuralSort loss for this subject
            subject_loss = self._neural_sort_loss(observed_risks, target_ranks)

            # Add to total loss weighted by event weights
            loss += subject_loss * torch.mean(self.weights[observed_indices + 1])
            valid_subjects += 1

        # Return average loss
        if valid_subjects > 0:
            return loss / valid_subjects
        # Return a small value connected to the computation graph
        return survival_probs.sum() * 0 + torch.tensor(
            1e-8, device=device, requires_grad=True
        )

    def _compute_inter_subject_ranking(self, events, durations, survival_probs):
        """Compute the NeuralSort loss for ranking events across different subjects.

        For each event type, we learn a soft permutation of subjects based on when they
        experienced the event using the NeuralSort operator.

        Args:
            events: Binary indicators of events. Shape [batch_size, num_events]
            durations: Time to events. Shape [batch_size, num_events]
            survival_probs: Survival probabilities. Shape [batch_size, num_events, num_time_points]

        Returns:
            loss: The computed inter-subject ranking loss
        """
        batch_size, num_events, num_time_points = survival_probs.shape
        device = durations.device

        # Initialize loss
        loss = torch.tensor(0.0, device=device)
        valid_event_types = 0

        # Compute risk scores as the negative mean survival probability
        # Higher risk means lower survival probability
        risk_scores = -torch.mean(survival_probs, dim=2)  # [batch_size, num_events]

        # For each event type
        for event_idx in range(num_events):
            event_mask = events[:, event_idx] > 0
            if (
                torch.sum(event_mask) <= 1
            ):  # Need at least 2 subjects with this event for comparison
                continue

            # Get data for subjects with observed events
            observed_durations = durations[event_mask, event_idx]
            observed_risks = risk_scores[event_mask, event_idx]

            # Skip if any durations are non-positive
            if torch.any(observed_durations <= 0):
                continue

            # Target ranks should put earlier events first (lower rank)
            # argsort returns indices that would sort the tensor in ascending order
            target_ranks = torch.argsort(observed_durations)

            # Compute NeuralSort loss for this event type
            event_loss = self._neural_sort_loss(observed_risks, target_ranks)

            # Add to total loss weighted by event weight
            loss += event_loss * self.weights[event_idx + 1]
            valid_event_types += 1

        # Return average loss
        if valid_event_types > 0:
            return loss / valid_event_types
        # Return a small value connected to the computation graph
        return survival_probs.sum() * 0 + torch.tensor(
            1e-8, device=device, requires_grad=True
        )
