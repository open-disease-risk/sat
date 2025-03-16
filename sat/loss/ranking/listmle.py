"""ListMLE: Listwise approach to ranking for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch

from sat.models.tasks.heads import SAOutput
from sat.loss import Loss


class ListMLERankingLoss(Loss):
    """ListMLE Ranking loss for survival analysis.

    A list-based ranking loss that directly optimizes the likelihood of the correct ranking order,
    rather than comparing pairs. This approach is much more efficient than pair-wise ranking.

    It has two components:
    1. Inter-subject component: for a single event type, orders subjects by their event times
    2. Intra-subject component: for a single subject, orders events by their occurrence times

    This implementation is inspired by the ListMLE loss from "Listwise Approach to Learning to Rank - Theory and Algorithm".
    """

    def __init__(
        self,
        temperature: float = 0.1,
        lambda_intra: float = 1.0,
        lambda_inter: float = 1.0,
        eps: float = 1e-10,
        importance_sample_weights: str = None,
        num_events: int = 1,
        balance_strategy="none",
        ema_momentum=0.9,
    ):
        """Initialize the ListMLE ranking loss.

        Args:
            temperature: Temperature parameter for softmax, controls how strict the ranking is (default: 0.1)
            lambda_intra: Weight for the intra-subject ranking loss (default: 1.0)
            lambda_inter: Weight for the inter-subject ranking loss (default: 1.0)
            eps: Small value for numerical stability (default: 1e-10)
            importance_sample_weights: Path to importance sampling weights (default: None)
            num_events: Number of event types (default: 1)
        """
        super().__init__(
            num_events, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )
        self.temperature = temperature
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter
        self.eps = eps  # Small constant for numerical stability

        # Load importance sampling weights if provided
        if importance_sample_weights is not None:
            import pandas as pd

            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

        # Initialize loss components for the loss balancing API
        self._init_loss_params(["intra", "inter"])

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Calculate the ListMLE ranking loss.

        Args:
            predictions: Survival analysis model outputs, containing survival probabilities
            references: Reference values (batch size x 4*num_events)

        Returns:
            torch.Tensor: The computed ranking loss
        """
        # Extract risk directly from SAOutput
        risk = predictions.risk  # Using risk (cumulative incidence function)

        # Extract events and durations from references (torch.Tensor)
        events = self.events(references)
        durations = self.durations(references)

        # Initialize both losses
        intra_loss = risk.sum() * 0
        inter_loss = risk.sum() * 0

        # Only compute intra-subject loss if lambda_intra > 0
        if self.lambda_intra > 0:
            intra_loss = self._compute_intra_subject_ranking(risk, events, durations)

        # Only compute inter-subject loss if lambda_inter > 0
        if self.lambda_inter > 0:
            inter_loss = self._compute_inter_subject_ranking(risk, events, durations)

        if self.lambda_intra > 0 and self.lambda_inter > 0:
            final_loss = self.balance_losses(
                {"intra": intra_loss, "inter": inter_loss},
                {"intra": self.lambda_intra, "inter": self.lambda_inter},
            )

            return final_loss
        elif self.lambda_intra > 0:
            return intra_loss
        elif self.lambda_inter > 0:
            return inter_loss
        else:
            # Return a small value connected to the computation graph to enable gradient flow
            return risk.sum() * 0 + 1e-6

    def _compute_intra_subject_ranking(
        self, risk: torch.Tensor, events: torch.Tensor, durations: torch.Tensor
    ) -> torch.Tensor:
        """Compute ListMLE ranking loss for events within the same subject using log-space operations.

        For each subject, this optimizes for the correct ordering of events based on their occurrence times.

        Args:
            risk: Risk scores [batch_size, num_events, time_points]
            events: Event indicators [batch_size, num_events]
            durations: Event times [batch_size, num_events]

        Returns:
            torch.Tensor: Intra-subject ranking loss
        """
        batch_size = events.size(0)

        # Skip if we only have one event type
        if self.num_events <= 1:
            # Return a small value connected to the computation graph
            return risk.sum() * 0 + 1e-8

        # Compute average risk across time points for each event type using einsum
        # This is more efficient than mean operation
        # Shape: [batch_size, num_events]
        risk_scores = torch.einsum("bep->be", risk) / risk.size(2)

        # Find subjects with at least 2 observed events
        # Shape: [batch_size]
        observed_event_counts = torch.sum(
            events.abs() > 0.5, dim=1
        )  # Use abs to handle potential floating point issues
        valid_subjects_mask = observed_event_counts > 1
        valid_subjects_count = valid_subjects_mask.sum().item()

        # If no valid subjects, return small loss connected to computation graph
        if valid_subjects_count == 0:
            return risk.sum() * 0 + 1e-8

        # For tracking valid subject losses
        valid_subject_losses = []

        # Get valid subject indices
        valid_indices = torch.nonzero(valid_subjects_mask).squeeze(-1)

        # Process subjects in parallel where possible
        for i in valid_indices:
            # Get observed events and their times
            observed_mask = events[i].abs() > 0.5  # More robust comparison
            observed_events_idx = torch.nonzero(observed_mask).squeeze(-1)

            # Check if we have valid observed events
            if observed_events_idx.dim() == 0 or observed_events_idx.size(0) <= 1:
                continue  # Skip if there are not enough observed events

            observed_times = durations[i, observed_mask]

            # Sort event indices by occurrence time
            time_order = torch.argsort(observed_times)
            ordered_event_idx = observed_events_idx[time_order]

            # Get risk scores for this subject
            subject_risk_scores = risk_scores[i]

            # Apply ListMLE loss on ordered risk scores and accumulate
            event_loss = self._listMLE_loss(subject_risk_scores, ordered_event_idx)
            valid_subject_losses.append(event_loss)

        # If we have valid subject losses, stack and average them
        if valid_subject_losses:
            # Stack losses into a tensor to maintain computational graph
            stacked_losses = torch.stack(valid_subject_losses)
            return stacked_losses.mean()
        else:
            # Fallback to a small value connected to the computation graph
            return risk.sum() * 0 + 1e-8

    def _compute_inter_subject_ranking(
        self, risk: torch.Tensor, events: torch.Tensor, durations: torch.Tensor
    ) -> torch.Tensor:
        """Compute ListMLE ranking loss comparing the same event across different subjects using log-space operations.

        For each event type, this optimizes for the correct ordering of subjects based on when they
        experienced the event. Uses log-space operations for improved numerical stability.

        Args:
            risk: Risk scores [batch_size, num_events, time_points]
            events: Event indicators [batch_size, num_events]
            durations: Event times [batch_size, num_events]

        Returns:
            torch.Tensor: Inter-subject ranking loss
        """
        # Compute average risk across time points for each event type using einsum
        # Shape: [batch_size, num_events]
        risk_scores = torch.einsum("bep->be", risk) / risk.size(2)

        # Find events with at least 2 subjects
        event_counts = torch.sum(
            events.abs() > 0.5, dim=0
        )  # [num_events] - more robust comparison
        valid_events_mask = event_counts > 1
        valid_events_count = valid_events_mask.sum().item()

        # If no valid events, return small loss connected to computation graph
        if valid_events_count == 0:

            return risk.sum() * 0 + 1e-8

        # For tracking valid event losses
        valid_event_losses = []
        valid_event_weights = []

        # Get valid event indices
        valid_event_indices = torch.nonzero(valid_events_mask).squeeze(-1)

        # Process each valid event
        for event_idx in valid_event_indices:
            # Get subjects with this event
            event_mask = events[:, event_idx].abs() > 0.5

            # Get subject indices and their event times
            subject_indices = torch.nonzero(event_mask).squeeze(-1)

            # Check if we have enough subjects with this event
            if subject_indices.dim() == 0 or subject_indices.size(0) <= 1:
                continue  # Skip if not enough subjects have this event

            event_times = durations[subject_indices, event_idx]

            # Sort subjects by event time
            time_order = torch.argsort(event_times)
            ordered_subjects = subject_indices[time_order]

            # Get risk scores for this event across all subjects
            event_risk_scores = risk_scores[:, event_idx]

            # Compute event weight
            event_weight = self.weights[event_idx + 1]

            # Apply ListMLE loss on ordered risk scores
            event_loss = self._listMLE_loss(event_risk_scores, ordered_subjects)

            # Store the loss and weight
            valid_event_losses.append(event_loss * event_weight)
            valid_event_weights.append(event_weight)

        # If we have valid event losses, compute weighted average
        if valid_event_losses:
            # Stack losses and compute weighted average
            stacked_losses = torch.stack(valid_event_losses)
            return stacked_losses.mean()
        else:
            # Fallback to a small value connected to the computation graph

            return risk.sum() * 0 + 1e-8

    def log_plackett_luce(self, scores, permutation):
        """Compute the Plackett-Luce model directly in log-space for numerical stability.

        The Plackett-Luce model defines the probability of a permutation as the product of
        conditional probabilities of selecting an item at each step, given the previous selections.
        By computing this directly in log-space, we avoid numerical underflow/overflow.

        More efficient implementation using masks and vectorized operations where possible.

        Args:
            scores: Model scores for each item [list_length]
            permutation: Target permutation - indices that would sort items by desired order [list_length]

        Returns:
            torch.Tensor: Negative log-likelihood (loss) for this permutation
        """
        list_length = len(permutation)

        # If permutation is empty or has just one element, the loss is undefined
        if list_length <= 1:
            # Return a small value connected to the computation graph
            return scores.sum() * 0 + 1e-8

        # Scale scores by temperature
        scaled_scores = scores / self.temperature

        # For storing the negative log probabilities
        neg_log_probs = []

        # The core Plackett-Luce computation still requires sequential processing
        # because each step depends on previous selections
        for i in range(list_length):
            # Get scores for remaining items - this is inherently sequential
            remaining_items = permutation[i:]
            remaining_scores = scaled_scores[remaining_items]

            # Compute log probability of selecting correct item at position i
            correct_score = scaled_scores[permutation[i]]
            log_prob = correct_score - torch.logsumexp(remaining_scores, dim=0)

            # Store the negative log probability
            neg_log_probs.append(-log_prob)

        # Sum all negative log probabilities to get the total loss
        # This approach maintains gradients better than incremental accumulation
        return torch.stack(neg_log_probs).sum()

    def _listMLE_loss(self, scores: torch.Tensor, target_ordering: torch.Tensor):
        """Compute ListMLE loss using log-space operations for numerical stability.

        This implementation computes the Plackett-Luce model directly in log-space,
        avoiding exponential operations that can cause numerical instability.

        Args:
            scores: Model scores for each item [list_length] or [batch_size, list_length]
            target_ordering: Target permutation - indices that would sort items by desired order [list_length]

        Returns:
            torch.Tensor: ListMLE loss for this list
        """
        # Handle batched case
        if len(scores.shape) > 1:
            return self._listMLE_loss_batched(scores, target_ordering)

        # Check if target ordering has enough elements
        if target_ordering.numel() <= 1:

            # Return a small loss connected to the computation graph
            return scores.sum() * 0 + 1e-8

        return self.log_plackett_luce(scores, target_ordering)

    def _listMLE_loss_batched(
        self, scores: torch.Tensor, target_ordering: torch.Tensor
    ):
        """Handle ListMLE loss for batched inputs using log-space operations.
        Optimized implementation with reduced loops where possible.

        Args:
            scores: Model scores with batch dim [batch_size, list_length]
            target_ordering: Target permutation [list_length] or [batch_size, list_length]

        Returns:
            torch.Tensor: Average ListMLE loss across the batch
        """
        batch_size = scores.size(0)

        # For storing individual losses
        batch_losses = []

        # Single target ordering for all batch items
        if len(target_ordering.shape) == 1:
            # Check if we have enough elements in target_ordering
            if target_ordering.numel() <= 1:
                return scores.sum() * 0 + 1e-8

            # We still need to iterate since each batch item needs its own Plackett-Luce calculation
            for i in range(batch_size):
                loss = self.log_plackett_luce(scores[i], target_ordering)
                batch_losses.append(loss)
        else:
            # Different target ordering for each batch item
            for i in range(batch_size):
                # Skip if target_ordering is empty or has just one element
                if target_ordering[i].numel() <= 1:
                    continue

                loss = self.log_plackett_luce(scores[i], target_ordering[i])
                batch_losses.append(loss)

        # If we collected any losses, stack them and compute mean
        if batch_losses:
            stacked_losses = torch.stack(batch_losses)
            return stacked_losses.mean()
        else:
            # Return a small value connected to the computation graph
            return scores.sum() * 0 + 1e-8
