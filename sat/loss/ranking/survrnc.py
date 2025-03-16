"""SurvRNC: Survival Rank-N-Contrast Loss for learning ordered representations"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd
import torch
import torch.nn.functional as F

from sat.models.tasks.heads import SAOutput
from sat.loss import Loss


class SurvRNCLoss(Loss):
    """SurvRNC: Survival Rank-N-Contrast Loss for learning ordered representations.

    This implementation is based on the SurvRNC algorithm proposed by Saeed et al. (2024)
    where patients are ranked with respect to their time-to-event differences with an anchor.

    The loss function enforces the learned representation to have higher similarity between
    an anchor patient and a positive pair patient (with similar time-to-event) compared to
    negative pairs (with larger time-to-event differences). It explicitly handles censored data
    by giving less weight to uncertain pairs where the real time difference is unknown.

    Mathematical formulation:
    Let A be an anchor patient and P be a positive pair with time-to-event T_A and T_P.
    Let N be a negative pair with time-to-event T_N, where |T_A - T_N| > |T_A - T_P|.

    The loss aims to maximize: sim(f(A), f(P)) > sim(f(A), f(N))
    where f(·) is the feature representation of a patient.

    References:
    - Saeed, N. et al. (2024). SurvRNC: Learning Ordered Representations for Survival
      Prediction using Rank-N-Contrast. In Proceedings of MICCAI 2024.
      (https://arxiv.org/abs/2403.10603)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.1,
        batch_samples: int = 16,
        importance_sample_weights: str = None,
        num_events: int = 1,
        balance_strategy: str = "none",
        ema_momentum: float = 0.9,
    ):
        """Initialize SurvRNCLoss.

        Args:
            temperature: Temperature parameter for scaling similarities. Lower values
                         make the model more sensitive to differences. Default: 0.1
            margin: Margin for the contrastive ranking loss. Default: 0.1
            batch_samples: Number of samples to use for contrastive pairs. Default: 16
            importance_sample_weights: Path to importance sampling weights. Default: None
            num_events: Number of event types. Default: 1
            balance_strategy: Strategy for balancing loss components ("none", "scale_normalization",
                             "grad_normalization", "homoscedastic", or "adaptive"). Default: "none"
            ema_momentum: Momentum for exponential moving average when using scale normalization. Default: 0.9
        """
        super().__init__(
            num_events, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )

        self.temperature = temperature
        self.margin = margin
        self.batch_samples = batch_samples

        # Load importance sampling weights if provided
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute the SurvRNC loss.

        Args:
            predictions: Model output containing survival probabilities.
                Shape: [batch_size, num_events, num_time_points]
            references: Target tensor containing event indicators and durations.
                Shape: [batch_size, 4*num_events]

        Returns:
            loss: The computed SurvRNC loss value
        """
        # Extract features from the model output
        features = predictions.hidden_states[-1]  # Shape: [batch_size, feature_dim]

        # Extract events and durations
        events = self.events(references)  # Shape: [batch_size, num_events]
        durations = self.durations(references)  # Shape: [batch_size, num_events]

        # Initialize the loss
        loss = torch.tensor(0.0, device=features.device)

        # Process each event type
        for event_idx in range(self.num_events):
            event_loss = self._compute_event_rnc_loss(
                features=features,
                events=events[:, event_idx],
                durations=durations[:, event_idx],
            )
            loss += event_loss * self.weights[event_idx + 1]

        return loss

    def _compute_event_rnc_loss(
        self, features: torch.Tensor, events: torch.Tensor, durations: torch.Tensor
    ) -> torch.Tensor:
        """Compute the SurvRNC loss for a single event type.

        Args:
            features: Feature representations. Shape: [batch_size, feature_dim]
            events: Binary indicators of events. Shape: [batch_size]
            durations: Time to events. Shape: [batch_size]

        Returns:
            loss: The computed contrastive ranking loss
        """
        batch_size = features.size(0)
        device = features.device

        # Skip loss calculation if batch is too small
        if batch_size < 3:  # Need at least anchor, positive, and negative
            return torch.tensor(0.0, device=device)

        # Limit samples to conserve memory
        actual_samples = min(self.batch_samples, batch_size)

        # Randomly sample anchors
        anchor_indices = torch.randperm(batch_size, device=device)[:actual_samples]
        anchor_features = features[anchor_indices]
        anchor_events = events[anchor_indices]
        anchor_durations = durations[anchor_indices]

        # Normalize features (for cosine similarity)
        features_norm = F.normalize(features, p=2, dim=1)
        anchor_features_norm = F.normalize(anchor_features, p=2, dim=1)

        # Compute pairwise similarities
        similarities = torch.matmul(
            anchor_features_norm, features_norm.t()
        )  # [samples, batch_size]

        # Compute time differences for ranking
        # For censored samples, we only know the lower bound of survival time
        expanded_durations = durations.unsqueeze(0).expand(
            actual_samples, -1
        )  # [samples, batch_size]
        expanded_events = events.unsqueeze(0).expand(
            actual_samples, -1
        )  # [samples, batch_size]
        anchor_durations_expanded = anchor_durations.unsqueeze(1).expand(
            -1, batch_size
        )  # [samples, batch_size]
        anchor_events_expanded = anchor_events.unsqueeze(1).expand(
            -1, batch_size
        )  # [samples, batch_size]

        # Compute absolute time differences
        time_diffs = torch.abs(
            anchor_durations_expanded - expanded_durations
        )  # [samples, batch_size]

        # Create masks for different pair types
        # We want to handle different censoring scenarios

        # Mask for valid pairs: both anchor and other sample are uncensored (known true time)
        valid_pairs_mask = (anchor_events_expanded == 1) & (
            expanded_events == 1
        )  # [samples, batch_size]

        # Mask for semi-certain pairs: anchor is uncensored but the other sample is censored
        # We can use these as lower-confidence negative pairs (we know minimum time difference)
        semi_certain_pairs_mask = (anchor_events_expanded == 1) & (
            expanded_events == 0
        )  # [samples, batch_size]

        # Self-mask to exclude anchors from being their own pairs
        self_mask = torch.zeros(
            actual_samples, batch_size, dtype=torch.bool, device=device
        )
        for i, idx in enumerate(anchor_indices):
            self_mask[i, idx] = True

        # Compute loss for each anchor
        loss = torch.tensor(0.0, device=device)
        count = 0

        for i in range(actual_samples):
            # Skip if anchor is censored and we don't have enough valid pairs
            if anchor_events[i] == 0 and valid_pairs_mask[i].sum() < 2:
                continue

            # Create mask for valid pairs for this anchor (excluding self)
            valid_mask = valid_pairs_mask[i] & ~self_mask[i]  # [batch_size]

            # Create mask for semi-certain pairs (censored samples) for this anchor
            semi_certain_mask = (
                semi_certain_pairs_mask[i] & ~self_mask[i]
            )  # [batch_size]

            # If we don't have enough valid pairs, we can include semi-certain pairs as well
            if valid_mask.sum() < 2:  # Need at least positive and negative
                if (valid_mask.sum() + semi_certain_mask.sum()) < 2:
                    continue
                # We have at least one valid and some semi-certain pairs
                valid_indices_list = []
                valid_time_diffs_list = []

                # Add valid pairs first
                if valid_mask.sum() > 0:
                    valid_indices = torch.nonzero(valid_mask).squeeze(-1)
                    valid_time_diffs = time_diffs[i][valid_mask]
                    valid_indices_list.append(valid_indices)
                    valid_time_diffs_list.append(valid_time_diffs)

                # Add semi-certain pairs (with a weight factor)
                if semi_certain_mask.sum() > 0:
                    semi_certain_indices = torch.nonzero(semi_certain_mask).squeeze(-1)
                    semi_certain_time_diffs = time_diffs[i][semi_certain_mask]
                    valid_indices_list.append(semi_certain_indices)
                    valid_time_diffs_list.append(semi_certain_time_diffs)

                # Combine both types of pairs
                valid_indices = torch.cat(valid_indices_list)
                valid_time_diffs = torch.cat(valid_time_diffs_list)
            else:

                # Get time differences for valid pairs
                valid_time_diffs = time_diffs[i][valid_mask]  # [valid_count]
                valid_indices = torch.nonzero(valid_mask).squeeze(-1)  # [valid_count]

            # Sort by time difference
            sorted_time_diffs, sort_indices = torch.sort(valid_time_diffs)
            sorted_valid_indices = valid_indices[sort_indices]

            # Select positive pair (smallest time difference)
            positive_idx = sorted_valid_indices[0]

            # Select negative pairs (larger time differences)
            negative_indices = sorted_valid_indices[1:]

            # Compute contrastive loss
            anchor_sim_positive = similarities[
                i, positive_idx
            ]  # Similarity with positive
            anchor_sim_negatives = similarities[
                i, negative_indices
            ]  # Similarities with negatives

            # Calculate loss using margin ranking
            # similarity(anchor, positive) should be higher than similarity(anchor, negative) + margin
            neg_count = anchor_sim_negatives.size(0)
            positive_expanded = anchor_sim_positive.expand(neg_count)

            # Check if we're using semi-certain pairs (censored samples)
            if semi_certain_mask.sum() > 0 and torch.any(
                torch.isin(
                    negative_indices, torch.nonzero(semi_certain_mask).squeeze(-1)
                )
            ):
                # Apply weight factor to uncertain pairs (lower confidence for censored samples)
                is_semi_certain = torch.tensor(
                    [
                        idx in torch.nonzero(semi_certain_mask).squeeze(-1)
                        for idx in negative_indices
                    ],
                    device=device,
                    dtype=torch.bool,
                )

                # Apply normal weight to certain pairs, reduced weight to semi-certain pairs
                weights = torch.ones_like(anchor_sim_negatives)
                weights[is_semi_certain] = 0.5  # Use half weight for censored samples

                ranking_loss = (
                    torch.relu(self.margin - positive_expanded + anchor_sim_negatives)
                    * weights
                )
                ranking_loss = ranking_loss.sum() / weights.sum()  # Weighted mean
            else:
                ranking_loss = torch.relu(
                    self.margin - positive_expanded + anchor_sim_negatives
                )
                ranking_loss = ranking_loss.mean()

            # Scale by temperature
            ranking_loss = ranking_loss / self.temperature

            loss += ranking_loss
            count += 1

        # Compute final loss
        if count > 0:
            return loss / count
        else:
            # Return small value connected to computation graph for gradient flow
            return features.sum() * 0 + torch.tensor(
                1e-8, device=device, requires_grad=True
            )
