"""Contrastive Time-to-Event Loss for survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torch.nn.functional as F

from sat.loss.base import Loss


class SurvContrastiveLoss(Loss):
    """Implementation of Contrastive Time-to-Event Loss.

    This loss creates positive pairs based on similar survival times
    and negative pairs based on different survival times to learn
    better representations for survival analysis.

    References:
        Maksoud, S.D., et al. (2023).
        SurvCLR: Self-Supervised Contrastive Learning for Survival Analysis.
        AAAI Conference on Artificial Intelligence.
    """

    def __init__(
        self,
        temperature=0.1,  # Temperature for softmax
        margin=0.1,  # Margin for contrastive loss
        time_threshold=30,  # Time threshold for considering similar times (in days)
        use_embeddings=True,  # Whether to use embeddings or predictions
        num_events=1,
    ):
        """Initialize Contrastive Time-to-Event Loss.

        Args:
            temperature: Temperature parameter for softmax
            margin: Margin for contrastive loss
            time_threshold: Threshold (in days) for considering two survival times similar
            use_embeddings: If True, use encoder embeddings. If False, use predictions.
            num_events: Number of competing events
        """
        super().__init__(
            num_events, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )
        self.temperature = temperature
        self.margin = margin
        self.time_threshold = time_threshold
        self.use_embeddings = use_embeddings

    def forward(self, predictions, references):
        """Compute the contrastive loss.

        Args:
            predictions: Model output with embeddings and predictions
            references: Target tensor containing event types and durations

        Returns:
            loss: The computed contrastive loss value
        """
        # Extract events and durations from references
        events = self.events(references)
        durations = self.durations(references)

        # Use embeddings or predictions based on configuration
        if self.use_embeddings:
            if hasattr(predictions, "embeddings"):
                features = predictions.embeddings
            else:
                raise ValueError(
                    "Predictions must have 'embeddings' attribute when use_embeddings=True"
                )
        else:
            # Use the model's output values
            if hasattr(predictions, "values"):
                features = predictions.values
            elif hasattr(predictions, "surv"):
                # Use survival function values at a reference time
                features = predictions.surv[:, :, 0]  # Use first time point
            else:
                raise ValueError(
                    "Predictions must have 'values' or 'surv' attribute when use_embeddings=False"
                )

        # Flatten features if needed
        if features.dim() > 2:
            batch_size = features.size(0)
            features = features.reshape(
                batch_size, -1
            )  # Flatten to [batch_size, feature_dim]

        # Normalize features for cosine similarity
        features_norm = F.normalize(features, p=2, dim=1)

        # Initialize loss
        loss = torch.zeros(1, device=features.device)

        # Handle each event type separately
        for event_type in range(self.num_events):
            # Get event indicator for current event
            event_indicator = (events == event_type + 1).float()

            # Skip if no events of this type
            if event_indicator.sum() == 0:
                continue

            # Get indices of patients with this event type
            event_indices = torch.where(event_indicator > 0)[0]

            # Compute contrastive loss for this event type
            event_loss = self._compute_contrastive_loss(
                features_norm, durations, event_indices, self.time_threshold
            )

            loss += event_loss

        # Handle censored observations
        censored_indicator = (events == 0).float()
        if censored_indicator.sum() > 0:
            censored_indices = torch.where(censored_indicator > 0)[0]
            censored_loss = self._compute_contrastive_loss(
                features_norm, durations, censored_indices, self.time_threshold
            )
            loss += censored_loss

        # Normalize by number of event types + 1 (for censoring)
        loss /= self.num_events + 1

        return self.balance_losses({"survcontrastive": loss}, {"survcontrastive": 1.0})

    def _compute_contrastive_loss(self, features, durations, indices, threshold):
        """Compute contrastive loss for a subset of patients.

        Args:
            features: Normalized feature vectors [batch_size, feature_dim]
            durations: Event/censoring times [batch_size]
            indices: Indices of patients to consider
            threshold: Time threshold for considering two times similar

        Returns:
            loss: Contrastive loss value
        """
        if len(indices) <= 1:
            return torch.zeros(1, device=features.device)

        # Get features and durations for the subset
        subset_features = features[indices]
        subset_durations = durations[indices]

        # Compute pairwise time differences
        time_diff = torch.abs(
            subset_durations.unsqueeze(1) - subset_durations.unsqueeze(0)
        )

        # Create similarity matrix based on time differences
        # Pairs with time difference <= threshold are similar (positive pairs)
        similarity_matrix = (time_diff <= threshold).float()

        # Remove self-comparisons
        identity_matrix = torch.eye(len(indices), device=features.device)
        similarity_matrix = similarity_matrix * (1 - identity_matrix)

        # Compute pairwise cosine similarity
        cosine_sim = torch.mm(subset_features, subset_features.t())

        # For numerical stability, clip values
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)

        # Scale by temperature
        cosine_sim = cosine_sim / self.temperature

        # Compute contrastive loss
        loss = 0.0

        # For each anchor
        for i in range(len(indices)):
            # Skip if no positive pairs
            if similarity_matrix[i].sum() == 0:
                continue

            # Get positive and negative pairs
            positives = torch.where(similarity_matrix[i] > 0)[0]
            negatives = torch.where(similarity_matrix[i] == 0)[0]

            # Skip if no negative pairs
            if len(negatives) == 0:
                continue

            # For each positive pair
            for pos_idx in positives:
                # Skip if same index (should not happen due to identity matrix)
                if pos_idx == i:
                    continue

                # Positive similarity
                pos_sim = cosine_sim[i, pos_idx]

                # Negative similarities
                neg_sims = cosine_sim[i, negatives]

                # InfoNCE loss: -log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sim))))
                pos_exp = torch.exp(pos_sim)
                neg_exp_sum = torch.sum(torch.exp(neg_sims))

                # Add to loss
                sample_loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-8))
                loss += sample_loss

            # Normalize by number of positive pairs
            loss /= len(positives)

        # Normalize by number of anchors with positive pairs
        num_valid_anchors = sum(
            1 for i in range(len(indices)) if similarity_matrix[i].sum() > 0
        )
        if num_valid_anchors > 0:
            loss /= num_valid_anchors

        return self.balance_losses({"survcontrastive": loss}, {"survcontrastive": 1.0})

    def _alternative_contrastive_loss(self, features, durations, event_indicator):
        """Alternative implementation using triplet loss formulation.

        This is another approach to contrastive learning that uses triplet loss
        with hard negative mining.

        Args:
            features: Normalized feature vectors [batch_size, feature_dim]
            durations: Event/censoring times [batch_size]
            event_indicator: Event indicators (1 if event, 0 if censored) [batch_size]

        Returns:
            loss: Triplet loss value
        """
        batch_size = features.size(0)

        # Compute pairwise time differences
        time_diff = torch.abs(durations.unsqueeze(1) - durations.unsqueeze(0))

        # Create similarity matrix based on time differences
        # Pairs with time difference <= threshold are similar (positive pairs)
        similarity_matrix = (time_diff <= self.time_threshold).float()

        # Remove self-comparisons
        identity_matrix = torch.eye(batch_size, device=features.device)
        similarity_matrix = similarity_matrix * (1 - identity_matrix)

        # Compute pairwise cosine similarity
        cosine_sim = torch.mm(features, features.t())

        # For numerical stability, clip values
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)

        # Initialize loss
        loss = torch.zeros(1, device=features.device)

        # For each anchor
        for i in range(batch_size):
            # Skip if no positive pairs
            if similarity_matrix[i].sum() == 0:
                continue

            # Get positive pairs (similar time)
            positives = torch.where(similarity_matrix[i] > 0)[0]

            # Get hardest positive (most dissimilar in feature space)
            pos_sims = cosine_sim[i, positives]
            hardest_pos_idx = positives[torch.argmin(pos_sims)]
            hardest_pos_sim = cosine_sim[i, hardest_pos_idx]

            # Get negative pairs (dissimilar time)
            negatives = torch.where(similarity_matrix[i] == 0)[0]

            # Skip if no negative pairs
            if len(negatives) == 0:
                continue

            # Get hardest negative (most similar in feature space)
            neg_sims = cosine_sim[i, negatives]
            hardest_neg_idx = negatives[torch.argmax(neg_sims)]
            hardest_neg_sim = cosine_sim[i, hardest_neg_idx]

            # Compute triplet loss with margin
            # triplet_loss = max(0, hardest_neg_sim - hardest_pos_sim + margin)
            triplet_loss = torch.relu(hardest_neg_sim - hardest_pos_sim + self.margin)

            # Add to loss
            loss += triplet_loss

        # Normalize by number of valid anchors
        num_valid_anchors = sum(
            1 for i in range(batch_size) if similarity_matrix[i].sum() > 0
        )
        if num_valid_anchors > 0:
            loss /= num_valid_anchors

        return self.balance_losses({"survcontrastive": loss}, {"survcontrastive": 1.0})
