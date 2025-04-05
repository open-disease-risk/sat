"""Survival Rank-N-Contrast (SurvRNC) loss for survival analysis"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch

from sat.models.heads import SAOutput
from sat.utils import logging

from ..base import RankingLoss

logger = logging.get_default_logger()


class SurvRNCLoss(RankingLoss):
    """
    Survival Rank-N-Contrast (SurvRNC) loss for survival analysis.

    This loss function combines ranking and contrastive learning approaches
    to learn better survival representations. It uses N-pair contrastive loss
    to enforce that patients with similar outcomes are closer in the embedding space
    than those with different outcomes.

    Performance characteristics:
    - More efficient than pairwise ranking losses (O(n) vs O(n²))
    - Provides more robust generalization by focusing on relative distances
    - Improved calibration compared to standard ranking approaches
    - Works well with mini-batch stochastic gradient descent
    - Optimized vectorized implementation for faster execution
    - Optional hard mining for improved performance with large batch sizes
    """

    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        num_events: int = 1,
        margin: float = 0.5,
        temperature: float = 0.1,
        reduction: str = "mean",
        use_hard_mining: bool = False,
        mining_ratio: float = 0.5,
    ):
        """
        Initialize SurvRNCLoss.

        Args:
            duration_cuts: Path to CSV file containing duration cut points
            importance_sample_weights: Optional path to CSV file with importance weights
            num_events: Number of competing events
            margin: Minimum margin between positive and negative pairs
            temperature: Temperature parameter to control the sharpness of similarity scores
            reduction: Reduction method ('mean', 'sum', or 'none')
            use_hard_mining: Whether to use hard negative/positive mining for large batches
            mining_ratio: Ratio of hardest examples to keep when mining is enabled
        """
        super(SurvRNCLoss, self).__init__(
            duration_cuts=duration_cuts,
            importance_sample_weights=importance_sample_weights,
            sigma=1.0,  # Not directly used in SurvRNC, but required by parent class
            num_events=num_events,
            margin=margin,
        )
        self.temperature = temperature
        self.reduction = reduction
        self.use_hard_mining = use_hard_mining
        self.mining_ratio = mining_ratio

    def compute_time_similarity_matrix(self, durations, event_indicators):
        """
        Efficiently compute similarity matrix between time-to-event values.

        Args:
            durations: Time-to-event values [batch_size, num_events]
            event_indicators: Event indicators [batch_size, num_events]

        Returns:
            Similarity matrix [batch_size, batch_size]
        """
        batch_size = durations.shape[0]
        device = durations.device

        # Calculate pairwise time differences efficiently
        # Using broadcasting to avoid loops
        time_diff = torch.abs(
            durations.unsqueeze(1) - durations.unsqueeze(0)
        )  # [batch, batch, num_events]

        # Scale time differences to [0, 1] using the max time in the dataset
        max_time = self.duration_cuts[-1].to(device)
        time_similarity = 1.0 - torch.clamp(time_diff / max_time, 0.0, 1.0)

        # Create event mask - only consider similarities for events that occurred
        event_mask = event_indicators.unsqueeze(1) * event_indicators.unsqueeze(
            0
        )  # [batch, batch, num_events]

        # Apply event mask and average over events - handle empty case
        masked_similarity = time_similarity * event_mask
        event_counts = torch.sum(event_mask, dim=2).clamp(min=1.0)

        # Compute weighted average over events
        similarity_matrix = torch.sum(masked_similarity, dim=2) / event_counts

        return similarity_matrix

    def interpolate_survival_batch(self, survival, durations, event_indicators):
        """
        Efficient batch interpolation of survival probabilities at event times.

        Args:
            survival: Survival probabilities [batch_size, num_events, num_time_bins+1]
            durations: Time-to-event values [batch_size, num_events]
            event_indicators: Event indicators [batch_size, num_events]

        Returns:
            Interpolated survival probabilities at event times [batch_size, num_events]
        """
        batch_size, num_events = durations.shape
        device = durations.device

        # Initialize output tensor
        survival_at_times = torch.zeros((batch_size, num_events), device=device)

        # Create mask for valid events
        valid_events = event_indicators == 1

        if not valid_events.any():
            return survival_at_times

        # Get indices for interpolation - vectorized implementation
        # Expand duration cuts for broadcasting
        cuts = self.duration_cuts.to(device)  # [num_cuts]

        # Reshape durations for broadcasting: [batch_size, num_events, 1]
        durations_expanded = durations.unsqueeze(2)

        # Expand cuts to match durations: [1, 1, num_cuts]
        cuts_expanded = cuts.view(1, 1, -1)

        # Create comparison matrix: [batch_size, num_events, num_cuts]
        index_smaller = cuts_expanded <= durations_expanded

        # Calculate indices for lower bound and upper bound
        t0_indices = torch.sum(index_smaller, dim=2) - 1  # [batch_size, num_events]
        t0_indices = torch.clamp(t0_indices, min=0)
        t1_indices = torch.clamp(t0_indices + 1, max=len(cuts) - 1)

        # Get time points for valid events
        flat_batch_indices = torch.arange(batch_size, device=device).repeat_interleave(
            num_events
        )
        flat_event_indices = torch.arange(num_events, device=device).repeat(batch_size)
        flat_valid_events = valid_events.view(-1)

        # Filter for valid events
        valid_batch_indices = flat_batch_indices[flat_valid_events]
        valid_event_indices = flat_event_indices[flat_valid_events]
        valid_t0_indices = t0_indices[valid_batch_indices, valid_event_indices]
        valid_t1_indices = t1_indices[valid_batch_indices, valid_event_indices]

        # Get time points for valid events
        t0 = cuts[valid_t0_indices]
        t1 = cuts[valid_t1_indices]

        # Get survival values for valid events
        s0 = survival[valid_batch_indices, valid_event_indices, valid_t0_indices]
        s1 = survival[valid_batch_indices, valid_event_indices, valid_t1_indices]

        # Calculate time differences
        dt = t1 - t0

        # Get valid durations
        valid_durations = durations[valid_batch_indices, valid_event_indices]

        # Interpolate survival values
        epsilon = 1e-6
        interpolated_survival = torch.zeros_like(s0)

        # Handle the case where dt > 0 (interpolation needed)
        interp_mask = dt > 0
        if interp_mask.any():
            # Compute hazard rate for interpolation
            log_s0 = torch.log(s0[interp_mask] + epsilon)
            log_s1 = torch.log(s1[interp_mask] + epsilon)
            h_star = (log_s0 - log_s1) / dt[interp_mask]

            # Interpolate
            interp_durations = valid_durations[interp_mask]
            interp_t0 = t0[interp_mask]
            interpolated_survival[interp_mask] = s0[interp_mask] * torch.exp(
                -(interp_durations - interp_t0) * h_star
            )

        # Handle dt == 0 case (no interpolation needed)
        interpolated_survival[~interp_mask] = s0[~interp_mask]

        # Update output tensor with interpolated values
        survival_at_times[valid_batch_indices, valid_event_indices] = (
            interpolated_survival
        )

        return survival_at_times

    def compute_risk_similarity_matrix(self, survival, durations, event_indicators):
        """
        Compute similarity matrix based on risk scores.

        Args:
            survival: Survival probabilities [batch_size, num_events, num_time_bins+1]
            durations: Time-to-event values [batch_size, num_events]
            event_indicators: Event indicators [batch_size, num_events]

        Returns:
            Risk similarity matrix [batch_size, batch_size]
        """
        # Get interpolated survival probabilities
        survival_at_times = self.interpolate_survival_batch(
            survival, durations, event_indicators
        )

        # Convert to risk scores (1 - survival)
        risk_at_times = 1.0 - survival_at_times  # [batch_size, num_events]

        # Calculate pairwise risk similarity
        risk_diff = torch.abs(
            risk_at_times.unsqueeze(1) - risk_at_times.unsqueeze(0)
        )  # [batch, batch, num_events]
        risk_similarity = 1.0 - torch.clamp(risk_diff, 0.0, 1.0)

        # Create event mask - only consider similarities for events that occurred
        event_mask = event_indicators.unsqueeze(1) * event_indicators.unsqueeze(
            0
        )  # [batch, batch, num_events]

        # Apply event mask and average over events
        masked_similarity = risk_similarity * event_mask
        event_counts = torch.sum(event_mask, dim=2).clamp(min=1.0)

        # Return weighted average
        similarity_matrix = torch.sum(masked_similarity, dim=2) / event_counts

        return similarity_matrix

    def compute_contrastive_loss(self, similarity_matrix, time_similarity_matrix):
        """
        Compute N-pair contrastive loss using efficient matrix operations.

        Args:
            similarity_matrix: Matrix of similarities between samples [batch_size, batch_size]
            time_similarity_matrix: Matrix of time similarities [batch_size, batch_size]

        Returns:
            N-pair contrastive loss
        """
        batch_size = similarity_matrix.shape[0]
        device = similarity_matrix.device

        # Apply temperature scaling
        scaled_similarity = similarity_matrix / self.temperature

        # Initialize loss
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_count = 0

        # For large batches, we can use approximate methods to improve performance
        if self.use_hard_mining and batch_size > 128:
            # Approximate with hard mining to reduce computation
            return self._compute_hard_mining_loss(
                scaled_similarity, time_similarity_matrix
            )

        # Create positive/negative masks based on time similarity
        for i in range(batch_size):
            # Get similarity with anchor i
            sim_i = scaled_similarity[i]
            time_sim_i = time_similarity_matrix[i]

            # Determine positive pairs (high time similarity) and negative pairs (low time similarity)
            # Use median-based thresholds
            pos_threshold = torch.median(time_sim_i) + self.margin / 2
            neg_threshold = torch.median(time_sim_i) - self.margin / 2

            # Create masks, excluding self
            pos_mask = (time_sim_i > pos_threshold) & (
                torch.arange(batch_size, device=device) != i
            )
            neg_mask = (time_sim_i < neg_threshold) & (
                torch.arange(batch_size, device=device) != i
            )

            # Skip if no positive or negative pairs
            if not pos_mask.any() or not neg_mask.any():
                continue

            # Get positive and negative similarities
            pos_sim = sim_i[pos_mask]
            neg_sim = sim_i[neg_mask]

            # Efficient N-pair contrastive loss computation
            # Use vectorized operations to compute log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sim))))
            neg_term = torch.logsumexp(neg_sim, dim=0)

            for pos_idx in range(len(pos_sim)):
                # Combine positive term with negative term using log-sum-exp trick
                combined_term = torch.logsumexp(
                    torch.cat([neg_sim, pos_sim[pos_idx].unsqueeze(0)]), dim=0
                )
                # Compute loss for this positive pair
                anchor_loss = -pos_sim[pos_idx] + combined_term
                loss = loss + anchor_loss
                valid_count += 1

        # Apply reduction
        if valid_count > 0:
            if self.reduction == "mean":
                loss = loss / valid_count
            elif self.reduction == "sum":
                pass  # Already summed
        else:
            loss = self.ensure_tensor(0.0, device=device)

        return loss

    def _compute_hard_mining_loss(self, scaled_similarity, time_similarity_matrix):
        """
        Compute loss using hard mining for efficiency with large batches.

        Args:
            scaled_similarity: Temperature-scaled similarity matrix [batch_size, batch_size]
            time_similarity_matrix: Time similarity matrix [batch_size, batch_size]

        Returns:
            Approximate contrastive loss
        """
        batch_size = scaled_similarity.shape[0]
        device = scaled_similarity.device

        # Create identity mask to exclude self-comparisons
        identity_mask = torch.eye(batch_size, device=device).bool()

        # Compute global median for thresholding
        median_sim = torch.median(time_similarity_matrix[~identity_mask])
        pos_threshold = median_sim + self.margin / 2
        neg_threshold = median_sim - self.margin / 2

        # Create global positive and negative masks
        pos_mask = (time_similarity_matrix > pos_threshold) & ~identity_mask
        neg_mask = (time_similarity_matrix < neg_threshold) & ~identity_mask

        # If either mask is empty, return zero loss
        if not pos_mask.any() or not neg_mask.any():
            return self.ensure_tensor(0.0, device=device)

        # For each anchor with positives, compute loss
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_count = 0

        # Find anchors with valid positive and negative pairs
        valid_anchors = (torch.sum(pos_mask, dim=1) > 0) & (
            torch.sum(neg_mask, dim=1) > 0
        )
        anchor_indices = torch.nonzero(valid_anchors).squeeze(-1)

        # If no valid anchors, return zero loss
        if len(anchor_indices) == 0:
            return self.ensure_tensor(0.0, device=device)

        # Hard mining ratio determines how many pairs to keep
        if self.mining_ratio < 1.0:
            # Calculate how many anchors to use
            num_anchors = max(1, int(len(anchor_indices) * self.mining_ratio))
            # Randomly select subset of anchors
            perm = torch.randperm(len(anchor_indices))
            anchor_indices = anchor_indices[perm[:num_anchors]]

        # Process each selected anchor
        for i in anchor_indices:
            # Get positive and negative pairs for this anchor
            anchor_pos = scaled_similarity[i][pos_mask[i]]
            anchor_neg = scaled_similarity[i][neg_mask[i]]

            # If using hard mining, select hardest positive and negative pairs
            if self.mining_ratio < 1.0:
                # For positives: select hardest (lowest similarity)
                if len(anchor_pos) > 1:
                    num_pos = max(1, int(len(anchor_pos) * self.mining_ratio))
                    anchor_pos = torch.topk(anchor_pos, k=num_pos, largest=False).values

                # For negatives: select hardest (highest similarity)
                if len(anchor_neg) > 1:
                    num_neg = max(1, int(len(anchor_neg) * self.mining_ratio))
                    anchor_neg = torch.topk(anchor_neg, k=num_neg, largest=True).values

            # Compute N-pair contrastive loss for this anchor
            neg_term = torch.logsumexp(anchor_neg, dim=0)

            for pos_sim in anchor_pos:
                combined_term = torch.logsumexp(
                    torch.cat([anchor_neg, pos_sim.unsqueeze(0)]), dim=0
                )
                anchor_loss = -pos_sim + combined_term
                loss = loss + anchor_loss
                valid_count += 1

        # Apply reduction
        if valid_count > 0:
            if self.reduction == "mean":
                loss = loss / valid_count
            elif self.reduction == "sum":
                pass  # Already summed
        else:
            loss = self.ensure_tensor(0.0, device=device)

        return loss

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the SurvRNC loss.

        Parameters:
            predictions (SAOutput): Predictions of the model with survival probabilities
            references (torch.Tensor): Reference values (dims: batch size x 4 * num_events)

        Returns:
            torch.Tensor: The computed loss value
        """
        # Extract event indicators and durations
        events = self.events(references)  # [batch_size, num_events]
        durations = self.durations(references)  # [batch_size, num_events]

        batch_size = events.shape[0]
        device = references.device

        # Skip if no events
        if torch.sum(events) == 0:
            return self.ensure_tensor(0.0, device=device)

        # Step 1: Compute time similarity matrix
        time_similarity = self.compute_time_similarity_matrix(durations, events)

        # Step 2: Compute risk similarity matrix
        risk_similarity = self.compute_risk_similarity_matrix(
            predictions.survival, durations, events
        )

        # Step 3: Compute contrastive loss
        loss = self.compute_contrastive_loss(risk_similarity, time_similarity)

        return loss
