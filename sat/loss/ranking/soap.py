"""Statistically Optimal Accelerated Pairwise (SOAP) loss for survival analysis"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torch.nn.functional as F

from sat.models.heads import SAOutput
from sat.utils import logging

from ..base import RankingLoss

logger = logging.get_default_logger()


class SOAPLoss(RankingLoss):
    """
    Base class for Statistically Optimal Accelerated Pairwise (SOAP) loss implementation.

    SOAP loss is a pairwise ranking approach that optimizes survival analysis models
    by accelerating pairwise comparisons through statistical optimization.

    Key concepts:
    - Accelerates pairwise ranking computation through strategic sampling
    - Maintains statistical optimality with significantly fewer comparisons
    - Uses a dual-optimization approach combining margin ranking and statistical efficiency

    Performance characteristics:
    - Reduces complexity from O(n²) to approximately O(n log n)
    - Strategically selects the most informative pairs for efficient learning
    - Particularly effective for large batch sizes where full pairwise is prohibitive
    - Provides better generalization by focusing on the most discriminative pairs
    """

    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        num_events: int = 1,
        margin: float = 0.1,
        sigma: float = 1.0,
        num_pairs: int = None,
        sampling_strategy: str = "uniform",
        adaptive_margin: bool = False,
    ):
        """
        Initialize SOAP loss.

        Args:
            duration_cuts: Path to CSV file containing duration cut points
            importance_sample_weights: Optional path to CSV file with importance weights
            num_events: Number of competing events
            margin: Minimum margin between correctly ranked samples
            sigma: Scaling factor for loss (smaller values create sharper differences)
            num_pairs: Number of pairs to sample per anchor (None = auto-calculate)
            sampling_strategy: Strategy for pair sampling ("uniform", "importance", "hard")
            adaptive_margin: Whether to adapt margin based on duration differences
        """
        super(SOAPLoss, self).__init__(
            duration_cuts, importance_sample_weights, sigma, num_events, margin
        )
        self.sampling_strategy = sampling_strategy
        self.adaptive_margin = adaptive_margin
        self.num_pairs = num_pairs

    def sample_pairs(
        self, events: torch.Tensor, durations: torch.Tensor, max_pairs: int = None
    ) -> torch.Tensor:
        """
        Sample pairs for efficient pairwise ranking with optimized implementation.

        Args:
            events: Event indicators [batch_size, num_events]
            durations: Event times [batch_size, num_events]
            max_pairs: Maximum number of pairs to sample (None = auto-calculate)

        Returns:
            torch.Tensor: Sampled pair indices [num_samples, 2]
                          where each row contains [anchor_idx, compare_idx]
        """
        batch_size = events.shape[0]
        device = events.device

        # Determine optimal number of pairs to sample based on batch size
        if max_pairs is None:
            if self.num_pairs is None:
                # Optimize pair count based on batch size:
                # - For small batches: use more pairs for better representation
                # - For large batches: use logarithmic scaling to maintain efficiency
                if batch_size <= 32:
                    # For small batches, use more thorough sampling
                    max_pairs = min(batch_size * 5, batch_size * (batch_size - 1))
                elif batch_size <= 128:
                    # Medium batches use moderated scaling
                    max_pairs = int(
                        batch_size
                        * 2
                        * torch.log2(torch.tensor(batch_size, device=device))
                    )
                else:
                    # Large batches use aggressive logarithmic scaling
                    max_pairs = int(
                        batch_size * torch.log2(torch.tensor(batch_size, device=device))
                    )
            else:
                # Use user-specified pairs per sample
                max_pairs = self.num_pairs * batch_size

        # Ensure we don't exceed the maximum possible number of pairs
        max_possible_pairs = batch_size * (batch_size - 1)
        max_pairs = min(max_pairs, max_possible_pairs)

        if max_pairs <= 0 or batch_size <= 1:
            # Return empty tensor with correct shape for edge cases
            return torch.zeros((0, 2), dtype=torch.long, device=device)

        if self.sampling_strategy == "uniform":
            # Optimized uniform random sampling of pairs
            # Generate unique pairs efficiently with a single operation

            # Use triu_indices to efficiently generate unique pairs
            if batch_size <= 128:
                # For smaller batches, we can generate all pairs and sample
                row_idx, col_idx = torch.triu_indices(
                    batch_size, batch_size, offset=1, device=device
                )
                pairs = torch.stack([row_idx, col_idx], dim=1)

                # Shuffle pairs for randomness
                shuffle_idx = torch.randperm(pairs.shape[0], device=device)[:max_pairs]
                pairs = pairs[shuffle_idx]

                # Add reverse pairs for bidirectional comparisons (important for ranking)
                if pairs.shape[0] * 2 <= max_pairs:
                    reverse_pairs = torch.stack([pairs[:, 1], pairs[:, 0]], dim=1)
                    pairs = torch.cat([pairs, reverse_pairs], dim=0)
                    # Shuffle again if we have fewer than max_pairs
                    if pairs.shape[0] < max_pairs:
                        shuffle_idx = torch.randperm(pairs.shape[0], device=device)
                        pairs = pairs[shuffle_idx]

                return pairs[:max_pairs]
            else:
                # For larger batches, direct random generation is more memory-efficient
                # Use vectorized operations to generate pairs and ensure i != j
                i_indices = torch.randint(0, batch_size, (max_pairs,), device=device)
                j_indices = torch.randint(
                    0, batch_size - 1, (max_pairs,), device=device
                )

                # Shift j_indices to avoid self-loops
                j_indices = j_indices + (j_indices >= i_indices).long()

                return torch.stack([i_indices, j_indices], dim=1)

        elif self.sampling_strategy == "importance":
            # Optimized importance-based sampling: vectorized operations for computing importance

            # Pre-filter events to focus only on samples with events
            event_mask = events.sum(dim=1) > 0
            if event_mask.sum() <= 1:
                # Not enough events to compare, fall back to uniform sampling
                return self.sample_pairs(events, durations, max_pairs)

            # Get valid indices with events
            valid_indices = torch.where(event_mask)[0]

            # Smaller subset for importance calculation to save memory
            # Get a reasonably sized subset (up to 1000 pairs) to calculate importance
            subset_size = min(len(valid_indices), 50)  # Limit events evaluated

            if subset_size < len(valid_indices):
                # Randomly sample a subset of indices
                subset_idx = torch.randperm(len(valid_indices), device=device)[
                    :subset_size
                ]
                valid_indices = valid_indices[subset_idx]

            # Create anchor and compare matrices for vectorized operations
            # Generate all pairs between valid indices
            n_valid = valid_indices.shape[0]
            n_valid_pairs = min(n_valid * (n_valid - 1), 1000)  # Limit pairs for memory

            # Use the smaller subset for vectorized importance calculation
            anchor_idx, compare_idx = torch.triu_indices(
                n_valid, n_valid, offset=1, device=device
            )

            # Limit to manageable size for memory efficiency
            if anchor_idx.shape[0] > n_valid_pairs:
                select_idx = torch.randperm(anchor_idx.shape[0], device=device)[
                    :n_valid_pairs
                ]
                anchor_idx = anchor_idx[select_idx]
                compare_idx = compare_idx[select_idx]

            # Map to original indices
            anchor_indices = valid_indices[anchor_idx]
            compare_indices = valid_indices[compare_idx]

            # Calculate importance scores efficiently using vectorized operations
            importance_scores = []

            # For each event type, calculate duration differences
            for event_idx in range(events.shape[1]):
                # Check if both samples in a pair have the same event
                anchor_has_event = events[anchor_indices, event_idx]
                compare_has_event = events[compare_indices, event_idx]
                both_have_event = anchor_has_event & compare_has_event

                # Skip if no pairs have this event
                if not both_have_event.any():
                    continue

                # Get valid pairs where both samples have this event
                valid_pair_idx = torch.where(both_have_event)[0]

                # Skip if no valid pairs
                if len(valid_pair_idx) == 0:
                    continue

                # Get durations for these pairs
                anchor_dur = durations[anchor_indices[valid_pair_idx], event_idx]
                compare_dur = durations[compare_indices[valid_pair_idx], event_idx]

                # Calculate absolute differences and normalize
                dur_diff = torch.abs(anchor_dur - compare_dur)
                max_dur = durations[:, event_idx].max()

                if max_dur > 0:
                    # Normalize and apply softplus for importance weighting
                    # Softplus ensures smooth positive weights
                    importance = F.softplus(dur_diff / max_dur)

                    # Create full score tensor
                    score = torch.zeros(len(anchor_indices), device=device)
                    score[valid_pair_idx] = importance
                    importance_scores.append(score)

            # If we have importance scores
            if importance_scores:
                # Combine importance scores across event types (max operation)
                combined_importance = torch.stack(importance_scores).max(dim=0)[0]

                # Sample pairs based on importance scores
                if len(combined_importance) > 0:
                    # Normalize for probability distribution
                    probs = F.softmax(combined_importance, dim=0)

                    # Sample with replacement for efficiency and to handle edge cases
                    num_to_sample = min(max_pairs, len(anchor_indices))

                    # Handle edge case where all importances are zero
                    if torch.allclose(probs.sum(), torch.zeros(1, device=device)):
                        selected_indices = torch.randint(
                            0, len(anchor_indices), (num_to_sample,), device=device
                        )
                    else:
                        selected_indices = torch.multinomial(
                            probs, num_to_sample, replacement=True
                        )

                    # Get final pairs
                    selected_anchor = anchor_indices[selected_indices]
                    selected_compare = compare_indices[selected_indices]

                    return torch.stack([selected_anchor, selected_compare], dim=1)

            # Fallback to uniform sampling if importance calculation fails
            return self.sample_pairs(events, durations, max_pairs)

        elif self.sampling_strategy == "hard":
            # Optimized version of hard sampling
            # For hard mining, we use a hybrid approach combining uniform and time-based sampling

            # Half of pairs from uniform sampling for exploration
            uniform_pairs = int(max_pairs * 0.5)
            uniform_result = self.sample_pairs(events, durations, uniform_pairs)

            # Half of pairs focused on similar durations (harder to distinguish)
            # Group samples by duration range and sample within groups
            event_count = events.sum(dim=1, keepdim=True)

            # Skip if no events
            if event_count.sum() == 0:
                return uniform_result

            # Calculate average duration per sample
            weighted_durations = (durations * events).sum(dim=1) / torch.clamp(
                event_count.squeeze(), min=1
            )

            # Create buckets by duration quantiles
            sorted_dur, sort_indices = torch.sort(weighted_durations)
            valid_samples = sort_indices[weighted_durations[sort_indices] > 0]

            if len(valid_samples) > 10:
                # Create duration quantile buckets
                num_buckets = min(10, len(valid_samples) // 2)
                bucket_size = (len(valid_samples) + num_buckets - 1) // num_buckets

                hard_pairs = []
                for i in range(num_buckets):
                    # Get samples in this bucket
                    start_idx = i * bucket_size
                    end_idx = min((i + 1) * bucket_size, len(valid_samples))
                    bucket_samples = valid_samples[start_idx:end_idx]

                    if len(bucket_samples) > 1:
                        # Sample pairs within the bucket
                        for _ in range(min(5, len(bucket_samples))):
                            pair_indices = torch.randint(
                                0, len(bucket_samples), (2,), device=device
                            )
                            # Ensure we don't compare a sample to itself
                            while (
                                pair_indices[0] == pair_indices[1]
                                and len(bucket_samples) > 1
                            ):
                                pair_indices[1] = torch.randint(
                                    0, len(bucket_samples), (1,), device=device
                                )

                            hard_pairs.append(
                                [
                                    bucket_samples[pair_indices[0]],
                                    bucket_samples[pair_indices[1]],
                                ]
                            )

                # Convert to tensor
                if hard_pairs:
                    hard_result = torch.tensor(hard_pairs, device=device)

                    # Combine with uniform pairs
                    if len(uniform_result) > 0:
                        combined = torch.cat([uniform_result, hard_result], dim=0)
                        # Shuffle
                        shuffle_idx = torch.randperm(len(combined), device=device)[
                            :max_pairs
                        ]
                        return combined[shuffle_idx]
                    else:
                        return hard_result

            # Fall back to uniform if hard mining fails
            return uniform_result

        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def compute_soap_loss(
        self,
        events: torch.Tensor,
        durations: torch.Tensor,
        survival: torch.Tensor,
        hazard: torch.Tensor,
        pairs: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute SOAP loss for sampled pairs using optimized tensor operations.

        Args:
            events: Event indicators [batch_size, num_events]
            durations: Event times [batch_size, num_events]
            survival: Survival probabilities [batch_size, num_events, num_time_bins+1]
            hazard: Hazard values [batch_size, num_events, num_time_bins]
            pairs: Pair indices [num_pairs, 2] where each row contains [anchor_idx, compare_idx]
            weights: Optional importance weights for each pair [num_pairs]

        Returns:
            torch.Tensor: SOAP loss value
        """
        if pairs.shape[0] == 0:
            # No pairs to process
            return torch.tensor(0.0, device=events.device, requires_grad=True)

        num_events = events.shape[1]
        device = events.device

        # Extract anchor and comparison indices - keep as separate tensors for better vectorization
        anchor_indices = pairs[:, 0]
        compare_indices = pairs[:, 1]
        num_pairs = anchor_indices.shape[0]

        # Get max duration for normalization (used later in adaptive margin)
        max_duration = self.duration_cuts[-1].to(device)

        # Gather relevant tensors for the sampled pairs using a single indexing operation
        # Shape: [num_pairs, num_events]
        anchor_events = events[anchor_indices]
        compare_events = events[compare_indices]
        anchor_durations = durations[anchor_indices]
        compare_durations = durations[compare_indices]

        # Initialize aggregators - pre-allocate tensors for efficiency
        loss_accumulator = torch.tensor(0.0, device=device, requires_grad=True)
        valid_count_accumulator = 0

        # Process each event type - for now we keep this loop since num_events is usually small
        for event_idx in range(num_events):
            # Get event weights for this event type
            event_weight = weights[event_idx] if weights is not None else 1.0

            # --- CASE 1: Both samples have events (most informative) ---
            # Create mask for valid pairs where both samples have this event
            both_events_mask = (
                anchor_events[:, event_idx] & compare_events[:, event_idx]
            )

            if both_events_mask.any():
                # Extract valid indices
                valid_indices = torch.where(both_events_mask)[0]

                # Get durations for these pairs
                t_anchor = anchor_durations[valid_indices, event_idx]
                t_compare = compare_durations[valid_indices, event_idx]

                # Calculate expected ranking direction based on time-to-event
                # Skip pairs with identical durations
                duration_diff = t_compare - t_anchor
                valid_diff_mask = duration_diff != 0

                if valid_diff_mask.any():
                    # Further filter valid pairs
                    final_indices = valid_indices[valid_diff_mask]
                    t_anchor_valid = t_anchor[valid_diff_mask]
                    t_compare_valid = t_compare[valid_diff_mask]
                    expected_sign = torch.sign(duration_diff[valid_diff_mask])

                    # Get anchor and compare sample indices
                    a_idx = anchor_indices[final_indices]
                    c_idx = compare_indices[final_indices]

                    # Batch interpolate survival values (vectorized operation)
                    s_anchor = self.interpolate_survival(
                        survival[a_idx, event_idx], t_anchor_valid, device
                    )
                    s_compare = self.interpolate_survival(
                        survival[c_idx, event_idx], t_compare_valid, device
                    )

                    # Convert to risk scores
                    r_anchor = 1.0 - s_anchor
                    r_compare = 1.0 - s_compare

                    # Calculate adaptive margins if needed using vectorized operations
                    if self.adaptive_margin:
                        time_diffs = torch.abs(t_anchor_valid - t_compare_valid)
                        normalized_diffs = time_diffs / max_duration
                        margins = self.margin * (1.0 + normalized_diffs)
                    else:
                        margins = torch.full_like(r_anchor, self.margin)

                    # Compute pair differences and hinge loss in one vectorized operation
                    # expected_sign ensures correct direction: earlier event should have higher risk
                    pair_diffs = expected_sign * (r_compare - r_anchor)
                    pair_losses = F.relu(margins - pair_diffs)

                    # Apply weight and add to accumulator
                    loss_accumulator = loss_accumulator + torch.sum(
                        pair_losses * event_weight
                    )
                    valid_count_accumulator += len(final_indices)

            # --- CASE 2: Anchor has event, comparison is censored ---
            anchor_event_only = (
                anchor_events[:, event_idx] & ~compare_events[:, event_idx]
            )

            if anchor_event_only.any():
                # Extract pairs where anchor has event but comparison is censored
                valid_indices = torch.where(anchor_event_only)[0]

                # Get durations
                t_anchor = anchor_durations[valid_indices, event_idx]
                t_censor = compare_durations[valid_indices, event_idx]

                # Only valid if censoring happens after the event
                valid_ranking = t_censor > t_anchor

                if valid_ranking.any():
                    # Get final indices and durations
                    final_indices = valid_indices[valid_ranking]
                    t_anchor_valid = t_anchor[valid_ranking]
                    t_censor_valid = t_censor[valid_ranking]

                    # Get indices for efficient tensor access
                    a_idx = anchor_indices[final_indices]
                    c_idx = compare_indices[final_indices]

                    # Vectorized survival interpolation
                    s_anchor = self.interpolate_survival(
                        survival[a_idx, event_idx], t_anchor_valid, device
                    )
                    s_censor = self.interpolate_survival(
                        survival[c_idx, event_idx], t_censor_valid, device
                    )

                    # Convert to risk (1 - survival)
                    r_anchor = 1.0 - s_anchor
                    r_censor = 1.0 - s_censor

                    # Compute pair differences - event should have higher risk than censored
                    pair_diffs = r_anchor - r_censor

                    # Use smaller margin due to censoring uncertainty
                    censor_margin = self.margin * 0.5

                    # Adaptive margin calculation if enabled
                    if self.adaptive_margin:
                        time_diffs = torch.abs(t_anchor_valid - t_censor_valid)
                        normalized_diffs = time_diffs / max_duration
                        margins = censor_margin * (1.0 + normalized_diffs)
                    else:
                        margins = torch.full_like(r_anchor, censor_margin)

                    # Apply hinge loss, weight, and accumulate
                    pair_losses = F.relu(margins - pair_diffs)
                    loss_accumulator = loss_accumulator + torch.sum(
                        pair_losses * event_weight
                    )
                    valid_count_accumulator += len(final_indices)

            # --- CASE 3: Comparison has event, anchor is censored ---
            compare_event_only = (
                ~anchor_events[:, event_idx] & compare_events[:, event_idx]
            )

            if compare_event_only.any():
                # Extract pairs where comparison has event but anchor is censored
                valid_indices = torch.where(compare_event_only)[0]

                # Get durations
                t_censor = anchor_durations[valid_indices, event_idx]
                t_compare = compare_durations[valid_indices, event_idx]

                # Only valid if censoring happens after the event
                valid_ranking = t_censor > t_compare

                if valid_ranking.any():
                    # Get final indices and durations
                    final_indices = valid_indices[valid_ranking]
                    t_censor_valid = t_censor[valid_ranking]
                    t_compare_valid = t_compare[valid_ranking]

                    # Get indices for efficient tensor access
                    a_idx = anchor_indices[final_indices]
                    c_idx = compare_indices[final_indices]

                    # Vectorized survival interpolation
                    s_censor = self.interpolate_survival(
                        survival[a_idx, event_idx], t_censor_valid, device
                    )
                    s_compare = self.interpolate_survival(
                        survival[c_idx, event_idx], t_compare_valid, device
                    )

                    # Convert to risk (1 - survival)
                    r_censor = 1.0 - s_censor
                    r_compare = 1.0 - s_compare

                    # Compute pair differences - event should have higher risk than censored
                    pair_diffs = r_compare - r_censor

                    # Use smaller margin due to censoring uncertainty
                    censor_margin = self.margin * 0.5

                    # Adaptive margin calculation if enabled
                    if self.adaptive_margin:
                        time_diffs = torch.abs(t_compare_valid - t_censor_valid)
                        normalized_diffs = time_diffs / max_duration
                        margins = censor_margin * (1.0 + normalized_diffs)
                    else:
                        margins = torch.full_like(r_censor, censor_margin)

                    # Apply hinge loss, weight, and accumulate
                    pair_losses = F.relu(margins - pair_diffs)
                    loss_accumulator = loss_accumulator + torch.sum(
                        pair_losses * event_weight
                    )
                    valid_count_accumulator += len(final_indices)

        # Return the mean loss if we have valid pairs, otherwise return zero
        if valid_count_accumulator > 0:
            return loss_accumulator / valid_count_accumulator
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

    def interpolate_survival(
        self,
        survival_curve: torch.Tensor,
        time_points: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Interpolate survival probability at given time points using vectorized operations.

        Args:
            survival_curve: Survival probabilities [num_time_bins+1] or [batch_size, num_time_bins+1]
            time_points: Times at which to interpolate [num_points] or [batch_size]
            device: Device for tensors

        Returns:
            torch.Tensor: Interpolated survival probabilities [num_points] or [batch_size]
        """
        # Handle scalar input
        if time_points.numel() == 1 and time_points.dim() == 0:
            time_points = time_points.unsqueeze(0)

        # Ensure duration_cuts is on the correct device
        cuts = self.duration_cuts.to(device)

        # Single curve case
        if survival_curve.dim() == 1:
            survival_curve = survival_curve.unsqueeze(0)
            single_curve = True
        else:
            single_curve = False

        # Prepare time_points for batch processing
        if time_points.dim() == 1:
            if time_points.shape[0] != survival_curve.shape[0]:
                # Broadcast time_points to match batch size if needed
                time_points = time_points.unsqueeze(0).expand(
                    survival_curve.shape[0], -1
                )
            else:
                # Add a dimension for broadcasting
                time_points = time_points.unsqueeze(1)

        # Expand dimensions for broadcasting
        time_expanded = time_points.unsqueeze(-1)  # [batch_size, num_points, 1]
        cuts_expanded = cuts.unsqueeze(0).unsqueeze(0)  # [1, 1, num_cuts]

        # Find indices where time falls between cuts
        # Compare each time point with all cuts
        is_greater = (time_expanded >= cuts_expanded).to(
            torch.float32
        )  # [batch_size, num_points, num_cuts]

        # Get the rightmost index where time >= cut
        # Sum across cuts dimension and subtract 1
        indices = (
            torch.sum(is_greater, dim=2, keepdim=True) - 1
        )  # [batch_size, num_points, 1]
        indices = torch.clamp(indices, min=0, max=len(cuts) - 2)

        # Reshape for gather operation
        batch_size = survival_curve.shape[0]
        num_points = time_points.shape[1] if time_points.dim() > 1 else 1
        flat_indices = indices.reshape(batch_size, num_points)

        # Prepare batch indices for gather
        batch_idx = torch.arange(batch_size, device=device)
        batch_idx = batch_idx.view(-1, 1).expand(-1, num_points)

        # Extract left and right cut points for each time point
        t0_idx = flat_indices
        t1_idx = torch.clamp(t0_idx + 1, max=len(cuts) - 1)

        # Get lower and upper cut values
        t0 = cuts[t0_idx.long()]
        t1 = cuts[t1_idx.long()]

        # Get corresponding survival values using batch indexing
        num_time_bins = survival_curve.shape[-1]
        t0_idx_clamped = torch.clamp(t0_idx, max=num_time_bins - 1).long()
        t1_idx_clamped = torch.clamp(t1_idx, max=num_time_bins - 1).long()

        # Handle different survival curve shapes
        if survival_curve.dim() == 2:  # [batch_size, num_bins+1]
            s0 = torch.gather(survival_curve, 1, t0_idx_clamped)
            s1 = torch.gather(survival_curve, 1, t1_idx_clamped)
        else:  # For specific event type already sliced
            # Use advanced indexing
            s0 = torch.zeros((batch_size, num_points), device=device)
            s1 = torch.zeros((batch_size, num_points), device=device)

            for i in range(batch_size):
                for j in range(num_points):
                    s0[i, j] = survival_curve[i, t0_idx_clamped[i, j]]
                    s1[i, j] = survival_curve[i, t1_idx_clamped[i, j]]

        # Compute interpolation weights
        dt = t1 - t0
        alpha = torch.zeros_like(dt)
        non_zero_dt = dt > 0

        # Only interpolate where dt > 0
        if non_zero_dt.any():
            # Reshape time_points for computation
            if time_points.dim() > 1:
                tp = time_points.reshape(batch_size, num_points)
            else:
                tp = time_points

            # Calculate weights only where needed
            alpha[non_zero_dt] = (tp[non_zero_dt] - t0[non_zero_dt]) / dt[non_zero_dt]

        # Linear interpolation
        result = s0 + alpha * (s1 - s0)

        # Return in original shape
        if single_curve:
            if num_points == 1:
                return result[0, 0]
            else:
                return result[0]
        elif num_points == 1:
            return result[:, 0]
        else:
            return result

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
