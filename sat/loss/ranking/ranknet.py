"""RankNet: Pairwise approach to ranking for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd
import torch

from sat.models.tasks.heads import SAOutput
from sat.loss.base import Loss


class RankNetLoss(Loss):
    """RankNet Loss for survival analysis ranking.

    This implementation is based on the RankNet algorithm proposed by Burges et al. (2005)
    and adapted for survival analysis. RankNet is a pairwise approach that learns to rank
    by minimizing a cross-entropy loss on pairs of examples.

    Mathematical formulation:
    For a pair of items i and j, where i should be ranked higher than j:

    P_ij = sigmoid(s_i - s_j) = 1 / (1 + exp(-(s_i - s_j)))

    where s_i and s_j are the model scores for items i and j.

    The loss is then defined as:
    L = -y_ij * log(P_ij) - (1 - y_ij) * log(1 - P_ij)

    where y_ij = 1 if i should be ranked higher than j, and 0 otherwise.

    In survival analysis context, items with earlier event times should be ranked higher,
    and censored items should be ranked lower than events occurring before the censoring time.

    References:
    - Burges, C. et al. (2005). Learning to rank using gradient descent.
      In Proceedings of the 22nd International Conference on Machine Learning.
    - Kvamme, H. et al. (2019). Time-to-event prediction with neural networks and Cox regression.
      Journal of Machine Learning Research.
    """

    def __init__(
        self,
        sigma=1.0,
        eps=1e-8,
        importance_sample_weights=None,
        num_events=1,
        balance_strategy="none",
        ema_momentum=0.9,
    ):
        """Initialize RankNet loss.

        Args:
            sigma (float): Controls the steepness of the sigmoid function used in pairwise comparisons.
            eps (float): Small constant to avoid numerical instability in logarithms.
            importance_sample_weights (str): Path to importance sampling weights (default: None)
            num_events (int): Number of event types (default: 1)
            balance_strategy (str): Strategy for balancing loss components ("none", "scale_normalization",
                                    "grad_normalization", "homoscedastic", or "adaptive")
            ema_momentum (float): Momentum for exponential moving average when using scale normalization
        """
        super().__init__(
            num_events, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )
        self.sigma = sigma
        self.eps = eps

        # Initialize loss balancing parameters for different loss components
        self._init_loss_params(["rank_censor", "rank_event"])

        # Load importance sampling weights if provided
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute the RankNet loss.

        Args:
            predictions: Model output containing survival probabilities.
                Shape: [batch_size, num_events, num_time_points]
            references: Target tensor containing event indicators and durations.
                Shape: [batch_size, 4*num_events]

        Returns:
            loss: The computed loss value
        """

        # Extract events and durations from references tensor using Loss class methods
        # These methods preserve gradient flow connections to the original tensor
        events = self.events(references)  # Event indicators
        durations = self.durations(references)  # Event times

        survival = predictions.survival

        # Compute both ranking components
        intra_subject_loss = self._compute_intra_subject_ranking(
            events, durations, survival
        )
        inter_subject_loss = self._compute_inter_subject_ranking(
            events, durations, survival
        )

        # Use loss balancing API
        loss_dict = {"intra": intra_subject_loss, "inter": inter_subject_loss}

        # The balance_losses method returns the balanced total loss directly
        return self.balance_losses(loss_dict)

    def _compute_intra_subject_ranking(self, events, durations, survival):
        """Compute the RankNet loss for ranking events within the same subject.

        For each subject with multiple events, we create pairs of events and compute
        the pairwise ranking loss. Events with earlier times should be ranked higher.

        Args:
            events: Binary indicators of events. Shape [batch_size, num_events]
            durations: Time to events. Shape [batch_size, num_events]
            survival: Survival probabilities. Shape [batch_size, num_events, num_time_points]

        Returns:
            loss: The computed intra-subject ranking loss
        """
        batch_size = events.size(0)
        device = events.device

        # Skip if we only have one event type
        if self.num_events <= 1:
            # Return a value connected to the gradient flow
            return survival.sum() * 0 + torch.tensor(
                1e-8, device=device, requires_grad=True
            )

        # Compute risk scores as the negative mean survival probability
        # Higher risk means lower survival probability
        risk_scores = -torch.mean(survival, dim=2)  # [batch_size, num_events]

        # For tracking valid pair losses
        all_pair_losses = []

        # For each subject
        for i in range(batch_size):
            subject_events = events[i]  # [num_events]
            subject_times = durations[i]  # [num_events]
            subject_risks = risk_scores[i]  # [num_events]

            # Get indices of observed events
            observed_mask = subject_events == 1
            if observed_mask.sum() <= 1:  # Need at least 2 events for comparison
                continue

            # Get observed events data
            observed_indices = torch.nonzero(observed_mask).squeeze(
                -1
            )  # [num_observed_events]
            observed_times = subject_times[observed_mask]  # [num_observed_events]
            observed_risks = subject_risks[observed_mask]  # [num_observed_events]

            # Create pairs using upper triangular indices
            num_observed = observed_indices.size(0)
            idx1, idx2 = torch.triu_indices(
                num_observed, num_observed, 1, device=device
            )

            # Skip if no pairs
            if idx1.numel() == 0:
                continue

            # Get pair data
            times1 = observed_times[idx1]  # [num_pairs]
            times2 = observed_times[idx2]  # [num_pairs]
            risks1 = observed_risks[idx1]  # [num_pairs]
            risks2 = observed_risks[idx2]  # [num_pairs]

            # Get original event indices (needed for weights)
            event_indices1 = observed_indices[idx1]  # [num_pairs]
            event_indices2 = observed_indices[idx2]  # [num_pairs]

            # Skip pairs with equal times
            not_equal_mask = times1 != times2
            if not not_equal_mask.any():
                continue

            # Filter to keep only pairs with different times
            times1 = times1[not_equal_mask]
            times2 = times2[not_equal_mask]
            risks1 = risks1[not_equal_mask]
            risks2 = risks2[not_equal_mask]
            event_indices1 = event_indices1[not_equal_mask]
            event_indices2 = event_indices2[not_equal_mask]

            # Create mask for time ordering: 1 where time1 < time2, 0 otherwise
            time_order = (times1 < times2).float()

            # Calculate risk differences based on time ordering
            # When time1 < time2: use risk1 - risk2 (risk1 should be higher)
            # When time1 > time2: use risk2 - risk1 (risk2 should be higher)
            risk_diffs = time_order * (risks1 - risks2) + (1 - time_order) * (
                risks2 - risks1
            )

            # Compute pair losses in a vectorized way
            pair_losses = -torch.log(torch.sigmoid(self.sigma * risk_diffs) + self.eps)

            # Apply event weights
            event_weights1 = self.weights[
                event_indices1 + 1
            ]  # +1 because weights are 1-indexed
            event_weights2 = self.weights[event_indices2 + 1]
            weighted_losses = pair_losses * event_weights1 * event_weights2

            # Add all pair losses from this subject to our collection
            all_pair_losses.append(weighted_losses)

        # If we collected any losses, stack them and compute mean
        if all_pair_losses:
            stacked_losses = torch.cat(all_pair_losses)
            return stacked_losses.mean()

        # Return a small value connected to the computation graph if no valid pairs found
        # Using survival tensor which definitely requires gradients
        return survival.sum() * 0 + torch.tensor(
            1e-8, device=device, requires_grad=True
        )

    def _compute_inter_subject_ranking(self, events, durations, survival):
        """Compute the RankNet loss for ranking events across different subjects.

        For each event type, we compare pairs of subjects and compute the pairwise ranking loss.
        Subjects with earlier event times should have higher risk scores.

        Args:
            events: Binary indicators of events. Shape [batch_size, num_events]
            durations: Time to events. Shape [batch_size, num_events]
            survival: Survival probabilities. Shape [batch_size, num_events, num_time_points]

        Returns:
            loss: The computed inter-subject ranking loss
        """
        batch_size = events.size(0)
        device = events.device

        # For tracking pair losses across all event types
        all_pair_losses = []

        # For each event type
        for event_idx in range(self.num_events):
            # Get risk scores for this event (negative mean survival probability)
            # Higher risk means lower survival probability
            event_risks = -torch.mean(survival[:, event_idx], dim=1)  # [batch_size]

            # Get event indicators and durations for this event
            event_indicators = events[:, event_idx]  # [batch_size]
            event_times = durations[:, event_idx]  # [batch_size]

            # Process observed-observed pairs
            observed_mask = event_indicators == 1
            observed_count = observed_mask.sum()

            if observed_count > 1:  # Need at least 2 subjects for comparison
                # Get indices, times and risks for observed subjects
                observed_indices = torch.nonzero(observed_mask).squeeze(
                    -1
                )  # [num_observed]
                observed_times = event_times[observed_mask]  # [num_observed]
                observed_risks = event_risks[observed_mask]  # [num_observed]

                # Create all pairs using upper triangular indices
                num_observed = observed_indices.size(0)
                idx1, idx2 = torch.triu_indices(
                    num_observed, num_observed, 1, device=device
                )

                if idx1.numel() > 0:  # Only proceed if we have pairs
                    # Get pair data
                    times1 = observed_times[idx1]  # [num_pairs]
                    times2 = observed_times[idx2]  # [num_pairs]
                    risks1 = observed_risks[idx1]  # [num_pairs]
                    risks2 = observed_risks[idx2]  # [num_pairs]

                    # Exclude pairs with equal times
                    not_equal_mask = times1 != times2
                    if not_equal_mask.any():
                        # Filter to keep only pairs with different times
                        times1 = times1[not_equal_mask]
                        times2 = times2[not_equal_mask]
                        risks1 = risks1[not_equal_mask]
                        risks2 = risks2[not_equal_mask]

                        # Create mask for time ordering: 1 where time1 < time2, 0 otherwise
                        time_order = (times1 < times2).float()

                        # Calculate risk differences based on time ordering
                        # When time1 < time2: use risk1 - risk2 (risk1 should be higher)
                        # When time1 > time2: use risk2 - risk1 (risk2 should be higher)
                        risk_diffs = time_order * (risks1 - risks2) + (
                            1 - time_order
                        ) * (risks2 - risks1)

                        # Compute pair losses
                        pair_losses = -torch.log(
                            torch.sigmoid(self.sigma * risk_diffs) + self.eps
                        )

                        # Apply event weight
                        weighted_losses = pair_losses * self.weights[event_idx + 1]

                        # Add to collection
                        all_pair_losses.append(weighted_losses)

            # Process observed-censored pairs
            censored_mask = (event_indicators == 0) & (event_times > 0)
            censored_count = censored_mask.sum()

            if observed_count > 0 and censored_count > 0:
                # Get observed subjects data
                observed_indices = torch.nonzero(observed_mask).squeeze(
                    -1
                )  # [num_observed]
                observed_times = event_times[observed_mask]  # [num_observed]
                observed_risks = event_risks[observed_mask]  # [num_observed]

                # Get censored subjects data
                censored_indices = torch.nonzero(censored_mask).squeeze(
                    -1
                )  # [num_censored]
                censored_times = event_times[censored_mask]  # [num_censored]
                censored_risks = event_risks[censored_mask]  # [num_censored]

                # Create all observed-censored pairs using meshgrid pattern
                num_observed = observed_indices.size(0)
                num_censored = censored_indices.size(0)

                # Create indices for all pairs
                obs_indices_expanded = torch.arange(
                    num_observed, device=device
                ).repeat_interleave(num_censored)
                cens_indices_expanded = torch.arange(
                    num_censored, device=device
                ).repeat(num_observed)

                # Get times and risks for all pairs
                obs_times = observed_times[
                    obs_indices_expanded
                ]  # [num_observed * num_censored]
                cens_times = censored_times[
                    cens_indices_expanded
                ]  # [num_observed * num_censored]
                obs_risks = observed_risks[
                    obs_indices_expanded
                ]  # [num_observed * num_censored]
                cens_risks = censored_risks[
                    cens_indices_expanded
                ]  # [num_observed * num_censored]

                # Only keep pairs where censoring time > observed time
                # These are the only cases where we know the ordering
                valid_ordering_mask = cens_times > obs_times

                if valid_ordering_mask.any():
                    # Get valid pairs
                    valid_obs_risks = obs_risks[valid_ordering_mask]
                    valid_cens_risks = cens_risks[valid_ordering_mask]

                    # For observed-censored pairs, observed should have higher risk
                    risk_diffs = valid_obs_risks - valid_cens_risks

                    # Compute losses
                    pair_losses = -torch.log(
                        torch.sigmoid(self.sigma * risk_diffs) + self.eps
                    )

                    # Apply event weight
                    weighted_losses = pair_losses * self.weights[event_idx + 1]

                    # Add to collection
                    all_pair_losses.append(weighted_losses)

        # If we collected any losses, stack them and compute mean
        if all_pair_losses:
            stacked_losses = torch.cat(all_pair_losses)
            return stacked_losses.mean()

        # Return a small value connected to the computation graph if no valid pairs found
        # Using survival tensor which definitely requires gradients
        return survival.sum() * 0 + torch.tensor(
            1e-8, device=device, requires_grad=True
        )
