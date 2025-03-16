"""DeepHit-style composite loss for survival analysis with competing risks."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import numpy as np
from sat.loss.base import Loss
from sat.models.tasks.heads import SAOutput


class DeepHitLoss(Loss):
    """Implementation of DeepHit loss combining likelihood and ranking components.

    DeepHit uses a custom likelihood function with an integrated ranking component
    for improved discrimination. This implementation supports both single event and
    competing risks scenarios.

    References:
        Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018).
        DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks.
        AAAI Conference on Artificial Intelligence.
    """

    def __init__(
        self,
        alpha=0.1,  # Weighting for ranking component
        sigma=0.1,  # Temperature parameter for ranking
        discrete_bins=None,
        importance_sample_weights=None,
        num_events=1,
        balance_strategy="none",  # Loss balancing strategy
        ema_momentum=0.9,
    ):  # EMA momentum for scale normalization
        """Initialize DeepHitLoss.

        Args:
            alpha: Weighting for ranking component (0 = only likelihood, 1 = only ranking)
            sigma: Temperature parameter for ranking component
            discrete_bins: Path to CSV file containing discrete time bins
            importance_sample_weights: Path to CSV file containing importance weights
            num_events: Number of competing events
            balance_strategy: Strategy for balancing loss components ("none", "scale_normalization",
                             "grad_normalization", "homoscedastic", or "adaptive")
            ema_momentum: Momentum for exponential moving average when using scale normalization
        """
        super().__init__(
            num_events, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )
        self.alpha = alpha
        self.sigma = sigma
        self.discrete_bins = discrete_bins

        # Initialize loss balancing parameters for likelihood and ranking components
        self._init_loss_params(["likelihood", "ranking"])

        # Load time bins if provided
        if discrete_bins is not None:
            import pandas as pd

            df = pd.read_csv(discrete_bins, header=None, names=["bins"])
            bins = torch.tensor(df.bins.values).to(torch.float32)
            self.register_buffer("time_bins", bins)
        else:
            self.register_buffer("time_bins", None)

        # Load importance sampling weights if provided
        if importance_sample_weights is not None:
            import pandas as pd

            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
            self.register_buffer("importance_weights", weights)
        else:
            self.register_buffer("importance_weights", torch.ones(num_events + 1))

    def likelihood_loss(self, hazard, survival, durations, events):
        """Compute the negative log-likelihood for discrete hazards.

        Direct PyTorch implementation of the original DeepHit TensorFlow code.

        Args:
            hazard: Hazard function values [batch_size, num_events, num_time_bins]
            survival: Survival function values [batch_size, num_events, num_time_bins]
            durations: Event times [batch_size]
            events: Event indicators [batch_size]

        Returns:
            loss: Negative log-likelihood loss
        """
        batch_size = len(durations)
        device = hazard.device
        EPSILON = 1e-8  # Small constant to avoid log(0)

        # Get time indices for each patient
        time_indices = self._get_time_indices(durations)

        # In original TensorFlow code:
        # I_1 = tf.sign(self.k) - event indicator (1 for events, 0 for censoring)
        I_1 = (events > 0).float().unsqueeze(1)  # Shape: [batch_size, 1]

        # Create masks for selecting values at specific time points
        # fc_mask1 - Used to extract probabilities for specific events at event times
        # fc_mask2 - Used to extract survival probabilities at censoring times

        # Vectorized mask creation
        # Create masks to select values at the right time points
        fc_mask1 = torch.zeros(
            (batch_size, self.num_events, hazard.shape[2]), device=device
        )
        fc_mask2 = torch.zeros((batch_size, hazard.shape[2]), device=device)

        # Check if indices are within bounds
        time_indices = (
            time_indices.squeeze(-1) if time_indices.dim() > 1 else time_indices
        )
        valid_indices = time_indices < hazard.shape[2]

        # Get event types for valid samples (convert 1-based to 0-based indexing)
        events = events.squeeze(-1) if events.dim() > 1 else events
        event_types = events - 1

        # Handle fc_mask2 (for all samples)
        for i in range(batch_size):
            if valid_indices[i]:
                fc_mask2[i, time_indices[i]] = 1.0

        # Handle fc_mask1 (only for uncensored samples with events)
        for i in range(batch_size):
            if valid_indices[i] and events[i] > 0:
                event_type = (
                    event_types[i].item()
                    if hasattr(event_types[i], "item")
                    else event_types[i]
                )
                fc_mask1[i, event_type, time_indices[i]] = 1.0

        # UNCENSORED: P(T=t, K=k|x)
        # Original: tmp1 = tf.reduce_sum(tf.reduce_sum(self.fc_mask1 * self.out, reduction_indices=2), reduction_indices=1, keep_dims=True)
        tmp1 = torch.sum(
            torch.sum(fc_mask1 * hazard * survival, dim=2), dim=1, keepdim=True
        )
        tmp1 = I_1 * torch.log(tmp1 + EPSILON)

        # CENSORED: P(T>t|x)
        # Original: tmp2 = tf.reduce_sum(tf.reduce_sum(self.fc_mask2 * self.out, reduction_indices=2), reduction_indices=1, keep_dims=True)

        # For censored observations, calculate overall survival probability
        # Original uses product: Total_Survival = tf.reduce_prod(self.S_out, reduction_indices=1, name='s_bar')
        # Product across all event types gives the overall survival probability with competing risks
        total_survival = torch.prod(survival, dim=1)  # Product across event types

        # Apply mask to select survival at censoring time
        tmp2 = torch.sum(fc_mask2 * total_survival, dim=1, keepdim=True)
        tmp2 = (1.0 - I_1) * torch.log(tmp2 + EPSILON)

        # Apply importance weights and compute negative log-likelihood
        # Vectorized approach to apply importance weights

        # Create an event-to-weight mapping for vectorized operation
        # Extract weights for each event type and map to each event in the batch
        event_weights = self.importance_weights[events].unsqueeze(
            1
        )  # Shape: [batch_size, 1]

        # Apply weights directly - I_1 selects only uncensored samples
        weighted_uncensored = I_1 * event_weights * tmp1

        # Apply weights for censored samples - (1-I_1) selects only censored samples
        censoring_weight = self.importance_weights[0]
        weighted_censored = (1.0 - I_1) * censoring_weight * tmp2

        # Compute weighted negative log-likelihood
        nll = -torch.mean(weighted_uncensored + weighted_censored)

        # Mean over batch
        return nll.mean()

    def ranking_loss(self, survival, durations, events):
        """Compute the ranking component of DeepHit loss.

        Fully vectorized implementation based directly on the original DeepHit TensorFlow code.

        Args:
            survival: Survival function values [batch_size, num_events, num_time_bins]
            durations: Event times [batch_size]
            events: Event indicators [batch_size]

        Returns:
            loss: Ranking loss component
        """
        batch_size = len(durations)
        device = survival.device
        sigma1 = self.sigma  # Smoothing parameter for loss

        # Get time indices for each patient
        time_indices = self._get_time_indices(durations)

        # Initialize event-specific loss components
        eta = []

        # Process each event type
        for event_type in range(self.num_events):
            # Skip if no events of this type
            event_indicator = (events == event_type + 1).float()
            if event_indicator.sum() == 0:
                continue

            # Create a matrix for event indicators (I_2 in original code)
            # Shape: [batch_size, 1] - column vector with event indicators
            I_2 = event_indicator.unsqueeze(1)

            # Get survival probabilities for this event type at each time index
            # Shape: [batch_size, num_time_bins]
            event_survival = survival[:, event_type, :]

            # Extract the risk for each subject at their event time
            # Create a mask to select the risk at the time point of interest for each subject
            # The mask operation is similar to the fc_mask2 in the original implementation
            mask = torch.zeros((batch_size, event_survival.shape[1]), device=device)

            # Make sure time_indices has the right shape
            time_indices_flat = (
                time_indices.squeeze(-1) if time_indices.dim() > 1 else time_indices
            )

            # Create a mask of valid indices (within bounds)
            valid_indices = time_indices_flat < event_survival.shape[1]

            # Get batch indices for valid entries
            batch_indices = torch.arange(batch_size, device=device)
            valid_batch_indices = batch_indices[valid_indices]
            valid_time_indices = time_indices_flat[valid_indices]

            # Set mask values for valid indices in one operation
            if valid_batch_indices.numel() > 0:  # Only set if there are valid indices
                mask[valid_batch_indices, valid_time_indices] = 1.0

            # Calculate risk for each subject at the time of interest
            # Shape: [batch_size, num_time_bins] -> batch matrix multiply -> [batch_size, batch_size]
            # This corresponds to R = tf.matmul(tmp_e, tf.transpose(self.fc_mask2)) in original code
            R = torch.matmul(event_survival, mask.t())

            # Get diagonal elements (risk of subject j at their own event time)
            # Shape: [batch_size, 1]
            diag_R = torch.diag(R).unsqueeze(1)

            # Create vector of ones for matrix operations
            one_vector = torch.ones((batch_size, 1), device=device)

            # Compute risk differences: r_j(T_j) - r_i(T_j) for all pairs
            # Shape: [batch_size, batch_size]
            # First compute: one_vector * diag_R^T - R
            R = torch.matmul(one_vector, diag_R.t()) - R
            # Then transpose to get: R_ij = r_i(T_i) - r_j(T_i)
            R = R.t()

            # Create time comparison matrix: T_ij = 1 if t_i < t_j, 0 otherwise
            # Shape: [batch_size, batch_size]
            # Using outer products of one_vector and durations
            # This is equivalent to: T = tf.nn.relu(tf.sign(tf.matmul(one_vector, tf.transpose(self.t)) - tf.matmul(self.t, tf.transpose(one_vector))))
            durations_col = durations.unsqueeze(1)  # [batch_size, 1]
            durations_row = durations.unsqueeze(0)  # [1, batch_size]
            T = torch.nn.functional.relu(torch.sign(durations_col - durations_row))

            # Filter time comparison matrix by event indicators
            # Only keep pairs where subject i had the event
            # Shape: [batch_size, batch_size]
            T = I_2 * T

            # Compute ranking loss: mean of exp(-R/sigma) for valid pairs
            # Shape: [batch_size, 1]
            # This is equivalent to: tmp_eta = tf.reduce_mean(T * tf.exp(-R/sigma1), reduction_indices=1, keep_dims=True)
            tmp_eta = torch.mean(T * torch.exp(-R / sigma1), dim=1, keepdim=True)

            # Add to list of event-specific losses
            eta.append(tmp_eta)

        # If no events of any type, return zero loss
        if len(eta) == 0:
            return torch.tensor(0.0, device=device)

        # Stack event-specific losses
        # Shape: [batch_size, num_events]
        eta = torch.cat(eta, dim=1)

        # Compute mean across events
        # Shape: [batch_size, 1]
        eta_mean = torch.mean(eta, dim=1, keepdim=True)

        # Sum across batch for final loss
        # This is equivalent to: self.LOSS_2 = tf.reduce_sum(eta)
        return torch.sum(eta_mean)

    def _get_time_indices(self, durations):
        """Convert continuous durations to indices in the discrete time grid.

        Args:
            durations: Continuous time durations [batch_size]

        Returns:
            indices: Time bin indices [batch_size]
        """
        if self.time_bins is None:
            # If no bins provided, assume durations are already indices
            return durations.long()

        # Find index of first bin greater than or equal to duration
        indices = torch.zeros_like(durations, dtype=torch.long)
        for i, time in enumerate(durations):
            # Find index of bin containing time
            idx = torch.searchsorted(self.time_bins, time.float())
            # Handle case where time is beyond last bin
            indices[i] = min(idx, len(self.time_bins) - 1)

        return indices

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute the DeepHit loss.

        Args:
            predictions: Model output of type SAOutput containing hazard and survival predictions
            references: Target tensor containing event types and durations

        Returns:
            loss: The computed loss value
        """
        # Extract hazard and survival directly from SAOutput
        hazard = predictions.hazard
        survival = (
            predictions.survival
        )  # Using survival instead of surv to match SAOutput definition

        # Extract events and durations from references (torch.Tensor)
        events = self.events(references)
        durations = self.durations(references)

        # Compute likelihood and ranking losses
        nll = self.likelihood_loss(hazard, survival, durations, events)
        rank = self.ranking_loss(survival, durations, events)

        # Create loss component dictionary
        loss_dict = {"likelihood": nll, "ranking": rank}

        # Define weights from alpha parameter
        weights = {"likelihood": 1.0 - self.alpha, "ranking": self.alpha}

        # Apply loss balancing with configured strategy
        loss = self.balance_losses(loss_dict, weights)

        return loss
