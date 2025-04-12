"""MENSA Loss implementation for multi-event survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import pandas as pd
import torch
import torch.nn.functional as F

from sat.distributions import LogNormalMixtureDistribution, WeibullMixtureDistribution
from sat.models.heads import SAOutput
from sat.utils import logging

from ..base import Loss

logger = logging.get_default_logger()


class MENSALoss(Loss):
    """
    MENSA loss for multi-event survival analysis.

    Implementation based on the paper "MENSA: Multi-Event Neural Survival Analysis."
    The loss function consists of multiple components:
    1. Negative log-likelihood for uncensored data using mixture distributions
    2. Negative log survival probability for censored data
    3. Regularization for the event dependency matrix
    4. Optional ELBO loss for mixture weights

    This loss is specifically designed for multi-event scenarios with event dependencies.
    """

    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        num_events: int = 1,
        distribution: str = "weibull",
        discount: float = 1.0,
        elbo: bool = False,
        dependency_regularization: float = 0.01,
    ):
        """
        Initialize MENSALoss.

        Args:
            duration_cuts: Path to CSV file containing duration cut points
            importance_sample_weights: Optional path to CSV file with importance weights
            num_events: Number of competing events
            distribution: Distribution type ('weibull' or 'lognormal')
            discount: Weight for the censored loss component (0.0-1.0)
            elbo: Whether to use Evidence Lower Bound for mixture model
            dependency_regularization: Weight for event dependency matrix regularization
        """
        super(MENSALoss, self).__init__(num_events)

        # Load the importance sampling weights if provided
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(num_events + 1)

        self.register_buffer("weights", weights)
        self.discount = discount
        self.distribution = distribution
        self.elbo = elbo
        self.dependency_regularization = dependency_regularization

        # Load duration cuts
        try:
            df = pd.read_csv(duration_cuts, header=None)
            self.duration_cuts = torch.tensor(df.values.squeeze(), dtype=torch.float32)
            self.num_durations = len(self.duration_cuts)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Loaded {self.num_durations} duration cuts from {duration_cuts}"
                )
        except Exception as e:
            logger.error(f"Error loading duration cuts from {duration_cuts}: {str(e)}")
            # Create default cuts as fallback
            self.duration_cuts = torch.linspace(0.1, 10.0, steps=20)
            self.num_durations = len(self.duration_cuts)
            logger.warning(
                f"Using default duration cuts with {self.num_durations} time points"
            )

    def _negative_log_likelihood(
        self,
        time: torch.Tensor,
        shape: torch.Tensor,
        scale: torch.Tensor,
        logits_g: torch.Tensor,
        event_type: str = None,
        use_expert_priors: bool = False,
        constrain_params: bool = False,
    ):
        """
        Compute negative log-likelihood for uncensored data.
        Uses distribution classes for numerical stability with optional expert knowledge.

        Args:
            time: Event times [batch_size]
            shape: Shape parameters [batch_size, num_mixtures]
            scale: Scale parameters [batch_size, num_mixtures]
            logits_g: Mixture weight logits [batch_size, num_mixtures]
            event_type: Optional event type for expert knowledge
            use_expert_priors: Whether to use expert knowledge priors
            constrain_params: Whether to apply parameter constraints

        Returns:
            torch.Tensor: Negative log-likelihood [batch_size]
        """
        # Create the appropriate mixture distribution based on distribution type
        if self.distribution.lower() == "weibull":
            mixture_dist = WeibullMixtureDistribution(
                shape,
                scale,
                logits_g,
                constrain_shape=constrain_params,
                event_type=event_type,
                use_expert_priors=use_expert_priors,
            )
        elif self.distribution.lower() == "lognormal":
            mixture_dist = LogNormalMixtureDistribution(
                scale,
                shape,
                logits_g,  # Note parameter order for lognormal: loc, scale
                constrain_params=constrain_params,
                event_type=event_type,
                use_expert_priors=use_expert_priors,
            )
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

        # Compute log-likelihood using the mixture distribution
        log_likelihood = mixture_dist.log_likelihood(time)

        # Return negative log likelihood
        return -log_likelihood

    def _negative_log_survival(
        self,
        time: torch.Tensor,
        shape: torch.Tensor,
        scale: torch.Tensor,
        logits_g: torch.Tensor,
        event_type: str = None,
        use_expert_priors: bool = False,
        constrain_params: bool = False,
    ):
        """
        Compute negative log survival probability for censored data.
        Uses distribution classes for numerical stability with optional expert knowledge.

        Args:
            time: Censoring times [batch_size]
            shape: Shape parameters [batch_size, num_mixtures]
            scale: Scale parameters [batch_size, num_mixtures]
            logits_g: Mixture weight logits [batch_size, num_mixtures]
            event_type: Optional event type for expert knowledge
            use_expert_priors: Whether to use expert knowledge priors
            constrain_params: Whether to apply parameter constraints

        Returns:
            torch.Tensor: Negative log survival probability [batch_size]
        """
        # Create the appropriate mixture distribution based on distribution type
        if self.distribution.lower() == "weibull":
            mixture_dist = WeibullMixtureDistribution(
                shape,
                scale,
                logits_g,
                constrain_shape=constrain_params,
                event_type=event_type,
                use_expert_priors=use_expert_priors,
            )
        elif self.distribution.lower() == "lognormal":
            mixture_dist = LogNormalMixtureDistribution(
                scale,
                shape,
                logits_g,  # Note parameter order for lognormal: loc, scale
                constrain_params=constrain_params,
                event_type=event_type,
                use_expert_priors=use_expert_priors,
            )
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

        # Reshape time for distribution API which expects [batch_size, num_times]
        time_reshaped = time.unsqueeze(-1)  # [batch_size, 1]

        # Compute log survival using the mixture distribution
        log_survival = mixture_dist.log_survival(time_reshaped)

        # Squeeze to get [batch_size] tensor
        log_survival = log_survival.squeeze(-1)

        # Return negative log survival
        return -log_survival

    def _dependency_regularization(self, event_dependency_matrix):
        """
        Compute regularization term for the event dependency matrix.

        This encourages the dependency matrix to be sparse by penalizing
        off-diagonal elements, which helps to identify the most important
        dependencies between events.

        Args:
            event_dependency_matrix: Event dependency matrix [num_events, num_events]

        Returns:
            torch.Tensor: Regularization term
        """
        if event_dependency_matrix is None:
            return torch.tensor(0.0, device=self.weights.device)

        # Extract the off-diagonal elements
        mask = ~torch.eye(
            event_dependency_matrix.size(0),
            dtype=torch.bool,
            device=event_dependency_matrix.device,
        )
        off_diag = event_dependency_matrix[mask]

        # Apply L1 regularization to off-diagonal elements
        reg = torch.sum(torch.abs(off_diag))

        return reg * self.dependency_regularization

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the MENSA loss for survival prediction.

        Args:
            predictions: MENSA model outputs containing distribution parameters
            references: Reference values with event indicators and times
                [batch_size, 4 * num_events]

        Returns:
            torch.Tensor: The loss value
        """
        # Extract event indicators and durations
        events = self.events(references)  # [batch_size, num_events]
        durations = self.durations(references)  # [batch_size, num_events]

        batch_size = events.shape[0]
        device = references.device

        # Create weights tensor if needed
        weights = None
        if self.weights is not None:
            # Skip the first weight (censoring) and use only event weights
            weights = self.weights[1:].to(device)

        # Initialize total loss
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_valid_events = 0

        # Process each event type separately
        for event_idx in range(self.num_events):
            # Get parameters for this event type
            # shape: [batch_size, num_mixtures]
            shape = predictions.shape[:, event_idx, :]
            # scale: [batch_size, num_mixtures]
            scale = predictions.scale[:, event_idx, :]
            # logits_g: [batch_size, num_mixtures]
            logits_g = predictions.logits_g[:, event_idx, :]

            # Get event indicators and durations for this event
            event_indicator = events[:, event_idx]  # [batch_size]
            event_duration = durations[:, event_idx]  # [batch_size]

            # Skip if no events of this type
            if not torch.any(event_indicator):
                continue

            # Clamp extreme durations to reasonable values for numerical stability
            eps = 1e-6
            max_duration = 1e6
            event_duration = torch.clamp(event_duration, min=eps, max=max_duration)

            # Calculate uncensored loss for samples with events
            uncensored_mask = event_indicator == 1
            if torch.any(uncensored_mask):
                uncensored_times = event_duration[uncensored_mask]
                uncensored_shape = shape[uncensored_mask]
                uncensored_scale = scale[uncensored_mask]
                uncensored_logits_g = logits_g[uncensored_mask]

                # Extract event type if available
                event_type = None
                if (
                    hasattr(predictions, "event_types")
                    and predictions.event_types is not None
                    and event_idx < len(predictions.event_types)
                ):
                    event_type = predictions.event_types[event_idx]

                # Extract expert knowledge flags from predictions if available
                use_expert_priors = False
                constrain_params = False
                if hasattr(predictions, "use_expert_priors"):
                    use_expert_priors = predictions.use_expert_priors
                if hasattr(predictions, "constrain_params"):
                    constrain_params = predictions.constrain_params

                uncensored_loss = self._negative_log_likelihood(
                    uncensored_times,
                    uncensored_shape,
                    uncensored_scale,
                    uncensored_logits_g,
                    event_type=event_type,
                    use_expert_priors=use_expert_priors,
                    constrain_params=constrain_params,
                )

                # Check for invalid values and replace with reasonable defaults
                uncensored_loss = torch.nan_to_num(
                    uncensored_loss, nan=1.0, posinf=10.0, neginf=-10.0
                )

                # Apply sample weighting if needed
                if weights is not None:
                    uncensored_loss = uncensored_loss * weights[event_idx]

                uncensored_loss = torch.mean(uncensored_loss)
            else:
                uncensored_loss = torch.tensor(0.0, device=device)

            # Calculate censored loss for samples without events
            censored_mask = event_indicator == 0
            if torch.any(censored_mask) and self.discount > 0:
                censored_times = event_duration[censored_mask]
                censored_shape = shape[censored_mask]
                censored_scale = scale[censored_mask]
                censored_logits_g = logits_g[censored_mask]

                # Find non-zero censoring times
                valid_times_mask = censored_times > 0
                if torch.any(valid_times_mask):
                    valid_censored_times = censored_times[valid_times_mask]
                    valid_censored_shape = censored_shape[valid_times_mask]
                    valid_censored_scale = censored_scale[valid_times_mask]
                    valid_censored_logits_g = censored_logits_g[valid_times_mask]

                    # Extract event type if available
                    event_type = None
                    if (
                        hasattr(predictions, "event_types")
                        and predictions.event_types is not None
                        and event_idx < len(predictions.event_types)
                    ):
                        event_type = predictions.event_types[event_idx]

                    # Extract expert knowledge flags from predictions if available
                    use_expert_priors = False
                    constrain_params = False
                    if hasattr(predictions, "use_expert_priors"):
                        use_expert_priors = predictions.use_expert_priors
                    if hasattr(predictions, "constrain_params"):
                        constrain_params = predictions.constrain_params

                    censored_loss = self._negative_log_survival(
                        valid_censored_times,
                        valid_censored_shape,
                        valid_censored_scale,
                        valid_censored_logits_g,
                        event_type=event_type,
                        use_expert_priors=use_expert_priors,
                        constrain_params=constrain_params,
                    )

                    # Check for invalid values and replace with reasonable defaults
                    censored_loss = torch.nan_to_num(
                        censored_loss, nan=1.0, posinf=10.0, neginf=-10.0
                    )

                    # Apply discount factor for censored samples
                    censored_loss = censored_loss * self.discount

                    # Apply sample weighting if needed
                    if weights is not None:
                        censored_loss = censored_loss * weights[event_idx]

                    censored_loss = torch.mean(censored_loss)
                else:
                    censored_loss = torch.tensor(0.0, device=device)
            else:
                censored_loss = torch.tensor(0.0, device=device)

            # Add regularization term for mixture weights if using ELBO
            if self.elbo:
                # KL divergence of mixture weights from uniform prior
                mixture_prior = torch.ones_like(logits_g) / logits_g.shape[1]
                log_mixture_prior = torch.log(mixture_prior)

                mixture_posterior = F.softmax(logits_g, dim=1)
                log_mixture_posterior = F.log_softmax(logits_g, dim=1)

                kl_div = torch.sum(
                    mixture_posterior * (log_mixture_posterior - log_mixture_prior),
                    dim=1,
                )

                # Handle any invalid KL values
                kl_div = torch.nan_to_num(kl_div, nan=0.0, posinf=1.0, neginf=0.0)

                kl_term = torch.mean(kl_div) * 0.1  # Weight the KL term
            else:
                kl_term = torch.tensor(0.0, device=device)

            # Combine loss components
            event_loss = uncensored_loss + censored_loss + kl_term

            # Ensure event loss is finite
            if not torch.isfinite(event_loss):
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        f"Non-finite event loss detected: {event_loss.item()}. Using default value."
                    )
                event_loss = torch.tensor(1.0, device=device, requires_grad=True)

            # Add to total loss
            total_loss = total_loss + event_loss
            total_valid_events += 1

        # Add dependency matrix regularization if available
        if (
            hasattr(predictions, "event_dependency_matrix")
            and predictions.event_dependency_matrix is not None
        ):
            dependency_reg = self._dependency_regularization(
                predictions.event_dependency_matrix
            )

            # Ensure regularization is finite
            dependency_reg = torch.nan_to_num(
                dependency_reg, nan=0.0, posinf=1.0, neginf=0.0
            )

            total_loss = total_loss + dependency_reg

        # Return average loss across event types
        if total_valid_events > 0:
            # Final check for invalid loss values
            final_loss = total_loss / total_valid_events

            # If the loss is still not finite, return a default value that can be backpropagated
            if not torch.isfinite(final_loss):
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        f"Non-finite final loss detected: {final_loss.item()}. Using default value."
                    )
                final_loss = torch.tensor(1.0, device=device, requires_grad=True)

            return final_loss
        else:
            # Return default loss if no events found
            return torch.tensor(1.0, device=device, requires_grad=True)
