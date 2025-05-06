"""Deep Survival Machines Loss implementation for survival analysis."""

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


class DSMLoss(Loss):
    """
    Deep Survival Machines loss for survival analysis.

    This implementation follows the paper:
    "Deep Survival Machines: Fully Parametric Survival Regression and
    Representation Learning for Censored Data with Competing Risks"
    by Nagpal et al.

    The loss function consists of two components:
    1. Negative log-likelihood for uncensored data
    2. Negative log survival probability for censored data

    For competing risks, separate models are learned for each event type.
    """

    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        num_events: int = 1,
        distribution: str = "weibull",
        discount: float = 1.0,
        elbo: bool = False,
    ):
        """
        Initialize DSMLoss.

        Args:
            duration_cuts: Path to CSV file containing duration cut points
            importance_sample_weights: Optional path to CSV file with importance weights
            num_events: Number of competing events
            distribution: Distribution type ('weibull' or 'lognormal')
            discount: Weight for the censored loss component (0.0-1.0)
            elbo: Whether to use Evidence Lower Bound for mixture model
        """
        super(DSMLoss, self).__init__(num_events)

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(num_events + 1)

        self.register_buffer("weights", weights)
        self.discount = discount
        self.distribution = distribution
        self.elbo = elbo

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
    ):
        """
        Compute negative log-likelihood for uncensored data.
        Uses distribution classes for numerical stability.

        Args:
            time: Event times [batch_size]
            shape: Shape parameters [batch_size, num_mixtures]
            scale: Scale parameters [batch_size, num_mixtures]
            logits_g: Mixture weight logits [batch_size, num_mixtures]

        Returns:
            torch.Tensor: Negative log-likelihood [batch_size]
        """
        _, num_mixtures = shape.shape
        device = time.device

        # Create the appropriate mixture distribution based on distribution type
        if self.distribution.lower() == "weibull":
            mixture_dist = WeibullMixtureDistribution(shape, scale, logits_g)
        elif self.distribution.lower() == "lognormal":
            mixture_dist = LogNormalMixtureDistribution(
                scale, shape, logits_g
            )  # Note parameter order for lognormal: loc, scale
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
    ):
        """
        Compute negative log survival probability for censored data.
        Uses distribution classes for numerical stability.

        Args:
            time: Censoring times [batch_size]
            shape: Shape parameters [batch_size, num_mixtures]
            scale: Scale parameters [batch_size, num_mixtures]
            logits_g: Mixture weight logits [batch_size, num_mixtures]

        Returns:
            torch.Tensor: Negative log survival probability [batch_size]
        """
        batch_size, num_mixtures = shape.shape

        # Create the appropriate mixture distribution based on distribution type
        if self.distribution.lower() == "weibull":
            mixture_dist = WeibullMixtureDistribution(shape, scale, logits_g)
        elif self.distribution.lower() == "lognormal":
            mixture_dist = LogNormalMixtureDistribution(
                scale, shape, logits_g
            )  # Note parameter order for lognormal: loc, scale
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

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the DSM loss for survival prediction.

        Args:
            predictions: DSM model outputs containing distribution parameters
            references: Reference values with event indicators and times
                [batch_size, 4 * num_events]

        Returns:
            torch.Tensor: The loss value
        """
        # Extract event indicators and durations
        events = self.events(references)  # [batch_size, num_events]
        durations = self.durations(references)  # [batch_size, num_events]

        # batch_size = events.shape[0]  # Used for debugging/logging if needed
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

            # Calculate uncensored loss for samples with events
            uncensored_mask = event_indicator == 1
            if torch.any(uncensored_mask):
                uncensored_times = event_duration[uncensored_mask]
                uncensored_shape = shape[uncensored_mask]
                uncensored_scale = scale[uncensored_mask]
                uncensored_logits_g = logits_g[uncensored_mask]

                uncensored_loss = self._negative_log_likelihood(
                    uncensored_times,
                    uncensored_shape,
                    uncensored_scale,
                    uncensored_logits_g,
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

                    censored_loss = self._negative_log_survival(
                        valid_censored_times,
                        valid_censored_shape,
                        valid_censored_scale,
                        valid_censored_logits_g,
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

                kl_term = torch.mean(kl_div) * 0.1  # Weight the KL term
            else:
                kl_term = torch.tensor(0.0, device=device)

            # Combine loss components
            event_loss = uncensored_loss + censored_loss + kl_term

            # Add to total loss if not zero
            if event_loss.item() != 0:
                total_loss = total_loss + event_loss
                total_valid_events += 1

        # Return average loss across event types
        if total_valid_events > 0:
            return total_loss / total_valid_events
        else:
            # Return zero loss if no events found
            return torch.tensor(0.0, device=device, requires_grad=True)
