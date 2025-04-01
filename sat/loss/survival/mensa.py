"""MENSA Loss implementation for multi-event survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from sat.models.heads import SAOutput
from sat.utils import logging
from ..base import Loss

logger = logging.get_default_logger()


class MENSALoss(Loss):
    """
    MENSA loss for multi-event survival analysis.

    This implementation follows the paper:
    "MENSA: Multi-Event Neural Survival Analysis"

    The loss function consists of multiple components:
    1. Negative log-likelihood for uncensored data using mixture of Weibull distributions
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
    ):
        """
        Compute negative log-likelihood for uncensored data.

        Args:
            time: Event times [batch_size]
            shape: Shape parameters [batch_size, num_mixtures]
            scale: Scale parameters [batch_size, num_mixtures]
            logits_g: Mixture weight logits [batch_size, num_mixtures]

        Returns:
            torch.Tensor: Negative log-likelihood [batch_size]
        """
        batch_size, num_mixtures = shape.shape
        device = time.device

        # Calculate log likelihood for each component distribution
        time_expanded = time.unsqueeze(1).expand(-1, num_mixtures)

        if self.distribution.lower() == "weibull":
            # Compute PDF of Weibull: (shape/scale)*(t/scale)^(shape-1)*exp(-(t/scale)^shape)
            # Log PDF is more stable numerically
            term1 = torch.log(shape) - torch.log(scale)
            term2 = (shape - 1) * (torch.log(time_expanded) - torch.log(scale))
            term3 = -torch.pow(time_expanded / scale, shape)
            log_pdf = term1 + term2 + term3
        elif self.distribution.lower() == "lognormal":
            # Compute PDF of lognormal distribution
            z = (torch.log(time_expanded) - scale) / shape
            log_pdf = (
                -torch.log(shape)
                - torch.log(time_expanded)
                - 0.5 * torch.log(2 * torch.tensor(np.pi, device=device))
                - 0.5 * z * z
            )
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

        # Use logsumexp for numerical stability when calculating mixture log likelihood
        log_weights = F.log_softmax(logits_g, dim=1)
        mixture_ll = torch.logsumexp(log_weights + log_pdf, dim=1)

        # Return negative log likelihood
        return -mixture_ll

    def _negative_log_survival(
        self,
        time: torch.Tensor,
        shape: torch.Tensor,
        scale: torch.Tensor,
        logits_g: torch.Tensor,
    ):
        """
        Compute negative log survival probability for censored data.

        Args:
            time: Censoring times [batch_size]
            shape: Shape parameters [batch_size, num_mixtures]
            scale: Scale parameters [batch_size, num_mixtures]
            logits_g: Mixture weight logits [batch_size, num_mixtures]

        Returns:
            torch.Tensor: Negative log survival probability [batch_size]
        """
        batch_size, num_mixtures = shape.shape
        device = time.device

        # Calculate survival function for each component
        time_expanded = time.unsqueeze(1).expand(-1, num_mixtures)

        if self.distribution.lower() == "weibull":
            # Weibull survival: exp(-(t/scale)^shape)
            log_surv = -torch.pow(time_expanded / scale, shape)
        elif self.distribution.lower() == "lognormal":
            # Log-normal survival function: 1 - CDF of normal distribution
            z = (torch.log(time_expanded) - scale) / shape
            surv = 0.5 - 0.5 * torch.erf(
                z / torch.sqrt(torch.tensor(2.0, device=device))
            )
            # Avoid log(0)
            surv = torch.clamp(surv, min=1e-7)
            log_surv = torch.log(surv)
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

        # Calculate mixture survival:
        # S(t) = Σ_k w_k * S_k(t)
        weights = F.softmax(logits_g, dim=1)

        # Calculate weighted sum for survival - we can't use logsumexp here
        # because survival is a weighted sum, not a product of components
        surv_per_component = torch.exp(log_surv)
        mixture_surv = torch.sum(weights * surv_per_component, dim=1)

        # Avoid log(0)
        mixture_surv = torch.clamp(mixture_surv, min=1e-7)
        log_mixture_surv = torch.log(mixture_surv)

        # Return negative log survival
        return -log_mixture_surv

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

        # Add dependency matrix regularization if available
        if (
            hasattr(predictions, "event_dependency_matrix")
            and predictions.event_dependency_matrix is not None
        ):
            dependency_reg = self._dependency_regularization(
                predictions.event_dependency_matrix
            )
            total_loss = total_loss + dependency_reg

        # Return average loss across event types
        if total_valid_events > 0:
            return total_loss / total_valid_events
        else:
            # Return zero loss if no events found
            return torch.tensor(0.0, device=device, requires_grad=True)
