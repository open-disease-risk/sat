"""Deep Survival Machines loss for parametric survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sat.loss.base import Loss
from sat.models.tasks.heads import SAOutput


class DSMLoss(Loss):
    """Implementation of Deep Survival Machines loss.

    DSM is a parametric approach that uses a mixture of distributions
    to model survival, with special handling for right-censored data.

    References:
        Nagpal, C., Li, X., & Dubrawski, A. (2021).
        Deep Survival Machines: Fully Parametric Survival Regression and
        Representation Learning for Censored Data with Competing Risks.
        IEEE Journal of Biomedical and Health Informatics.
    """

    def __init__(
        self,
        alpha=0.1,  # Weighting for ranking component
        k=4,  # Number of mixture components
        distribution="Weibull",
        discrete_bins=None,
        importance_sample_weights=None,
        num_events=1,
        balance_strategy="none",
        ema_momentum=0.9,
    ):
        """Initialize DSM Loss.

        Args:
            alpha: Weighting for ranking component (0 = only likelihood, 1 = only ranking)
            k: Number of mixture components
            distribution: Type of distribution ('Weibull' or 'LogNormal')
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

        # Initialize loss balancing parameters for different loss components
        self._init_loss_params(["likelihood", "ranking"])

        self.alpha = alpha
        self.k = k  # Number of mixture components
        self.distribution = distribution
        self.discrete_bins = discrete_bins

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

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute the DSM loss.

        Args:
            predictions (SAOutput): Model output containing distribution objects
            references (torch.Tensor): Target tensor containing event types and durations

        Returns:
            loss (torch.Tensor): The computed DSM loss value
        """
        # Extract events and durations from references
        events = self.events(references)
        durations = self.durations(references)

        # Initialize loss
        loss = torch.zeros(1, device=events.device)

        # For each event type
        for event_type in range(self.num_events):
            # Get event indicator for current event (1 if event occurred, 0 if censored)
            event_indicator = (events == event_type + 1).float()
            censoring_indicator = 1 - event_indicator

            # Get distribution for current event
            if hasattr(predictions, "dist"):
                # Single distribution object
                dist = predictions.dist
            else:
                # Multiple distribution objects (one per event)
                if hasattr(predictions, "dists"):
                    dist = predictions.dists[event_type]
                else:
                    # Extract parameters and create distribution manually
                    if self.distribution == "Weibull":
                        from sat.models.tasks.distributions.base import (
                            WeibullDistribution,
                        )

                        if predictions.params.dim() == 2:
                            dist = WeibullDistribution(predictions.params)
                        else:
                            dist = WeibullDistribution(
                                predictions.params[:, event_type, :]
                            )
                    elif self.distribution == "LogNormal":
                        from sat.models.tasks.distributions.base import (
                            LogNormalDistribution,
                        )

                        if predictions.params.dim() == 2:
                            dist = LogNormalDistribution(predictions.params)
                        else:
                            dist = LogNormalDistribution(
                                predictions.params[:, event_type, :]
                            )
                    else:
                        raise ValueError(
                            f"Unsupported distribution: {self.distribution}"
                        )

            # Apply importance weights if available
            if hasattr(self, "importance_weights"):
                uncensored_weight = self.importance_weights[event_type + 1]
                censored_weight = self.importance_weights[0]
            else:
                uncensored_weight = 1.0
                censored_weight = 1.0

            # Compute loss for uncensored data (using PDF)
            if event_indicator.sum() > 0:
                uncensored_idx = torch.where(event_indicator > 0)[0]
                uncensored_times = durations[uncensored_idx].unsqueeze(
                    1
                )  # [n_uncensored, 1]

                # Compute PDF at uncensored times
                pdf = dist.pdf(uncensored_times).squeeze(1)  # [n_uncensored]

                # Compute negative log-likelihood
                uncensored_nll = -torch.log(pdf + 1e-8)
                loss += uncensored_weight * uncensored_nll.mean()

            # Compute loss for censored data (using survival function)
            if censoring_indicator.sum() > 0:
                censored_idx = torch.where(censoring_indicator > 0)[0]
                censored_times = durations[censored_idx].unsqueeze(1)  # [n_censored, 1]

                # Compute survival at censored times
                survival = dist.survival(censored_times).squeeze(1)  # [n_censored]

                # Compute negative log-likelihood
                censored_nll = -torch.log(survival + 1e-8)
                loss += censored_weight * censored_nll.mean()

            # Add ranking regularization if requested
            if self.alpha > 0 and event_indicator.sum() > 0:
                ranking_loss = self._compute_ranking_loss(
                    dist, durations, events, event_type
                )
                # Store ranking loss separately for balancing
                if "ranking_total" not in locals():
                    ranking_total = ranking_loss / self.num_events
                else:
                    ranking_total += ranking_loss / self.num_events

        # Normalize likelihood by number of events
        likelihood_loss = loss / self.num_events

        # Create loss dict with separate components
        loss_dict = {"likelihood": likelihood_loss}
        weights_dict = {"likelihood": 1.0 - self.alpha}

        # Add ranking loss if used
        if self.alpha > 0 and "ranking_total" in locals():
            loss_dict["ranking"] = ranking_total
            weights_dict["ranking"] = self.alpha

        return self.balance_losses(loss_dict, weights_dict)

    def _compute_ranking_loss(self, dist, durations, events, event_type):
        """Compute a ranking regularization component.

        This penalizes when patients with events have higher survival
        probabilities than patients who live longer.

        Args:
            dist: Distribution object for current event
            durations: Event times
            events: Event indicators
            event_type: Current event type

        Returns:
            ranking_loss: Regularization term for improved discrimination
        """
        # Get event indicator for current event
        event_indicator = (events == event_type + 1).float()

        # Get indices of patients with events
        event_indices = torch.where(event_indicator > 0)[0]

        # Initialize ranking loss
        ranking_loss = torch.zeros(1, device=durations.device)

        # For each patient with an event
        for i in event_indices:
            # Get time of event
            time_i = durations[i]

            # Find patients with longer survival times (risk set)
            risk_set = torch.where(durations > time_i)[0]

            # Skip if empty risk set
            if len(risk_set) == 0:
                continue

            # Compute survival at time_i
            surv_i = dist.survival(time_i.unsqueeze(0)).squeeze()

            # Get survival for risk set at time_i
            surv_j = dist.survival(time_i.unsqueeze(0).expand(len(risk_set))).squeeze()

            # Patients with events should have lower survival than those who lived longer
            # Using hinge loss: max(0, surv_i - surv_j + margin)
            margin = 0.1
            pairwise_loss = torch.relu(surv_i - surv_j + margin)

            # Sum over risk set
            ranking_loss += pairwise_loss.mean()

        # Normalize by number of events
        if len(event_indices) > 0:
            ranking_loss /= len(event_indices)

        return ranking_loss
