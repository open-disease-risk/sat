"""
Base classes for survival analysis distributions.

This module defines the interface for survival distributions and mixture models.
All specialized distributions should inherit from these base classes.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class SurvivalDistribution(ABC):
    """
    Base class for all survival distributions.

    This abstract class defines the interface that all survival distributions
    must implement, including methods for computing survival function,
    hazard function, and log-likelihood.
    """

    @abstractmethod
    def survival_function(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute the survival function S(t) = P(T > t).

        Args:
            time: Time points at which to evaluate the survival function [batch_size, num_times]

        Returns:
            Survival function values [batch_size, num_times]
        """
        pass

    @abstractmethod
    def hazard_function(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute the hazard function h(t) = -d/dt log(S(t)).

        Args:
            time: Time points at which to evaluate the hazard function [batch_size, num_times]

        Returns:
            Hazard function values [batch_size, num_times]
        """
        pass

    @abstractmethod
    def log_likelihood(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute the log likelihood of the observed times.

        Args:
            time: Observed event times [batch_size]

        Returns:
            Log likelihood values [batch_size]
        """
        pass

    @abstractmethod
    def log_survival(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute the log survival function log(S(t)).

        Args:
            time: Time points at which to evaluate the log survival [batch_size, num_times]

        Returns:
            Log survival values [batch_size, num_times]
        """
        pass

    @abstractmethod
    def cumulative_hazard(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute the cumulative hazard function H(t) = -log(S(t)).

        Args:
            time: Time points at which to evaluate the cumulative hazard [batch_size, num_times]

        Returns:
            Cumulative hazard values [batch_size, num_times]
        """
        pass

    @abstractmethod
    def mean(self) -> torch.Tensor:
        """
        Compute the mean of the distribution.

        Returns:
            Mean values [batch_size]
        """
        pass


class MixtureDistribution(SurvivalDistribution):
    """
    Base class for mixture distributions used in survival analysis.

    This class implements the common functionality of mixture distributions,
    where the final distribution is a weighted sum of multiple component distributions.
    """

    def __init__(self, num_mixtures: int, eps: float = 1e-7):
        """
        Initialize the mixture distribution.

        Args:
            num_mixtures: Number of mixture components
            eps: Small constant for numerical stability
        """
        self.num_mixtures = num_mixtures
        self.eps = eps

    @abstractmethod
    def get_component_distributions(self, **kwargs) -> Tuple[SurvivalDistribution]:
        """
        Get the component distributions of this mixture.

        Returns:
            Tuple of component distributions
        """
        pass

    @abstractmethod
    def get_mixture_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute mixture weights from logits.

        Args:
            logits: Raw logits for mixture weights [batch_size, num_mixtures]

        Returns:
            Normalized mixture weights [batch_size, num_mixtures]
        """
        pass

    def survival_function(
        self,
        time: torch.Tensor,
        distributions: Tuple[SurvivalDistribution],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the mixture survival function.

        Args:
            time: Time points [batch_size, num_times]
            distributions: Component distributions
            weights: Mixture weights [batch_size, num_mixtures]

        Returns:
            Mixture survival function [batch_size, num_times]
        """
        batch_size, num_times = time.shape

        # Initialize survival tensor
        survival = torch.zeros(batch_size, num_times, device=time.device)

        # Expand weights for broadcasting
        weights_expanded = weights.unsqueeze(-1)  # [batch_size, num_mixtures, 1]

        # Compute survival for each component and weight the results
        for i, dist in enumerate(distributions):
            component_survival = dist.survival_function(time)  # [batch_size, num_times]
            component_survival = torch.clamp(component_survival, min=0.0, max=1.0)
            survival += weights_expanded[:, i, :] * component_survival

        # Ensure survival probabilities are valid
        survival = torch.clamp(survival, min=0.0, max=1.0)

        return survival

    def log_survival(
        self,
        time: torch.Tensor,
        distributions: Tuple[SurvivalDistribution],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the log of the mixture survival function.

        Note: This cannot use logsumexp directly since survival is a weighted sum,
        not a product of component distributions.

        Args:
            time: Time points [batch_size, num_times]
            distributions: Component distributions
            weights: Mixture weights [batch_size, num_mixtures]

        Returns:
            Log mixture survival function [batch_size, num_times]
        """
        # Compute regular survival (weighted sum) then take log
        survival = self.survival_function(time, distributions, weights)

        # Safe log
        log_survival = torch.log(torch.clamp(survival, min=self.eps))

        return log_survival

    def hazard_function(
        self,
        time: torch.Tensor,
        distributions: Tuple[SurvivalDistribution],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the mixture hazard function.

        For mixture distributions, hazard can be computed as a weighted sum of component
        hazards, weighted by their posterior probabilities.

        Args:
            time: Time points [batch_size, num_times]
            distributions: Component distributions
            weights: Mixture weights [batch_size, num_mixtures]

        Returns:
            Mixture hazard function [batch_size, num_times]
        """
        batch_size, num_times = time.shape

        # Compute survival for each component
        component_survivals = []
        for dist in distributions:
            component_survival = dist.survival_function(time)  # [batch_size, num_times]
            component_survival = torch.clamp(
                component_survival, min=self.eps, max=1.0 - self.eps
            )
            component_survivals.append(component_survival)

        # Compute mixture survival
        mixture_survival = torch.zeros(batch_size, num_times, device=time.device)
        for i, component_survival in enumerate(component_survivals):
            mixture_survival += weights[:, i].unsqueeze(-1) * component_survival

        # Ensure valid probability
        mixture_survival = torch.clamp(
            mixture_survival, min=self.eps, max=1.0 - self.eps
        )

        # Compute mixture hazard using weighted hazards
        mixture_hazard = torch.zeros(batch_size, num_times, device=time.device)
        for i, dist in enumerate(distributions):
            # Weight by posterior probability of the component
            posterior_weight = (
                weights[:, i].unsqueeze(-1) * component_survivals[i]
            ) / mixture_survival
            mixture_hazard += posterior_weight * dist.hazard_function(time)

        # Ensure non-negative hazard
        mixture_hazard = torch.clamp(mixture_hazard, min=0.0)

        return mixture_hazard

    def log_likelihood(
        self,
        time: torch.Tensor,
        distributions: Tuple[SurvivalDistribution],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the log likelihood for observed event times.

        For mixture distributions, we can use logsumexp for stable computation:
        log(sum_i w_i * f_i(t)) = logsumexp(log(w_i) + log(f_i(t)))

        Args:
            time: Observed event times [batch_size]
            distributions: Component distributions
            weights: Mixture weights [batch_size, num_mixtures]

        Returns:
            Log likelihood values [batch_size]
        """
        # Get batch size for potential debugging/logging if needed
        # batch_size = time.shape[0]

        # Initialize log likelihood tensor
        log_weights = torch.log(
            torch.clamp(weights, min=self.eps)
        )  # [batch_size, num_mixtures]

        # Compute component log likelihoods
        component_log_likelihoods = []
        for dist in distributions:
            component_ll = dist.log_likelihood(time)  # [batch_size]
            component_log_likelihoods.append(component_ll)

        # Stack component log likelihoods
        stacked_lls = torch.stack(
            component_log_likelihoods, dim=1
        )  # [batch_size, num_mixtures]

        # Use logsumexp for numerical stability
        log_likelihood = torch.logsumexp(
            log_weights + stacked_lls, dim=1
        )  # [batch_size]

        return log_likelihood

    def cumulative_hazard(
        self,
        time: torch.Tensor,
        distributions: Tuple[SurvivalDistribution],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the cumulative hazard function H(t) = -log(S(t)).

        Args:
            time: Time points [batch_size, num_times]
            distributions: Component distributions
            weights: Mixture weights [batch_size, num_mixtures]

        Returns:
            Cumulative hazard values [batch_size, num_times]
        """
        log_survival = self.log_survival(time, distributions, weights)
        cumulative_hazard = -log_survival
        return cumulative_hazard

    def mean(
        self, distributions: Tuple[SurvivalDistribution], weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the mean of the mixture distribution.

        Args:
            distributions: Component distributions
            weights: Mixture weights [batch_size, num_mixtures]

        Returns:
            Mean values [batch_size]
        """
        # Compute weighted sum of component means
        batch_size = weights.shape[0]
        mixture_mean = torch.zeros(batch_size, device=weights.device)

        for i, dist in enumerate(distributions):
            component_mean = dist.mean()  # [batch_size]
            mixture_mean += weights[:, i] * component_mean

        return mixture_mean
