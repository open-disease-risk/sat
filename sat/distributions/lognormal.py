"""
LogNormal distribution implementation for survival analysis.

This module provides numerically stable implementations of LogNormal distribution
and LogNormal mixture distribution for survival analysis tasks.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .base import MixtureDistribution, SurvivalDistribution


class LogNormalDistribution(SurvivalDistribution):
    """
    Numerically stable implementation of LogNormal distribution with statistical theory constraints.

    The LogNormal distribution is parameterized by location (μ) and scale (σ) parameters.
    - Survival function: S(t) = 1 - Φ((log(t) - μ)/σ)
    - Hazard function: h(t) = f(t)/S(t)
    - PDF: f(t) = 1/(t*σ*sqrt(2π)) * exp(-(log(t) - μ)²/(2σ²))

    Where Φ is the standard normal CDF.

    Statistical properties:
    - Median survival time: exp(μ)
    - Mean survival time: exp(μ + σ²/2)
    - Variance increases with both μ and σ
    - Has a non-monotonic hazard that increases then decreases

    Common values in survival analysis literature:
    - Cancer: μ ≈ 1.5-3.0, σ ≈ 0.7-1.5
      References:
      - Royston & Parmar (2002). "Flexible parametric proportional-hazards and proportional-odds models for censored survival data." Statistics in Medicine, 21(15), 2175-2197.
      - Zelen (1966). "Application of exponential models to problems in cancer research." Journal of the Royal Statistical Society, 29(3), 368-398.
    - Chronic diseases: μ ≈ 2.0-4.0, σ ≈ 0.8-1.2
      References:
      - Farewell & Prentice (1977). "A study of distributional shape in life testing." Technometrics, 19(1), 69-75.
      - Cox et al. (2007). "Parametric survival analysis and taxonomy of hazard functions for the generalized gamma distribution." Statistics in Medicine, 26(23), 4352-4374.
    - Organ transplant: μ ≈ 2.0-3.5, σ ≈ 0.6-1.2
      References:
      - Klein et al. (2001). "Survival analysis: State of the art." Kluwer Academic Publishers, pp. 219-253.
      - Jackson et al. (2010). "Comparison of MELD and Child-Pugh scores to predict survival after chemoembolization for hepatocellular carcinoma." Journal of Vascular and Interventional Radiology, 21(11), 1725-1732.
    """

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        eps: float = 1e-7,
        constrain_params: bool = False,
        event_type: str = None,
        use_expert_priors: bool = False,
    ):
        """
        Initialize LogNormal distribution with location and scale parameters.

        Args:
            loc: Location parameter (μ) [batch_size]
            scale: Scale parameter (σ) [batch_size]
            eps: Small constant for numerical stability
            constrain_params: Whether to apply expert constraints
            event_type: Optional event type for specific priors
            use_expert_priors: Whether to use expert knowledge for initialization
        """
        # Apply expert knowledge if requested
        if use_expert_priors and event_type is not None:
            loc, scale = self._apply_expert_priors(loc, scale, event_type)

        # Apply constraints based on statistical knowledge
        if constrain_params:
            self.loc, self.scale = self._constrain_parameters(loc, scale, event_type)
        else:
            # Basic constraints for numerical stability
            self.loc = torch.clamp(
                loc, min=-100.0, max=100.0
            )  # Location can be negative
            self.scale = torch.clamp(
                scale, min=eps, max=100.0
            )  # Scale should be positive

        self.eps = eps

    def _apply_expert_priors(
        self, loc: torch.Tensor, scale: torch.Tensor, event_type: str
    ) -> tuple:
        """
        Apply expert knowledge priors to location and scale parameters.

        Args:
            loc: Location parameter (μ) [batch_size]
            scale: Scale parameter (σ) [batch_size]
            event_type: Type of event for specific priors

        Returns:
            Tuple of (loc, scale) with expert knowledge applied
        """
        # Apply different priors based on event type
        if event_type == "cancer":
            # Cancer often follows lognormal with specific parameters
            # Median survival for many cancers is around 3-20 months
            prior_loc = 2.0  # log(e^2) ≈ 7.4 months median
            prior_scale = 1.0  # moderate variance

            # Blend with existing values
            loc_blend = 0.8  # Keep 80% of original, blend 20% with prior
            loc = loc_blend * loc + (1 - loc_blend) * prior_loc

            scale_blend = 0.85
            scale = scale_blend * scale + (1 - scale_blend) * prior_scale

        elif event_type == "chronic_disease":
            # Chronic diseases often have longer survival times
            prior_loc = 3.0  # log(e^3) ≈ 20 months median
            prior_scale = 0.9  # lower variance

            loc_blend = 0.75
            loc = loc_blend * loc + (1 - loc_blend) * prior_loc

            scale_blend = 0.8
            scale = scale_blend * scale + (1 - scale_blend) * prior_scale

        elif event_type == "transplant":
            # Organ transplant survival has specific patterns
            prior_loc = 2.5  # log(e^2.5) ≈ 12 months median
            prior_scale = 0.7  # lower variance

            loc_blend = 0.7
            loc = loc_blend * loc + (1 - loc_blend) * prior_loc

            scale_blend = 0.7
            scale = scale_blend * scale + (1 - scale_blend) * prior_scale

        # Other event types could be added here

        return loc, scale

    def _constrain_parameters(
        self, loc: torch.Tensor, scale: torch.Tensor, event_type: str = None
    ) -> tuple:
        """
        Apply statistical theory constraints to parameters based on event type.

        Args:
            loc: Location parameter (μ) [batch_size]
            scale: Scale parameter (σ) [batch_size]
            event_type: Type of event for specific constraints

        Returns:
            Tuple of constrained (loc, scale) parameters
        """
        # Apply basic constraints first
        loc_constrained = torch.clamp(loc, min=-100.0, max=100.0)
        scale_constrained = torch.clamp(scale, min=self.eps, max=100.0)

        # Apply event-specific constraints if event type is provided
        if event_type is not None:
            if event_type == "cancer":
                # Most cancer studies show lognormal parameters in specific ranges
                loc_constrained = torch.clamp(loc_constrained, min=0.5, max=5.0)
                scale_constrained = torch.clamp(scale_constrained, min=0.5, max=1.5)

            elif event_type == "chronic_disease":
                # Chronic diseases typically have higher location parameter
                # indicating longer survival times
                loc_constrained = torch.clamp(loc_constrained, min=1.5, max=5.0)

            elif event_type == "transplant":
                # Transplant survival often has moderate loc and lower scale
                loc_constrained = torch.clamp(loc_constrained, min=1.0, max=4.0)
                scale_constrained = torch.clamp(scale_constrained, min=0.3, max=1.2)

        return loc_constrained, scale_constrained

    def survival_function(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute LogNormal survival function S(t) = 1 - Φ((log(t) - μ)/σ).

        Args:
            time: Time points [batch_size, num_times]

        Returns:
            Survival function values [batch_size, num_times]
        """
        # Ensure time is positive
        time_safe = torch.clamp(time, min=self.eps)

        # Handle broadcasting - reshape loc and scale for broadcasting
        loc_expanded = self.loc.unsqueeze(-1)  # [batch_size, 1]
        scale_expanded = self.scale.unsqueeze(-1)  # [batch_size, 1]

        # Compute standard normal CDF argument
        log_time = torch.log(time_safe)
        z = (log_time - loc_expanded) / scale_expanded

        # Clamp z to avoid numerical issues with erf
        z_clamped = torch.clamp(z, min=-8.0, max=8.0)

        # Compute survival using relationship with normal CDF
        # S(t) = 1 - Φ(z) = 0.5 - 0.5 * erf(z/sqrt(2))
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=time.device))
        survival = 0.5 - 0.5 * torch.erf(z_clamped / sqrt_2)

        # Ensure valid probabilities
        survival = torch.clamp(survival, min=0.0, max=1.0)

        return survival

    def log_survival(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute log of LogNormal survival function log(S(t)).

        Args:
            time: Time points [batch_size, num_times]

        Returns:
            Log survival values [batch_size, num_times]
        """
        # Compute survival then take log
        survival = self.survival_function(time)
        log_survival = torch.log(torch.clamp(survival, min=self.eps))

        return log_survival

    def hazard_function(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute LogNormal hazard function h(t) = f(t)/S(t).

        Args:
            time: Time points [batch_size, num_times]

        Returns:
            Hazard function values [batch_size, num_times]
        """
        # Ensure time is positive
        time_safe = torch.clamp(time, min=self.eps)

        # Handle broadcasting
        loc_expanded = self.loc.unsqueeze(-1)  # [batch_size, 1]
        scale_expanded = self.scale.unsqueeze(-1)  # [batch_size, 1]

        # Compute PDF
        log_time = torch.log(time_safe)
        z = (log_time - loc_expanded) / scale_expanded
        z_squared = z * z

        # Clamp to avoid overflow/underflow
        z_squared_clamped = torch.clamp(z_squared, max=30.0)

        # LogNormal PDF computation
        # f(t) = 1/(t*σ*sqrt(2π)) * exp(-(log(t) - μ)²/(2σ²))
        # Compute in log domain for stability
        log_pdf = (
            -torch.log(time_safe)
            - torch.log(scale_expanded)
            - 0.5 * torch.log(2 * torch.tensor(np.pi, device=time.device))
            - 0.5 * z_squared_clamped
        )
        pdf = torch.exp(log_pdf)

        # Compute survival
        survival = self.survival_function(time)

        # Avoid division by zero
        survival_safe = torch.clamp(survival, min=self.eps)

        # Compute hazard = pdf / survival
        hazard = pdf / survival_safe

        # Ensure non-negative
        hazard = torch.clamp(hazard, min=0.0)

        return hazard

    def log_likelihood(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute log likelihood of observed times.

        Log likelihood = log(f(t))

        Args:
            time: Observed event times [batch_size]

        Returns:
            Log likelihood values [batch_size]
        """
        # Ensure time is positive and bounded
        time_safe = torch.clamp(time, min=self.eps, max=1e10)

        # Ensure scale is positive and loc is within bounds
        scale_safe = torch.clamp(self.scale, min=self.eps, max=100.0)
        loc_safe = torch.clamp(self.loc, min=-100.0, max=100.0)

        # Compute z-score with safety checks
        log_time = torch.log(time_safe)
        z = (log_time - loc_safe) / scale_safe

        # Apply reasonable bounds to z
        z_clamped = torch.clamp(z, min=-10.0, max=10.0)
        z_squared = z_clamped * z_clamped

        # Clamp to avoid overflow/underflow
        z_squared_clamped = torch.clamp(z_squared, min=0.0, max=30.0)

        # Compute log PDF directly with numerical stability
        log_time_safe = torch.log(time_safe)
        log_scale_safe = torch.log(scale_safe)
        log_2pi = torch.log(2 * torch.tensor(np.pi, device=time.device))

        # Handle any potential NaN values
        log_time_safe = torch.nan_to_num(
            log_time_safe, nan=0.0, posinf=10.0, neginf=-10.0
        )
        log_scale_safe = torch.nan_to_num(
            log_scale_safe, nan=0.0, posinf=5.0, neginf=-5.0
        )

        # Compute log likelihood components
        term1 = -log_time_safe
        term2 = -log_scale_safe
        term3 = -0.5 * log_2pi
        term4 = -0.5 * z_squared_clamped

        # Combine terms with bounds
        log_likelihood = term1 + term2 + term3 + term4

        # Final safety check for numerical stability
        log_likelihood = torch.clamp(log_likelihood, min=-100.0, max=100.0)
        log_likelihood = torch.nan_to_num(
            log_likelihood, nan=-10.0, posinf=10.0, neginf=-10.0
        )

        return log_likelihood

    def cumulative_hazard(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute LogNormal cumulative hazard function H(t) = -log(S(t)).

        Args:
            time: Time points [batch_size, num_times]

        Returns:
            Cumulative hazard values [batch_size, num_times]
        """
        log_survival = self.log_survival(time)
        cumulative_hazard = -log_survival

        return cumulative_hazard

    def mean(self) -> torch.Tensor:
        """
        Compute the mean of the LogNormal distribution E[T] = exp(μ + σ²/2).

        Returns:
            Mean values [batch_size]
        """
        # Compute mean safely in log domain
        variance_term = 0.5 * self.scale * self.scale
        log_mean = self.loc + variance_term

        # Clamp to avoid overflow
        log_mean_clamped = torch.clamp(log_mean, max=20.0)
        mean = torch.exp(log_mean_clamped)

        return mean


class LogNormalMixtureDistribution(MixtureDistribution):
    """
    Mixture of LogNormal distributions for flexible survival modeling.

    This class represents a mixture of LogNormal distributions, providing
    methods to compute survival, hazard, and likelihood functions.

    The LogNormal mixture model is particularly useful for:
    - Capturing multiple risk phases (early vs late mortality)
    - Modeling heterogeneous patient populations
    - Representing diverse failure mechanisms

    Research in survival analysis has found that mixtures of log-normal distributions
    can effectively model complex survival patterns in various diseases and medical
    scenarios, often outperforming single-distribution models.

    Key references:
    - McLachlan & Peel (2000). "Finite Mixture Models," Wiley-Interscience, Chapter 6.
    - Yu et al. (2001). "Analysis of multivariate survival data using mixture models," Biometrics, 57(2), 521-529.
    - Farewell & Prentice (1977). "A study of distributional shape in life testing," Technometrics, 19(1), 69-75.
    - Jackson et al. (2016). "Flexible parametric accelerated failure time models," The Stata Journal, 16(2), 310-328.
    """

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        logits: torch.Tensor,
        eps: float = 1e-7,
        constrain_params: bool = False,
        event_type: str = None,
        use_expert_priors: bool = False,
    ):
        """
        Initialize a LogNormal mixture distribution.

        Args:
            loc: Location parameters [batch_size, num_mixtures]
            scale: Scale parameters [batch_size, num_mixtures]
            logits: Mixture weight logits [batch_size, num_mixtures]
            eps: Small constant for numerical stability
            constrain_params: Whether to apply expert constraints
            event_type: Optional event type for specific constraints
            use_expert_priors: Whether to use expert knowledge for initialization
        """
        batch_size, num_mixtures = loc.shape
        super().__init__(num_mixtures, eps)

        # Store original parameters
        self.loc_orig = loc
        self.scale_orig = scale
        self.logits = logits

        # Apply expert knowledge if requested - with component-specific logic
        if use_expert_priors and event_type is not None:
            # For mixture models, different components can represent distinct risk groups

            # First component often represents the primary risk profile
            if num_mixtures >= 1:
                first_loc = loc[:, 0:1]
                first_scale = scale[:, 0:1]

                if event_type == "cancer":
                    # First component: primary cancer mortality
                    # Moderate median survival (exp(2.0) ≈ 7.4 months)
                    first_loc = torch.clamp(first_loc, min=1.5, max=3.0)
                    first_scale = torch.clamp(first_scale, min=0.7, max=1.3)
                elif event_type == "chronic_disease":
                    # First component: primary disease progression
                    # Longer median survival (exp(3.0) ≈ 20 months)
                    first_loc = torch.clamp(first_loc, min=2.5, max=4.0)
                    first_scale = torch.clamp(first_scale, min=0.6, max=1.0)

                loc[:, 0:1] = first_loc
                scale[:, 0:1] = first_scale

            # Second component can represent a secondary risk group
            if num_mixtures >= 2:
                second_loc = loc[:, 1:2]
                second_scale = scale[:, 1:2]

                if event_type == "cancer":
                    # Second component: long-term survivors
                    # Higher median survival (exp(3.5) ≈ 33 months)
                    second_loc = torch.clamp(second_loc, min=3.0, max=4.5)
                    second_scale = torch.clamp(second_scale, min=0.5, max=0.9)
                elif event_type == "chronic_disease":
                    # Second component: rapid progressors
                    # Lower median survival (exp(1.5) ≈ 4.5 months)
                    second_loc = torch.clamp(second_loc, min=1.0, max=2.0)
                    second_scale = torch.clamp(second_scale, min=0.4, max=0.8)

                loc[:, 1:2] = second_loc
                scale[:, 1:2] = second_scale

        # Store parameters with appropriate constraints for numerical stability
        self.loc = torch.clamp(loc, min=-100.0, max=100.0)
        self.scale = torch.clamp(scale, min=eps, max=100.0)

        # Pre-compute weights
        self.weights = self.get_mixture_weights(logits)

    def get_component_distributions(self) -> Tuple[LogNormalDistribution]:
        """
        Create component LogNormal distributions for each mixture.

        Returns:
            Tuple of LogNormal distributions
        """
        batch_size, num_mixtures = self.loc.shape
        distributions = []

        for i in range(num_mixtures):
            dist = LogNormalDistribution(
                loc=self.loc[:, i],  # [batch_size]
                scale=self.scale[:, i],  # [batch_size]
                eps=self.eps,
            )
            distributions.append(dist)

        return tuple(distributions)

    def get_mixture_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute mixture weights from logits using softmax.

        Args:
            logits: Raw logits for mixture weights [batch_size, num_mixtures]

        Returns:
            Normalized mixture weights [batch_size, num_mixtures]
        """
        # Apply softmax to get normalized weights
        weights = F.softmax(logits, dim=1)

        # Ensure weights sum to 1 and are non-negative
        weights = torch.clamp(weights, min=self.eps)
        weights = weights / weights.sum(dim=1, keepdim=True)

        return weights

    def survival_function(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute mixture survival function.

        Args:
            time: Time points [batch_size, num_times]

        Returns:
            Mixture survival function [batch_size, num_times]
        """
        distributions = self.get_component_distributions()

        # Initialize survival tensor
        batch_size, num_times = time.shape
        survival = torch.zeros(batch_size, num_times, device=time.device)

        # Expand weights for broadcasting
        weights_expanded = self.weights.unsqueeze(-1)  # [batch_size, num_mixtures, 1]

        # Compute survival for each component and weight the results
        for i, dist in enumerate(distributions):
            component_survival = dist.survival_function(time)  # [batch_size, num_times]
            component_survival = torch.clamp(component_survival, min=0.0, max=1.0)
            survival += weights_expanded[:, i, :] * component_survival

        # Ensure survival probabilities are valid
        survival = torch.clamp(survival, min=0.0, max=1.0)

        return survival

    def hazard_function(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute mixture hazard function.

        Args:
            time: Time points [batch_size, num_times]

        Returns:
            Mixture hazard function [batch_size, num_times]
        """
        distributions = self.get_component_distributions()
        batch_size, num_times = time.shape

        # Ensure time is positive and within reasonable bounds
        time_safe = torch.clamp(time, min=self.eps, max=1e10)

        # Ensure weights are normalized and valid
        weights_safe = torch.clamp(self.weights, min=self.eps, max=1.0 - self.eps)
        weights_sum = weights_safe.sum(dim=1, keepdim=True)
        weights_normalized = weights_safe / weights_sum

        # Compute survival for each component with careful handling
        component_survivals = []
        for dist in distributions:
            component_survival = dist.survival_function(
                time_safe
            )  # [batch_size, num_times]
            component_survival = torch.clamp(
                component_survival, min=self.eps, max=1.0 - self.eps
            )
            # Check for NaN or Inf values
            component_survival = torch.nan_to_num(
                component_survival, nan=0.5, posinf=1.0 - self.eps, neginf=self.eps
            )
            component_survivals.append(component_survival)

        # Compute mixture survival
        mixture_survival = torch.zeros(batch_size, num_times, device=time.device)
        for i, component_survival in enumerate(component_survivals):
            mixture_survival += (
                weights_normalized[:, i].unsqueeze(-1) * component_survival
            )

        # Ensure valid probability
        mixture_survival = torch.clamp(
            mixture_survival, min=self.eps, max=1.0 - self.eps
        )

        # Compute mixture hazard using weighted hazards
        mixture_hazard = torch.zeros(batch_size, num_times, device=time.device)
        for i, dist in enumerate(distributions):
            # Get component hazard with careful handling
            component_hazard = dist.hazard_function(time_safe)
            component_hazard = torch.clamp(component_hazard, min=0.0, max=1e3)
            component_hazard = torch.nan_to_num(
                component_hazard, nan=0.0, posinf=1e3, neginf=0.0
            )

            # Weight by posterior probability of the component
            posterior_weight = (
                weights_normalized[:, i].unsqueeze(-1) * component_survivals[i]
            ) / mixture_survival
            # Ensure posterior weights are valid
            posterior_weight = torch.clamp(posterior_weight, min=0.0, max=1.0)
            posterior_weight = torch.nan_to_num(
                posterior_weight, nan=1.0 / self.num_mixtures, posinf=1.0, neginf=0.0
            )

            mixture_hazard += posterior_weight * component_hazard

        # Ensure non-negative and bounded hazard
        mixture_hazard = torch.clamp(mixture_hazard, min=0.0, max=1e3)

        # Final check for any remaining NaN or Inf values
        mixture_hazard = torch.nan_to_num(
            mixture_hazard, nan=1.0, posinf=1e3, neginf=0.0
        )

        return mixture_hazard

    def log_likelihood(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute mixture log likelihood.

        Args:
            time: Observed event times [batch_size]

        Returns:
            Mixture log likelihood [batch_size]
        """
        distributions = self.get_component_distributions()
        # batch_size = time.shape[0]  # For potential debugging/logging

        # Ensure time is positive and within reasonable bounds
        time_safe = torch.clamp(time, min=self.eps, max=1e10)

        # Initialize log weights with careful clamping
        weights_safe = torch.clamp(self.weights, min=self.eps, max=1.0 - self.eps)
        weights_sum = weights_safe.sum(dim=1, keepdim=True)
        weights_normalized = weights_safe / weights_sum
        log_weights = torch.log(weights_normalized)

        # Compute component log likelihoods with careful handling of extreme values
        component_log_likelihoods = []
        for dist in distributions:
            component_ll = dist.log_likelihood(time_safe)  # [batch_size]
            # Apply reasonable bounds to log-likelihood to prevent extreme values
            component_ll = torch.clamp(component_ll, min=-100.0, max=100.0)
            component_log_likelihoods.append(component_ll)

        # Stack component log likelihoods
        stacked_lls = torch.stack(
            component_log_likelihoods, dim=1
        )  # [batch_size, num_mixtures]

        # Combine log weights and log likelihoods with careful handling of extreme values
        combined_values = log_weights + stacked_lls
        # Remove any NaN or Inf values before logsumexp
        combined_values_safe = torch.where(
            torch.isfinite(combined_values),
            combined_values,
            torch.tensor(-100.0, device=combined_values.device),
        )

        # Calculate max value for stability in logsumexp
        max_values, _ = torch.max(combined_values_safe, dim=1, keepdim=True)

        # Avoid underflow in exp by subtracting max values
        exp_values = torch.exp(combined_values_safe - max_values)

        # Sum of exponentials
        sum_exp = torch.sum(exp_values, dim=1)

        # Compute stable log-likelihood
        log_likelihood = max_values.squeeze(1) + torch.log(sum_exp)

        # Handle any remaining NaN or Inf values
        log_likelihood = torch.nan_to_num(
            log_likelihood, nan=-10.0, posinf=10.0, neginf=-10.0
        )

        return log_likelihood

    def log_survival(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute log mixture survival function.

        Args:
            time: Time points [batch_size, num_times]

        Returns:
            Log mixture survival function [batch_size, num_times]
        """
        # Compute regular survival (weighted sum) then take log
        survival = self.survival_function(time)

        # Safe log
        log_survival = torch.log(torch.clamp(survival, min=self.eps))

        return log_survival

    def cumulative_hazard(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute mixture cumulative hazard function.

        Args:
            time: Time points [batch_size, num_times]

        Returns:
            Mixture cumulative hazard [batch_size, num_times]
        """
        log_survival = self.log_survival(time)
        cumulative_hazard = -log_survival
        return cumulative_hazard

    def mean(self) -> torch.Tensor:
        """
        Compute mean of the mixture distribution.

        Returns:
            Mean values [batch_size]
        """
        distributions = self.get_component_distributions()

        # Compute weighted sum of component means
        batch_size = self.weights.shape[0]
        mixture_mean = torch.zeros(batch_size, device=self.weights.device)

        for i, dist in enumerate(distributions):
            component_mean = dist.mean()  # [batch_size]
            mixture_mean += self.weights[:, i] * component_mean

        return mixture_mean
