"""Base classes for parametric survival distributions."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torch.nn as nn
import numpy as np


class SurvivalDistribution:
    """Base class for parametric survival distributions.

    This serves as an abstract base class that defines the interface
    for all parametric survival distributions.
    """

    def __init__(self, params):
        """Initialize with distribution parameters.

        Args:
            params: Tensor of parameters specific to each distribution type
        """
        self.params = params

    def hazard(self, time_points):
        """Return hazard at specified time points.

        Args:
            time_points: Tensor of time points to evaluate

        Returns:
            hazard: Hazard function values at specified time points
        """
        raise NotImplementedError

    def survival(self, time_points):
        """Return survival probability at specified time points.

        Args:
            time_points: Tensor of time points to evaluate

        Returns:
            survival: Survival function values at specified time points
        """
        raise NotImplementedError

    def pdf(self, time_points):
        """Return probability density at specified time points.

        Args:
            time_points: Tensor of time points to evaluate

        Returns:
            pdf: Probability density values at specified time points
        """
        raise NotImplementedError

    def mean(self):
        """Return mean survival time.

        Returns:
            mean: Expected survival time
        """
        raise NotImplementedError

    def quantile(self, q):
        """Return q-th quantile of survival time.

        Args:
            q: Quantile level(s) to evaluate (between 0 and 1)

        Returns:
            quantile: q-th quantile of the survival distribution
        """
        raise NotImplementedError


class WeibullDistribution(SurvivalDistribution):
    """Weibull distribution for survival analysis.

    Parameterized by shape (k) and scale (lambda) parameters.
    Survival function: S(t) = exp(-(t/lambda)^k)
    Hazard function: h(t) = (k/lambda) * (t/lambda)^(k-1)

    This is one of the most commonly used parametric models in survival analysis
    due to its flexibility in representing different hazard shapes.
    """

    def __init__(self, params):
        """Initialize Weibull distribution with shape and scale parameters.

        Args:
            params: Tensor of shape [batch_size, 2] or [batch_size, num_events, 2]
                   where params[..., 0] is the shape (k) and params[..., 1] is the scale (lambda)
        """
        super().__init__(params)

        # Extract shape and scale parameters
        if params.dim() == 2:
            self.shape = params[:, 0].clamp(min=1e-5)  # k > 0
            self.scale = params[:, 1].clamp(min=1e-5)  # lambda > 0
        else:  # params.dim() == 3
            self.shape = params[:, :, 0].clamp(min=1e-5)  # k > 0
            self.scale = params[:, :, 1].clamp(min=1e-5)  # lambda > 0

    def hazard(self, time_points):
        """Compute Weibull hazard function h(t) = (k/lambda) * (t/lambda)^(k-1).

        Args:
            time_points: Time points to evaluate hazard function

        Returns:
            hazard: Hazard function values at specified time points
        """
        if time_points.dim() == 1:
            time_points = time_points.unsqueeze(0).repeat(self.shape.size(0), 1)

        # Ensure time_points are positive
        time_points = time_points.clamp(min=1e-10)

        if self.shape.dim() == 1:
            # Single event case
            shape = self.shape.unsqueeze(1)  # [batch_size, 1]
            scale = self.scale.unsqueeze(1)  # [batch_size, 1]

            # Compute hazard
            return (shape / scale) * (time_points / scale) ** (shape - 1)
        else:
            # Multiple events case
            shape = self.shape.unsqueeze(2)  # [batch_size, num_events, 1]
            scale = self.scale.unsqueeze(2)  # [batch_size, num_events, 1]

            # Expand time_points for broadcasting
            time_points = time_points.unsqueeze(1).repeat(1, shape.size(1), 1)

            # Compute hazard
            return (shape / scale) * (time_points / scale) ** (shape - 1)

    def survival(self, time_points):
        """Compute Weibull survival function S(t) = exp(-(t/lambda)^k).

        Args:
            time_points: Time points to evaluate survival function

        Returns:
            survival: Survival function values at specified time points
        """
        if time_points.dim() == 1:
            time_points = time_points.unsqueeze(0).repeat(self.shape.size(0), 1)

        # Ensure time_points are positive
        time_points = time_points.clamp(min=1e-10)

        if self.shape.dim() == 1:
            # Single event case
            shape = self.shape.unsqueeze(1)  # [batch_size, 1]
            scale = self.scale.unsqueeze(1)  # [batch_size, 1]

            # Compute survival
            return torch.exp(-((time_points / scale) ** shape))
        else:
            # Multiple events case
            shape = self.shape.unsqueeze(2)  # [batch_size, num_events, 1]
            scale = self.scale.unsqueeze(2)  # [batch_size, num_events, 1]

            # Expand time_points for broadcasting
            time_points = time_points.unsqueeze(1).repeat(1, shape.size(1), 1)

            # Compute survival
            return torch.exp(-((time_points / scale) ** shape))

    def pdf(self, time_points):
        """Compute Weibull probability density function.

        PDF: f(t) = (k/lambda) * (t/lambda)^(k-1) * exp(-(t/lambda)^k)

        Args:
            time_points: Time points to evaluate probability density

        Returns:
            pdf: Probability density function values
        """
        if time_points.dim() == 1:
            time_points = time_points.unsqueeze(0).repeat(self.shape.size(0), 1)

        # Ensure time_points are positive
        time_points = time_points.clamp(min=1e-10)

        # Get hazard and survival
        hazard_values = self.hazard(time_points)
        survival_values = self.survival(time_points)

        # PDF = hazard * survival
        return hazard_values * survival_values

    def mean(self):
        """Compute mean survival time E[T] = lambda * Gamma(1 + 1/k).

        Returns:
            mean: Mean survival time
        """
        if self.shape.dim() == 1:
            # Gamma function approximation for (1 + 1/k)
            gamma_term = torch.exp(torch.lgamma(1 + 1 / self.shape))
            return self.scale * gamma_term
        else:
            # Multiple events case
            gamma_term = torch.exp(torch.lgamma(1 + 1 / self.shape))
            return self.scale * gamma_term

    def quantile(self, q):
        """Compute q-th quantile of survival time.

        F^(-1)(q) = lambda * (-ln(1-q))^(1/k)

        Args:
            q: Quantile level(s) between 0 and 1

        Returns:
            quantile: q-th quantile of the survival distribution
        """
        if not (0 <= q <= 1):
            raise ValueError("Quantile q must be between 0 and 1")

        if self.shape.dim() == 1:
            return self.scale * (-torch.log(1 - q)) ** (1 / self.shape)
        else:
            # Multiple events case
            return self.scale * (-torch.log(1 - q)) ** (1 / self.shape)


class LogNormalDistribution(SurvivalDistribution):
    """Log-Normal distribution for survival analysis.

    Parameterized by location (mu) and scale (sigma) parameters.
    Survival function: S(t) = 1 - Phi((ln(t) - mu) / sigma)
    where Phi is the standard normal CDF.

    The Log-Normal distribution is useful when the logarithm of survival times
    follows a normal distribution, common in many biological processes.
    """

    def __init__(self, params):
        """Initialize Log-Normal distribution with location and scale parameters.

        Args:
            params: Tensor of shape [batch_size, 2] or [batch_size, num_events, 2]
                   where params[..., 0] is the location (mu) and params[..., 1] is the scale (sigma)
        """
        super().__init__(params)

        # Extract location and scale parameters
        if params.dim() == 2:
            self.loc = params[:, 0]  # mu
            self.scale = params[:, 1].clamp(min=1e-5)  # sigma > 0
        else:  # params.dim() == 3
            self.loc = params[:, :, 0]  # mu
            self.scale = params[:, :, 1].clamp(min=1e-5)  # sigma > 0

    def hazard(self, time_points):
        """Compute Log-Normal hazard function h(t) = f(t) / S(t).

        Args:
            time_points: Time points to evaluate hazard function

        Returns:
            hazard: Hazard function values at specified time points
        """
        if time_points.dim() == 1:
            time_points = time_points.unsqueeze(0).repeat(self.loc.size(0), 1)

        # Ensure time_points are positive
        time_points = time_points.clamp(min=1e-10)

        # Compute PDF and survival, then divide to get hazard
        pdf_values = self.pdf(time_points)
        surv_values = self.survival(time_points)

        # Avoid division by zero
        return pdf_values / (surv_values + 1e-10)

    def pdf(self, time_points):
        """Compute Log-Normal probability density function.

        PDF: f(t) = (1 / (t * sigma * sqrt(2pi))) * exp(-0.5 * ((ln(t) - mu) / sigma)^2)

        Args:
            time_points: Time points to evaluate probability density

        Returns:
            pdf: Probability density function values
        """
        if time_points.dim() == 1:
            time_points = time_points.unsqueeze(0).repeat(self.loc.size(0), 1)

        # Ensure time_points are positive
        time_points = time_points.clamp(min=1e-10)

        if self.loc.dim() == 1:
            # Single event case
            loc = self.loc.unsqueeze(1)  # [batch_size, 1]
            scale = self.scale.unsqueeze(1)  # [batch_size, 1]

            # Compute standardized variable
            z = (torch.log(time_points) - loc) / scale

            # Compute PDF
            return torch.exp(-0.5 * z**2) / (time_points * scale * np.sqrt(2 * np.pi))
        else:
            # Multiple events case
            loc = self.loc.unsqueeze(2)  # [batch_size, num_events, 1]
            scale = self.scale.unsqueeze(2)  # [batch_size, num_events, 1]

            # Expand time_points for broadcasting
            time_points = time_points.unsqueeze(1).repeat(1, loc.size(1), 1)

            # Compute standardized variable
            z = (torch.log(time_points) - loc) / scale

            # Compute PDF
            return torch.exp(-0.5 * z**2) / (time_points * scale * np.sqrt(2 * np.pi))

    def survival(self, time_points):
        """Compute Log-Normal survival function S(t) = 1 - Phi((ln(t) - mu) / sigma).

        Args:
            time_points: Time points to evaluate survival function

        Returns:
            survival: Survival function values at specified time points
        """
        if time_points.dim() == 1:
            time_points = time_points.unsqueeze(0).repeat(self.loc.size(0), 1)

        # Ensure time_points are positive
        time_points = time_points.clamp(min=1e-10)

        if self.loc.dim() == 1:
            # Single event case
            loc = self.loc.unsqueeze(1)  # [batch_size, 1]
            scale = self.scale.unsqueeze(1)  # [batch_size, 1]

            # Compute standardized variable
            z = (torch.log(time_points) - loc) / scale

            # Compute survival using complement of normal CDF
            return 0.5 * torch.erfc(z / np.sqrt(2))
        else:
            # Multiple events case
            loc = self.loc.unsqueeze(2)  # [batch_size, num_events, 1]
            scale = self.scale.unsqueeze(2)  # [batch_size, num_events, 1]

            # Expand time_points for broadcasting
            time_points = time_points.unsqueeze(1).repeat(1, loc.size(1), 1)

            # Compute standardized variable
            z = (torch.log(time_points) - loc) / scale

            # Compute survival using complement of normal CDF
            return 0.5 * torch.erfc(z / np.sqrt(2))

    def mean(self):
        """Compute mean survival time E[T] = exp(mu + sigma^2/2).

        Returns:
            mean: Mean survival time
        """
        if self.loc.dim() == 1:
            return torch.exp(self.loc + self.scale**2 / 2)
        else:
            return torch.exp(self.loc + self.scale**2 / 2)

    def quantile(self, q):
        """Compute q-th quantile of survival time.

        F^(-1)(q) = exp(mu + sigma * Phi^(-1)(q))
        where Phi^(-1) is the inverse of the standard normal CDF.

        Args:
            q: Quantile level(s) between 0 and 1

        Returns:
            quantile: q-th quantile of the survival distribution
        """
        if not (0 <= q <= 1):
            raise ValueError("Quantile q must be between 0 and 1")

        # Compute inverse of standard normal CDF
        if self.loc.dim() == 1:
            return torch.exp(
                self.loc + self.scale * torch.erfinv(2 * q - 1) * np.sqrt(2)
            )
        else:
            return torch.exp(
                self.loc + self.scale * torch.erfinv(2 * q - 1) * np.sqrt(2)
            )
