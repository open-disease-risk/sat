"""Loss balancing strategies for multi-objective optimization."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torch.nn as nn
from enum import Enum
from typing import Dict, List, Optional, Union

from sat.utils import logging

logger = logging.get_default_logger()


class BalancingStrategy(Enum):
    """Enumeration of supported loss balancing strategies."""

    FIXED = "fixed"  # Fixed coefficient weighting (standard approach)
    SCALE = "scale"  # Scale normalization
    GRAD = "grad"  # Gradient-based normalization
    UNCERTAINTY = "uncertainty"  # Homoscedastic uncertainty weighting
    ADAPTIVE = "adaptive"  # Adaptive weighting based on loss dynamics


class LossBalancer(nn.Module):
    """Base class for loss balancing strategies."""

    def __init__(self, num_losses: int):
        """
        Initialize the loss balancer.

        Args:
            num_losses: Number of losses to balance
        """
        super().__init__()
        self.num_losses = num_losses

    def forward(
        self, losses: List[torch.Tensor], iteration: Optional[int] = None
    ) -> torch.Tensor:
        """
        Balance multiple loss components.

        Args:
            losses: List of individual loss tensors
            iteration: Current training iteration (for adaptive methods)

        Returns:
            Balanced total loss
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def get_weights(self) -> List[float]:
        """
        Get the current loss weights.

        Returns:
            List of current loss weights
        """
        raise NotImplementedError("Subclasses must implement get_weights method")

    @staticmethod
    def create(
        strategy: Union[str, BalancingStrategy],
        num_losses: int,
        coeffs: Optional[List[float]] = None,
        **kwargs,
    ) -> "LossBalancer":
        """
        Factory method to create a loss balancer based on strategy.

        Args:
            strategy: Balancing strategy name or enum value
            num_losses: Number of losses to balance
            coeffs: Initial coefficients (for fixed weighting)
            **kwargs: Additional strategy-specific parameters

        Returns:
            Instantiated loss balancer
        """
        if isinstance(strategy, str):
            try:
                strategy = BalancingStrategy(strategy)
            except ValueError:
                logger.warning(
                    f"Unknown balancing strategy: {strategy}. Using fixed weighting."
                )
                strategy = BalancingStrategy.FIXED

        if strategy == BalancingStrategy.FIXED:
            if coeffs is None:
                coeffs = [1.0] * num_losses
            return FixedWeightBalancer(coeffs)

        elif strategy == BalancingStrategy.SCALE:
            return ScaleNormalizationBalancer(num_losses, **kwargs)

        elif strategy == BalancingStrategy.GRAD:
            return GradientNormalizationBalancer(num_losses, **kwargs)

        elif strategy == BalancingStrategy.UNCERTAINTY:
            return UncertaintyWeightBalancer(num_losses, **kwargs)

        elif strategy == BalancingStrategy.ADAPTIVE:
            return AdaptiveWeightBalancer(num_losses, **kwargs)

        else:
            logger.warning(
                f"Unknown balancing strategy: {strategy}. Using fixed weighting."
            )
            if coeffs is None:
                coeffs = [1.0] * num_losses
            return FixedWeightBalancer(coeffs)


class FixedWeightBalancer(LossBalancer):
    """Fixed coefficient weighting strategy."""

    def __init__(self, coeffs: List[float]):
        """
        Initialize with fixed weights.

        Args:
            coeffs: Loss coefficients
        """
        super().__init__(len(coeffs))
        self.register_buffer("coeffs", torch.tensor(coeffs, dtype=torch.float32))

    def forward(
        self, losses: List[torch.Tensor], iteration: Optional[int] = None
    ) -> torch.Tensor:
        """Apply fixed weights to losses."""
        assert (
            len(losses) == self.num_losses
        ), f"Expected {self.num_losses} losses, got {len(losses)}"

        total_loss = 0.0
        for i, loss in enumerate(losses):
            total_loss += self.coeffs[i] * loss

        return total_loss

    def get_weights(self) -> List[float]:
        """Return fixed weights."""
        return self.coeffs.cpu().tolist()


class ScaleNormalizationBalancer(LossBalancer):
    """Scale normalization balancing strategy."""

    def __init__(self, num_losses: int, alpha: float = 0.9, eps: float = 1e-8):
        """
        Initialize scale normalization balancer.

        Args:
            num_losses: Number of losses
            alpha: EMA decay factor for loss scale tracking
            eps: Small constant for numerical stability
        """
        super().__init__(num_losses)
        self.alpha = alpha
        self.eps = eps
        self.register_buffer("loss_scales", torch.ones(num_losses, dtype=torch.float32))
        self.register_buffer("coeffs", torch.ones(num_losses, dtype=torch.float32))

    def forward(
        self, losses: List[torch.Tensor], iteration: Optional[int] = None
    ) -> torch.Tensor:
        """Balance losses using scale normalization."""
        assert (
            len(losses) == self.num_losses
        ), f"Expected {self.num_losses} losses, got {len(losses)}"

        # Update loss scales with EMA
        with torch.no_grad():
            for i, loss in enumerate(losses):
                if iteration is None or iteration > 0:  # Skip first iteration
                    curr_scale = loss.detach().abs()
                    # Update with exponential moving average
                    self.loss_scales[i] = (
                        self.alpha * self.loss_scales[i] + (1 - self.alpha) * curr_scale
                    )
                    self.coeffs[i] = 1.0 / (self.loss_scales[i] + self.eps)

        # Apply normalization
        total_loss = 0.0
        for i, loss in enumerate(losses):
            total_loss += self.coeffs[i] * loss

        return total_loss

    def get_weights(self) -> List[float]:
        """Return current weights."""
        return self.coeffs.cpu().tolist()


class GradientNormalizationBalancer(LossBalancer):
    """Gradient-based normalization balancing strategy."""

    def __init__(self, num_losses: int, alpha: float = 0.9, eps: float = 1e-8):
        """
        Initialize gradient normalization balancer.

        Args:
            num_losses: Number of losses
            alpha: EMA decay factor for gradient norm tracking
            eps: Small constant for numerical stability
        """
        super().__init__(num_losses)
        self.alpha = alpha
        self.eps = eps
        self.register_buffer("grad_norms", torch.ones(num_losses, dtype=torch.float32))
        self.register_buffer("coeffs", torch.ones(num_losses, dtype=torch.float32))

    def forward(
        self, losses: List[torch.Tensor], iteration: Optional[int] = None
    ) -> torch.Tensor:
        """Balance losses using gradient normalization."""
        assert (
            len(losses) == self.num_losses
        ), f"Expected {self.num_losses} losses, got {len(losses)}"

        # Compute individual loss gradients
        weighted_losses = []
        for i, loss in enumerate(losses):
            # Allow gradient computation for this loss
            if hasattr(loss, "grad_fn") and loss.grad_fn is not None:
                # We don't actually compute the full gradient here, as that would be inefficient
                # Instead, we use the current loss value as a proxy for gradient magnitude
                # This is a simplification, but works well in practice
                curr_grad_norm = loss.detach().abs()

                # Update gradient norm with EMA
                if iteration is None or iteration > 0:  # Skip first iteration
                    self.grad_norms[i] = (
                        self.alpha * self.grad_norms[i]
                        + (1 - self.alpha) * curr_grad_norm
                    )
                    self.coeffs[i] = 1.0 / (self.grad_norms[i] + self.eps)

                weighted_losses.append(self.coeffs[i] * loss)
            else:
                weighted_losses.append(loss)

        # Sum losses
        total_loss = sum(weighted_losses)
        return total_loss

    def get_weights(self) -> List[float]:
        """Return current weights."""
        return self.coeffs.cpu().tolist()


class UncertaintyWeightBalancer(LossBalancer):
    """Homoscedastic uncertainty weighting balancer."""

    def __init__(self, num_losses: int, init_sigma: float = 1.0):
        """
        Initialize uncertainty weighting balancer.

        Args:
            num_losses: Number of losses
            init_sigma: Initial uncertainty value
        """
        super().__init__(num_losses)
        # We parameterize log(sigma^2) for numerical stability
        log_sigma_sq = torch.ones(num_losses) * torch.log(torch.tensor(init_sigma**2))
        self.log_sigma_sq = nn.Parameter(log_sigma_sq)

    def forward(
        self, losses: List[torch.Tensor], iteration: Optional[int] = None
    ) -> torch.Tensor:
        """Balance losses using learned uncertainty weighting."""
        assert (
            len(losses) == self.num_losses
        ), f"Expected {self.num_losses} losses, got {len(losses)}"

        total_loss = 0.0
        for i, loss in enumerate(losses):
            # Weight = 1/2σ² * L + log(σ)
            # This is derived from maximizing the log likelihood of a Gaussian with uncertainty σ
            precision = torch.exp(-self.log_sigma_sq[i])
            total_loss += precision * loss + 0.5 * self.log_sigma_sq[i]

        return total_loss

    def get_weights(self) -> List[float]:
        """Return current weights (precisions)."""
        with torch.no_grad():
            precisions = torch.exp(-self.log_sigma_sq).cpu().tolist()
        return precisions


class AdaptiveWeightBalancer(LossBalancer):
    """Adaptive weighting based on loss dynamics."""

    def __init__(
        self,
        num_losses: int,
        alpha: float = 0.9,
        eps: float = 1e-8,
        window_size: int = 10,
        adaptation_rate: float = 0.01,
    ):
        """
        Initialize adaptive weighting balancer.

        Args:
            num_losses: Number of losses
            alpha: EMA decay factor for loss tracking
            eps: Small constant for numerical stability
            window_size: Window size for trend detection
            adaptation_rate: Rate at which to adjust weights
        """
        super().__init__(num_losses)
        self.alpha = alpha
        self.eps = eps
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate

        # Initialize tracking buffers
        self.register_buffer("loss_history", torch.zeros(num_losses, window_size))
        self.register_buffer("current_losses", torch.ones(num_losses))
        self.register_buffer("coeffs", torch.ones(num_losses))
        self.register_buffer("iteration_counter", torch.tensor(0))

    def forward(
        self, losses: List[torch.Tensor], iteration: Optional[int] = None
    ) -> torch.Tensor:
        """Balance losses using adaptive weighting."""
        assert (
            len(losses) == self.num_losses
        ), f"Expected {self.num_losses} losses, got {len(losses)}"

        # Update loss tracking
        with torch.no_grad():
            # Update current losses with EMA
            for i, loss in enumerate(losses):
                curr_loss = loss.detach()
                if self.iteration_counter > 0:
                    self.current_losses[i] = (
                        self.alpha * self.current_losses[i]
                        + (1 - self.alpha) * curr_loss
                    )
                else:
                    self.current_losses[i] = curr_loss

            # Update history buffer (circular buffer)
            if iteration is not None:
                idx = iteration % self.window_size
            else:
                idx = int(self.iteration_counter) % self.window_size

            self.loss_history[:, idx] = self.current_losses

            # Compute loss trends if we have enough history
            if self.iteration_counter >= self.window_size:
                # Compute average rate of change over window
                start_losses = self.loss_history[:, (idx + 1) % self.window_size]
                end_losses = self.current_losses

                # Compute relative improvement for each loss
                improvements = (start_losses - end_losses) / (start_losses + self.eps)

                # Adjust weights - give more weight to losses that are improving less
                # This encourages balanced optimization across all objectives
                min_improvement = torch.min(improvements)
                improvement_gaps = improvements - min_improvement

                # Reduce weight for losses that are improving faster
                weight_adjustments = -self.adaptation_rate * improvement_gaps
                self.coeffs = torch.clamp(
                    self.coeffs + weight_adjustments, min=0.1, max=10.0
                )

                # Normalize weights to sum to num_losses
                self.coeffs = self.coeffs * (
                    self.num_losses / (self.coeffs.sum() + self.eps)
                )

            # Increment counter
            self.iteration_counter += 1

        # Apply weights to losses
        total_loss = 0.0
        for i, loss in enumerate(losses):
            total_loss += self.coeffs[i] * loss

        return total_loss

    def get_weights(self) -> List[float]:
        """Return current weights."""
        return self.coeffs.cpu().tolist()
