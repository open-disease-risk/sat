"""Loss balancing strategies for multi-objective optimization."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torch.nn as nn
from enum import Enum
from typing import List, Optional, Union

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
    """
    Fixed coefficient weighting strategy with extra safeguards.

    This implementation ensures no in-place operations that could interfere with the
    computation graph, particularly on MPS devices.
    """

    def __init__(self, coeffs: List[float]):
        """
        Initialize with fixed weights.

        Args:
            coeffs: Loss coefficients
        """
        super().__init__(len(coeffs))
        # Store coefficient values as a Python list to avoid computational graph issues
        self._coeff_values = [float(c) for c in coeffs]
        # Register buffer for state saving/loading
        self.register_buffer("coeffs", torch.tensor(coeffs, dtype=torch.float32))

    def forward(
        self, losses: List[torch.Tensor], iteration: Optional[int] = None
    ) -> torch.Tensor:
        """Apply fixed weights to losses safely without in-place operations."""
        assert (
            len(losses) == self.num_losses
        ), f"Expected {self.num_losses} losses, got {len(losses)}"

        # Collect weighted losses
        weighted_losses = []
        for i, loss in enumerate(losses):
            # Use the Python float value to avoid computation graph issues
            weighted_losses.append(self._coeff_values[i] * loss)

        # Stack and sum for a clean non-in-place operation
        if weighted_losses:
            # Handle different tensor shapes by using sum instead of stack
            result = weighted_losses[0]
            for wl in weighted_losses[1:]:
                result = result + wl
            return result
        else:
            # Return empty loss (should never happen in practice)
            return torch.tensor(0.0, device=losses[0].device if losses else None)

    def get_weights(self) -> List[float]:
        """Return fixed weights."""
        return (
            self._coeff_values.copy()
        )  # Return a copy to avoid potential modifications


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
    """Gradient-based normalization balancing strategy with Apple Silicon compatibility."""

    def __init__(self, num_losses: int, alpha: float = 0.9, eps: float = 1e-8):
        """
        Initialize gradient normalization balancer with MPS-friendly operations.

        Args:
            num_losses: Number of losses
            alpha: EMA decay factor for gradient norm tracking
            eps: Small constant for numerical stability
        """
        super().__init__(num_losses)
        self.alpha = alpha
        self.eps = eps

        # Store values as Python lists to avoid computational graph issues
        self._grad_norms = [1.0] * num_losses
        self._coeff_values = [1.0] * num_losses

        # Register buffers for state dict saving/loading
        self.register_buffer("grad_norms", torch.ones(num_losses, dtype=torch.float32))
        self.register_buffer("coeffs", torch.ones(num_losses, dtype=torch.float32))
        self.register_buffer("iter_counter", torch.tensor(0))

    def forward(
        self, losses: List[torch.Tensor], iteration: Optional[int] = None
    ) -> torch.Tensor:
        """Balance losses using gradient normalization with clean operations."""
        assert (
            len(losses) == self.num_losses
        ), f"Expected {self.num_losses} losses, got {len(losses)}"

        # Update coefficients without affecting the computation graph
        with torch.no_grad():
            self.iter_counter += 1

            # Update gradient norms and coefficients
            for i, loss in enumerate(losses):
                if hasattr(loss, "grad_fn") and loss.grad_fn is not None:
                    # Use detached loss value as proxy for gradient magnitude
                    curr_grad_norm = float(loss.detach().abs().item())

                    # Update with EMA (using Python floats)
                    if self.iter_counter > 1:  # Skip first iteration
                        self._grad_norms[i] = (
                            self.alpha * self._grad_norms[i]
                            + (1.0 - self.alpha) * curr_grad_norm
                        )
                        self._coeff_values[i] = 1.0 / (self._grad_norms[i] + self.eps)

                        # Update buffers for visualization and state saving
                        self.grad_norms[i] = self._grad_norms[i]
                        self.coeffs[i] = self._coeff_values[i]

        # Apply weights to losses with clean operations
        weighted_losses = []
        for i, loss in enumerate(losses):
            # Use Python float value to avoid computation graph issues
            weighted_losses.append(self._coeff_values[i] * loss)

        # Combine losses safely
        if weighted_losses:
            result = weighted_losses[0]
            for wl in weighted_losses[1:]:
                result = result + wl  # Clean non-inplace addition
            return result
        else:
            # Fallback case
            device = losses[0].device if losses else None
            return torch.tensor(0.0, device=device, requires_grad=True)

    def get_weights(self) -> List[float]:
        """Return current weights."""
        return self._coeff_values.copy()  # Return a copy to avoid modification


class SafeUncertaintyFunction(torch.autograd.Function):
    """
    Custom autograd function for uncertainty-based loss balancing.

    This ensures proper gradient flow by explicitly defining forward and backward passes.
    """

    @staticmethod
    def forward(ctx, loss, log_var):
        """Forward pass: weight loss by precision (1/variance) and add log variance term."""
        # Save values for backward pass
        ctx.save_for_backward(loss, log_var)

        # Compute precision (detached to avoid circular gradient paths)
        precision = torch.exp(-log_var)

        # Return weighted loss + log_var regularization term
        return precision * loss + 0.5 * log_var

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: compute gradients for loss and log variance."""
        loss, log_var = ctx.saved_tensors

        # Precision (1/variance)
        precision = torch.exp(-log_var)

        # Gradient for loss: precision * grad_output
        grad_loss = precision * grad_output

        # Gradient for log variance:
        # d(precision * loss + 0.5 * log_var)/d(log_var) =
        # -precision * loss * grad_output + 0.5 * grad_output
        grad_log_var = (-precision * loss + 0.5) * grad_output

        return grad_loss, grad_log_var


class UncertaintyWeightBalancer(LossBalancer):
    """
    Homoscedastic uncertainty weighting balancer that uses SafeUncertaintyFunction
    to learn optimal weights through gradient descent.

    This implementation is compatible with Apple Silicon (MPS) by using
    a custom autograd function that explicitly defines forward and backward
    passes with proper gradient flow.
    """

    # Class variable to track tensorboard writer instances
    _tb_writers = {}  # Maps logging_dir to SummaryWriter instances
    _global_step = 0  # Global step counter for logging
    _log_dir = None  # Optional explicit log directory

    @classmethod
    def get_tb_writer(cls, logging_dir=None):
        """Get or create a TensorBoard writer for the given logging directory."""
        # Use the provided logging_dir, or the class-level _log_dir if set
        if logging_dir is None:
            # If class has a configured log_dir, use it
            if cls._log_dir is not None:
                logging_dir = cls._log_dir
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Using configured log_dir: {logging_dir}")
            else:
                # Look in common places
                import os

                potential_dirs = ["./runs", "./outputs", "./logs", "./tensorboard"]

                for d in potential_dirs:
                    if os.path.exists(d):
                        # Find most recent subdirectory by modification time
                        subdirs = [
                            os.path.join(d, sd)
                            for sd in os.listdir(d)
                            if os.path.isdir(os.path.join(d, sd))
                        ]
                        if subdirs:
                            most_recent = max(subdirs, key=os.path.getmtime)
                            logging_dir = most_recent
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(
                                    f"Found existing log directory: {logging_dir}"
                                )
                            break

                if logging_dir is None:
                    # No existing TB dirs found, create one
                    fallback_dir = "./tb_logs/uncertainty_weights"
                    os.makedirs(fallback_dir, exist_ok=True)
                    logging_dir = fallback_dir
                    logger.info(f"Created fallback logging directory: {logging_dir}")

        # Create writer if needed
        if logging_dir not in cls._tb_writers:
            try:
                from torch.utils.tensorboard import SummaryWriter

                cls._tb_writers[logging_dir] = SummaryWriter(log_dir=logging_dir)
                logger.info(f"Created TensorBoard writer at {logging_dir}")
            except Exception as e:
                logger.warning(f"Failed to create TensorBoard writer: {e}")
                return None

        return cls._tb_writers[logging_dir]

    def __init__(
        self,
        num_losses: int,
        init_sigma: float = 1.0,
        log_interval: int = 10,
        log_dir: str = None,
    ):
        """
        Initialize uncertainty weighting balancer.

        Args:
            num_losses: Number of losses
            init_sigma: Initial uncertainty value (higher means lower initial weight)
            log_interval: How often to log weights to tensorboard (in steps)
            log_dir: Explicit directory to use for tensorboard logging
        """
        super().__init__(num_losses)

        # Store the explicit logging directory if provided
        if log_dir is not None:
            self.__class__._log_dir = log_dir
            logger.info(f"UncertaintyWeightBalancer will log to: {log_dir}")

            # Create the directory if it doesn't exist
            import os

            os.makedirs(log_dir, exist_ok=True)

        # Initialize log variance parameters
        # Using Parameter ensures they get updated through gradient descent
        init_log_var = torch.ones(num_losses) * (
            2.0 * torch.log(torch.tensor(init_sigma))
        )
        self.log_var = nn.Parameter(init_log_var, requires_grad=True)

        # For state tracking and visualization
        self.register_buffer("iteration", torch.tensor(0))
        self.log_interval = log_interval

        # Initial precision = 1/variance = 1/sigma^2 = exp(-log_var)
        init_precision = torch.exp(-init_log_var).tolist()

        # Log to console
        logger.info(f"Using SafeUncertaintyFunction Balancer with {num_losses} losses")
        logger.info(f"Initial weights: {init_precision}")

        # Direct TensorBoard logging
        self._log_to_tensorboard(step=0, force=True)

    def forward(
        self, losses: List[torch.Tensor], iteration: Optional[int] = None
    ) -> torch.Tensor:
        """
        Balance losses using uncertainty weighting with SafeUncertaintyFunction.

        Args:
            losses: List of loss tensors to balance
            iteration: Optional iteration number for tracking

        Returns:
            Balanced total loss with uncertainty regularization terms
        """
        assert (
            len(losses) == self.num_losses
        ), f"Expected {self.num_losses} losses, got {len(losses)}"

        # Processing each loss with the uncertainty function
        total_loss = None
        device = losses[0].device

        # Move the log variances to the same device as the losses
        log_var_device = self.log_var.to(device)

        for i, loss in enumerate(losses):
            # Get individual log variance for this loss
            log_var_i = log_var_device[i].reshape(1)

            # Apply uncertainty weighting using our custom autograd function
            weighted_loss = SafeUncertaintyFunction.apply(loss, log_var_i)

            # Accumulate the losses
            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss

        # Increment iteration and log to tensorboard
        with torch.no_grad():
            # Use passed iteration if provided, otherwise use internal counter
            if iteration is not None:
                step = iteration
            else:
                step = self.__class__._global_step
                self.__class__._global_step += 1

            self.iteration += 1

            # Periodically log to TensorBoard
            if step % self.log_interval == 0:
                self._log_to_tensorboard(step)

        return total_loss

    def _log_to_tensorboard(self, step, force=False):
        """Log current weights to TensorBoard."""
        if force or step % self.log_interval == 0:
            # Get weights
            weights = self.get_weights()

            # Try to get TensorBoard writer
            writer = self.get_tb_writer()
            if writer is not None:
                # Log weights
                for i, weight in enumerate(weights):
                    writer.add_scalar(f"uncertainty_weights/weight_{i}", weight, step)

                # If we have multiple weights, log their ratios
                if len(weights) > 1:
                    for i in range(len(weights)):
                        for j in range(i + 1, len(weights)):
                            ratio = weights[i] / (weights[j] + 1e-8)
                            writer.add_scalar(
                                f"uncertainty_weights/ratio_{i}_{j}", ratio, step
                            )

                # Also log raw log_var values
                log_vars = self.log_var.detach().cpu().tolist()
                for i, log_var in enumerate(log_vars):
                    writer.add_scalar(f"uncertainty_weights/log_var_{i}", log_var, step)

                # Flush to ensure all events are written
                writer.flush()

                # Console logging
                if step % (self.log_interval * 5) == 0:
                    logger.info(
                        f"Step {step}: weights = {weights}, log_vars = {log_vars}"
                    )

    def get_weights(self) -> List[float]:
        """
        Get current precision weights (1/variance) for logging.

        The weight is e^(-log_var), which corresponds to precision (1/variance).
        Higher precision means higher weight in the loss function.
        """
        with torch.no_grad():
            # Convert log variances to precisions: precision = 1/variance = e^(-log_var)
            precisions = torch.exp(-self.log_var).detach().cpu().tolist()
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

        # Calculate weights without modifying the computation graph
        weights = self._calculate_weights(losses, iteration)

        # Apply weights to losses using a non-inplace accumulation approach
        weighted_losses = []
        for i, loss in enumerate(losses):
            # Use detached weight to avoid connecting to computation graph
            weighted_losses.append(weights[i] * loss)

        # Sum losses using a clean operation
        if weighted_losses:
            # Use torch.stack for a clean non-inplace operation
            total_loss = torch.stack(weighted_losses).sum(dim=0)
        else:
            # Fallback for empty list
            device = (
                next(iter(self.parameters())).device
                if self.parameters()
                else torch.device("cpu")
            )
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss

    def _calculate_weights(
        self, losses: List[torch.Tensor], iteration: Optional[int] = None
    ) -> List[float]:
        """
        Calculate weights based on loss history without modifying the computation graph.
        Returns Python list of weights to avoid gradient issues.
        """
        # Create a detached copy of the coeffs for returning
        current_coeffs = self.coeffs.detach().clone().cpu().tolist()

        # Only update tracking during evaluation (not during backward pass)
        with torch.no_grad():
            # Create new tensors for updates to avoid in-place operations
            new_current_losses = self.current_losses.clone()
            new_loss_history = self.loss_history.clone()
            new_coeffs = self.coeffs.clone()
            new_counter = self.iteration_counter.clone()

            # Update current losses with EMA
            for i, loss in enumerate(losses):
                curr_loss = loss.detach()
                if self.iteration_counter > 0:
                    new_current_losses[i] = (
                        self.alpha * self.current_losses[i]
                        + (1 - self.alpha) * curr_loss
                    )
                else:
                    new_current_losses[i] = curr_loss

            # Update history buffer (circular buffer)
            if iteration is not None:
                idx = iteration % self.window_size
            else:
                idx = int(self.iteration_counter) % self.window_size

            new_loss_history[:, idx] = new_current_losses

            # Compute loss trends if we have enough history
            if self.iteration_counter >= self.window_size:
                # Compute average rate of change over window
                start_losses = new_loss_history[:, (idx + 1) % self.window_size]
                end_losses = new_current_losses

                # Compute relative improvement for each loss
                improvements = (start_losses - end_losses) / (start_losses + self.eps)

                # Adjust weights - give more weight to losses that are improving less
                # This encourages balanced optimization across all objectives
                min_improvement = torch.min(improvements)
                improvement_gaps = improvements - min_improvement

                # Reduce weight for losses that are improving faster
                weight_adjustments = -self.adaptation_rate * improvement_gaps
                new_coeffs = torch.clamp(
                    new_coeffs + weight_adjustments, min=0.1, max=10.0
                )

                # Normalize weights to sum to num_losses
                total_weight = new_coeffs.sum() + self.eps
                new_coeffs = new_coeffs * (self.num_losses / total_weight)

            # Increment counter
            new_counter += 1

            # Update buffers with new values
            self.current_losses.copy_(new_current_losses)
            self.loss_history.copy_(new_loss_history)
            self.coeffs.copy_(new_coeffs)
            self.iteration_counter.copy_(new_counter)

        # Return weights as Python list (completely detached from computation graph)
        return current_coeffs

    def get_weights(self) -> List[float]:
        """Return current weights."""
        return self.coeffs.cpu().tolist()
