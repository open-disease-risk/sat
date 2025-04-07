"""Momentum Buffer for survival analysis loss functions"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import math
from collections import deque
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers.utils import ModelOutput

from sat.utils import logging

from .base import Loss

logger = logging.get_default_logger()


class MomentumBuffer:
    """
    Momentum buffer that maintains a queue of embeddings and their associated metadata
    for use in survival analysis with momentum contrast.

    This buffer:
    1. Stores embeddings from past batches along with their event/duration info
    2. Maintains a FIFO queue of features and metadata
    3. Provides combined embeddings (current + buffered) for loss computation
    """

    def __init__(
        self,
        embedding_dim: int,
        buffer_size: int = 2048,
        num_events: int = 1,
        dynamic_growth: bool = True,
        initial_size: Optional[int] = None,
        growth_factor: float = 1.5,
        growth_steps: int = 5,
        track_variance: bool = True,
        variance_window: int = 10,
        adaptive_buffer_adjustment: bool = True,
    ):
        """
        Initialize a momentum buffer for survival analysis.

        Args:
            embedding_dim: Dimension of embeddings to store
            buffer_size: Maximum number of embeddings to maintain in buffer
            num_events: Number of events in the survival analysis task
            dynamic_growth: Whether to grow buffer size during training
            initial_size: Initial buffer size if using dynamic growth
            growth_factor: Factor by which to grow buffer in each step
            growth_steps: Number of growth steps to reach full buffer size
            track_variance: Whether to track loss variance to detect instability
            variance_window: Window size for variance calculation
            adaptive_buffer_adjustment: Whether to adapt buffer size based on variance
        """
        self.embedding_dim = embedding_dim
        self.max_buffer_size = buffer_size
        self.num_events = num_events

        # Dynamic buffer growth settings
        self.dynamic_growth = dynamic_growth
        self.initial_size = initial_size or max(128, buffer_size // 8)
        self.growth_factor = growth_factor
        self.growth_steps = growth_steps
        self.current_step = 0

        # Variance tracking for stability detection
        self.track_variance = track_variance
        self.variance_window = variance_window
        self.adaptive_buffer_adjustment = adaptive_buffer_adjustment
        self.loss_history = deque(maxlen=variance_window * 2)
        self.loss_variance_history = []
        self.buffer_adjustment_history = []

        # Actual buffer size will start small and grow
        self.current_buffer_size = self.initial_size if dynamic_growth else buffer_size

        # Create the buffer queues for embeddings and metadata
        # We use deque for efficient FIFO operations
        self.reset_buffer()

        logger.info(
            f"MomentumBuffer initialized with embedding_dim={embedding_dim}, "
            f"max_buffer_size={buffer_size}"
        )

        if dynamic_growth:
            logger.info(
                f"Using dynamic buffer growth: initial={self.initial_size}, "
                f"growth_factor={growth_factor}, steps={growth_steps}"
            )

        if track_variance:
            logger.info(
                f"Tracking loss variance with window size {variance_window}, "
                f"adaptive adjustment: {adaptive_buffer_adjustment}"
            )

    def reset_buffer(self):
        """Reset the buffer queues."""
        self.embedding_queue = deque(maxlen=self.max_buffer_size)
        self.reference_queue = deque(maxlen=self.max_buffer_size)

        # Track buffer stats
        self.buffer_size_history = []
        self.uncensored_events_count = []

    def track_loss(self, loss_value: float):
        """
        Track loss values to calculate variance for stability detection.

        Args:
            loss_value: Current batch loss value

        Returns:
            Optional variance if we have enough history
        """
        if not self.track_variance:
            return None

        # Add loss to history
        self.loss_history.append(loss_value)

        # We need at least a window size of history to calculate variance
        if len(self.loss_history) < self.variance_window:
            return None

        # Calculate variance over the window
        recent_losses = list(self.loss_history)[-self.variance_window :]
        variance = torch.tensor(recent_losses).var().item()
        self.loss_variance_history.append(variance)

        # Check if we need to adjust buffer size based on variance
        if self.adaptive_buffer_adjustment and len(self.loss_variance_history) >= 2:
            self._adjust_buffer_based_on_variance(variance)

        return variance

    def _adjust_buffer_based_on_variance(self, current_variance: float):
        """
        Dynamically adjust buffer size based on loss variance.

        Args:
            current_variance: Current loss variance
        """
        # Only adjust if we have enough history (at least 2 variance calculations)
        if len(self.loss_variance_history) < 2:
            return

        # Get previous variance
        prev_variance = self.loss_variance_history[-2]

        # Calculate relative change
        if prev_variance > 0:
            rel_change = (current_variance - prev_variance) / prev_variance
        else:
            rel_change = 1.0 if current_variance > 0 else 0.0

        # High increase in variance indicates instability
        variance_threshold = 0.5  # 50% increase in variance

        if rel_change > variance_threshold:
            # Significant increase in variance - need larger buffer
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Detected loss instability (variance +{rel_change:.1%}), increasing buffer size"
                )

            old_size = self.current_buffer_size
            # Increase buffer by 50% up to max size
            new_size = min(int(self.current_buffer_size * 1.5), self.max_buffer_size)

            if new_size > old_size:
                self.current_buffer_size = new_size
                self._resize_queues(new_size)

                # Record adjustment
                self.buffer_adjustment_history.append(
                    (
                        len(self.loss_variance_history),
                        old_size,
                        new_size,
                        "increase",
                        current_variance,
                    )
                )

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Increased buffer size due to instability: {old_size} → {new_size}"
                    )

        elif len(self.loss_variance_history) >= 5 and rel_change < -variance_threshold:
            # Significant decrease in variance - could reduce buffer to save memory
            # But only if we have a stable trend (5+ measurements)

            # Don't shrink below initial size or half of current
            min_allowed = max(self.initial_size, self.current_buffer_size // 2)

            # Only shrink if we're significantly above minimum size
            if self.current_buffer_size > min_allowed * 1.5:
                old_size = self.current_buffer_size
                # Decrease buffer by 25%
                new_size = max(int(self.current_buffer_size * 0.75), min_allowed)

                self.current_buffer_size = new_size
                self._resize_queues(new_size)

                # Record adjustment
                self.buffer_adjustment_history.append(
                    (
                        len(self.loss_variance_history),
                        old_size,
                        new_size,
                        "decrease",
                        current_variance,
                    )
                )

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Decreased buffer size due to stability: {old_size} → {new_size}"
                    )

    def update_buffer_size(self, iteration: int):
        """
        Update buffer size based on training progress.

        Args:
            iteration: Current training iteration
        """
        if not self.dynamic_growth:
            return

        # Only update on specific iterations
        growth_interval = max(5000, iteration // self.growth_steps)
        if iteration % growth_interval != 0:
            return

        # Don't exceed max size or update beyond growth steps
        if (
            self.current_buffer_size >= self.max_buffer_size
            or self.current_step >= self.growth_steps
        ):
            return

        # Compute new buffer size with growth factor
        old_size = self.current_buffer_size
        self.current_step += 1

        if self.current_step >= self.growth_steps:
            self.current_buffer_size = self.max_buffer_size
        else:
            # Exponential growth
            self.current_buffer_size = min(
                self.max_buffer_size,
                int(self.initial_size * (self.growth_factor**self.current_step)),
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Increasing buffer size: {old_size} → {self.current_buffer_size} "
                f"(step {self.current_step}/{self.growth_steps})"
            )

        # Update queue maxlen - need to recreate queues with new size
        self._resize_queues(self.current_buffer_size)

        # Store history for analysis
        self.buffer_size_history.append((iteration, self.current_buffer_size))

    def _resize_queues(self, new_size: int):
        """
        Resize all buffer queues to the new size.

        Args:
            new_size: New maximum queue size
        """
        # Create new queues with new size
        new_embedding_queue = deque(maxlen=new_size)
        new_reference_queue = deque(maxlen=new_size)

        # Copy over existing items
        new_embedding_queue.extend(list(self.embedding_queue)[-new_size:])
        new_reference_queue.extend(list(self.reference_queue)[-new_size:])

        # Replace old queues
        self.embedding_queue = new_embedding_queue
        self.reference_queue = new_reference_queue

    def update(
        self,
        outputs: Union[ModelOutput, Dict[str, torch.Tensor]],
        references: torch.Tensor,
    ):
        """
        Update buffer with a new batch of model outputs and references.

        Args:
            outputs: Model outputs containing logits, hazard, etc.
            references: Reference tensor with event and duration information
        """
        # For logits, we store them directly
        if isinstance(outputs, Dict) and "logits" in outputs:
            embeddings = outputs["logits"].detach().cpu()
        elif hasattr(outputs, "logits") and outputs.logits is not None:
            embeddings = outputs.logits.detach().cpu()
        else:
            logger.warning("No logits found in outputs, using hazard or other features")
            # Fallback to whatever is available
            if isinstance(outputs, Dict) and "hazard" in outputs:
                embeddings = outputs["hazard"].detach().cpu()
            elif hasattr(outputs, "hazard") and outputs.hazard is not None:
                embeddings = outputs.hazard.detach().cpu()
            else:
                logger.error("No suitable embeddings found in outputs")
                return

        # Store references
        references_cpu = references.detach().cpu()

        batch_size = embeddings.shape[0]

        # Count uncensored events before adding to buffer
        try:
            events = references[:, self.num_events : 2 * self.num_events]
            uncensored_count = (events > 0).sum().item()
        except:
            uncensored_count = 0

        # Add to buffer
        for i in range(batch_size):
            # Add new items to queues
            if len(self.embedding_queue) < self.current_buffer_size:
                self.embedding_queue.append(embeddings[i])
                self.reference_queue.append(references_cpu[i])

        # Update statistics
        self.uncensored_events_count.append(uncensored_count)

        if logger.isEnabledFor(logging.DEBUG) and len(self.embedding_queue) % 100 == 0:
            logger.debug(
                f"Buffer: size={len(self.embedding_queue)}/{self.current_buffer_size}, "
                f"uncensored_events_this_batch={uncensored_count}"
            )

    def get_buffer_contents(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current buffer contents as tensors.

        Returns:
            Tuple of (embeddings, references)
        """
        if not self.embedding_queue:
            return None, None

        # Convert queues to tensors
        buffer_embeddings = torch.stack(list(self.embedding_queue))
        buffer_references = torch.stack(list(self.reference_queue))

        return buffer_embeddings, buffer_references

    def combine_with_current_batch(
        self,
        current_outputs: Union[ModelOutput, Dict[str, torch.Tensor]],
        current_references: torch.Tensor,
    ) -> Tuple[ModelOutput, torch.Tensor]:
        """
        Combine current batch with buffer contents.

        Args:
            current_outputs: Current batch model outputs
            current_references: Current batch references

        Returns:
            Tuple of (combined_outputs, combined_references)
        """
        if len(self.embedding_queue) == 0:
            # Buffer is empty, return current batch only
            return current_outputs, current_references

        # Get buffer contents as tensors
        buffer_embeddings, buffer_references = self.get_buffer_contents()

        # Move buffer to same device as current batch
        device = current_references.device
        buffer_embeddings = buffer_embeddings.to(device)
        buffer_references = buffer_references.to(device)

        # Extract current batch embeddings
        if isinstance(current_outputs, Dict) and "logits" in current_outputs:
            current_embeddings = current_outputs["logits"]
        elif hasattr(current_outputs, "logits") and current_outputs.logits is not None:
            current_embeddings = current_outputs.logits
        else:
            logger.warning("No logits found in outputs, using hazard or other features")
            # Fallback
            if isinstance(current_outputs, Dict) and "hazard" in current_outputs:
                current_embeddings = current_outputs["hazard"]
            elif (
                hasattr(current_outputs, "hazard")
                and current_outputs.hazard is not None
            ):
                current_embeddings = current_outputs.hazard
            else:
                logger.error("No suitable embeddings found in outputs")
                return current_outputs, current_references

        # Combine embeddings
        combined_embeddings = torch.cat([current_embeddings, buffer_embeddings], dim=0)
        combined_references = torch.cat([current_references, buffer_references], dim=0)

        # Create a new output object with combined embeddings
        if isinstance(current_outputs, ModelOutput):
            # Create a new ModelOutput without deepcopy
            # Extract the class from the instance to create a new one
            output_class = current_outputs.__class__
            # Create a new instance with only the combined logits
            combined_outputs = output_class(logits=combined_embeddings)

            # Copy over any other attributes that aren't tensors requiring gradients
            for key, value in vars(current_outputs).items():
                if key != "logits" and not (
                    isinstance(value, torch.Tensor) and value.requires_grad
                ):
                    try:
                        setattr(combined_outputs, key, value)
                    except Exception:
                        # Skip if attribute can't be set
                        pass

            # Set hazard, risk, and survival to None as they would need recalculation
            combined_outputs.hazard = None
            combined_outputs.risk = None
            combined_outputs.survival = None
        else:
            # For dictionary outputs, create a new dict
            combined_outputs = dict(current_outputs)  # Shallow copy
            combined_outputs["logits"] = combined_embeddings

            # Clear derived fields that would need recalculation
            if "hazard" in combined_outputs:
                combined_outputs["hazard"] = None
            if "risk" in combined_outputs:
                combined_outputs["risk"] = None
            if "survival" in combined_outputs:
                combined_outputs["survival"] = None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Combined batch size: {combined_embeddings.shape[0]} "
                f"(current: {current_embeddings.shape[0]}, buffer: {buffer_embeddings.shape[0]})"
            )

        return combined_outputs, combined_references

    def get_stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics for monitoring.

        Returns:
            Dictionary of buffer statistics
        """
        stats = {
            "buffer_size": len(self.embedding_queue),
            "max_buffer_size": self.current_buffer_size,
            "buffer_utilization": len(self.embedding_queue)
            / max(1, self.current_buffer_size),
        }

        # Event statistics if we have any buffer content
        if self.uncensored_events_count:
            stats["latest_batch_uncensored"] = self.uncensored_events_count[-1]

        # Add variance statistics if tracking
        if self.track_variance and self.loss_variance_history:
            stats["loss_variance"] = self.loss_variance_history[-1]
            stats["buffer_adjustments"] = len(self.buffer_adjustment_history)

        return stats

    @staticmethod
    def estimate_optimal_buffer_size(
        num_samples: int,
        censoring_rate: float,
        min_events_per_batch: int = 5,
        batch_size: int = 32,
    ) -> int:
        """
        Estimate optimal buffer size based on dataset characteristics.

        Args:
            num_samples: Total samples in dataset
            censoring_rate: Proportion of censored samples (0-1)
            min_events_per_batch: Minimum desired events per effective batch
            batch_size: Batch size used in training

        Returns:
            Recommended buffer size
        """
        # Calculate expected events per batch
        events_per_batch = batch_size * (1 - censoring_rate)

        if events_per_batch >= min_events_per_batch:
            # Already enough events in each batch
            recommended_buffer = batch_size * 3
        else:
            # Need buffer to reach desired events per effective batch
            multiplier = math.ceil(min_events_per_batch / max(0.01, events_per_batch))
            recommended_buffer = batch_size * multiplier

        # Cap at reasonable size relative to dataset
        max_reasonable_size = min(num_samples // 2, 4096)
        recommended_buffer = min(recommended_buffer, max_reasonable_size)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Estimated optimal buffer size: {recommended_buffer} "
                f"(events_per_batch={events_per_batch:.1f}, "
                f"target={min_events_per_batch})"
            )

        return recommended_buffer


class MoCoSurvivalLoss(Loss):
    """
    Enhanced survival loss that uses a memory buffer to simulate larger batch sizes.

    This loss wrapper works with any existing survival loss function and enhances it with:
    1. A memory buffer of past embeddings for more stable risk sets
    2. Weighted combination of loss on current batch and loss on combined data
    """

    def __init__(
        self,
        base_loss: Loss,
        buffer_size: int = 2048,
        num_events: int = 1,
        embedding_dim: int = 128,
        use_buffer: bool = True,
        current_batch_weight: float = 1.0,
        buffer_weight: float = 1.0,
        dynamic_buffer: bool = True,
        initial_buffer_size: Optional[int] = None,
        track_variance: bool = True,
        adaptive_buffer: bool = True,
    ):
        """
        Initialize MoCo-enhanced survival loss.

        Args:
            base_loss: Base survival loss to enhance
            buffer_size: Maximum buffer size
            num_events: Number of events
            embedding_dim: Dimension of embeddings
            use_buffer: Whether to use buffer data in loss
            current_batch_weight: Weight for current batch loss
            buffer_weight: Weight for combined data loss
            dynamic_buffer: Whether to grow buffer size during training
            initial_buffer_size: Initial buffer size if using dynamic growth
            track_variance: Whether to track loss variance
            adaptive_buffer: Whether to adjust buffer size based on variance
        """
        super(MoCoSurvivalLoss, self).__init__(num_events=num_events)

        self.base_loss = base_loss
        self.use_buffer = use_buffer
        self.current_batch_weight = current_batch_weight
        self.buffer_weight = buffer_weight

        # Create momentum buffer
        self.buffer = MomentumBuffer(
            embedding_dim=embedding_dim,
            buffer_size=buffer_size,
            num_events=num_events,
            dynamic_growth=dynamic_buffer,
            initial_size=initial_buffer_size,
            track_variance=track_variance,
            adaptive_buffer_adjustment=adaptive_buffer,
        )

        self.iteration = 0

        # Register weights as buffers for state_dict
        self.register_buffer("_current_weight", torch.tensor(current_batch_weight))
        self.register_buffer("_buffer_weight", torch.tensor(buffer_weight))

    def forward(
        self, predictions: ModelOutput, references: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss with MoCo enhancement.

        Args:
            predictions: Model predictions
            references: Ground truth references

        Returns:
            torch.Tensor: The computed loss value
        """
        # Update iterations
        self.iteration += 1

        # Update buffer size based on training progress
        if self.training:
            self.buffer.update_buffer_size(self.iteration)

        # Compute loss on current batch
        current_loss = self.base_loss(predictions, references)

        # Track loss variance to detect instability if in training mode
        if self.training:
            self.buffer.track_loss(current_loss.item())

        # Initialize total loss with current batch loss
        total_loss = self.current_batch_weight * current_loss

        # Only use buffer in training mode and if enabled
        if self.training and self.use_buffer and self.buffer_weight > 0:
            # Update buffer with current batch data
            self.buffer.update(predictions, references)

            # Only use buffer if it contains data
            if len(self.buffer.embedding_queue) > 0:
                # Get combined batch (current + buffer)
                combined_predictions, combined_references = (
                    self.buffer.combine_with_current_batch(predictions, references)
                )

                # Skip if no combined data
                if combined_predictions is not None and combined_references is not None:
                    # Compute loss on combined data
                    combined_loss = self.base_loss(
                        combined_predictions, combined_references
                    )

                    # Add weighted combined loss to total
                    total_loss = total_loss + self.buffer_weight * combined_loss

        # Normalize weights
        if self.training and self.use_buffer and self.buffer_weight > 0:
            total_loss = total_loss / (self.current_batch_weight + self.buffer_weight)

        return total_loss

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics for monitoring."""
        return self.buffer.get_stats()

    def reset_buffer(self):
        """Reset the buffer."""
        self.buffer.reset_buffer()


class AdaptiveMoCoLoss(MoCoSurvivalLoss):
    """
    Enhanced MoCo survival loss with automatic stability detection and
    buffer size adjustments based on loss variance analysis.
    """

    def __init__(
        self,
        base_loss: Loss,
        buffer_size: int = 2048,
        num_events: int = 1,
        embedding_dim: int = 128,
        use_buffer: bool = True,
        current_batch_weight: float = 1.0,
        buffer_weight: float = 1.0,
        dynamic_buffer: bool = True,
        initial_buffer_size: Optional[int] = None,
        variance_window: int = 10,
        variance_threshold: float = 0.1,
        min_buffer_ratio: float = 0.25,
        max_buffer_ratio: float = 1.0,
    ):
        """
        Initialize adaptive MoCo loss with variance monitoring.

        Args:
            base_loss: Base survival loss to enhance
            buffer_size: Maximum buffer size
            num_events: Number of events
            embedding_dim: Dimension of embeddings
            use_buffer: Whether to use buffer data in loss
            current_batch_weight: Weight for current batch loss
            buffer_weight: Weight for combined data loss
            dynamic_buffer: Whether to grow buffer size during training
            initial_buffer_size: Initial buffer size if using dynamic growth
            variance_window: Window size for variance calculation
            variance_threshold: Threshold for significant variance change
            min_buffer_ratio: Minimum buffer ratio relative to max size
            max_buffer_ratio: Maximum buffer ratio relative to max size
        """
        super(AdaptiveMoCoLoss, self).__init__(
            base_loss=base_loss,
            buffer_size=buffer_size,
            num_events=num_events,
            embedding_dim=embedding_dim,
            use_buffer=use_buffer,
            current_batch_weight=current_batch_weight,
            buffer_weight=buffer_weight,
            dynamic_buffer=dynamic_buffer,
            initial_buffer_size=initial_buffer_size,
            track_variance=True,
            adaptive_buffer=True,
        )

        # Additional parameters for adaptive behavior
        self.variance_window = variance_window
        self.variance_threshold = variance_threshold
        self.min_buffer_size = int(buffer_size * min_buffer_ratio)
        self.max_buffer_size = int(buffer_size * max_buffer_ratio)

        # Override buffer configuration
        self.buffer = MomentumBuffer(
            embedding_dim=embedding_dim,
            buffer_size=buffer_size,
            num_events=num_events,
            dynamic_growth=dynamic_buffer,
            initial_size=initial_buffer_size,
            track_variance=True,
            variance_window=variance_window,
            adaptive_buffer_adjustment=True,
        )


class DynamicWeightMoCoLoss(MoCoSurvivalLoss):
    """
    MoCo survival loss with dynamic weighting between batch and buffer components
    based on training progress.
    """

    def __init__(
        self,
        base_loss: Loss,
        buffer_size: int = 2048,
        num_events: int = 1,
        embedding_dim: int = 128,
        use_buffer: bool = True,
        initial_batch_weight: float = 1.0,
        final_batch_weight: float = 0.5,
        initial_buffer_weight: float = 0.0,
        final_buffer_weight: float = 1.0,
        warmup_steps: int = 1000,
        dynamic_buffer: bool = True,
        initial_buffer_size: Optional[int] = None,
        track_variance: bool = True,
        adaptive_buffer: bool = True,
    ):
        """
        Initialize MoCo loss with dynamic weight adjustment.

        Args:
            base_loss: Base survival loss to enhance
            buffer_size: Maximum buffer size
            num_events: Number of events
            embedding_dim: Dimension of embeddings
            use_buffer: Whether to use buffer data in loss
            initial_batch_weight: Initial weight for batch loss
            final_batch_weight: Final weight for batch loss
            initial_buffer_weight: Initial weight for buffer loss
            final_buffer_weight: Final weight for buffer loss
            warmup_steps: Steps to transition from initial to final weights
            dynamic_buffer: Whether to grow buffer size during training
            initial_buffer_size: Initial buffer size if using dynamic growth
            track_variance: Whether to track loss variance
            adaptive_buffer: Whether to adjust buffer size based on variance
        """
        super(DynamicWeightMoCoLoss, self).__init__(
            base_loss=base_loss,
            buffer_size=buffer_size,
            num_events=num_events,
            embedding_dim=embedding_dim,
            use_buffer=use_buffer,
            current_batch_weight=initial_batch_weight,
            buffer_weight=initial_buffer_weight,
            dynamic_buffer=dynamic_buffer,
            initial_buffer_size=initial_buffer_size,
            track_variance=track_variance,
            adaptive_buffer=adaptive_buffer,
        )

        # Store weight transition parameters
        self.initial_batch_weight = initial_batch_weight
        self.final_batch_weight = final_batch_weight
        self.initial_buffer_weight = initial_buffer_weight
        self.final_buffer_weight = final_buffer_weight
        self.warmup_steps = warmup_steps

    def _update_weights(self):
        """Update weights based on current iteration."""
        if self.iteration >= self.warmup_steps:
            # Use final weights after warmup
            self.current_batch_weight = self.final_batch_weight
            self.buffer_weight = self.final_buffer_weight
        else:
            # Linear interpolation during warmup
            progress = self.iteration / self.warmup_steps
            self.current_batch_weight = self.initial_batch_weight + progress * (
                self.final_batch_weight - self.initial_batch_weight
            )
            self.buffer_weight = self.initial_buffer_weight + progress * (
                self.final_buffer_weight - self.initial_buffer_weight
            )

        # Update buffer tensors
        self._current_weight.fill_(self.current_batch_weight)
        self._buffer_weight.fill_(self.buffer_weight)

    def forward(
        self, predictions: ModelOutput, references: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss with dynamically weighted MoCo enhancement.

        Args:
            predictions: Model predictions
            references: Ground truth references

        Returns:
            torch.Tensor: The computed loss value
        """
        # Update iterations
        self.iteration += 1

        # Update weights based on training progress
        if self.training:
            self._update_weights()

        # Call parent implementation with updated weights
        return super().forward(predictions, references)
