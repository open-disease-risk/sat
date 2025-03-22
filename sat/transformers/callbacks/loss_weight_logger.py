"""Callback for logging loss weights to TensorBoard."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from typing import Dict, Any, Optional, List

import torch
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)

from sat.utils import logging

logger = logging.get_default_logger()


class LossWeightLoggerCallback(TrainerCallback):
    """
    Callback for logging loss weights to TensorBoard during training.

    This callback checks if the model's loss function implements get_loss_weights()
    and logs each weight to TensorBoard, allowing you to monitor how the weights
    change during training with different balancing strategies.
    """

    def __init__(
        self,
        log_freq: int = 1,
        prefix: str = "loss_weights",
        log_eval: bool = True,
        log_train: bool = True,
    ):
        """
        Initialize the callback.

        Args:
            log_freq: Logging frequency (every N evaluation steps)
            prefix: Prefix for the logged metrics
            log_eval: Whether to log during evaluation
            log_train: Whether to log during training
        """
        super().__init__()
        self.log_freq = log_freq
        self.prefix = prefix
        self.log_eval = log_eval
        self.log_train = log_train
        self.step_counter = 0

    def _log_weights(
        self,
        args: TrainingArguments,
        state: TrainerState,
        model: torch.nn.Module,
        phase: str,
    ) -> None:
        """
        Log the current loss weights to TensorBoard.

        Args:
            args: Training arguments
            state: Trainer state
            model: The model being trained
            phase: 'train' or 'eval'
        """
        # Try to get the loss weights through different possible paths
        loss_weights = None
        loss_fn = None

        # Check if model has a loss_fn attribute
        if hasattr(model, "loss_fn"):
            loss_fn = model.loss_fn
        # Check if model has a module attribute (for distributed training wrappers)
        elif hasattr(model, "module") and hasattr(model.module, "loss_fn"):
            loss_fn = model.module.loss_fn
        # Check if model has get_loss_weights directly
        elif hasattr(model, "get_loss_weights"):
            loss_weights = model.get_loss_weights()

        # If we found a loss function, try to get weights from it
        if loss_fn is not None and hasattr(loss_fn, "get_loss_weights"):
            try:
                loss_weights = loss_fn.get_loss_weights()
            except Exception as e:
                logger.warning(f"Failed to get loss weights: {e}")

        # Log weights if we found them
        if loss_weights is not None:
            if not isinstance(loss_weights, (list, tuple)):
                logger.warning(f"Expected list of weights, got {type(loss_weights)}")
                return

            log_metrics = {}
            for i, weight in enumerate(loss_weights):
                # Create a metric name that includes phase and weight index
                metric_name = f"{self.prefix}/{phase}/weight_{i}"
                log_metrics[metric_name] = weight

            # Log batch of metrics
            if hasattr(model, "log") and callable(model.log):
                model.log(log_metrics)

            # Additionally try to use accelerator logging if available
            if hasattr(args, "accelerator") and hasattr(args.accelerator, "log"):
                args.accelerator.log(log_metrics)

            # Log to console as fallback if requested
            logger.debug(f"Loss weights ({phase}): {loss_weights}")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module,
        **kwargs,
    ) -> None:
        """
        Log weights during evaluation.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            model: The model being trained
        """
        if not self.log_eval:
            return

        if self.step_counter % self.log_freq == 0:
            self._log_weights(args, state, model, "eval")

        self.step_counter += 1

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module,
        **kwargs,
    ) -> None:
        """
        Log weights at the end of each training step.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            model: The model being trained
        """
        if not self.log_train:
            return

        # Log less frequently during training to avoid slowing things down
        if state.global_step % (self.log_freq * 10) == 0:
            self._log_weights(args, state, model, "train")
