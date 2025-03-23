"""Learning rate finder utility for finding optimal learning rates."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import math
import numpy as np
import torch
from torch.optim import AdamW
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from sat.utils import logging

logger = logging.get_default_logger()


class LRFinder:
    """
    Learning Rate Finder that works with any transformer trainer.
    Inspired by the fast.ai learning rate finder approach.
    """

    def __init__(
        self,
        trainer,
        model,
        optimizer=None,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_steps: int = 100,
        smooth_factor: float = 0.05,
        diverge_threshold: float = 4.0,
    ):
        """
        Initialize the learning rate finder.

        Args:
            trainer: The trainer instance that will be used for training
            model: The model to use for finding the learning rate
            optimizer: Optional optimizer instance to use (will create one if not provided)
            start_lr: Starting learning rate
            end_lr: Maximum learning rate to try
            num_steps: Number of steps to take between start_lr and end_lr
            smooth_factor: Smoothing factor for loss curve
            diverge_threshold: Loss ratio threshold that indicates divergence
        """
        self.trainer = trainer
        self.model = model
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_steps = num_steps
        self.smooth_factor = smooth_factor
        self.diverge_threshold = diverge_threshold

        # Store results
        self.lrs = []
        self.losses = []
        self.best_lr = None

        # Store the original learning rate to restore it after finding
        self.original_lr = None

    def _prepare_optimizer(self):
        """Create optimizer if one wasn't provided."""
        if self.optimizer is None:
            # Follow the same pattern as the trainer for setting up optimizer
            decay_parameters = [
                name
                for name, p in self.model.named_parameters()
                if not any(nd in name for nd in ["bias", "LayerNorm.weight"])
            ]

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in decay_parameters
                    ],
                    "weight_decay": self.trainer.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in decay_parameters
                    ],
                    "weight_decay": 0.0,
                },
            ]

            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                betas=(self.trainer.args.adam_beta1, self.trainer.args.adam_beta2),
                eps=self.trainer.args.adam_epsilon,
                lr=self.start_lr,
            )

        return self.optimizer

    def _get_dataloader(self):
        """Get dataloader from trainer."""
        return self.trainer.get_train_dataloader()

    def _compute_lr_step(self, step):
        """Compute LR for the current step."""
        if self.num_steps <= 1:
            return self.end_lr
        return self.start_lr * (self.end_lr / self.start_lr) ** (
            step / (self.num_steps - 1)
        )

    def _update_lr(self, lr):
        """Update learning rate in optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _smooth_losses(self):
        """Apply exponential smoothing to losses."""
        smoothed_losses = []
        loss_history = []

        for i, loss in enumerate(self.losses):
            if i == 0:
                smoothed_losses.append(loss)
                loss_history.append(loss)
            else:
                smoothed_losses.append(
                    self.smooth_factor * loss
                    + (1 - self.smooth_factor) * smoothed_losses[-1]
                )
                # Apply median-based smoothing for better stability
                if len(loss_history) >= 5:
                    loss_history.pop(0)
                loss_history.append(loss)

        return smoothed_losses

    def find(self, save_plot_path: Optional[str] = None):
        """
        Run the learning rate finder and optionally save plot.

        Args:
            save_plot_path: Path to save the LR finder plot. If None, no plot is saved.

        Returns:
            Tuple of (best_lr, lrs, losses)
        """
        # Store original learning rate from trainer
        self.original_lr = self.trainer.args.learning_rate

        # Set model to training mode
        self.model.train()

        # Prepare optimizer
        self._prepare_optimizer()

        # Get dataloader
        dataloader = self._get_dataloader()

        # Track best loss for divergence checking
        best_loss = float("inf")

        # Start finder
        logger.info(
            f"Starting learning rate finder from {self.start_lr} to {self.end_lr}"
        )

        # Store initial weights to restore after search
        state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        step = 0
        try:
            for batch_idx, batch in enumerate(dataloader):
                # Compute and update learning rate
                lr = self._compute_lr_step(step)
                self._update_lr(lr)

                # Move batch to device
                batch = self.trainer._prepare_inputs(batch)

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                # Check if loss is valid
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        f"Invalid loss detected at lr={lr}. Stopping LR search."
                    )
                    break

                # Check for divergence
                if loss.item() > best_loss * self.diverge_threshold:
                    logger.info(f"Loss diverged at lr={lr}. Stopping LR search.")
                    break

                # Update best loss
                if loss.item() < best_loss:
                    best_loss = loss.item()

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Store results
                self.lrs.append(lr)
                self.losses.append(loss.item())

                # Log progress
                if (batch_idx + 1) % 5 == 0:
                    logger.info(
                        f"Step {step+1}/{self.num_steps}: lr={lr:.8f}, loss={loss.item():.4f}"
                    )

                # Increment step and check if finished
                step += 1
                if step >= self.num_steps:
                    break

        except Exception as e:
            logger.error(f"Error during learning rate search: {str(e)}")
        finally:
            # Restore original model state
            self.model.load_state_dict(state_dict)

        # Determine the best learning rate - typically the point with steepest loss descent
        if len(self.lrs) > 10:
            # Apply smoothing
            smoothed_losses = self._smooth_losses()

            # Compute derivatives
            derivatives = [
                (smoothed_losses[i + 1] - smoothed_losses[i])
                / (self.lrs[i + 1] - self.lrs[i])
                for i in range(len(smoothed_losses) - 1)
            ]

            # Find the point with the steepest negative gradient before divergence
            min_derivative_idx = None
            for i, derivative in enumerate(derivatives):
                if derivative < 0 and (
                    min_derivative_idx is None
                    or derivative < derivatives[min_derivative_idx]
                ):
                    min_derivative_idx = i

            if min_derivative_idx is not None:
                # Use point with steepest negative gradient, divided by factor for stability
                self.best_lr = self.lrs[min_derivative_idx] / 10.0
            else:
                # Fallback if no clear minimum was found
                self.best_lr = self.lrs[len(self.lrs) // 2] / 10.0
        elif len(self.lrs) > 0:
            # If we have very few points, use a simple heuristic
            self.best_lr = self.lrs[0]
        else:
            # If the finder failed, return the original learning rate
            self.best_lr = self.original_lr

        logger.info(f"Recommended learning rate: {self.best_lr:.8f}")

        # Create and save plot if requested
        if save_plot_path and len(self.lrs) > 0:
            self._create_plot(save_plot_path)

        return self.best_lr, self.lrs, self.losses

    def _create_plot(self, save_path):
        """Create and save learning rate finder plot."""
        plt.figure(figsize=(10, 6))
        plt.semilogx(self.lrs, self.losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.axvline(
            x=self.best_lr,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Best LR: {self.best_lr:.8f}",
        )
        plt.title("Learning Rate Finder")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        logger.info(f"Learning rate finder plot saved to {save_path}")


def find_optimal_lr(
    trainer,
    model,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    num_steps: int = 100,
    save_plot_path: Optional[str] = None,
) -> float:
    """
    High-level function to find the optimal learning rate for a model.

    Args:
        trainer: The trainer instance
        model: The model to train
        start_lr: Lower bound of learning rate search range
        end_lr: Upper bound of learning rate search range
        num_steps: Number of steps to use
        save_plot_path: Where to save the LR finder plot

    Returns:
        The optimal learning rate found
    """
    finder = LRFinder(
        trainer=trainer,
        model=model,
        start_lr=start_lr,
        end_lr=end_lr,
        num_steps=num_steps,
    )

    # Run the finder
    best_lr, _, _ = finder.find(save_plot_path=save_plot_path)

    return best_lr
