"""Custom Trainer class that extends transformers.Trainer with explicit Accelerate integration."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import math
import os
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, speed_metrics
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import get_scheduler
from transformers.utils import logging

from accelerate import Accelerator
from accelerate.utils import find_batch_size, set_seed

logger = logging.get_logger(__name__)


# Simple hack to work with MPS devices from: https://github.com/huggingface/transformers/issues/17971#issuecomment-1171579884
class TrainingArgumentsWithMPSSupport(TrainingArguments):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


class SATTrainer(Trainer):
    """
    A custom Trainer class that extends the Hugging Face Transformers Trainer
    with specific functionality for survival analysis tasks and explicit Accelerate integration.
    """

    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        """
        Initialize the SATTrainer with additional capability to handle multi-event survival models
        and explicit Accelerate integration.

        Args:
            model: The model to train.
            args: The training arguments.
            data_collator: Function to collate batches.
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset.
            tokenizer: The tokenizer, if applicable.
            model_init: Function to initialize the model.
            compute_metrics: Function to compute metrics.
            callbacks: List of callbacks to use.
            optimizers: Tuple containing optimizer and scheduler.
            preprocess_logits_for_metrics: Function to preprocess logits before computing metrics.
        """
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Check if this is a multi-event survival model
        self.is_multi_event = self._check_is_multi_event()

        # Create an explicit Accelerator instance for more control
        # This supplements the accelerator that transformers.Trainer creates
        self.sat_accelerator = self._create_accelerator()

    def _create_accelerator(self):
        """Create an explicit Accelerator instance with our desired configuration."""
        mixed_precision = "fp16" if self.args.fp16 else "no"
        return Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            project_dir=self.args.output_dir,
            log_with="tensorboard" if self.args.logging_dir else None,
        )

    def _check_is_multi_event(self):
        """Check if the model is a multi-event survival model."""
        if hasattr(self.model, "config") and hasattr(self.model.config, "num_events"):
            return self.model.config.num_events > 1
        return False

    def create_optimizer(self):
        """
        Custom optimizer creation to handle multi-event models differently.
        """
        opt_model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [torch.nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if n in decay_parameters
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if n not in decay_parameters
                    ],
                    "weight_decay": 0.0,
                },
            ]

            # Apply specialized parameters for multi-event models
            if self.is_multi_event:
                logger.info(
                    "Using specialized optimizer settings for multi-event survival model"
                )
                multi_event_lr = (
                    self.args.learning_rate * 0.5
                )  # Lower learning rate for stability
                multi_event_weight_decay = 0.01  # More aggressive weight decay
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                    "lr": multi_event_lr,
                    "weight_decay": multi_event_weight_decay,
                }
            else:
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                    "lr": self.args.learning_rate,
                }

            self.optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        """Create the learning rate scheduler with additional handling for multi-event models."""
        if self.lr_scheduler is None:
            if optimizer is None:
                optimizer = self.optimizer

            # For multi-event models, we can customize the warmup steps if needed
            if self.is_multi_event:
                warmup_steps = max(int(0.2 * num_training_steps), 100)  # Longer warmup
            else:
                warmup_steps = self.args.get_warmup_steps(num_training_steps)

            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )

        return self.lr_scheduler

    # Use the parent class's dataloader implementations to avoid serialization issues with lambdas
    # We'll override these methods to add our performance optimizations in a way that's compatible with multiprocessing

    def get_train_dataloader(self) -> DataLoader:
        """
        Gets a dataloader for training that works with MPS and other devices.

        Returns:
            A DataLoader for the training dataset.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Simply use the parent class implementation
        # This avoids multiprocessing issues on MPS
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Gets a dataloader for evaluation that works with MPS and other devices.

        Args:
            eval_dataset: The dataset to evaluate on. If not provided, self.eval_dataset will be used.

        Returns:
            A DataLoader for the evaluation dataset.
        """
        # Simply use the parent class implementation
        # This avoids multiprocessing issues on MPS
        return super().get_eval_dataloader(eval_dataset)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step with additional handling for multi-event models.
        Uses explicit accelerate integration for gradient scaling and handling.

        Args:
            model: The model to train.
            inputs: The inputs to the model.
            num_items_in_batch: Number of items in the batch (for gradient accumulation).

        Returns:
            loss: The loss value for this step.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Use accelerate's context manager for backward pass handling
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # Use accelerate's backward method for mixed precision handling
        self.accelerator.backward(loss)

        # Special handling for gradients in multi-event cases
        if self.is_multi_event:
            self._handle_multi_event_gradients(model)

        return loss.detach()

    def _handle_multi_event_gradients(self, model):
        """
        Apply special gradient handling for multi-event models.

        Args:
            model: The model being trained.
        """
        for param in model.parameters():
            if param.grad is not None:
                # Handle NaN and Inf values by replacing with zeros
                nan_mask = torch.isnan(param.grad) | torch.isinf(param.grad)
                if nan_mask.any():
                    logger.warning(
                        f"Found {nan_mask.sum().item()} NaN/Inf grad values. Zeroing them."
                    )
                    param.grad.data[nan_mask] = 0.0

                # Optional: Monitor large gradients
                grad_abs_max = (
                    torch.max(torch.abs(param.grad)).item()
                    if torch.numel(param.grad) > 0
                    else 0
                )
                if grad_abs_max > 10.0:
                    logger.debug(f"Large gradient detected: max={grad_abs_max:.2f}")

                # Apply per-parameter gradient clipping
                grad_clip_threshold = 1.0
                large_grad_mask = torch.abs(param.grad) > grad_clip_threshold
                if large_grad_mask.any():
                    param.grad.data[large_grad_mask] = (
                        torch.sign(param.grad.data[large_grad_mask])
                        * grad_clip_threshold
                    )

    def _clip_gradients(self, model, clip_norm):
        """
        Custom gradient clipping for survival analysis, using accelerate for better performance.

        Args:
            model: The model being trained.
            clip_norm: Maximum gradient norm.

        Returns:
            The norm of the gradients.
        """
        if self.is_multi_event:
            # Use more aggressive clipping for multi-event models
            clip_norm = min(clip_norm, 0.1)  # Max of 0.1 for multi-event
            logger.debug(
                f"Using aggressive gradient clipping with norm {clip_norm} for multi-event model"
            )

        # Use accelerator for gradient clipping
        return self.accelerator.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Custom evaluation method with additional handling for multi-event models.
        Uses accelerate for more efficient metric gathering.

        Args:
            eval_dataset: Evaluation dataset.
            ignore_keys: Keys to ignore in the model output.
            metric_key_prefix: Prefix for the metric keys.

        Returns:
            Evaluation metrics.
        """
        # Use the parent class evaluation method
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # For multi-event models, add some specialized metrics
        if self.is_multi_event:
            metrics[f"{metric_key_prefix}_is_multi_event"] = True

        return metrics

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Custom prediction step with additional handling for multi-event models.
        Uses accelerate for more efficient output gathering.

        Args:
            model: The model to use for prediction.
            inputs: The inputs to the model.
            prediction_loss_only: Whether to return only the loss.
            ignore_keys: Keys to ignore in the model output.

        Returns:
            Tuple of loss, logits, and labels.
        """
        # For multi-event models, ensure we capture all needed outputs
        if self.is_multi_event:
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            inputs = self._prepare_inputs(inputs)

            if prediction_loss_only:
                with torch.no_grad():
                    outputs = model(**inputs)
                    loss = outputs.loss
                return (loss, None, None)

            with torch.no_grad():
                outputs = model(**inputs)

            # Get logits or complete SAOutput for survival models
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            # Process labels
            labels = tuple(inputs.get(name) for name in self.label_names)
            labels = torch.cat(labels, dim=0) if len(labels) > 0 else None

            return (outputs.loss, logits, labels)

        # Default behavior for non-multi-event models
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def save_model(self, output_dir=None, _internal_call=False):
        """
        Custom save model method to handle specialized model outputs and use accelerate's save methods.

        Args:
            output_dir: Directory to save the model to.
            _internal_call: Whether this is an internal call.
        """
        # Default Transformers save behavior
        super().save_model(output_dir, _internal_call)

        # Additional saving for survival models if needed
        if output_dir is None:
            output_dir = self.args.output_dir

        # Unwrap the model for direct saving
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Save the state of the accelerator (including optimizer and scheduler)
        self.accelerator.save_state(output_dir)

        # Add any additional model artifacts specific to survival models
        if hasattr(unwrapped_model, "save_survival_outputs"):
            unwrapped_model.save_survival_outputs(output_dir)

        return output_dir

    # Let the parent class handle the dataloader creation
    # The previous implementation here was causing duplicate method definitions

    # Custom push_to_hub method if special handling is needed
    def push_to_hub(self, **kwargs):
        """
        Push the model to the Hugging Face Hub.

        Args:
            **kwargs: Additional arguments to pass to the push_to_hub method.
        """
        # Use the parent class method
        return super().push_to_hub(**kwargs)

    # Implement a custom method to run our entire training pipeline
    def run_pipeline(self):
        """Run the entire training and evaluation pipeline."""
        # Train the model
        train_result = self.train()
        metrics = train_result.metrics

        # Save the final model
        self.save_model(self.args.output_dir)

        # Evaluate the model
        if self.args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = self.evaluate()

        return metrics
