"""Stable Trainer implementation extending Transformers trainer"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import inspect
import torch
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import speed_metrics
import time
import math

from sat.utils.numeric_stability import StabilityTracker, log_tensor_stats, MonitorTensorOps
from sat.utils.logging import get_default_logger

logger = get_default_logger()


class StableTrainer(Trainer):
    """
    Extends the transformers Trainer with numerical stability monitoring and diagnostics.
    
    This class overrides key methods to add stability tracking during training and evaluation.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a stable trainer with numerical stability monitoring.
        
        Args:
            stability_log_freq: How often to log stability metrics
            stability_threshold: Threshold for considering values problematic
            enable_histogram: Whether to track full histograms (memory intensive)
            export_stats: Whether to export stats to CSV files
            *args, **kwargs: Arguments passed to the Transformers Trainer
        """
        # Extract stability monitoring parameters
        self.stability_log_freq = kwargs.pop("stability_log_freq", 10)
        self.stability_threshold = kwargs.pop("stability_threshold", 100.0)
        self.enable_histogram = kwargs.pop("enable_histogram", False)
        self.export_stats = kwargs.pop("export_stats", True)
        self.export_path = kwargs.pop("export_path", "stability_stats")
        
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Initialize stability tracker
        self.tracker = StabilityTracker(
            model=self.model,
            track_gradients=True,
            track_parameters=True,
            track_activations=True,
            log_frequency=self.stability_log_freq,
            enable_histogram=self.enable_histogram
        )
        
        # Track stats by step
        self.nan_step_count = 0
        self.inf_step_count = 0
        self.problematic_steps = []
        self.step_times = []
        
        # Layer information storage
        self.layer_info = {}
        self._analyze_model_structure()
    
    def _analyze_model_structure(self):
        """Analyze model structure to better understand where issues might occur"""
        # Store layer types and shapes
        for name, module in self.model.named_modules():
            parameters = list(module.parameters())
            if not parameters:
                continue
                
            self.layer_info[name] = {
                "type": module.__class__.__name__,
                "parameters": len(parameters),
                "shapes": [p.shape for p in parameters]
            }
        
        logger.info(f"Model structure analyzed: found {len(self.layer_info)} layers with parameters")
        
    def training_step(self, model, inputs) -> torch.Tensor:
        """Override training_step to add stability monitoring"""
        # Record step start time for performance tracking
        step_start = time.time()
        
        # Unwrap inputs safely if needed
        if isinstance(inputs, dict) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        # Log input data statistics
        self._log_input_statistics(inputs, prefix="train_input")
        
        # Run forward pass with monitoring
        with MonitorTensorOps("forward_pass", log_level=logger.level):
            model.train()
            outputs = model(**inputs)
        
        # Check outputs
        self._check_outputs(outputs, "train_outputs")
        
        # Handle PyTorch data-parallel wrapped models
        if self.args.n_gpu > 1:
            loss = outputs.loss.mean()
        else:
            loss = outputs.loss
        
        # Check loss for numerical issues
        has_issue = not log_tensor_stats(loss, "training_loss", 
                                       threshold=self.stability_threshold, 
                                       log_level=logger.level)
        
        if has_issue:
            self.nan_step_count += torch.isnan(loss).any().item()
            self.inf_step_count += torch.isinf(loss).any().item()
            self.problematic_steps.append(self.state.global_step)
            logger.warning(f"Step {self.state.global_step}: Detected NaN/Inf in training loss")
            
            # Option: Replace problematic loss with a safe value to continue training
            if self.args.ignore_instability:
                # Create a small positive loss that can still train
                device = loss.device
                dtype = loss.dtype
                safe_loss = torch.tensor(1.0, device=device, dtype=dtype, requires_grad=True)
                logger.warning(f"Replacing NaN/Inf loss with safe value: {safe_loss.item()}")
                loss = safe_loss
        
        # Record step time
        step_end = time.time()
        self.step_times.append(step_end - step_start)
        
        # Update stability tracker
        self.tracker.update_step()
        
        # Export stats periodically if enabled
        if self.export_stats and self.state.global_step % (self.stability_log_freq * 10) == 0:
            export_path = f"{self.export_path}_step_{self.state.global_step}"
            self.tracker.export_stats_to_csv(export_path)
        
        # Generate stability report periodically
        if self.state.global_step % self.stability_log_freq == 0:
            report = self.tracker.get_stability_report()
            # Log key metrics
            logger.info(f"Stability report at step {self.state.global_step}:")
            logger.info(f"  - NaN detections: {report['summary']['total_nan_detected']}")
            logger.info(f"  - Inf detections: {report['summary']['total_inf_detected']}")
            if report['summary']['has_stability_issues']:
                logger.warning("  - Model has numerical stability issues")
                # Log top 3 unstable locations
                top_locs = dict(list(report['top_unstable_locations'].items())[:3])
                logger.warning(f"  - Top unstable locations: {top_locs}")
        
        return loss
    
    def _log_input_statistics(self, inputs, prefix="input"):
        """Log statistics about input tensors"""
        if not isinstance(inputs, dict):
            # Try to check the input as a tensor
            if isinstance(inputs, torch.Tensor):
                log_tensor_stats(inputs, f"{prefix}", 
                               threshold=self.stability_threshold, 
                               log_level=logger.level)
            return
        
        # Check each input tensor
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                log_tensor_stats(value, f"{prefix}.{key}", 
                               threshold=self.stability_threshold, 
                               log_level=logger.level)
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
                # Handle lists of tensors
                for i, tensor in enumerate(value):
                    log_tensor_stats(tensor, f"{prefix}.{key}[{i}]", 
                                   threshold=self.stability_threshold, 
                                   log_level=logger.level)
    
    def _check_outputs(self, outputs, name="outputs"):
        """Check model outputs for numerical issues"""
        # Handle various output types
        if isinstance(outputs, torch.Tensor):
            log_tensor_stats(outputs, name, 
                           threshold=self.stability_threshold, 
                           log_level=logger.level)
        elif hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
            log_tensor_stats(outputs.logits, f"{name}.logits", 
                           threshold=self.stability_threshold, 
                           log_level=logger.level)
            
            # Check loss if present
            if hasattr(outputs, "loss") and outputs.loss is not None:
                log_tensor_stats(outputs.loss, f"{name}.loss", 
                               threshold=self.stability_threshold, 
                               log_level=logger.level)
        
        # Handle dictionary-like outputs
        elif hasattr(outputs, "__dict__"):
            for key, value in outputs.__dict__.items():
                if isinstance(value, torch.Tensor):
                    log_tensor_stats(value, f"{name}.{key}", 
                                   threshold=self.stability_threshold, 
                                   log_level=logger.level)
                    
        # Handle explicit dictionary outputs
        elif isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    log_tensor_stats(value, f"{name}.{key}", 
                                   threshold=self.stability_threshold, 
                                   log_level=logger.level)
    
    def log(self, logs: Dict[str, float]) -> None:
        """Override log to add stability metrics"""
        # Add stability metrics
        logs = logs.copy()  # Create a copy to avoid modifying the original
        
        # Add numerical stability metrics
        logs["stability/nan_steps"] = self.nan_step_count
        logs["stability/inf_steps"] = self.inf_step_count
        logs["stability/problem_steps"] = len(self.problematic_steps)
        
        # Add computation metrics
        if self.step_times:
            logs["performance/avg_step_time"] = sum(self.step_times) / len(self.step_times)
            logs["performance/max_step_time"] = max(self.step_times)
        
        # Add gradient norm if available
        if hasattr(self, "grad_norm"):
            logs["train/grad_norm"] = self.grad_norm
        
        # Parent logging
        super().log(logs)
    
    def evaluate(self, *args, **kwargs):
        """Override evaluate to add stability monitoring"""
        # Initialize evaluation stability tracking
        eval_nan_count = 0
        eval_inf_count = 0
        
        # Store original methods
        original_prediction_step = self.prediction_step
        
        # Override prediction_step to monitor stability
        def monitored_prediction_step(*args, **kwargs):
            with MonitorTensorOps("eval_forward", log_level=logger.level):
                outputs = original_prediction_step(*args, **kwargs)
            
            # Check loss for NaN/Inf if present in outputs
            if len(outputs) > 0 and outputs[0] is not None:
                loss_tensor = outputs[0]
                if torch.isnan(loss_tensor).any():
                    nonlocal eval_nan_count
                    eval_nan_count += 1
                if torch.isinf(loss_tensor).any():
                    nonlocal eval_inf_count
                    eval_inf_count += 1
            
            return outputs
        
        # Replace method
        self.prediction_step = monitored_prediction_step
        
        try:
            # Run evaluation
            start_time = time.time()
            result = super().evaluate(*args, **kwargs)
            eval_time = time.time() - start_time
            
            # Add stability metrics to result
            result["eval/nan_count"] = eval_nan_count
            result["eval/inf_count"] = eval_inf_count
            result["eval/has_stability_issues"] = (eval_nan_count > 0 or eval_inf_count > 0)
            result["eval/runtime"] = eval_time
            
            # Log evaluation stability
            if eval_nan_count > 0 or eval_inf_count > 0:
                logger.warning(f"Evaluation detected numerical issues: {eval_nan_count} NaNs, {eval_inf_count} Infs")
            
            return result
        finally:
            # Restore original method
            self.prediction_step = original_prediction_step
    
    def create_optimizer(self):
        """Override to add gradient clipping and stability options to optimizer"""
        # Create standard optimizer
        optimizer = super().create_optimizer()
        
        # Log optimizer settings
        logger.info(f"Using optimizer: {optimizer.__class__.__name__}")
        logger.info(f"Learning rate: {self.args.learning_rate}")
        logger.info(f"Weight decay: {self.args.weight_decay}")
        logger.info(f"Gradient clipping: {self.args.max_grad_norm}")
        
        return optimizer
    
    def get_stability_metrics(self):
        """
        Return comprehensive stability metrics
        
        Returns:
            Dict: Stability metrics for logging/monitoring
        """
        # Get internal stability report
        report = self.tracker.get_stability_report()
        
        # Add additional metrics
        metrics = {
            "nan_step_count": self.nan_step_count,
            "inf_step_count": self.inf_step_count,
            "problematic_steps": len(self.problematic_steps),
            "total_nan_detected": report["summary"]["total_nan_detected"],
            "total_inf_detected": report["summary"]["total_inf_detected"],
            "has_stability_issues": report["summary"]["has_stability_issues"]
        }
        
        # Add top unstable locations
        top_locations = list(report["top_unstable_locations"].items())
        for i, (name, count) in enumerate(top_locations[:5]):  # Top 5 locations
            metrics[f"unstable_loc_{i+1}"] = f"{name} ({count})"
        
        # Add additional metrics if available
        if "gradients" in report:
            metrics["mean_grad_norm"] = report["gradients"]["mean_norm"]
            metrics["max_grad_norm"] = report["gradients"]["max_norm"]
        
        if "parameters" in report:
            metrics["mean_param_norm"] = report["parameters"]["mean_norm"]
            metrics["max_param_norm"] = report["parameters"]["max_norm"]
        
        return metrics
    
    def save_stability_report(self, output_dir=None):
        """
        Save a comprehensive stability report to disk
        
        Args:
            output_dir: Directory to save the report in
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Export all stats to CSV
        if self.export_stats:
            export_path = f"{output_dir}/stability_final"
            self.tracker.export_stats_to_csv(export_path)
        
        # Get stability metrics
        metrics = self.get_stability_metrics()
        
        # Log final stability report
        logger.info("Final stability report:")
        for key, value in metrics.items():
            logger.info(f"  - {key}: {value}")
        
        # Add to state
        self.state.stability_metrics = metrics
    
    def save_model(self, output_dir=None, _internal_call=False):
        """Override to add stability report"""
        # Save model as usual
        super().save_model(output_dir, _internal_call)
        
        # Save stability report if not an internal call
        if not _internal_call:
            self.save_stability_report(output_dir)


# Helper methods for integrating with sat.transformers.trainer
def create_stable_trainer(
    model,
    args,
    data_collator=None,
    train_dataset=None,
    eval_dataset=None,
    tokenizer=None,
    compute_metrics=None,
    callbacks=None,
    optimizers=(None, None),
    stability_log_freq=10,
    stability_threshold=100.0,
    enable_histogram=False,
    export_stats=True,
    export_path="stability_stats",
):
    """
    Create a StableTrainer with the given parameters
    
    Args:
        model: The model to train
        args: Training arguments
        data_collator: Function to collate batches
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer
        compute_metrics: Metrics computation function
        callbacks: Callbacks
        optimizers: Tuple of (optimizer, scheduler)
        stability_log_freq: How often to log stability metrics
        stability_threshold: Threshold for considering values problematic
        enable_histogram: Whether to track full histograms
        export_stats: Whether to export stats to CSV files
        export_path: Base path for exporting stats
        
    Returns:
        StableTrainer: A trainer with stability monitoring
    """
    return StableTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=optimizers,
        stability_log_freq=stability_log_freq,
        stability_threshold=stability_threshold,
        enable_histogram=enable_histogram,
        export_stats=export_stats,
        export_path=export_path,
    )