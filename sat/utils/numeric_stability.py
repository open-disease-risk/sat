"""Utilities for diagnosing and monitoring numerical stability"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import logging

# Get logger
from sat.utils.logging import get_default_logger
logger = get_default_logger()

class StabilityTracker:
    """
    Track numerical stability issues during model training
    
    This class provides methods to:
    1. Monitor tensors for NaN/Inf values
    2. Track gradients and parameter values over time
    3. Log stability metrics at different stages of model computation
    4. Produce summary statistics for debugging
    """
    
    def __init__(
        self, 
        model: Optional[torch.nn.Module] = None,
        track_gradients: bool = True,
        track_parameters: bool = True,
        track_activations: bool = True,
        log_frequency: int = 10,
        log_level: int = logging.WARNING,
        enable_histogram: bool = False,
        threshold_values: bool = True,
        enable_hooks: bool = True
    ):
        """
        Initialize the stability tracker
        
        Args:
            model: Optional model to monitor
            track_gradients: Whether to track gradient statistics
            track_parameters: Whether to track parameter statistics
            track_activations: Whether to track activation statistics
            log_frequency: How often to log statistics (in steps)
            log_level: Logging level for stability issues
            enable_histogram: Whether to track full histograms (memory intensive)
            threshold_values: Whether to report cases exceeding thresholds
            enable_hooks: Whether to automatically register hooks on the model
        """
        self.model = model
        self.track_gradients = track_gradients
        self.track_parameters = track_parameters
        self.track_activations = track_activations
        self.log_frequency = log_frequency
        self.log_level = log_level
        self.enable_histogram = enable_histogram
        self.threshold_values = threshold_values
        self.enable_hooks = enable_hooks
        
        # Counters
        self.step = 0
        self.nan_count = defaultdict(int)
        self.inf_count = defaultdict(int)
        self.instability_locations = defaultdict(int)
        
        # Statistics tracking
        self.grad_stats = defaultdict(list)
        self.param_stats = defaultdict(list)
        self.activation_stats = defaultdict(list)
        
        # Thresholds for reporting
        self.grad_norm_threshold = 1000.0
        self.param_norm_threshold = 1000.0
        self.activation_norm_threshold = 1000.0
        
        # Register hooks if model is provided and hooks are enabled
        if model is not None and enable_hooks:
            self.register_hooks(model)
    
    def register_hooks(self, model: torch.nn.Module):
        """
        Register forward and backward hooks on all model parameters
        
        Args:
            model: The model to monitor
        """
        # Store handles to unregister later if needed
        self.hook_handles = []
        
        # Register hooks on all modules
        for name, module in model.named_modules():
            # Forward hook to track activations
            if self.track_activations:
                handle = module.register_forward_hook(
                    lambda m, inp, out, name=name: self._activation_hook(m, inp, out, name)
                )
                self.hook_handles.append(handle)
            
            # Register hooks for parameters
            if hasattr(module, 'weight') and module.weight is not None:
                # Track weight gradients
                if self.track_gradients:
                    handle = module.weight.register_hook(
                        lambda grad, name=f"{name}.weight": self._gradient_hook(grad, name)
                    )
                    self.hook_handles.append(handle)
                
                # Track weight parameters
                if self.track_parameters:
                    # Update stats at step time
                    pass
            
            if hasattr(module, 'bias') and module.bias is not None:
                # Track bias gradients
                if self.track_gradients:
                    handle = module.bias.register_hook(
                        lambda grad, name=f"{name}.bias": self._gradient_hook(grad, name)
                    )
                    self.hook_handles.append(handle)
    
    def _activation_hook(self, module: torch.nn.Module, inp: Tuple[torch.Tensor], 
                         out: Union[torch.Tensor, Tuple[torch.Tensor]], name: str):
        """
        Hook for tracking activation statistics
        
        Args:
            module: The module being monitored
            inp: Input tensors
            out: Output tensors
            name: Module name
        """
        # Only process on some steps to reduce overhead
        if self.step % self.log_frequency != 0:
            return
        
        # Handle different output types
        if isinstance(out, tuple):
            tensors = [t for t in out if isinstance(t, torch.Tensor)]
        elif isinstance(out, torch.Tensor):
            tensors = [out]
        else:
            return
        
        # Check each output tensor
        for i, tensor in enumerate(tensors):
            tensor_name = f"{name}.output{i if len(tensors) > 1 else ''}"
            
            # Check for NaN/Inf
            has_nan = torch.isnan(tensor).any()
            has_inf = torch.isinf(tensor).any()
            
            if has_nan:
                self.nan_count[tensor_name] += 1
                self.instability_locations[tensor_name] += 1
                if self.nan_count[tensor_name] % self.log_frequency == 1:
                    logger.log(self.log_level, f"NaN detected in activation: {tensor_name} (count: {self.nan_count[tensor_name]})")
            
            if has_inf:
                self.inf_count[tensor_name] += 1
                self.instability_locations[tensor_name] += 1
                if self.inf_count[tensor_name] % self.log_frequency == 1:
                    logger.log(self.log_level, f"Inf detected in activation: {tensor_name} (count: {self.inf_count[tensor_name]})")
            
            # Compute statistics if no NaN/Inf or track anyway
            try:
                with torch.no_grad():
                    stats = {
                        'mean': tensor.mean().item(),
                        'std': tensor.std().item(),
                        'min': tensor.min().item(),
                        'max': tensor.max().item(),
                        'norm': tensor.norm().item(),
                        'step': self.step
                    }
                    
                    # Check for concerning statistics
                    if self.threshold_values and (
                        abs(stats['mean']) > 10 or
                        stats['std'] > 10 or
                        abs(stats['min']) > 100 or
                        abs(stats['max']) > 100 or
                        stats['norm'] > self.activation_norm_threshold
                    ):
                        logger.log(self.log_level, 
                                 f"Activation {tensor_name} has concerning values: " +
                                 f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, " +
                                 f"min={stats['min']:.4f}, max={stats['max']:.4f}, " +
                                 f"norm={stats['norm']:.4f}")
                    
                    # Track histogram if enabled
                    if self.enable_histogram:
                        tensor_np = tensor.detach().cpu().numpy()
                        if tensor_np.size > 0:  # Only compute if tensor has elements
                            hist, bin_edges = np.histogram(
                                tensor_np.flatten(), 
                                bins=10, 
                                range=(float(stats['min']), float(stats['max']))
                            )
                            stats['histogram'] = hist.tolist()
                            stats['bin_edges'] = bin_edges.tolist()
                    
                    self.activation_stats[tensor_name].append(stats)
            except Exception as e:
                logger.error(f"Error computing activation statistics for {tensor_name}: {str(e)}")
    
    def _gradient_hook(self, grad: torch.Tensor, name: str):
        """
        Hook for tracking gradient statistics
        
        Args:
            grad: The gradient tensor
            name: Parameter name
        """
        # Skip frequent logging to reduce overhead
        if self.step % self.log_frequency != 0:
            return grad
        
        # Check for NaN/Inf
        has_nan = torch.isnan(grad).any()
        has_inf = torch.isinf(grad).any()
        
        if has_nan:
            self.nan_count[f"{name}.grad"] += 1
            self.instability_locations[f"{name}.grad"] += 1
            if self.nan_count[f"{name}.grad"] % self.log_frequency == 1:
                logger.log(self.log_level, f"NaN detected in gradient: {name} (count: {self.nan_count[f'{name}.grad']})")
        
        if has_inf:
            self.inf_count[f"{name}.grad"] += 1
            self.instability_locations[f"{name}.grad"] += 1
            if self.inf_count[f"{name}.grad"] % self.log_frequency == 1:
                logger.log(self.log_level, f"Inf detected in gradient: {name} (count: {self.inf_count[f'{name}.grad']})")
        
        # Compute statistics
        try:
            with torch.no_grad():
                stats = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'min': grad.min().item(),
                    'max': grad.max().item(),
                    'norm': grad.norm().item(),
                    'step': self.step
                }
                
                # Check for concerning values
                if self.threshold_values and (
                    abs(stats['mean']) > 10 or
                    stats['std'] > 10 or
                    abs(stats['min']) > 100 or
                    abs(stats['max']) > 100 or
                    stats['norm'] > self.grad_norm_threshold
                ):
                    logger.log(self.log_level, 
                             f"Gradient {name} has concerning values: " +
                             f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, " +
                             f"min={stats['min']:.4f}, max={stats['max']:.4f}, " +
                             f"norm={stats['norm']:.4f}")
                
                # Track histogram if enabled
                if self.enable_histogram:
                    grad_np = grad.detach().cpu().numpy()
                    if grad_np.size > 0:  # Only compute if tensor has elements
                        hist, bin_edges = np.histogram(
                            grad_np.flatten(), 
                            bins=10, 
                            range=(float(stats['min']), float(stats['max']))
                        )
                        stats['histogram'] = hist.tolist()
                        stats['bin_edges'] = bin_edges.tolist()
                
                self.grad_stats[name].append(stats)
        except Exception as e:
            logger.error(f"Error computing gradient statistics for {name}: {str(e)}")
        
        return grad
    
    def update_step(self):
        """Increment step counter and update parameter statistics"""
        self.step += 1
        
        # Only compute parameter statistics periodically
        if self.track_parameters and self.model is not None and self.step % self.log_frequency == 0:
            self._update_parameter_stats()
    
    def _update_parameter_stats(self):
        """Update statistics for all model parameters"""
        if self.model is None:
            return
        
        for name, param in self.model.named_parameters():
            if param is None:
                continue
                
            # Check for NaN/Inf
            has_nan = torch.isnan(param).any()
            has_inf = torch.isinf(param).any()
            
            if has_nan:
                self.nan_count[name] += 1
                self.instability_locations[name] += 1
                if self.nan_count[name] % self.log_frequency == 1:
                    logger.log(self.log_level, f"NaN detected in parameter: {name} (count: {self.nan_count[name]})")
            
            if has_inf:
                self.inf_count[name] += 1
                self.instability_locations[name] += 1
                if self.inf_count[name] % self.log_frequency == 1:
                    logger.log(self.log_level, f"Inf detected in parameter: {name} (count: {self.inf_count[name]})")
            
            # Compute statistics
            try:
                with torch.no_grad():
                    stats = {
                        'mean': param.mean().item(),
                        'std': param.std().item(),
                        'min': param.min().item(),
                        'max': param.max().item(),
                        'norm': param.norm().item(),
                        'step': self.step
                    }
                    
                    # Check for concerning values
                    if self.threshold_values and (
                        abs(stats['mean']) > 10 or
                        stats['std'] > 10 or
                        abs(stats['min']) > 100 or
                        abs(stats['max']) > 100 or
                        stats['norm'] > self.param_norm_threshold
                    ):
                        logger.log(self.log_level, 
                                 f"Parameter {name} has concerning values: " +
                                 f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, " +
                                 f"min={stats['min']:.4f}, max={stats['max']:.4f}, " +
                                 f"norm={stats['norm']:.4f}")
                    
                    # Track histogram if enabled
                    if self.enable_histogram:
                        param_np = param.detach().cpu().numpy()
                        if param_np.size > 0:  # Only compute if tensor has elements
                            hist, bin_edges = np.histogram(
                                param_np.flatten(), 
                                bins=10, 
                                range=(float(stats['min']), float(stats['max']))
                            )
                            stats['histogram'] = hist.tolist()
                            stats['bin_edges'] = bin_edges.tolist()
                    
                    self.param_stats[name].append(stats)
            except Exception as e:
                logger.error(f"Error computing parameter statistics for {name}: {str(e)}")
    
    def check_tensor(self, tensor: torch.Tensor, name: str) -> bool:
        """
        Check a tensor for numerical issues
        
        Args:
            tensor: The tensor to check
            name: The name for logging purposes
            
        Returns:
            bool: True if the tensor has no numerical issues
        """
        # Check for NaN/Inf
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        has_issue = has_nan or has_inf
        
        # Update counters
        if has_nan:
            self.nan_count[name] += 1
            self.instability_locations[name] += 1
            logger.log(self.log_level, f"NaN detected in tensor: {name} (count: {self.nan_count[name]})")
        
        if has_inf:
            self.inf_count[name] += 1
            self.instability_locations[name] += 1
            logger.log(self.log_level, f"Inf detected in tensor: {name} (count: {self.inf_count[name]})")
        
        # Compute statistics
        if not has_issue or self.log_frequency > 0:
            try:
                with torch.no_grad():
                    stats = {
                        'mean': tensor.mean().item(),
                        'std': tensor.std().item(),
                        'min': tensor.min().item(),
                        'max': tensor.max().item(),
                        'norm': tensor.norm().item(),
                        'step': self.step
                    }
                    
                    # Log concerning statistics
                    if has_issue or (self.threshold_values and (
                        abs(stats['mean']) > 10 or
                        stats['std'] > 10 or
                        abs(stats['min']) > 100 or
                        abs(stats['max']) > 100
                    )):
                        logger.log(self.log_level, 
                                 f"Tensor {name} statistics: " +
                                 f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, " +
                                 f"min={stats['min']:.4f}, max={stats['max']:.4f}, " +
                                 f"norm={stats['norm']:.4f}")
            except Exception as e:
                logger.error(f"Error computing tensor statistics for {name}: {str(e)}")
        
        return not has_issue
    
    def get_stability_report(self) -> Dict:
        """
        Generate a comprehensive stability report
        
        Returns:
            Dict: A dictionary with stability statistics
        """
        # Summary stats
        total_nan = sum(self.nan_count.values())
        total_inf = sum(self.inf_count.values())
        
        report = {
            'summary': {
                'total_steps': self.step,
                'total_nan_detected': total_nan,
                'total_inf_detected': total_inf,
                'has_stability_issues': total_nan > 0 or total_inf > 0,
                'num_tracked_locations': len(self.instability_locations),
            },
            'top_unstable_locations': dict(sorted(
                self.instability_locations.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]),  # Top 10 locations with issues
        }
        
        # Add gradient stats if available
        if self.track_gradients and self.grad_stats:
            # Compute summary across all gradients
            grad_norms = []
            for name, stats_list in self.grad_stats.items():
                if stats_list:  # Skip empty lists
                    latest_stats = stats_list[-1]
                    grad_norms.append(latest_stats.get('norm', 0))
            
            if grad_norms:
                report['gradients'] = {
                    'mean_norm': np.mean(grad_norms),
                    'max_norm': np.max(grad_norms),
                    'min_norm': np.min(grad_norms),
                }
        
        # Add parameter stats if available
        if self.track_parameters and self.param_stats:
            # Compute summary across all parameters
            param_norms = []
            for name, stats_list in self.param_stats.items():
                if stats_list:  # Skip empty lists
                    latest_stats = stats_list[-1]
                    param_norms.append(latest_stats.get('norm', 0))
            
            if param_norms:
                report['parameters'] = {
                    'mean_norm': np.mean(param_norms),
                    'max_norm': np.max(param_norms),
                    'min_norm': np.min(param_norms),
                }
        
        return report
    
    def export_stats_to_csv(self, prefix: str = "stability_stats"):
        """
        Export all statistics to CSV files
        
        Args:
            prefix: Prefix for the CSV filenames
        """
        # Export gradient statistics
        if self.grad_stats:
            all_grad_stats = []
            for name, stats_list in self.grad_stats.items():
                for stats in stats_list:
                    stats_copy = stats.copy()
                    stats_copy['name'] = name
                    # Remove histogram data as it complicates CSV format
                    if 'histogram' in stats_copy:
                        del stats_copy['histogram']
                    if 'bin_edges' in stats_copy:
                        del stats_copy['bin_edges']
                    all_grad_stats.append(stats_copy)
            
            if all_grad_stats:
                df = pd.DataFrame(all_grad_stats)
                df.to_csv(f"{prefix}_gradients.csv", index=False)
        
        # Export parameter statistics
        if self.param_stats:
            all_param_stats = []
            for name, stats_list in self.param_stats.items():
                for stats in stats_list:
                    stats_copy = stats.copy()
                    stats_copy['name'] = name
                    # Remove histogram data
                    if 'histogram' in stats_copy:
                        del stats_copy['histogram']
                    if 'bin_edges' in stats_copy:
                        del stats_copy['bin_edges']
                    all_param_stats.append(stats_copy)
            
            if all_param_stats:
                df = pd.DataFrame(all_param_stats)
                df.to_csv(f"{prefix}_parameters.csv", index=False)
        
        # Export activation statistics
        if self.activation_stats:
            all_activation_stats = []
            for name, stats_list in self.activation_stats.items():
                for stats in stats_list:
                    stats_copy = stats.copy()
                    stats_copy['name'] = name
                    # Remove histogram data
                    if 'histogram' in stats_copy:
                        del stats_copy['histogram']
                    if 'bin_edges' in stats_copy:
                        del stats_copy['bin_edges']
                    all_activation_stats.append(stats_copy)
            
            if all_activation_stats:
                df = pd.DataFrame(all_activation_stats)
                df.to_csv(f"{prefix}_activations.csv", index=False)
        
        # Export instability locations
        if self.instability_locations:
            df = pd.DataFrame([
                {'name': name, 'count': count}
                for name, count in self.instability_locations.items()
            ])
            df.to_csv(f"{prefix}_instability_locations.csv", index=False)
    
    def __del__(self):
        """Clean up hooks when the tracker is deleted"""
        if hasattr(self, 'hook_handles'):
            for handle in self.hook_handles:
                handle.remove()


def wrap_loss_with_stability_checking(loss_fn):
    """
    Decorator to add stability checking to a loss function
    
    Args:
        loss_fn: The loss function to wrap
        
    Returns:
        A wrapped loss function with stability checking
    """
    # Create a stability tracker specifically for this loss function
    tracker = StabilityTracker(
        track_gradients=False,
        track_parameters=False,
        track_activations=False,
        log_frequency=1
    )
    
    def wrapped_loss_fn(*args, **kwargs):
        """Wrapped loss function with stability checking"""
        # First, check inputs for numerical issues
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                tracker.check_tensor(arg, f"loss_input_{i}")
        
        for name, arg in kwargs.items():
            if isinstance(arg, torch.Tensor):
                tracker.check_tensor(arg, f"loss_kwarg_{name}")
        
        # Compute the loss
        try:
            loss = loss_fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in loss function: {str(e)}")
            raise
        
        # Check output for numerical issues
        if isinstance(loss, torch.Tensor):
            is_ok = tracker.check_tensor(loss, "loss_output")
            if not is_ok:
                # Log where the loss came from for debugging
                logger.error(f"Loss function produced NaN/Inf with inputs: {args} and kwargs: {kwargs}")
                # Option 1: Return a default "safe" loss
                # return torch.tensor(1.0, device=loss.device, dtype=loss.dtype, requires_grad=True)
                # Option 2: Return the problematic loss and let caller handle it
                pass
        
        # Increment the step counter
        tracker.update_step()
        
        return loss
    
    return wrapped_loss_fn


def log_tensor_stats(tensor, name, threshold=100.0, log_level=logging.INFO):
    """
    Log statistics about a tensor
    
    Args:
        tensor: The tensor to log stats for
        name: Name for the logging
        threshold: Threshold to consider values concerning
        log_level: Logging level to use
    """
    logger = get_default_logger()
    
    # Check for NaN/Inf
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan or has_inf:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        logger.log(logging.WARNING, 
                 f"[STABILITY] Tensor '{name}' has {nan_count} NaNs and {inf_count} Infs out of {tensor.numel()} elements")
        
        # Get indices of NaN/Inf values for closer inspection
        if has_nan and tensor.numel() < 1000:  # Only for reasonably sized tensors
            nan_indices = torch.where(torch.isnan(tensor))
            logger.log(logging.WARNING, f"[STABILITY] NaN indices in '{name}': {nan_indices}")
        
        if has_inf and tensor.numel() < 1000:
            inf_indices = torch.where(torch.isinf(tensor))
            logger.log(logging.WARNING, f"[STABILITY] Inf indices in '{name}': {inf_indices}")
    
    # Compute statistics if tensor is valid
    if not has_nan and not has_inf:
        try:
            with torch.no_grad():
                mean_val = tensor.mean().item()
                std_val = tensor.std().item()
                min_val = tensor.min().item()
                max_val = tensor.max().item()
                norm_val = tensor.norm().item()
                
                # Determine if values are concerning
                is_concerning = (
                    abs(mean_val) > threshold or 
                    std_val > threshold or 
                    abs(min_val) > threshold * 10 or 
                    abs(max_val) > threshold * 10 or
                    norm_val > threshold * 100
                )
                
                # Log with appropriate level
                level = logging.WARNING if is_concerning else log_level
                logger.log(level, 
                         f"[STABILITY] Tensor '{name}' stats: " +
                         f"shape={tensor.shape}, mean={mean_val:.6f}, std={std_val:.6f}, " +
                         f"min={min_val:.6f}, max={max_val:.6f}, norm={norm_val:.6f}")
                
                # Additional warning if values are concerning
                if is_concerning:
                    logger.warning(f"[STABILITY] Tensor '{name}' has values exceeding threshold {threshold}")
        except Exception as e:
            logger.error(f"[STABILITY] Error computing statistics for '{name}': {str(e)}")
    
    return not (has_nan or has_inf)


def filter_nan_predictions(predictions, references=None, train_set=None, logger=None):
    """
    Filter out samples with NaN values in predictions.
    
    Args:
        predictions: numpy array of predictions, shape (n_samples, ...)
        references: optional numpy array of references to filter accordingly
        train_set: optional training set data to filter
        logger: optional logger for logging information
    
    Returns:
        tuple of filtered (predictions, references, train_set, removed_indices, removal_rate)
    """
    if logger is None:
        logger = get_default_logger()
    
    # Check if input is a PyTorch tensor and convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
        
    # Identify NaN/Inf values in predictions (checking first axis only)
    if len(predictions.shape) > 1:
        # For multi-dimensional predictions, check if any dimension has NaN
        has_nan_or_inf = np.isnan(predictions).any(axis=tuple(range(1, len(predictions.shape)))) | \
                         np.isinf(predictions).any(axis=tuple(range(1, len(predictions.shape))))
    else:
        has_nan_or_inf = np.isnan(predictions) | np.isinf(predictions)
    
    # Get indices of valid predictions (no NaN or Inf)
    valid_indices = ~has_nan_or_inf
    invalid_indices = np.where(has_nan_or_inf)[0]
    
    # Calculate removal statistics
    total_samples = len(predictions)
    removed_count = len(invalid_indices)
    removal_rate = removed_count / total_samples if total_samples > 0 else 0
    
    # Log the filtering operation
    if removed_count > 0:
        logger.warning(f"Removed {removed_count}/{total_samples} samples ({removal_rate:.2%}) with NaN/Inf values")
        
        # Detailed logging for small datasets
        if len(invalid_indices) < 50:  # Only log detailed indices for smaller datasets
            logger.info(f"Removed indices: {invalid_indices.tolist()}")
    else:
        logger.info("No NaN/Inf values found in predictions, no filtering performed")
    
    # Filter predictions
    filtered_predictions = predictions[valid_indices]
    
    # Filter references if provided
    filtered_references = None
    if references is not None:
        if len(references) != len(predictions):
            logger.warning(f"References length ({len(references)}) doesn't match predictions length ({len(predictions)})")
        else:
            filtered_references = references[valid_indices]
    
    # Filter train_set if provided
    filtered_train_set = None
    if train_set is not None:
        if len(train_set) != len(predictions):
            logger.warning(f"Train set length ({len(train_set)}) doesn't match predictions length ({len(predictions)})")
        else:
            filtered_train_set = train_set[valid_indices]
    
    return (filtered_predictions, filtered_references, filtered_train_set, 
            invalid_indices, removal_rate)


def debug_backward_with_logger(loss, module_dict=None):
    """
    Run backward pass with detailed logging for debugging
    
    Args:
        loss: The loss tensor to backpropagate
        module_dict: Optional dict mapping names to modules for parameter inspection
    """
    logger = get_default_logger()
    logger.info("[BACKWARD] Starting backward pass")
    
    # Log the loss value
    log_tensor_stats(loss, "loss")
    
    # Set hooks if modules are provided
    hooks = []
    if module_dict is not None:
        for name, module in module_dict.items():
            # Add hooks for all parameters
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    full_name = f"{name}.{param_name}"
                    
                    def make_hook(param_name):
                        def hook(grad):
                            log_tensor_stats(grad, f"{param_name}.grad")
                            return grad
                        return hook
                    
                    hook_handle = param.register_hook(make_hook(full_name))
                    hooks.append(hook_handle)
    
    # Run backward pass
    try:
        loss.backward()
        logger.info("[BACKWARD] Backward pass completed successfully")
    except Exception as e:
        logger.error(f"[BACKWARD] Error in backward pass: {str(e)}")
        raise
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    # Log gradients for key parameters if module_dict provided
    if module_dict is not None:
        for name, module in module_dict.items():
            for param_name, param in module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    full_name = f"{name}.{param_name}"
                    log_tensor_stats(param.grad, f"{full_name}.grad (after backward)")


def monitor_training_step(model, loss, optimizer, step, log_every=10):
    """
    Monitor a complete training step (forward + backward + optimize)
    
    Args:
        model: The model being trained
        loss: The loss tensor
        optimizer: The optimizer
        step: Current training step
        log_every: How often to log detailed statistics
    """
    logger = get_default_logger()
    
    # Only do detailed logging periodically
    if step % log_every == 0:
        # Check the loss
        has_issue = torch.isnan(loss).any() or torch.isinf(loss).any()
        
        if has_issue:
            logger.warning(f"[TRAINING] Step {step}: Loss has NaN/Inf values: {loss.item()}")
            return False
        
        logger.info(f"[TRAINING] Step {step}: Loss = {loss.item():.6f}")
        
        # Run backward with logging
        module_dict = {name: module for name, module in model.named_children()}
        debug_backward_with_logger(loss, module_dict)
        
        # Check for NaN/Inf in gradients
        has_nan_grad = False
        has_inf_grad = False
        max_grad_norm = 0.0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    has_nan_grad = True
                    logger.warning(f"[TRAINING] NaN gradient in {name}")
                
                if torch.isinf(param.grad).any():
                    has_inf_grad = True  
                    logger.warning(f"[TRAINING] Inf gradient in {name}")
                
                grad_norm = param.grad.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)
                
                if grad_norm > 1000:
                    logger.warning(f"[TRAINING] Large gradient norm in {name}: {grad_norm:.2f}")
        
        logger.info(f"[TRAINING] Max gradient norm: {max_grad_norm:.6f}")
        
        if has_nan_grad or has_inf_grad:
            logger.warning(f"[TRAINING] Step {step}: Detected NaN/Inf in gradients")
            return False
        
        # Proceed with optimizer step in the caller function
        return True
    
    # For non-logging steps, just check for basic issues
    has_issue = torch.isnan(loss).any() or torch.isinf(loss).any()
    if has_issue:
        logger.warning(f"[TRAINING] Step {step}: Loss has NaN/Inf values")
        return False
    
    return True


# Context manager for monitoring tensor operations
class MonitorTensorOps:
    """
    Context manager for monitoring tensor operations
    
    Usage:
    ```
    with MonitorTensorOps("forward_pass"):
        outputs = model(inputs)
    ```
    """
    
    def __init__(self, name, tensor_names=None, log_level=logging.INFO):
        """
        Initialize the context manager
        
        Args:
            name: Name for this monitored section
            tensor_names: Optional list of names to use for tensors
            log_level: Logging level to use
        """
        self.name = name
        self.tensor_names = tensor_names or []
        self.log_level = log_level
        self.logger = get_default_logger()
    
    def __enter__(self):
        self.logger.log(self.log_level, f"[MONITOR] Entering {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(f"[MONITOR] Exception in {self.name}: {exc_val}")
        else:
            self.logger.log(self.log_level, f"[MONITOR] Completed {self.name}")
    
    def check(self, tensor, name=None):
        """Check a tensor for numerical issues"""
        if name is None:
            if self.tensor_names:
                name = self.tensor_names.pop(0)
            else:
                name = f"{self.name}_tensor_{len(self.tensor_names)}"
        
        return log_tensor_stats(tensor, f"{self.name}.{name}", log_level=self.log_level)