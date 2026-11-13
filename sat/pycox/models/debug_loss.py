"""Debugging utilities for tracing numerical stability issues in the loss functions"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional

from sat.utils.logging import get_default_logger
logger = get_default_logger()


def check_tensor(tensor: Tensor, name: str) -> bool:
    """
    Check a tensor for NaN/Inf values and log statistics
    
    Args:
        tensor: The tensor to check
        name: Name for logging
        
    Returns:
        bool: True if tensor is valid (no NaN/Inf)
    """
    # Check for None
    if tensor is None:
        logger.warning(f"[STABILITY] {name} is None")
        return False
        
    # Check for NaN/Inf
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    # Compute basic statistics
    with torch.no_grad():
        try:
            nan_count = torch.isnan(tensor).sum().item() if has_nan else 0
            inf_count = torch.isinf(tensor).sum().item() if has_inf else 0
            
            # Log statistics
            if has_nan or has_inf:
                logger.warning(f"[STABILITY] {name}: Shape {tensor.shape}, {nan_count} NaNs, {inf_count} Infs")
                return False
            
            # If no NaN/Inf, log more detailed stats
            mean = tensor.mean().item()
            std = tensor.std().item()
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            
            # Check for concerning values
            is_concerning = (
                abs(mean) > 100 or
                std > 100 or
                abs(min_val) > 1000 or
                abs(max_val) > 1000
            )
            
            if is_concerning:
                logger.warning(
                    f"[STABILITY] {name}: Concerning values - " + 
                    f"Mean: {mean:.4g}, Std: {std:.4g}, Min: {min_val:.4g}, Max: {max_val:.4g}"
                )
            else:
                logger.debug(
                    f"[STABILITY] {name} OK - " +
                    f"Mean: {mean:.4g}, Std: {std:.4g}, Min: {min_val:.4g}, Max: {max_val:.4g}"
                )
            
            return not is_concerning
            
        except Exception as e:
            logger.error(f"[STABILITY] Error checking {name}: {e}")
            return False


def debug_nll_pc_hazard_loss(
    phi: Tensor,
    idx_durations: Tensor,
    events: Tensor,
    interval_frac: Tensor,
) -> Dict[str, Any]:
    """
    Debug version of nll_pc_hazard_loss with extensive stability monitoring
    
    Args:
        phi: Estimates in (-inf, inf), where hazard = sigmoid(phi)
        idx_durations: Event times as indices
        events: Event indicators (1.0) or censoring (0.0)
        interval_frac: Fraction of last interval
        
    Returns:
        Dict: Results with loss and stability info
    """
    logger.info("[DEBUG_LOSS] Starting debug_nll_pc_hazard_loss")
    
    # Track each step result
    steps = {}
    
    # Step 0: Check inputs
    steps["input_phi_valid"] = check_tensor(phi, "input_phi")
    steps["input_idx_valid"] = check_tensor(idx_durations, "input_idx_durations")
    steps["input_events_valid"] = check_tensor(events, "input_events")
    steps["input_frac_valid"] = check_tensor(interval_frac, "input_interval_frac")
    
    # Convert bool events to float if needed
    if events.dtype is torch.bool:
        events = events.float()
        logger.debug("[DEBUG_LOSS] Converted events from bool to float")
    
    # Reshape inputs
    idx_durations = idx_durations.view(-1, 1)
    events = events.view(-1)
    interval_frac = interval_frac.view(-1)
    
    # Step 1: Filter valid indices
    keep = idx_durations.view(-1) >= 0
    phi_filtered = phi[keep, :]
    idx_durations_filtered = idx_durations[keep, :]
    events_filtered = events[keep]
    interval_frac_filtered = interval_frac[keep]
    
    # Check filtered tensors
    steps["filtered_phi_valid"] = check_tensor(phi_filtered, "filtered_phi")
    steps["filtered_idx_valid"] = check_tensor(idx_durations_filtered, "filtered_idx_durations")
    steps["filtered_events_valid"] = check_tensor(events_filtered, "filtered_events")
    steps["filtered_frac_valid"] = check_tensor(interval_frac_filtered, "filtered_interval_frac")
    
    try:
        # Step 2: log_h_e computation using log_softplus (phi.gather -> log_softplus -> mul(events))
        gathered_phi = phi_filtered.gather(1, idx_durations_filtered)
        steps["gathered_phi_valid"] = check_tensor(gathered_phi, "gathered_phi")
        
        gathered_phi_view = gathered_phi.view(-1)
        steps["gathered_phi_view_valid"] = check_tensor(gathered_phi_view, "gathered_phi_view")
        
        # Apply log_softplus - This is a critical step for stability
        try:
            from sat.pycox.models.utils import log_softplus
            log_softplus_values = log_softplus(gathered_phi_view)
            steps["log_softplus_valid"] = check_tensor(log_softplus_values, "log_softplus_values")
        except Exception as e:
            logger.error(f"[DEBUG_LOSS] Error in log_softplus: {e}")
            # Fallback to direct computation
            softplus_values = F.softplus(gathered_phi_view)
            steps["softplus_valid"] = check_tensor(softplus_values, "softplus_values")
            
            # Add small epsilon to prevent log(0)
            softplus_values = softplus_values + 1e-8
            log_values = torch.log(softplus_values)
            steps["log_values_valid"] = check_tensor(log_values, "log_values")
            log_softplus_values = log_values
        
        # Multiply by events
        log_h_e = log_softplus_values.mul(events_filtered)
        steps["log_h_e_valid"] = check_tensor(log_h_e, "log_h_e")
        
        # Step 3: scaled_h_e computation (haz.gather -> mul(interval_frac))
        haz = F.softplus(phi_filtered)
        steps["haz_valid"] = check_tensor(haz, "haz")
        
        gathered_haz = haz.gather(1, idx_durations_filtered)
        steps["gathered_haz_valid"] = check_tensor(gathered_haz, "gathered_haz")
        
        gathered_haz_view = gathered_haz.view(-1)
        steps["gathered_haz_view_valid"] = check_tensor(gathered_haz_view, "gathered_haz_view")
        
        scaled_h_e = gathered_haz_view.mul(interval_frac_filtered)
        steps["scaled_h_e_valid"] = check_tensor(scaled_h_e, "scaled_h_e")
        
        # Step 4: sum_haz computation (pad_col -> cumsum -> gather -> view)
        try:
            from sat.pycox.models.utils import pad_col
            padded_haz = pad_col(haz, where="start")
            steps["padded_haz_valid"] = check_tensor(padded_haz, "padded_haz")
        except Exception as e:
            logger.error(f"[DEBUG_LOSS] Error in pad_col: {e}")
            # Create fallback padded version
            pad = torch.zeros_like(haz[:, :1]) + 1e-6
            padded_haz = torch.cat([pad, haz], dim=1)
            steps["padded_haz_fallback_valid"] = check_tensor(padded_haz, "padded_haz_fallback")
        
        # Compute cumulative sum
        cumsum_haz = padded_haz.cumsum(1)
        steps["cumsum_haz_valid"] = check_tensor(cumsum_haz, "cumsum_haz")
        
        gathered_cumsum = cumsum_haz.gather(1, idx_durations_filtered)
        steps["gathered_cumsum_valid"] = check_tensor(gathered_cumsum, "gathered_cumsum")
        
        sum_haz = gathered_cumsum.view(-1)
        steps["sum_haz_valid"] = check_tensor(sum_haz, "sum_haz")
        
        # Step 5: Final loss computation (-log_h_e.sub(scaled_h_e).sub(sum_haz))
        loss = -log_h_e.sub(scaled_h_e).sub(sum_haz)
        steps["loss_valid"] = check_tensor(loss, "loss")
        
        # Take mean for final result
        mean_loss = loss.mean()
        steps["mean_loss_valid"] = check_tensor(mean_loss, "mean_loss")
        
        # Track final loss value
        steps["loss_value"] = mean_loss.item()
        steps["success"] = True
        
        logger.info(f"[DEBUG_LOSS] Loss computation successful, value = {mean_loss.item():.6f}")
        
    except Exception as e:
        logger.error(f"[DEBUG_LOSS] Error during loss computation: {e}")
        steps["success"] = False
        steps["error"] = str(e)
    
    return steps


def debug_loss_function(loss_fn, model_outputs, references):
    """
    Debug wrapper to trace through loss function execution
    
    Args:
        loss_fn: The loss function to debug
        model_outputs: Outputs from the model
        references: Reference values
        
    Returns:
        Tuple: (loss value, debug info)
    """
    logger.info("[DEBUG] Starting loss function debugging")
    
    debug_info = {
        "inputs_valid": True,
        "outputs_valid": True,
        "loss_valid": True,
        "steps": {}
    }
    
    # Check inputs
    if hasattr(model_outputs, "logits"):
        debug_info["logits_valid"] = check_tensor(model_outputs.logits, "model_outputs.logits")
        if not debug_info["logits_valid"]:
            debug_info["inputs_valid"] = False
    
    # Check references
    debug_info["references_valid"] = check_tensor(references, "references")
    if not debug_info["references_valid"]:
        debug_info["inputs_valid"] = False
    
    # Compute loss with monitoring
    try:
        # Create a new stability-enhanced model outputs if needed
        if hasattr(model_outputs, "logits") and (torch.isnan(model_outputs.logits).any() or torch.isinf(model_outputs.logits).any()):
            logger.warning("[DEBUG] Cleaning NaN/Inf values in model outputs")
            fixed_logits = torch.nan_to_num(
                model_outputs.logits, 
                nan=0.0, 
                posinf=10.0, 
                neginf=-10.0
            )
            model_outputs.logits = torch.clamp(fixed_logits, -100.0, 100.0)
        
        # Compute loss
        loss = loss_fn(model_outputs, references)
        
        # Check loss
        debug_info["loss_valid"] = check_tensor(loss, "loss")
        debug_info["loss_value"] = loss.item() if debug_info["loss_valid"] else None
        
        return loss, debug_info
    
    except Exception as e:
        logger.error(f"[DEBUG] Error in loss computation: {e}")
        debug_info["loss_valid"] = False
        debug_info["error"] = str(e)
        
        # Return a dummy loss for training to continue
        device = next(loss_fn.parameters()).device if hasattr(loss_fn, "parameters") else torch.device("cpu")
        dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
        
        return dummy_loss, debug_info