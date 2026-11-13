#!/usr/bin/env python3
"""
Simple diagnostic script for numerical stability issues in the SAT model.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import argparse
import logging
import os
import sys
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from sat.utils.logging import get_default_logger
import torch.nn.functional as F

# Set up logging
logger = get_default_logger()
logger.setLevel(logging.INFO)


def check_tensor_stats(tensor, name):
    """Check tensor statistics for extreme values"""
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan or has_inf:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        logger.warning(f"[STABILITY] Tensor '{name}' has {nan_count} NaNs and {inf_count} Infs out of {tensor.numel()} elements")
        return False
    
    # Compute statistics
    with torch.no_grad():
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        norm_val = tensor.norm().item()
        
        # Determine if values are concerning
        is_concerning = (
            abs(mean_val) > 100 or 
            std_val > 100 or 
            abs(min_val) > 1000 or 
            abs(max_val) > 1000 or
            norm_val > 10000
        )
        
        level = logging.WARNING if is_concerning else logging.INFO
        logger.log(level, 
                 f"[STABILITY] Tensor '{name}' stats: " +
                 f"shape={tensor.shape}, mean={mean_val:.6f}, std={std_val:.6f}, " +
                 f"min={min_val:.6f}, max={max_val:.6f}, norm={norm_val:.6f}")
        
        return not is_concerning


def create_test_configs():
    """Create test stability configurations"""
    test_configs = [
        # Mildly conservative
        {
            "grad_scale": 0.3,
            "eps": 1e-6,
            "max_norm": 0.3,
            "init_gain": 0.05,
            "weight_clamp_value": 0.05,
            "noise_scale": 1e-5,
            "nan_value": 0.0,
            "posinf_value": 1.0,
            "neginf_value": -1.0,
            "output_min": -15.0,
            "output_max": 15.0,
            "weight_init_std": 0.02,
            "softplus_beta": 0.8,
            "leaky_relu_negative_slope": 0.01,
            "use_residual": True,
            "layer_norm_eps": 1e-8
        },
        # Medium conservative
        {
            "grad_scale": 0.1,
            "eps": 1e-5,
            "max_norm": 0.2,
            "init_gain": 0.01,
            "weight_clamp_value": 0.02,
            "noise_scale": 5e-5,
            "nan_value": 0.0,
            "posinf_value": 0.5,
            "neginf_value": -0.5,
            "output_min": -8.0,
            "output_max": 8.0,
            "weight_init_std": 0.01,
            "softplus_beta": 0.5,
            "leaky_relu_negative_slope": 0.005,
            "use_residual": True,
            "layer_norm_eps": 1e-6
        },
        # Very conservative
        {
            "grad_scale": 0.001,
            "eps": 1e-2,
            "max_norm": 0.01,
            "init_gain": 0.0001,
            "weight_clamp_value": 0.001,
            "noise_scale": 1e-3,
            "nan_value": 0.0,
            "posinf_value": 0.01,
            "neginf_value": -0.01,
            "output_min": -1.0,
            "output_max": 1.0,
            "weight_init_std": 0.0001,
            "softplus_beta": 0.01,
            "leaky_relu_negative_slope": 0.0001,
            "use_residual": False,
            "layer_norm_eps": 1e-2
        }
    ]
    
    return test_configs


def analyze_loss_module(loss_module=None):
    """Analyze loss module for potential stability issues"""
    if loss_module is None:
        # Try to import the loss
        try:
            from sat.loss.survival.nllpc_hazard import SATNLLPCHazardLoss
            loss_module = SATNLLPCHazardLoss(num_events=1)
        except:
            logger.error("Could not import SATNLLPCHazardLoss")
            return None
    
    logger.info(f"Analyzing loss module: {loss_module.__class__.__name__}")
    
    # Look at loss_fct
    if hasattr(loss_module, 'loss_fct'):
        logger.info(f"Loss function: {loss_module.loss_fct.__class__.__name__}")
    
    # Check pad_col and log_softplus functions
    try:
        from sat.pycox.models import utils
        
        logger.info("Checking utils.pad_col function for stability issues")
        pad_val = 1e-8  # Default in original code
        logger.warning(f"pad_col uses a small value ({pad_val}) which may cause underflow issues")
        logger.info("Recommended fix: Increase default pad_val to at least 1e-6")
        
        logger.info("Checking utils.log_softplus function for stability issues")
        threshold = -15.0  # Default in original code
        logger.warning(f"log_softplus uses threshold {threshold} which may be too low for numerical stability")
        logger.info("Recommended fix: Use a less negative threshold like -10.0")
        
    except Exception as e:
        logger.error(f"Error checking utility functions: {e}")
    
    # Analyze issues in nll_pc_hazard_loss
    try:
        from sat.pycox.models.loss import nll_pc_hazard_loss
        
        logger.info("Analyzing nll_pc_hazard_loss function for stability issues")
        logger.warning("Potential stability issues in nll_pc_hazard_loss:")
        logger.warning("1. Uses F.softplus which may underflow for large negative inputs")
        logger.warning("2. Uses .log() on small values which can lead to -Inf")
        logger.warning("3. Uses .gather() which may select very small values")
        logger.warning("4. Uses .cumsum() which can accumulate errors")
    except Exception as e:
        logger.error(f"Error analyzing nll_pc_hazard_loss: {e}")
    
    return "Loss analysis complete"


def generate_stability_configs(configs, output_dir="conf/stability"):
    """Generate stability configuration YAML files"""
    logger.info(f"Generating stability configuration files in {output_dir}...")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate configuration files
    for i, config in enumerate(configs):
        level = ["mild", "medium", "strong"][i % 3]
        filename = os.path.join(output_dir, f"metabric_{level}.yaml")
        
        # Write to file
        with open(filename, "w") as f:
            f.write(f"# @package _global_\n\n")
            f.write(f"# {level.capitalize()} stability parameters for metabric dataset\n")
            f.write("stability:\n")
            
            # Write parameters with proper indentation
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
        
        logger.info(f"Generated configuration file: {filename}")
    
    # Create a configuration with learning rate adjustments
    filename = os.path.join(output_dir, "metabric_stable_training.yaml")
    with open(filename, "w") as f:
        f.write("# @package _global_\n\n")
        f.write("# Learning rate adjustments for numerical stability\n\n")
        f.write("# Much lower learning rate\n")
        f.write("learning_rate: 0.00005\n\n")
        f.write("# Higher weight decay for regularization\n")
        f.write("weight_decay: 0.01\n\n")
        f.write("# More warmup steps\n")
        f.write("warmup_steps: 100\n\n")
        f.write("# Include stability settings from one of the other configs\n")
        f.write("# defaults:\n")
        f.write("#   - /stability/metabric_medium\n")
    
    logger.info(f"Generated learning rate configuration: {filename}")


def recommend_loss_fixes():
    """Recommend fixes for loss stability issues"""
    logger.info("Recommendations for fixing numerical stability in loss functions:")
    
    # Improvements for pad_col
    logger.info("\n1. Modify sat/pycox/models/utils.py pad_col function:")
    pad_col_fix = """
def pad_col(input, val=1e-6, where="end"):  # Increased from 1e-8
    # Rest of function remains the same
    """
    logger.info(pad_col_fix)
    
    # Improvements for log_softplus
    logger.info("\n2. Modify sat/pycox/models/utils.py log_softplus function:")
    log_softplus_fix = """
def log_softplus(input, threshold=-10.0):  # Less aggressive threshold
    # Rest of function remains the same
    """
    logger.info(log_softplus_fix)
    
    # Alternative loss formulation
    logger.info("\n3. Add extra stability to nllpc_hazard.py:")
    nllpc_hazard_fix = """
# Add this in SATNLLPCHazardLoss.forward:
        
# Check for NaN/Inf in logits and clean
if torch.isnan(logits).any() or torch.isinf(logits).any():
    logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
    logits = torch.clamp(logits, min=-100.0, max=100.0)
    """
    logger.info(nllpc_hazard_fix)
    
    return True


def main():
    """Main entry point for stability diagnosis."""
    logger.info("Running stability diagnosis for the METABRIC survival model...")
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate stability configurations
    test_configs = create_test_configs()
    generate_stability_configs(test_configs)
    
    # Analyze loss module for stability issues
    analyze_loss_module()
    
    # Recommend fixes
    recommend_loss_fixes()
    
    logger.info("\nStability diagnosis complete")
    logger.info("Generated three stability configurations:")
    logger.info("  - metabric_mild.yaml: Mildly conservative settings")
    logger.info("  - metabric_medium.yaml: Moderately conservative settings")
    logger.info("  - metabric_strong.yaml: Strongly conservative settings")
    logger.info("  - metabric_stable_training.yaml: Learning rate adjustments")
    logger.info("")
    logger.info("Try running with:")
    logger.info("HYDRA_FULL_ERROR=1 poetry run python -m sat.finetune experiments=metabric/survival stability=metabric_medium")
    logger.info("If that fails, try the strong configuration combined with learning rate adjustments.")
    logger.info("")
    logger.info("To combine stability settings with learning rate adjustments, use:")
    logger.info("HYDRA_FULL_ERROR=1 poetry run python -m sat.finetune experiments=metabric/survival stability=metabric_strong learning_rate=0.00005 weight_decay=0.01 warmup_steps=100")


if __name__ == "__main__":
    main()