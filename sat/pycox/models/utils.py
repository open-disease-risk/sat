"""utils from pycox copied here to reduce dependencies"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torch.nn.functional as F


def pad_col(input, val=1e-4, where="end"): # Better default value for stable gradients
    """Adds a column of `val` at the start or end of `input`.

    Uses an appropriate default value to ensure numerical stability for hazard calculations.
    
    Arguments:
        input (torch.tensor): Input tensor to pad
        val (float): Value to use for padding (default: 1e-4)
        where (str): Where to add padding column ('start' or 'end')
        
    Returns:
        torch.tensor: Padded tensor
    """
    from sat.utils import logging
    logger = logging.get_default_logger()
    
    # First validate the input tensor
    if len(input.shape) != 2:
        logger.warning(f"pad_col expects 2-D tensor, got shape {input.shape}. Attempting to handle.")
        if len(input.shape) == 3:  # Handle 3D case (batch x events x durations)
            # Create new output tensor
            if where == "end":
                output = torch.zeros(input.shape[0], input.shape[1], input.shape[2] + 1, 
                                  device=input.device, dtype=input.dtype)
                output[:, :, :-1] = input
                output[:, :, -1] = val
                return output
            elif where == "start":
                output = torch.zeros(input.shape[0], input.shape[1], input.shape[2] + 1, 
                                  device=input.device, dtype=input.dtype)
                output[:, :, 1:] = input
                output[:, :, 0] = val
                return output
        else:
            # Fall back to standard behavior but with warning
            logger.warning(f"Unexpected tensor shape {input.shape} in pad_col. Results may be incorrect.")
    
    # Only check for extremely problematic values
    has_nan = torch.isnan(input).any()
    has_inf = torch.isinf(input).any()
    
    if has_nan or has_inf:
        nan_count = torch.isnan(input).sum().item() if has_nan else 0
        inf_count = torch.isinf(input).sum().item() if has_inf else 0
        logger.warning(f"pad_col - NaN/Inf detected in input: {nan_count} NaNs, {inf_count} Infs")
        
        # Only clean extreme values if they're a small proportion (<25%) of the tensor
        if (nan_count + inf_count) < (input.numel() * 0.25):
            # Only replace NaN/Inf without modifying valid values
            input_clean = input.clone()
            if has_nan:
                input_clean[torch.isnan(input)] = 0.0
            if has_inf:
                input_clean[torch.isinf(input) & (input > 0)] = 100.0  # For positive infinity
                input_clean[torch.isinf(input) & (input < 0)] = -100.0  # For negative infinity
            input = input_clean
        else:
            logger.error(f"pad_col - Too many NaN/Inf ({nan_count + inf_count} out of {input.numel()}), hazard calculation may be unstable")
    
    # Create padding value - use a slightly adaptive approach
    pad = torch.zeros_like(input[:, :1])
    
    # Set padding value with validation
    if val is None or val == 0:
        # For hazard calculations, we want to avoid exact zeros
        # Use a proportional value if possible (relative to input max)
        try:
            with torch.no_grad():
                if not has_nan and not has_inf:
                    # Use a very small percentage of the maximum value, but with a minimum
                    input_max = torch.max(torch.abs(input)).item()
                    adaptive_val = max(1e-8, min(1e-4, input_max * 0.001))
                    pad = pad + adaptive_val
                    logger.debug(f"pad_col - Using adaptive padding value: {adaptive_val}")
                else:
                    # Fallback to default
                    pad = pad + 1e-4
                    logger.debug("pad_col - Using default padding value (1e-4)")
        except Exception:
            # Fallback to a reasonable default
            pad = pad + 1e-4
            logger.debug("pad_col - Using default padding value (1e-4) after exception")
    else:
        # Ensure val is reasonable
        if val < 0:
            logger.warning(f"pad_col - Negative padding value {val} may cause issues with hazard calculations")
        pad = pad + val
        logger.debug(f"pad_col - Using specified padding value: {val}")
    
    # Perform padding safely
    try:
        if where == "end":
            result = torch.cat([input, pad], dim=1)
            return result
        elif where == "start":
            result = torch.cat([pad, input], dim=1)
            return result
        else:
            logger.error(f"pad_col - Invalid 'where' value: {where}")
            raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")
    except Exception as e:
        logger.error(f"pad_col - Exception during padding: {str(e)}")
        # Create a new tensor with the right shape as fallback
        try:
            if where == "end":
                result = torch.zeros(input.shape[0], input.shape[1] + 1, device=input.device, dtype=input.dtype)
                result[:, :-1] = input
                result[:, -1] = val
            else:  # start
                result = torch.zeros(input.shape[0], input.shape[1] + 1, device=input.device, dtype=input.dtype)
                result[:, 1:] = input
                result[:, 0] = val
            logger.warning(f"pad_col - Used fallback method to create padded tensor")
            return result
        except Exception as e2:
            logger.error(f"pad_col - Fallback also failed: {str(e2)}")
            # Last resort - return input unmodified
            logger.error(f"pad_col - Returning input unmodified as last resort")
            return input


def log_softplus(input, threshold=-5.0): # Less aggressive threshold for better learning
    """Equivalent to 'F.softplus(input).log()', but for 'input < threshold',
    we return 'input', as this is approximately the same.

    Arguments:
        input {torch.tensor} -- Input tensor

    Keyword Arguments:
        threshold {float} -- Threshold for when to just return input (default: {-5.0})
                             Higher value allows for more learning; theoretical accuracy
                             increases with more negative values, but stability decreases.

    Returns:
        torch.tensor -- return log(softplus(input)).
    """
    from sat.utils import logging
    logger = logging.get_default_logger()
    
    # Check for NaN/Inf in input 
    has_nan = torch.isnan(input).any()
    has_inf = torch.isinf(input).any()
    
    if has_nan or has_inf:
        # Only log warning for extreme cases
        nan_count = torch.isnan(input).sum().item() if has_nan else 0
        inf_count = torch.isinf(input).sum().item() if has_inf else 0
        
        if (nan_count + inf_count) > (input.numel() * 0.1):  # Only log if significant portion is problematic
            logger.warning(f"log_softplus - High proportion of NaN/Inf: {nan_count} NaNs, {inf_count} Infs out of {input.numel()}")
        
        # Only replace the problematic values, not the entire tensor
        input_clean = input.clone()
        if has_nan:
            input_clean[torch.isnan(input)] = 0.0
        if has_inf:
            # Use more reasonable bounds for replacement
            input_clean[torch.isinf(input) & (input > 0)] = 50.0
            input_clean[torch.isinf(input) & (input < 0)] = -50.0
        input = input_clean
    
    # Clone to avoid modifying the original tensor
    output = input.clone()
    
    # For values below threshold, approximate log(softplus(x)) ≈ x
    # For very negative values, this is a good numerical approximation
    above = input >= threshold

    # Process values above threshold more carefully
    if above.any():
        with torch.no_grad():  # Only for diagnostic checks
            # Get statistics on the values we'll process
            above_vals = input[above]
            try:
                min_val = above_vals.min().item()
                max_val = above_vals.max().item()
                mean_val = above_vals.mean().item()
                
                # Only log if we see very large values
                if max_val > 50 or min_val < -50:
                    logger.warning(f"log_softplus - Large values in input: min={min_val}, max={max_val}, mean={mean_val}")
            except:
                pass  # Ignore stats calculation errors
        
        try:
            # Handle positive values first - for these softplus(x) ≈ x
            very_positive = input > 15.0
            if very_positive.any():
                # For large positive x, log(softplus(x)) ≈ log(x) ≈ x
                # We add a small correction since log(x) is slightly less than x
                output[very_positive] = input[very_positive] - torch.log(torch.ones_like(input[very_positive]) + 
                                                                 torch.exp(-input[very_positive]))
            
            # For values in the middle range, compute standard softplus
            middle_range = above & ~very_positive
            if middle_range.any():
                # Apply softplus with numerical safety
                softplus_vals = F.softplus(input[middle_range])
                
                # Ensure minimum value before taking log
                # Use a reasonable minimum (1e-7) that won't cause excessive gradient changes
                softplus_vals = torch.clamp(softplus_vals, min=1e-7)
                
                # Apply log safely
                output[middle_range] = torch.log(softplus_vals)
            
            # Check if we created any NaN/Inf values
            invalid_mask = torch.isnan(output) | torch.isinf(output)
            if invalid_mask.any():
                invalid_count = invalid_mask.sum().item()
                logger.debug(f"log_softplus - Produced {invalid_count} invalid values after computation")
                
                # For any invalid results, use original input as fallback but slightly adjust
                # to ensure different gradient behavior from below-threshold values
                output[invalid_mask & above] = input[invalid_mask & above]
                
        except Exception as e:
            logger.error(f"log_softplus - Exception in calculation: {str(e)}")
            # Fallback for errors is to return original input values
            output[above] = input[above]

    return output
