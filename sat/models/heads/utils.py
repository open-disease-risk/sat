"""Utilities for the heads for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import torch


def pad_col(input, val=0, where="end"):
    """Adds a column of `val` at the start or end of `input` with optimized memory usage.

    This optimized version avoids unnecessary tensor allocations.
    """
    if len(input.shape) != 3:
        raise ValueError(f"Only works for `phi` tensor that is 3-D.")

    # Get shape for new tensor
    batch_size, num_events, seq_len = input.shape
    new_shape = (batch_size, num_events, seq_len + 1)

    # Create output tensor directly with correct size (avoids intermediate allocations)
    result = torch.zeros(new_shape, dtype=input.dtype, device=input.device)

    # Fill with data efficiently
    if where == "end":
        result[:, :, :seq_len].copy_(input)
        if val != 0:
            result[:, :, -1] = val
    elif where == "start":
        result[:, :, 1:].copy_(input)
        if val != 0:
            result[:, :, 0] = val
    else:
        raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

    return result
