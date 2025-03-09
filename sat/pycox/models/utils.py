"""utils from pycox copied here to reduce dependencies"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torch.nn.functional as F


def pad_col(input, val=1e-8, where="end"):
    """Adds a column of `val` at the start or end of `input`.

    Uses a small positive default value to ensure numerical stability for hazard calculations.
    """
    if len(input.shape) != 2:
        raise ValueError(f"Only works for `phi` tensor that is 2-D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    else:
        # For hazard calculations, we want to avoid exact zeros
        # Use a small value instead of zero for numerical stability
        pad = pad + 1e-8

    if where == "end":
        return torch.cat([input, pad], dim=1)
    elif where == "start":
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")


def log_softplus(input, threshold=-15.0):
    """Equivalent to 'F.softplus(input).log()', but for 'input < threshold',
    we return 'input', as this is approximately the same.

    Arguments:
        input {torch.tensor} -- Input tensor

    Keyword Arguments:
        threshold {float} -- Treshold for when to just return input (default: {-15.})

    Returns:
        torch.tensor -- return log(softplus(input)).
    """
    output = input.clone()
    above = input >= threshold

    # Add numerical stability for softplus calculation
    if above.any():
        # Add a small epsilon to avoid log(0) issues
        softplus_vals = F.softplus(input[above])
        # Ensure minimum value before taking log
        softplus_vals = torch.clamp(softplus_vals, min=1e-8)
        output[above] = softplus_vals.log()

    return output
