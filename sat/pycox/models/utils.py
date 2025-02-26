"""utils from pycox copied here to reduce dependencies"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torch.nn.functional as F


def pad_col(input, val=0, where="end"):
    """Addes a column of `val` at the start of end of `input`."""
    if len(input.shape) != 2:
        raise ValueError(f"Only works for `phi` tensor that is 2-D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
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
    output[above] = F.softplus(input[above]).log()
    return output
