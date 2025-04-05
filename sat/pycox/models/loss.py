"""Loss functions from pycox copied here to reduce dependencies"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torch.nn.functional as F
from torch import Tensor

from sat.pycox.models import utils


def _reduction(loss: Tensor, reduction: str = "mean") -> Tensor:
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    raise ValueError(
        f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'."
    )


def nll_pc_hazard_loss(
    phi: Tensor,
    idx_durations: Tensor,
    events: Tensor,
    interval_frac: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Negative log-likelihood of the PC-Hazard parametrization model [1].

    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
        interval_frac {torch.tensor} -- Fraction of last interval before event/censoring.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.

    Returns:
        torch.tensor -- The negative log-likelihood.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    if events.dtype is torch.bool:
        events = events.float()
    idx_durations = idx_durations.view(-1, 1)
    events = events.view(-1)
    interval_frac = interval_frac.view(-1)

    keep = idx_durations.view(-1) >= 0
    phi = phi[keep, :]
    idx_durations = idx_durations[keep, :]
    events = events[keep]
    interval_frac = interval_frac[keep]

    # log_h_e = F.softplus(phi.gather(1, idx_durations).view(-1)).log().mul(events)
    log_h_e = utils.log_softplus(phi.gather(1, idx_durations).view(-1)).mul(events)

    # Apply softplus with numerical stability
    haz = F.softplus(phi)
    # Ensure no zeros in hazard that could cause issues
    haz = torch.clamp(haz, min=1e-8)

    # Ensure interval_frac is valid (no NaN, Inf) before multiplication
    safe_interval_frac = torch.nan_to_num(
        interval_frac, nan=0.0, posinf=1.0, neginf=0.0
    )
    safe_interval_frac = torch.clamp(
        safe_interval_frac, min=0.0, max=1.0
    )  # Ensure it's between 0 and 1
    scaled_h_e = haz.gather(1, idx_durations).view(-1).mul(safe_interval_frac)
    haz = utils.pad_col(haz, where="start")
    sum_haz = haz.cumsum(1).gather(1, idx_durations).view(-1)

    # Compute loss with added stability
    loss = -log_h_e.sub(scaled_h_e).sub(sum_haz)

    # Check for invalid values and replace with reasonable defaults
    invalid_mask = torch.isnan(loss) | torch.isinf(loss)
    if invalid_mask.any():
        # Log warning for debugging
        print(
            f"Warning: Found {invalid_mask.sum().item()} invalid loss values. Replacing with zeros."
        )
        loss[invalid_mask] = 0.0
    return _reduction(loss, reduction)


class _Loss(torch.nn.Module):
    """Generic loss function.

    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction


class NLLPCHazardLoss(_Loss):
    def forward(
        self,
        phi: Tensor,
        idx_durations: Tensor,
        events: Tensor,
        interval_frac: Tensor,
        reduction: str = "mean",
    ) -> Tensor:
        """Negative log-likelihood of the PC-Hazard parametrization model.
        See `loss.nll_pc_hazard_loss` for details.

        Arguments:
            reduction {string} -- How to reduce the loss.
                'none': No reduction.
                'mean': Mean of tensor.
                'sum: sum.

        Returns:
            torch.tensor -- The negative log-likelihood loss.
        """
        return nll_pc_hazard_loss(
            phi, idx_durations, events, interval_frac, self.reduction
        )
