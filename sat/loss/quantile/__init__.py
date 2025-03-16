"""Quantile loss functions for regression analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from sat.loss.quantile.quantile import QuantileLoss
from sat.loss.quantile.smooth_quantile import SmoothQuantileLoss

__all__ = ["QuantileLoss", "SmoothQuantileLoss"]
