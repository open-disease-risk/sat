"""Regression loss functions."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from sat.loss.regression.l1 import L1Loss
from sat.loss.regression.mse import MSELoss

__all__ = ["L1Loss", "MSELoss"]
