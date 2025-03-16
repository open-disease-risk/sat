"""Classification loss functions."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from sat.loss.classification.cross_entropy import CrossEntropyLoss
from sat.loss.classification.focal_loss import FocalLoss

__all__ = ["CrossEntropyLoss", "FocalLoss"]
