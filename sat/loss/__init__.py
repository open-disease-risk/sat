"""Initialization of the package"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from .base import Loss, RankingLoss
from .meta import MetaLoss

from .classification.bce import CrossEntropyLoss

from .ranking.sample import SampleRankingLoss
from .ranking.multievent import MultiEventRankingLoss

from .regression.l1 import L1Loss
from .regression.mse import MSELoss
from .regression.quantile import QuantileLoss

from .survival.nllpchazard import SATNLLPCHazardLoss
from .survival.focal import SurvivalFocalLoss
from .survival.deephit import (
    DeepHitLikelihoodLoss,
    DeepHitCalibrationLoss,
)
from .survival.dsm import DSMLoss

__all__ = [
    "Loss",
    "RankingLoss",
    "MetaLoss",
    "CrossEntropyLoss",
    "SampleRankingLoss",
    "MultiEventRankingLoss",
    "L1Loss",
    "MSELoss",
    "QuantileLoss",
    "SATNLLPCHazardLoss",
    "SurvivalFocalLoss",
    "DeepHitLikelihoodLoss",
    "DeepHitCalibrationLoss",
    "DSMLoss",
]
