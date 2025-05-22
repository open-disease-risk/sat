"""Initialization of the package"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from .base import Loss, RankingLoss
from .classification.bce import CrossEntropyLoss
from .meta import MetaLoss
from .ranking.multievent import MultiEventRankingLoss
from .ranking.sample import SampleRankingLoss
from .regression.l1 import L1Loss
from .regression.mse import MSELoss
from .regression.quantile import QuantileLoss
from .survival.deephit import DeepHitCalibrationLoss, DeepHitLikelihoodLoss
from .survival.dsm import DSMLoss
from .survival.focal import SurvivalFocalLoss
from .survival.nllpchazard import SATNLLPCHazardLoss
from .survival.mismatch_rank import IntraEventRankingLoss

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
    "IntraEventRankingLoss",
]
