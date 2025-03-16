"""Loss functions for survival analysis

This module provides various loss functions for survival analysis tasks.
It has been refactored from a monolithic file into a more modular structure
while maintaining backward compatibility with existing code.
"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

# Import base classes
from sat.loss.base import Loss

# Import meta loss
from sat.loss.meta import MetaLoss

# Import classification losses
from sat.loss.classification.cross_entropy import CrossEntropyLoss
from sat.loss.classification.focal_loss import FocalLoss

# Import contrastive losses
from sat.loss.contrastive.surv_contrastive import SurvContrastiveLoss

# Import quantile losses
from sat.loss.quantile.quantile import QuantileLoss
from sat.loss.quantile.smooth_quantile import SmoothQuantileLoss

# Import ranking losses
from sat.loss.ranking.listmle import ListMLERankingLoss
from sat.loss.ranking.neuralsort import NeuralSortLoss
from sat.loss.ranking.ranknet import RankNetLoss
from sat.loss.ranking.soap import SOAPLoss
from sat.loss.ranking.survrnc import SurvRNCLoss

# Import regression losses
from sat.loss.regression.l1 import L1Loss
from sat.loss.regression.mse import MSELoss

# Import survival losses
from sat.loss.survival.deephit import DeepHitLoss
from sat.loss.survival.dsm import DSMLoss
from sat.loss.survival.mismatch import MismatchLoss
from sat.loss.survival.nllpc_hazard import SATNLLPCHazardLoss
from sat.loss.survival.ranking import RankingLoss

# For backward compatibility, re-export all losses at the top level
__all__ = [
    # Base classes
    "Loss",
    # Meta loss
    "MetaLoss",
    # Classification losses
    "CrossEntropyLoss",
    "FocalLoss",
    # Contrastive losses
    "SurvContrastiveLoss",
    # Quantile losses
    "QuantileLoss",
    "SmoothQuantileLoss",
    # Ranking losses
    "BaseRankingLoss",
    "ListMLERankingLoss",
    "NeuralSortLoss",
    "RankNetLoss",
    "SOAPLoss",
    "SurvRNCLoss",
    # Regression losses
    "L1Loss",
    "MSELoss",
    # Survival losses
    "DeepHitLoss",
    "DSMLoss",
    "MismatchLoss",
    "RankingLoss",
    "SATNLLPCHazardLoss",
]
