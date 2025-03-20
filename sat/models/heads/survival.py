"""Survival Task heads for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import hydra
import torch

import torch.nn.functional as F

from sat.models.nets import CauseSpecificNet, CauseSpecificNetCompRisk, SimpleMLP
from sat.utils import logging

from .config import SurvivalConfig
from .base import SurvivalTask
from .output import SAOutput
from .utils import pad_col

logger = logging.get_default_logger()


class SurvivalTaskHead(SurvivalTask):
    config_class = SurvivalConfig

    def __init__(self, config: SurvivalConfig):
        super().__init__(config)

        if self.config.num_events > 1:
            self.nets = CauseSpecificNetCompRisk(
                in_features=self.config.num_features,
                shared_intermediate_size=self.config.intermediate_size,
                shared_num_hidden_layers=self.config.num_hidden_layers,
                indiv_intermediate_size=self.config.indiv_intermediate_size,
                indiv_num_hidden_layers=self.config.indiv_num_hidden_layers,
                bias=self.config.bias,
                batch_norm=self.config.batch_norm,
                dropout=self.config.hidden_dropout_prob,
                out_features=self.config.num_labels,
                num_events=self.config.num_events,
            )
        else:
            self.nets = CauseSpecificNet(
                in_features=self.config.num_features,
                intermediate_size=self.config.intermediate_size,
                num_hidden_layers=self.config.num_hidden_layers,
                bias=self.config.bias,
                batch_norm=self.config.batch_norm,
                dropout=self.config.hidden_dropout_prob,
                out_features=self.config.num_labels,
                num_events=self.config.num_events,
            )

        # Initialize weights when created as a standalone model (not as part of MTL)
        # If we're part of an MTL model, the MTL model will handle calling post_init
        # We can detect if we're standalone by checking our parent class
        if self.__class__.__name__ == "SurvivalTaskHead":
            logger.debug("Standalone SurvivalTaskHead - initializing weights")
            self.post_init()
        else:
            logger.debug(
                "SurvivalTaskHead created as part of MTL - will be initialized by MTL"
            )

        loss = config.loss[config.model_type]
        logger.debug(f"Instantiate the loss {loss}")
        self.loss = hydra.utils.instantiate(loss)

    def forward(self, sequence_output, labels=None, **kwargs):
        logits = self.nets(sequence_output)  # num events x batch x duration cuts
        hazard = F.softplus(logits)
        hazard = pad_col(hazard, where="start")
        # Optimized tensor operations: fuse cumsum+mul+exp into a single operation
        surv = (
            -hazard.cumsum(dim=2)
        ).exp()  # More efficient than cumsum().mul(-1).exp()
        # Use in-place operation to create risk from survival
        risk = torch.ones_like(surv).sub_(
            surv
        )  # Equivalent to 1.0 - surv but more efficient

        output = SAOutput(
            loss=None,
            logits=logits,
            hazard=hazard,
            risk=risk,
            survival=surv,
            hidden_states=sequence_output,
        )

        if labels is not None:
            logger.debug(
                f"Computing loss with logits {logits[0].shape} and labels {labels.shape}"
            )
            output.loss = self.loss(
                output,
                labels,
            )

        return output
