"""Survival Task heads for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import hydra
import torch
import torch.nn.functional as F

from sat.models.nets import CauseSpecificNet, CauseSpecificNetCompRisk
from sat.utils import logging

from .base import BaseConfig, SurvivalTask
from .output import SAOutput
from .utils import pad_col


class SurvivalConfig(BaseConfig):
    model_type = "sat-transformer"

    def __init__(
        self,
        num_inputs: int = 32,
        intermediate_size: int = 64,
        num_hidden_layers: int = 0,
        indiv_intermediate_size: int = 64,
        indiv_num_hidden_layers: int = 0,
        batch_norm: bool = True,
        hidden_dropout_prob: float = 0.05,
        bias: bool = True,
        max_time=400,
        duration_cuts=[],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_inputs = num_inputs
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.indiv_intermediate_size = indiv_intermediate_size
        self.indiv_num_hidden_layers = indiv_num_hidden_layers
        self.batch_norm = batch_norm
        self.hidden_dropout_prob = hidden_dropout_prob
        self.bias = bias
        self.max_time = max_time
        self.duration_cuts = duration_cuts


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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Standalone SurvivalTaskHead - initializing weights")
            self.post_init()
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "SurvivalTaskHead created as part of MTL - will be initialized by MTL"
                )

        loss = config.loss[config.model_type]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Instantiate the loss {loss}")
        self.loss = hydra.utils.instantiate(loss)

    def forward(self, sequence_output, labels=None, **kwargs):
        # Compute network output
        logits = self.nets(sequence_output)  # num events x batch x duration cuts

        # Apply activation and add padding column
        hazard = F.softplus(logits)
        hazard = pad_col(hazard, where="start")

        # Optimized single-op tensor calculation for survival
        cumsum = hazard.cumsum(dim=2)
        surv = torch.exp(-cumsum)
        risk = 1.0 - surv

        # Create output container
        output = SAOutput(
            loss=None,
            logits=logits,
            hazard=hazard,
            risk=risk,
            survival=surv,
            hidden_states=sequence_output,
        )

        # Compute loss if labels are provided
        if labels is not None:
            # Debug logging can be expensive in production - consider disabling
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Computing loss with logits {logits[0].shape} and labels {labels.shape}"
                )
            output.loss = self.loss(output, labels)

        return output
