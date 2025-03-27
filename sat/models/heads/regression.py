"""Regression Task heads for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import hydra

import torch
from torch import nn

from sat.models.nets import CauseSpecificNet, CauseSpecificNetCompRisk, SimpleMLP
from sat.utils import logging
import logging as py_logging

from .base import BaseConfig, RegressionTask
from .output import TaskOutput


class EventDurationTaskConfig(BaseConfig):
    model_type = "sat-transformer-event-duration"

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


logger = logging.get_default_logger()


class EventDurationTaskHead(RegressionTask):
    config_class = EventDurationTaskConfig

    def __init__(self, config: EventDurationTaskConfig):
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

        loss = config.loss[config.model_type]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Instantiate the loss {loss}")
        self.loss = hydra.utils.instantiate(loss)

    def forward(self, sequence_output, labels=None, **kwargs):
        logits = nn.ReLU()(self.nets(sequence_output))
        predictions = torch.squeeze(logits, dim=2)  # num events x batch x predictions

        loss = None
        output = TaskOutput(loss=loss, logits=logits, predictions=predictions)
        if labels is not None:
            loss = self.loss(
                output,
                labels,
            )
            output.loss = loss

        return output
