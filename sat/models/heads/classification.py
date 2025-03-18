"""Classification Task heads for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import hydra
import torch

from sat.models.nets import CauseSpecificNet, CauseSpecificNetCompRisk, SimpleMLP
from sat.utils import logging

from .config import EventClassificationTaskConfig
from .base import SurvivalPreTrainedModel
from .output import TaskOutput


logger = logging.get_default_logger()


class EventClassificationTaskHead(SurvivalPreTrainedModel):
    config_class = EventClassificationTaskConfig

    def __init__(self, config: EventClassificationTaskConfig):
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
        logger.debug(f"Instantiate the loss {loss}")
        self.loss = hydra.utils.instantiate(loss)

    def forward(self, sequence_output, labels=None, **kwargs):
        logits = self.nets(sequence_output)  # num events x batch x 1
        predictions = torch.sigmoid(logits)

        loss = None
        output = TaskOutput(loss=loss, logits=logits, predictions=predictions)
        if labels is not None:
            loss = self.loss(
                output,
                labels,
            )
            output.loss = loss

        return output
