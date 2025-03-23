"""Mismatch Metrics."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import datasets
import torch

import evaluate

from sat.utils import logging
from sat.models.tasks.loss import L1Loss
from sat.models.tasks.heads import TaskOutput

logger = logging.get_default_logger()


class L1(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=self._get_feature_types(),
            reference_urls=[],
        )

    def _get_feature_types(self):
        return datasets.Features(
            {
                "predictions": datasets.Sequence(datasets.Value("float")),
                "references": datasets.Sequence(datasets.Value("float")),
            }
        )

    def _compute(
        self,
        references,
        predictions,
        training_set,
        importance_sample_weights,
        l1_type,
        num_events,
        per_event=False,
    ):
        l1_loss = L1Loss(
            training_set=training_set,
            importance_sample_weights=importance_sample_weights,
            l1_type=l1_type,
            num_events=num_events,
        )

        references = torch.Tensor(references)
        preds = torch.Tensor(predictions)  # batch x variables

        event_losses = []
        if per_event:
            for event in range(num_events):
                event_loss = l1_loss.l1(preds, references, event)
                event_losses.append(event_loss)

        output = TaskOutput(predictions=preds)
        loss = l1_loss(output, references)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Computed hazard loss: {loss}")

        return loss, event_losses
