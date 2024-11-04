"""Mismatch Metrics."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import datasets
import torch

from logging import DEBUG, ERROR

import evaluate

from sat.utils import logging
from sat.models.tasks.loss import CrossEntropyLoss
from sat.models.tasks.heads import TaskOutput
from sat.models.utils import get_device

logger = logging.get_default_logger()


class CrossEntropyScores(evaluate.Metric):
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
        event_time_thr,
        num_events,
        training_set,
        l_type,
        importance_sample_weights,
        per_event=False,
    ):
        ce_loss = CrossEntropyLoss(
            event_time_thr=event_time_thr,
            num_events=num_events,
            training_set=training_set,
            l_type=l_type,
            importance_sample_weights=importance_sample_weights,
        )

        references = torch.Tensor(references)
        preds = torch.Tensor(predictions)  # batch x variables
        num_events = preds.shape[1]

        event_losses = []
        if per_event:
            for event in range(num_events):
                event_loss = ce_loss.ce(preds, references, event)
                event_losses.append(event_loss)

        output = TaskOutput(predictions=preds)
        loss = ce_loss(output, references)
        logger.debug(f"Computed Cross Entropy loss: {loss}")

        return loss, event_losses
