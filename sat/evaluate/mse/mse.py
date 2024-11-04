"""Mismatch Metrics."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import datasets
import torch

import evaluate

from sat.utils import logging
from sat.models.tasks.loss import MSELoss
from sat.models.tasks.heads import TaskOutput
from sat.models.utils import get_device

logger = logging.get_default_logger()


class MSEScores(evaluate.Metric):
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
        l2_type,
        num_events,
        per_event=False,
    ):
        mse_loss = MSELoss(
            training_set=training_set,
            importance_sample_weights=importance_sample_weights,
            l2_type=l2_type,
            num_events=num_events,
        )

        references = torch.Tensor(references)
        preds = torch.Tensor(predictions)  # batch x variables
        num_events = preds.shape[1]

        event_losses = []
        if per_event:
            for event in range(num_events):
                event_loss = mse_loss.mse(preds, references, event)
                event_losses.append(event_loss)

        output = TaskOutput(predictions=preds)
        loss = mse_loss(output, references)
        logger.debug(f"Computed MSE loss: {loss}")

        return loss, event_losses
