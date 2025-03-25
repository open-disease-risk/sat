"""Mismatch Metrics."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import datasets
import torch

import evaluate

from sat.utils import logging
from sat.models.tasks.loss import MismatchLoss
from sat.models.tasks.heads import SAOutput

logger = logging.get_default_logger()


class MismatchScores(evaluate.Metric):
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
                "predictions": datasets.Sequence(
                    datasets.Sequence(datasets.Sequence(datasets.Value("float")))
                ),
                "references": datasets.Sequence(datasets.Value("float")),
            }
        )

    def _compute(self, references, predictions, duration_cuts, max_time):
        device = references.device
        mismatch_loss = MismatchLoss(duration_cuts, max_time)

        logits = torch.permute(
            torch.Tensor(predictions, device=device)[:, 0], (1, 0, 2)
        )
        num_events = logits.shape[0]
        logits = [logits[e].to(device) for e in range(num_events)]

        output = SAOutput(logits=logits)
        loss = mismatch_loss(output, references)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Computed hazard loss: {loss}")

        return loss
