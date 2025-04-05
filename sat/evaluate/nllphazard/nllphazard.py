"""Mismatch Metrics."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import datasets
import evaluate
import torch

from sat.models.tasks.heads import SAOutput
from sat.models.tasks.loss import SATNLLPCHazardLoss
from sat.utils import logging

logger = logging.get_default_logger()


class NLLPHazardScores(evaluate.Metric):
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

    def _compute(
        self,
        references,
        predictions,
        importance_sampling_weights,
        per_event=False,
    ):
        hazard_loss = SATNLLPCHazardLoss(
            importance_sample_weights=importance_sampling_weights
        )

        references = torch.Tensor(references)
        logits = torch.permute(torch.Tensor(predictions)[:, 0], (1, 0, 2))
        num_events = logits.shape[0]
        logits = [logits[e] for e in range(num_events)]

        event_losses = []
        if per_event:
            for event in range(num_events):
                event_loss = hazard_loss.nllp_hazard_loss(logits, references, event)
                event_losses.append(event_loss)

        output = SAOutput(logits=logits)
        loss = hazard_loss(output, references)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Computed hazard loss: {loss}")

        return loss, event_losses
