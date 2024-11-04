"""TTE Ranking Count Metrics."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import datasets
import torch

from logging import DEBUG, ERROR

import evaluate

from sat.utils import logging
from sat.models.utils import get_device

logger = logging.get_default_logger()


class TTERankingCounts(evaluate.Metric):
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

    def _compute(self, references, predictions, num_events):
        references = torch.Tensor(references)
        preds = torch.Tensor(predictions)  # batch x variables

        count = 0
        event_counts = []
        for event in range(num_events):
            events = references[:, (1 * self.cfg.data.num_events + event)].to(bool)
            durations = references[:, (3 * self.cfg.data.num_events + event)]
            censor_indicators = ~events
            predictions = preds[:, event]
            errors = durations[censor_indicators] - predictions[censor_indicators]
            event_count = torch.sum(errors > 0)
            count += event_count
            event_counts.append(event_count)

        logger.debug(f"Computed TTE Ranking Counts: {count}")

        return count, event_counts
