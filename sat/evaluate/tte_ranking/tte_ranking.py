"""TTE Ranking Count Metrics."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import datasets
import torch

import evaluate

from sat.utils import logging

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

        # Create indices for events and durations - compute once
        event_indices = torch.tensor(
            [1 * self.cfg.data.num_events + e for e in range(num_events)]
        )
        duration_indices = torch.tensor(
            [3 * self.cfg.data.num_events + e for e in range(num_events)]
        )

        # Extract all events and durations at once - more efficient
        all_events = references[:, event_indices].to(bool)
        all_durations = references[:, duration_indices]

        # Create censor mask (inverted events mask)
        censor_mask = ~all_events  # [batch_size, num_events]

        # Vectorized error calculation
        event_counts = []
        count = 0

        # Still process event by event, but with more vectorized operations
        for event in range(num_events):
            # Get durations and predictions for this event
            event_durations = all_durations[:, event]
            event_predictions = preds[:, event]
            event_censor_mask = censor_mask[:, event]

            # Calculate errors for censored samples
            if event_censor_mask.any():
                # Vectorized subtraction for only censored samples
                errors = (
                    event_durations[event_censor_mask]
                    - event_predictions[event_censor_mask]
                )
                # Count positive errors
                event_count = torch.sum(errors > 0)
                count += event_count
                event_counts.append(event_count)
            else:
                # No censored samples for this event
                event_counts.append(torch.tensor(0))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Computed TTE Ranking Counts: {count}")

        return count, event_counts
