"""One-Calibration Metric.

cite: https://arxiv.org/abs/1811.11347
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import datasets

import numpy as np

from scipy.stats import chi2

import evaluate

from sat.utils import logging
from sat.utils.types import NumericArrayLike
from sat.utils.km import (
    KaplanMeier,
    to_array,
)

logger = logging.get_default_logger()


def one_calibration(
    event_times: NumericArrayLike,
    event_indicators: NumericArrayLike,
    predictions: NumericArrayLike,
    time: float,
    bins: int = 10,
) -> dict:
    event_times = to_array(event_times)
    event_indicators = to_array(event_indicators, to_boolean=True)
    predictions = 1 - to_array(predictions)

    prediction_order = np.argsort(-predictions)
    predictions = predictions[prediction_order]
    event_times = event_times[prediction_order]
    event_indicators = event_indicators[prediction_order]

    # Can't do np.mean since split array may be of different sizes.
    binned_event_times = np.array_split(event_times, bins)
    binned_event_indicators = np.array_split(event_indicators, bins)
    probability_means = [np.mean(x) for x in np.array_split(predictions, bins)]
    hosmer_lemeshow = 0
    observed_probabilities = list()
    expected_probabilities = list()
    for b in range(bins):
        prob = probability_means[b]
        if prob == 1.0:
            raise ValueError(
                "One-Calibration is not well defined: the risk"
                f"probability of the {b}th bin was {prob}."
            )
        km_model = KaplanMeier(binned_event_times[b], binned_event_indicators[b])
        event_probability = 1 - km_model.predict(time)
        bin_count = len(binned_event_times[b])
        hosmer_lemeshow += (bin_count * event_probability - bin_count * prob) ** 2 / (
            bin_count * prob * (1 - prob)
        )
        observed_probabilities.append(event_probability)
        expected_probabilities.append(prob)

    return dict(
        p_value=1 - chi2.cdf(hosmer_lemeshow, bins - 1),
        observed=observed_probabilities,
        expected=expected_probabilities,
    )


class OneCalibration(evaluate.Metric):
    def _info(self):
        if self.config_name not in [
            "survival",
            "classification",
        ]:
            raise KeyError(
                "You should supply a configuration name selected in "
                '["survival", "classification"]'
            )

        return evaluate.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=self._get_feature_types(),
            reference_urls=[],
        )

    def _get_feature_types(self):
        if self.config_name == "survival":
            return datasets.Features(
                {
                    "predictions": datasets.Sequence(
                        datasets.Sequence(datasets.Value("float"))
                    ),
                    "references": datasets.Sequence(datasets.Value("float")),
                }
            )
        elif self.config_name == "classification":
            return datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float")),
                    "references": datasets.Sequence(datasets.Value("float")),
                }
            )
        else:
            raise ValueError(
                "You should supply a configuration name selected in "
                '["survival", "classification"]'
            )

    def _compute(
        self, references, predictions, bins, num_events, duration_cuts, event_time_thr
    ):
        references = np.array(references)
        predictions = np.array(predictions)
        horizons = np.arange(1, len(duration_cuts) + 1) * 1.0 / (len(duration_cuts) + 1)

        metric_dict = {}
        for e in range(num_events):
            event_indicator = references[:, (1 * self.cfg.data.num_events + e)].to(bool)
            durations = references[:, (3 * self.cfg.data.num_events + e)]
            if self.config_name == "survival":
                # iterate over all labels except the last one
                for j in range(len(duration_cuts)):
                    metric_dict[
                        f"1-calibration_{e}th_event_{horizons[j]}"
                    ] = one_calibration(
                        durations,
                        event_indicator,
                        predictions[:, e, j],
                        time=duration_cuts[j],
                        bins=bins,
                    )
            elif self.config_name == "classification":
                metric_dict[f"1-calibration_{e}th_event"] = one_calibration(
                    durations,
                    event_indicator,
                    predictions[:, e],
                    time=event_time_thr,
                    bins=bins,
                )

        return metric_dict
