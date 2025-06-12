"""Brier Score Metric"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import datasets
import evaluate
import numpy as np
import torch
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw

from sat.utils import logging

logger = logging.get_default_logger()


def to_tensor(arr, dtype=torch.float32):
    """Convert array to tensor with specified dtype"""
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype)
    return torch.as_tensor(arr, dtype=dtype)


_CITATION = """
@article{Uno2011OnTC,
  title={On the C‐statistics for evaluating overall adequacy of risk prediction procedures with censored survival data},
  author={Hajime Uno and Tianxi Cai and Michael J. Pencina and Ralph B. D'Agostino and L. J. Wei},
  journal={Statistics in Medicine},
  year={2011},
  volume={30}
}

@article{torchsurv2023,
  title={TorchSurv: A PyTorch Package for Accelerated Development of Survival Analysis Methods},
  author={Zhu, Peng and Kong, Youben and Wever, Marcel and Schulz, Christian and Zitnik, Marinka and Cheng, Jie-Xun and Bachl, Fabian and Theis, Fabian J. and Ray, Pratyush},
  journal={arXiv preprint arXiv:2302.03686},
  year={2023}
}"""

_DESCRIPTION = """
Concordance Index for right-censored data based on inverse probability of censoring weights (IPCW), implemented using torchsurv.

This implementation is based on the C-statistics proposed by Uno et al. (2011) for evaluating the
overall performance of risk prediction procedures with censored survival data. It provides an unbiased
estimate that is free of censoring and does not depend on the censoring distribution in the test data.

The metric calculates C-index at multiple time horizons (duration cuts) and provides both individual
time point C-indices and an integrated C-index weighted by the number of events at each time horizon.

Unlike traditional C-index which cannot handle censored observations properly, the IPCW-based C-index
properly accounts for censoring by weighting each comparable pair of observations using the inverse
probability of censoring weight at the observed event time.

Implementation is based on torchsurv's ConcordanceIndex class which provides GPU-acceleration
for faster computation on large datasets.

References:
    Uno, H., Cai, T., Pencina, M. J., D'Agostino, R. B., & Wei, L. J. (2011).
    "On the C-statistics for evaluating overall adequacy of risk prediction
    procedures with censored survival data".
    Statistics in Medicine, 30(10), 1105–1117.

    Zhu, P. et al. (2023). "TorchSurv: A PyTorch Package for Accelerated Development of
    Survival Analysis Methods". arXiv preprint arXiv:2302.03686.
"""

_KWARGS_DESCRIPTION = """

Parameters
----------
references : array-like
    Ground truth survival data containing event indicators and times.
    Can be provided as a list of tuples (event, time) or as a list of dictionaries
    with keys 'e' for event indicator and 't' for time.
    Event indicators should be boolean (True for events, False for censored).
    Times should be positive floats representing time-to-event or time-to-censoring.

predictions : array-like
    Model predictions representing risk scores.
    Higher values should indicate higher risk (shorter survival time).
    Can be a single risk score per subject or multiple values for different time horizons.
    Shape can be (n_samples,) for a single prediction per subject or
    (n_samples, n_horizons) for risk scores at multiple time points.

duration_cuts : array-like
    Time horizons at which to evaluate the C-index.
    Each value represents a specific time point for evaluation.

per_horizon : bool, optional, default=False
    If True, return C-indices at each time horizon specified in duration_cuts.
    If False, only return the integrated C-index.

Returns
-------
integrated_cindex : float
    The integrated C-index across all time horizons, weighted by the number of events at each time point.
    Range is typically [0, 1] where 0.5 represents random prediction and 1.0 perfect prediction.

cindeces_np : array-like or empty list
    If per_horizon=True, returns a numpy array containing C-indices for each time horizon in duration_cuts.
    If per_horizon=False, returns an empty list.

"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CIIPCW(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._get_feature_types(),
            reference_urls=[
                "https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.brier_score.html"
            ],
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
        duration_cuts,
        per_horizon=False,
    ):
        # Handle both dictionary and tuple formats for references
        if isinstance(references[0], dict):
            # For dictionary format (with 'e' and 't' keys)
            e_test_np = np.array([x["e"] for x in references])
            t_test_np = np.array([x["t"] for x in references])
        else:
            # For tuple format (event, time)
            e_test_np = np.array([x[0] for x in references])
            t_test_np = np.array([x[1] for x in references])

        # Convert test data to tensors
        e_test = to_tensor(e_test_np, dtype=torch.bool)
        t_test = to_tensor(t_test_np, dtype=torch.float32)

        # Convert predictions to tensor
        preds = to_tensor(predictions)

        # Add detailed shape diagnostics for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Tensor shapes before C-Index calculation:")
            logger.debug(f"  Test events (e_test): {e_test.shape}")
            logger.debug(f"  Test times (t_test): {t_test.shape}")
            logger.debug(f"  Predictions (preds): {preds.shape}")
            logger.debug(f"  Eval times: {len(duration_cuts)}")

        # Initialize C-Index calculator
        c_index_calculator = ConcordanceIndex()
        ipcw = get_ipcw(e_test, t_test)

        # Calculate C-Index for each duration cut
        cindeces = []
        total_events = 0
        event_counts = []

        for i, tau in enumerate(duration_cuts):
            try:
                # Extract predictions for this time point
                if preds.dim() > 1 and i < preds.shape[1]:
                    pred_at_time = preds[:, i]
                else:
                    pred_at_time = preds

                # Calculate number of events before this time for weighting
                event_mask = (e_test_np == 1) & (t_test_np <= tau)
                event_count = event_mask.sum()
                event_counts.append(event_count)
                total_events += event_count

                # Calculate C-Index for this timepoint
                cindex = c_index_calculator(
                    event=e_test,
                    time=t_test,
                    estimate=pred_at_time,
                    weight=ipcw,
                )

                cindeces.append(float(cindex.item()))

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"C-index at tau={tau}: {cindeces[-1]} with {event_count} events"
                    )

            except Exception as e:
                logger.error(f"Error calculating C-Index at tau={tau}: {str(e)}")
                cindeces.append(float("nan"))
                event_counts.append(0)

        # Calculate integrated/weighted C-index
        if total_events > 0:
            # Weight by number of events at each time point
            weights = np.array(event_counts) / total_events
            integrated_cindex = float(np.sum(np.array(cindeces) * weights))
        else:
            integrated_cindex = 0.5  # Default for no events

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Computed C-indices: {cindeces}")
            logger.debug(f"Event counts: {event_counts}")
            logger.debug(f"Integrated C-index: {integrated_cindex}")

        # Convert to numpy arrays for the evaluate API
        if per_horizon:
            cindeces_np = np.array(cindeces)
        else:
            cindeces_np = []

        return integrated_cindex, cindeces_np
