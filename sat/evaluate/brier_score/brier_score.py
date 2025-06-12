"""Brier Score Metric."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import datasets
import evaluate
import numpy as np
import torch
from torchsurv.metrics.brier_score import BrierScore as TorchsurvBrierScore
from torchsurv.stats.ipcw import get_ipcw

from sat.utils import logging

logger = logging.get_default_logger()


def to_tensor(arr, dtype=torch.float32):
    """Convert array to tensor with specified dtype"""
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype)
    return torch.as_tensor(arr, dtype=dtype)


_CITATION = r"""
@article{https://doi.org/10.1002/(SICI)1097-0258(19990915/30)18:17/18<2529::AID-SIM274>3.0.CO;2-5,
author = {Graf, Erika and Schmoor, Claudia and Sauerbrei, Willi and Schumacher, Martin},
title = {Assessment and comparison of prognostic classification schemes for survival data},
journal = {Statistics in Medicine},
}

@article{mogensen2012evaluating,
  title={Evaluating random forests for survival analysis using prediction error curves.},
  author={Mogensen, Ulla B and Ishwaran, Hemant and Gerds, Thomas A},
  journal={Journal of statistical software},
  volume={50},
  pages={1},
  year={2012},
  publisher={NIH Public Access}
}

@article{torchsurv2023,
  title={TorchSurv: A PyTorch Package for Accelerated Development of Survival Analysis Methods},
  author={Zhu, Peng and Kong, Youben and Wever, Marcel and Schulz, Christian and Zitnik, Marinka and Cheng, Jie-Xun and Bachl, Fabian and Theis, Fabian J. and Ray, Pratyush},
  journal={arXiv preprint arXiv:2302.03686},
  year={2023}
}"""

_DESCRIPTION = """
Brier Score for survival prediction, implemented using torchsurv.

The Brier score measures the accuracy of probabilistic predictions by calculating the mean squared error
between predicted survival probabilities and the actual survival status at specific time points.

This implementation calculates both:
1. Time-dependent Brier scores at each specified time horizon (duration cut)
2. Integrated Brier Score (IBS) - the weighted average of Brier scores across all time points

To account for censoring in survival data, Inverse Probability of Censoring Weights (IPCW) are used.
The weights are estimated from the test data to ensure compatibility regardless of dataset sizes.

The implementation leverages torchsurv's BrierScore class which provides GPU-acceleration for faster
computation on large datasets and proper handling of censored observations through IPCW.

References:
    Graf, E., Schmoor, C., Sauerbrei, W., & Schumacher, M. (1999).
    "Assessment and comparison of prognostic classification schemes for survival data".
    Statistics in Medicine, 18(17-18), 2529-2545.

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
    Time horizons at which to evaluate the brier score.
    Each value represents a specific time point for evaluation.

per_horizon : bool, optional, default=False
    If True, return brier scores at each time horizon specified in duration_cuts.
    If False, only return the integrated brier score.

Returns
-------
integrated_bs : float
    The integrated Brier score (IBS) across all time horizons, weighted by the number of events at each time point.
    Range is typically [0, 1] where 0 represents perfect prediction and higher values indicate worse performance.
    For binary outcomes, a random prediction would yield around 0.25.

brier_scores_np : array-like or empty list
    If per_horizon=True, returns a numpy array containing brier scores for each time horizon in duration_cuts.
    If per_horizon=False, returns an empty list.

"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BrierScore(evaluate.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        # Handle both dictionary and tuple formats that might come from HF
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

        # Convert predictions and times to tensors
        preds = to_tensor(predictions)
        times = to_tensor(duration_cuts)

        # Add detailed shape diagnostics to help debug size mismatches
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Tensor shapes before Brier score calculation:")
            logger.debug(f"  Test events (e_test): {e_test.shape}")
            logger.debug(f"  Test times (t_test): {t_test.shape}")
            logger.debug(f"  Predictions (preds): {preds.shape}")
            logger.debug(f"  Eval times: {times.shape}")

        brier_scorer = TorchsurvBrierScore()

        # Compute IPCWs from the test set itself
        test_ipcw = get_ipcw(e_test, t_test)  # ipcw at time
        test_ipcw_new_time = get_ipcw(e_test, t_test, times)  # ipcw at new time

        if logger.isEnabledFor(logging.DEBUG):
            if hasattr(test_ipcw, "shape"):
                logger.debug(f"  IPCW shape: {test_ipcw.shape}")
            if hasattr(test_ipcw_new_time, "shape"):
                logger.debug(f"  IPCW new time shape: {test_ipcw_new_time.shape}")

        try:
            # Compute Brier scores using torchsurv with test-based weights
            brier_scores = brier_scorer(
                estimate=preds,
                event=e_test,
                time=t_test,
                new_time=times,
                weight=test_ipcw,  # Use test-based weights
                weight_new_time=test_ipcw_new_time,  # Use test-based weights
            )
        except RuntimeError as e:
            logger.error(f"Error calculating Brier score: {str(e)}")
            raise

        # For integrated Brier score, use the built-in integral method
        # Note: this will use the weights that were provided in the previous call
        ibs = brier_scorer.integral()
        integrated_bs = float(ibs.cpu().item())

        # Convert to numpy arrays for the evaluate API
        if per_horizon:
            bs_numpy = brier_scores.cpu().numpy()
        else:
            bs_numpy = []

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Computed brier scores: {bs_numpy}")
            logger.debug(f"Computed integrated brier score: {integrated_bs}")

        return integrated_bs, bs_numpy
