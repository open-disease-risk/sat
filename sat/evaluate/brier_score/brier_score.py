"""Brier Score Metric."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import datasets
import evaluate
import numpy as np
import torch

from torchsurv.stats.ipcw import get_ipcw
from torchsurv.metrics.brier_score import BrierScore as TorchsurvBrierScore

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
volume = {18},
number = {17-18},
pages = {2529-2545},
doi = {https://doi.org/10.1002/(SICI)1097-0258(19990915/30)18:17/18<2529::AID-SIM274>3.0.CO;2-5},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2819990915/30%2918%3A17/18%3C2529%3A%3AAID-SIM274%3E3.0.CO%3B2-5},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/%28SICI%291097-0258%2819990915/30%2918%3A17/18%3C2529%3A%3AAID-SIM274%3E3.0.CO%3B2-5},
abstract = {Abstract Prognostic classification schemes have often been used in medical applications, but rarely subjected to a rigorous examination of their adequacy. For survival data, the statistical methodology to assess such schemes consists mainly of a range of ad hoc approaches, and there is an alarming lack of commonly accepted standards in this field. We review these methods and develop measures of inaccuracy which may be calculated in a validation study in order to assess the usefulness of estimated patient-specific survival probabilities associated with a prognostic classification scheme. These measures are meaningful even when the estimated probabilities are misspecified, and asymptotically they are not affected by random censorship. In addition, they can be used to derive R2-type measures of explained residual variation. A breast cancer study will serve for illustration throughout the paper. Copyright © 1999 John Wiley \& Sons, Ltd.},
year = {1999}
}
"""

_DESCRIPTION = """
Estimate the time-dependent Brier score for right censored data.

The time-dependent Brier score is the mean squared error at time point :math:`t`:
.. math::
\\mathrm{BS}^c(t) = \\frac{1}{n} \\sum_{i=1}^n I(y_i \\leq t \\land \\delta_i = 1)
\\frac{(0 - \\hat{\\pi}(t | \\mathbf{x}_i))^2}{\\hat{G}(y_i)} + I(y_i > t)
\\frac{(1 - \\hat{\\pi}(t | \\mathbf{x}_i))^2}{\\hat{G}(t)} ,

where :math:`\\hat{\\pi}(t | \\mathbf{x})` is the predicted probability of
remaining event-free up to time point :math:`t` for a feature vector :math:`\\mathbf{x}`,
and :math:`1/\\hat{G}(t)` is a inverse probability of censoring weight, estimated by
the Kaplan-Meier estimator.

See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Time-dependent-Brier-Score>`
and [1]_ for details.
"""


_KWARGS_DESCRIPTION = """
    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.
    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.
    estimate : array-like, shape = (n_samples, n_times)
        Estimated risk of experiencing an event for test data at `times`.
        The i-th column must contain the estimated probability of
        remaining event-free up to the i-th time point.
    times : array-like, shape = (n_times,)
        The time points for which to estimate the Brier score.
        Values must be within the range of follow-up times of
        the test data `survival_test`.

    Returns
    -------
    times : array, shape = (n_times,)
        Unique time points at which the brier scores was estimated.
    brier_scores : array , shape = (n_times,)
        Values of the brier score.

    Examples
    --------
    >>> from sksurv.datasets import load_gbsg2
    >>> from sksurv.linear_model import CoxPHSurvivalAnalysis
    >>> from sksurv.metrics import brier_score
    >>> from sksurv.preprocessing import OneHotEncoder
    Load and prepare data.
    >>> X, y = load_gbsg2()
    >>> X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)
    >>> Xt = OneHotEncoder().fit_transform(X)
    Fit a Cox model.
    >>> est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)
    Retrieve individual survival functions and get probability
    of remaining event free up to 5 years (=1825 days).
    >>> survs = est.predict_survival_function(Xt)
    >>> preds = [fn(1825) for fn in survs]
    Compute the Brier score at 5 years.
    >>> times, score = brier_score(y, y, preds, 1825)
    >>> print(score)
    [0.20881843]

    See also
    --------
    integrated_brier_score
        Computes the average Brier score over all time points.

    References
    ----------
    .. [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher,
           "Assessment and comparison of prognostic classification schemes for survival data,"
           Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545, 1999.
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BrierScore(evaluate.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the torchsurv BrierScore calculator once
        self._brier_scorer = TorchsurvBrierScore()

        # Initialize cache
        self._tensor_cache = {}

    def _get_tensor(self, data, key, is_event=False):
        """Convert data to appropriate tensors and cache them"""
        cache_key = f"{key}_{id(data)}_{len(data)}"

        if cache_key in self._tensor_cache:
            return self._tensor_cache[cache_key]

        # Extract event indicators and times from data
        if isinstance(data[0], dict):
            if is_event:
                arr = np.array([x["e"] for x in data])
            else:  # time
                arr = np.array([x["t"] for x in data])
        else:
            # Assume data is a list of (event, time) tuples
            if is_event:
                arr = np.array([x[0] for x in data])
            else:  # time
                arr = np.array([x[1] for x in data])

        # Convert to tensor
        dtype = torch.bool if is_event else torch.float32
        tensor = to_tensor(arr, dtype=dtype)

        # Cache result
        self._tensor_cache[cache_key] = tensor
        return tensor

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
            brier_scores = self._brier_scorer(
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
        ibs = self._brier_scorer.integral()
        integrated_bs = float(ibs.cpu().item())

        # Convert to numpy arrays for the evaluate API
        if per_horizon:
            bs_numpy = brier_scores.cpu().numpy()
        else:
            bs_numpy = []

        # Clear cache if needed
        if len(self._tensor_cache) > 20:
            self._tensor_cache.clear()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Computed brier scores: {bs_numpy}")
            logger.debug(f"Computed integrated brier score: {integrated_bs}")

        return integrated_bs, bs_numpy
