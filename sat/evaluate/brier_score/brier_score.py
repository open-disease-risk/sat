"""Brier Score Metric."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import datasets
import numpy as np

import evaluate

from sksurv.metrics import brier_score, integrated_brier_score
from sat.utils import logging

logger = logging.get_default_logger()


_CITATION = """
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
        # Initialize cache for struct arrays and predictions
        self._array_cache = {}
        self._prediction_cache = {}

    def _clear_cache(self):
        """Clear the cache to free memory"""
        self._array_cache.clear()
        self._prediction_cache.clear()

    def _get_struct_array(self, data, key):
        """Get or create a structured array for survival data"""
        # Use cache key based on data id and shape
        cache_key = f"{key}_{id(data)}_{len(data)}"

        if cache_key in self._array_cache:
            return self._array_cache[cache_key]

        # Create the structured array
        arr = np.array([tuple(x) for x in data], dtype=[("e", bool), ("t", float)])
        # Store in cache
        self._array_cache[cache_key] = arr
        return arr

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
        train_set,
        duration_cuts,
        per_horizon=False,
    ):
        # Get cached arrays or create new ones
        et_test = self._get_struct_array(references, "test")
        et_train = self._get_struct_array(train_set, "train")

        # Convert predictions to numpy array for efficiency
        # Use cache based on predictions ID
        pred_cache_key = f"pred_{id(predictions)}_{len(predictions)}"
        if pred_cache_key in self._prediction_cache:
            preds_np = self._prediction_cache[pred_cache_key]
        else:
            preds_np = np.array(predictions)
            self._prediction_cache[pred_cache_key] = preds_np

        # Compute scores
        if per_horizon:
            # Calculate per-horizon Brier scores
            brs = brier_score(et_train, et_test, preds_np, duration_cuts)[1]
        else:
            brs = []

        # Calculate integrated Brier score
        ibrs = integrated_brier_score(et_train, et_test, preds_np, duration_cuts)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Computed brier score: {brs}")

        # Clear cache if it's getting too large (optional)
        if len(self._array_cache) > 10 or len(self._prediction_cache) > 10:
            self._clear_cache()

        return ibrs, brs
