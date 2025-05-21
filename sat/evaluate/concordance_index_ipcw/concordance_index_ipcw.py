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

_CITATION = """\
@article{Uno2011OnTC,
  title={On the C‐statistics for evaluating overall adequacy of risk prediction procedures with censored survival data},
  author={Hajime Uno and Tianxi Cai and Michael J. Pencina and Ralph B. D'Agostino and L. J. Wei},
  journal={Statistics in Medicine},
  year={2011},
  volume={30}
}"""

_DESCRIPTION = """\
Concordance index for right-censored data based on inverse probability of censoring weights.
    This is an alternative to the estimator in :func:`concordance_index_censored`
    that does not depend on the distribution of censoring times in the test data.
    Therefore, the estimate is unbiased and consistent for a population concordance
    measure that is free of censoring.
    It is based on inverse probability of censoring weights, thus requires
    access to survival times from the training data to estimate the censoring
    distribution. Note that this requires that survival times `survival_test`
    lie within the range of survival times `survival_train`. This can be
    achieved by specifying the truncation time `tau`.
    The resulting `cindex` tells how well the given prediction model works in
    predicting events that occur in the time range from 0 to `tau`.
    The estimator uses the Kaplan-Meier estimator to estimate the
    censoring survivor function. Therefore, it is restricted to
    situations where the random censoring assumption holds and
    censoring is independent of the features.
    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb>`
    and [1]_ for further description.

    See also
    --------
    concordance_index_censored
        Simpler estimator of the concordance index.
    as_concordance_index_ipcw_scorer
        Wrapper class that uses :func:`concordance_index_ipcw`
        in its ``score`` method instead of the default
        :func:`concordance_index_censored`.

    References
    ----------
    .. [1] Uno, H., Cai, T., Pencina, M. J., D’Agostino, R. B., & Wei, L. J. (2011).
           "On the C-statistics for evaluating overall adequacy of risk prediction
           procedures with censored survival data".
           Statistics in Medicine, 30(10), 1105–1117.
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
    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event of test data.
    tau : float, optional
        Truncation time. The survival function for the underlying
        censoring time distribution :math:`D` needs to be positive
        at `tau`, i.e., `tau` should be chosen such that the
        probability of being censored after time `tau` is non-zero:
        :math:`P(D > \\tau) > 0`. If `None`, no truncation is performed.
    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

    Returns
    -------
    cindex : float
        Concordance index
    concordant : int
        Number of concordant pairs
    discordant : int
        Number of discordant pairs
    tied_risk : int
        Number of pairs having tied estimated risks
    tied_time : int
        Number of comparable pairs sharing the same time
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
                    logger.debug(f"C-index at tau={tau}: {cindeces[-1]} with {event_count} events")
                    
            except Exception as e:
                logger.error(f"Error calculating C-Index at tau={tau}: {str(e)}")
                cindeces.append(float('nan'))
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
