"""Within-Subject Concordance Index Metric"""

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
Within-Subject Concordance Index for competing risks evaluation based on inverse probability of censoring weights (IPCW).

This metric evaluates the model's ability to correctly rank competing events within individual subjects,
complementing the traditional cross-patient C-index. For each patient with multiple competing events,
it calculates how well the model ranks the actual event order based on predicted risks.

Unlike the traditional C-index which compares risk predictions across different patients for the same event type,
the within-subject C-index compares risk predictions across different event types for the same patient.
This is particularly useful for competing risks scenarios where understanding the relative risk of different
events for an individual patient is crucial for personalized treatment decisions.

The implementation uses IPCW to handle censored observations properly and leverages torchsurv's
GPU-accelerated C-index calculation by reshaping tensors to treat events as the comparison dimension.

Key features:
- Evaluates event ranking within each patient separately
- Handles censored events using IPCW
- Provides both per-patient and aggregated metrics
- GPU-accelerated through torchsurv

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
    Ground truth survival data in multi-event format.
    Expected shape: (n_patients, n_events)
    Each element should be a tuple (event_indicator, time) or dictionary with keys 'e' and 't'.
    Event indicators should be boolean (True for events, False for censored).
    Times should be positive floats representing time-to-event or time-to-censoring.

predictions : array-like
    Model risk predictions for each event type per patient.
    Shape: (n_patients, n_events) or (n_patients, n_events, n_horizons)
    Higher values should indicate higher risk (shorter survival time).

duration_cuts : array-like
    Time horizons at which to evaluate the within-subject C-index.
    Each value represents a specific time point for evaluation.

per_horizon : bool, optional, default=False
    If True, return within-subject C-indices at each time horizon.
    If False, only return the integrated within-subject C-index.

min_events_per_patient : int, optional, default=2
    Minimum number of uncensored events required per patient to compute C-index.
    Patients with fewer events are excluded from the calculation.

Returns
-------
integrated_within_cindex : float
    The integrated within-subject C-index across all time horizons and patients.
    Range is typically [0, 1] where 0.5 represents random prediction and 1.0 perfect prediction.

within_cindeces_np : array-like or empty list
    If per_horizon=True, returns a numpy array containing within-subject C-indices
    for each time horizon in duration_cuts.
    If per_horizon=False, returns an empty list.

"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class WithinSubjectCIIPCW(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._get_feature_types(),
            reference_urls=[
                "https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_ipcw.html"
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
        min_events_per_patient=2,
        n_samples=None,
        n_events=None,
    ):
        """
        Compute within-subject concordance index with IPCW.

        The key insight is to reshape the data from [patients, events] to [events, patients]
        and then calculate C-index for each patient across their events.
        """
        # Handle the direct array case (from eval_modules.py)
        if (
            n_samples is not None
            and n_events is not None
            and hasattr(references, "shape")
        ):
            # Convert to numpy arrays
            references = np.array(references)
            predictions = np.array(predictions)

            # Extract event indicators and times from the references
            # References layout: [events, hazards, risks, durations] × num_events
            e_matrix = np.zeros((n_samples, n_events), dtype=bool)
            t_matrix = np.zeros((n_samples, n_events), dtype=np.float32)

            for sample_idx in range(n_samples):
                for event_idx in range(n_events):
                    # Extract event indicator and time for this event
                    e_matrix[sample_idx, event_idx] = references[
                        sample_idx, 1 * n_events + event_idx
                    ].astype(bool)
                    t_matrix[sample_idx, event_idx] = references[
                        sample_idx, 3 * n_events + event_idx
                    ]

            # Extract predictions: shape [batch, pred_type, event, time_horizons]
            # We want the risk predictions (pred_type=1)
            pred_matrix = predictions[:, 1, :, :]  # [batch, events, time_horizons]

            n_patients = n_samples
        else:
            # Handle the direct dictionary/tuple format (for testing)
            n_patients = len(references)
            n_events_test = len(references[0]) if n_patients > 0 else 0

            if n_patients == 0 or n_events_test == 0:
                logger.warning(
                    "No data provided for within-subject C-index calculation"
                )
                return 0.5, []

            # Extract event indicators and times
            e_matrix = np.zeros((n_patients, n_events_test), dtype=bool)
            t_matrix = np.zeros((n_patients, n_events_test), dtype=np.float32)

            for i, patient_refs in enumerate(references):
                for j, ref in enumerate(patient_refs):
                    if isinstance(ref, dict):
                        e_matrix[i, j] = ref["e"]
                        t_matrix[i, j] = ref["t"]
                    else:
                        e_matrix[i, j] = ref[0]
                        t_matrix[i, j] = ref[1]

            # Convert predictions to numpy array
            pred_matrix = np.array(predictions)

            # Update n_events for the rest of the computation
            if n_events is None:
                n_events = n_events_test

        # Log shapes for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Within-subject C-index tensor shapes:")
            logger.debug(f"  Event matrix: {e_matrix.shape}")
            logger.debug(f"  Time matrix: {t_matrix.shape}")
            logger.debug(f"  Prediction matrix: {pred_matrix.shape}")
            logger.debug(f"  Duration cuts: {len(duration_cuts)}")

        # Initialize C-Index calculator
        c_index_calculator = ConcordanceIndex()

        # Calculate within-subject C-index for each time horizon
        horizon_cindeces = []

        for h_idx, tau in enumerate(duration_cuts):
            patient_cindeces = []
            patient_weights = []

            for patient_idx in range(n_patients):
                # Extract patient's data
                patient_events = e_matrix[patient_idx]
                patient_times = t_matrix[patient_idx]

                # Extract predictions for this time horizon
                if pred_matrix.ndim == 3:
                    patient_preds = pred_matrix[patient_idx, :, h_idx]
                else:
                    patient_preds = pred_matrix[patient_idx]

                # Filter to events that occurred before tau
                valid_mask = (
                    (patient_events) & (patient_times <= tau) & (patient_times > 0)
                )
                n_valid_events = valid_mask.sum()

                if n_valid_events >= min_events_per_patient:
                    try:
                        # Convert to tensors for this patient
                        patient_events_tensor = to_tensor(
                            patient_events[valid_mask], dtype=torch.bool
                        )
                        patient_times_tensor = to_tensor(
                            patient_times[valid_mask], dtype=torch.float32
                        )
                        patient_preds_tensor = to_tensor(patient_preds[valid_mask])

                        # Calculate IPCW for this patient's events
                        patient_ipcw = get_ipcw(
                            patient_events_tensor, patient_times_tensor
                        )

                        # Calculate C-index for this patient
                        patient_cindex = c_index_calculator(
                            event=patient_events_tensor,
                            time=patient_times_tensor,
                            estimate=patient_preds_tensor,
                            weight=patient_ipcw,
                        )

                        patient_cindeces.append(float(patient_cindex.item()))
                        patient_weights.append(n_valid_events)

                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"Patient {patient_idx} at tau={tau}: "
                                f"C-index={patient_cindeces[-1]:.4f} with {n_valid_events} events"
                            )

                    except Exception as e:
                        logger.error(
                            f"Error calculating within-subject C-index for patient {patient_idx} "
                            f"at tau={tau}: {str(e)}"
                        )

            # Calculate weighted average C-index for this time horizon
            if patient_cindeces:
                weights = np.array(patient_weights) / np.sum(patient_weights)
                horizon_cindex = float(np.sum(np.array(patient_cindeces) * weights))
            else:
                horizon_cindex = 0.5  # Default when no valid patients

            horizon_cindeces.append(horizon_cindex)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Horizon tau={tau}: {len(patient_cindeces)} valid patients, "
                    f"weighted C-index={horizon_cindex:.4f}"
                )

        # Calculate integrated within-subject C-index
        # Weight by number of events at each horizon
        total_events_per_horizon = []
        for h_idx, tau in enumerate(duration_cuts):
            event_count = np.sum((e_matrix) & (t_matrix <= tau) & (t_matrix > 0))
            total_events_per_horizon.append(event_count)

        total_events = sum(total_events_per_horizon)
        if total_events > 0:
            horizon_weights = np.array(total_events_per_horizon) / total_events
            integrated_within_cindex = float(
                np.sum(np.array(horizon_cindeces) * horizon_weights)
            )
        else:
            integrated_within_cindex = 0.5

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Horizon C-indices: {horizon_cindeces}")
            logger.debug(f"Event counts per horizon: {total_events_per_horizon}")
            logger.debug(
                f"Integrated within-subject C-index: {integrated_within_cindex}"
            )

        # Return results
        if per_horizon:
            within_cindeces_np = np.array(horizon_cindeces)
        else:
            within_cindeces_np = []

        return integrated_within_cindex, within_cindeces_np
