__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

from sat.data.dataset.femr_extensions.schema import LabelType

logger = logging.getLogger(__name__)


class CohortOMOP:
    """
    Modular pipeline for OMOP cohort construction with provenance tracking.

    Steps:
        1. Load data
        2. Group events by patient (if needed)
        3. Apply labelers (index date, outcome, eligibility, etc.)
        4. Save cohort dataset and metadata (provenance)
    """

    def __init__(
        self,
        source: str,
        name: str = "cohort_omop",
        processed_dir: str = "",
        labelers: Optional[List[Any]] = None,
        primary_key: str = "patient_id",
        time_field: str = "time",
        date_diff_unit: str = "days",  # Unit for date differences
    ):
        """
        Parameters:
            date_diff_unit: str
                Unit for time differences when subtracting datetimes (e.g., 'days', 'hours', 'minutes'). Default: 'days'.
        """
        self.name = name
        self.source = source
        self.processed_dir = processed_dir
        self.labelers = labelers or []
        self.primary_key = primary_key
        self.time_field = time_field
        self.metadata: List[Dict[str, Any]] = []
        self.date_diff_unit = date_diff_unit

    def _record_metadata(self, step: str, ds: Any):
        """Record provenance information for each pipeline step."""
        self.metadata.append(
            {
                "step": step,
                "num_patients": len(ds) if hasattr(ds, "__len__") else None,
                "columns": getattr(ds, "column_names", None),
            }
        )

    def load_data(self) -> Dataset:
        logger.info(f"Loading OMOP-format parquet file from {self.source}")
        ds = Dataset.from_parquet(self.source)
        self._record_metadata("Loaded raw data", ds)
        return ds

    def group_events(self, ds: Dataset) -> Dataset:
        if "events" not in ds.column_names:
            df = ds.to_pandas()
            grouped = (
                df.groupby(self.primary_key)
                .apply(
                    lambda x: x[[self.time_field, "code", "numeric_value"]].to_dict(
                        orient="records"
                    )
                )
                .reset_index(name="events")
            )
            ds = Dataset.from_pandas(grouped)
            self._record_metadata("Grouped events by patient", ds)
        return ds

    def apply_labelers(self, ds: Dataset) -> Tuple[Dict[str, List[Any]], List[Any]]:
        """
        Apply labelers in the following order based on LabelType:
        1. ANCHOR (index date labeler) - must be present and run first
        2. OUTCOME (e.g. event/censoring)
        3. INCLUSION (eligibility)
        4. EXCLUSION (eligibility)
        Raises error if no index (ANCHOR) labeler is found.
        """
        # Partition labelers by LabelType
        anchor_labelers = [
            labeler
            for labeler in self.labelers
            if getattr(labeler, "label_type", None) == LabelType.ANCHOR
        ]
        remaining_labelers = [
            labeler
            for labeler in self.labelers
            if getattr(labeler, "label_type", None) != LabelType.ANCHOR
        ]

        if not anchor_labelers:
            raise ValueError(
                "At least one ANCHOR (index date) labeler must be provided."
            )

        # Apply anchor labelers first and attach anchor_time
        anchor_times = []
        labels_dict = {}
        for labeler in anchor_labelers:
            logger.info(f"Applying labeler: {labeler}")
            anchor_label_output = labeler.apply(ds)
            # Each element in anchor_label_output is a list of labels for that patient
            anchor_time_flat = []
            for labels in anchor_label_output:
                if labels and isinstance(labels, list) and len(labels) > 0:
                    anchor_time_flat.append(labels[0]["prediction_time"])
                else:
                    anchor_time_flat.append(None)
            anchor_times.extend(anchor_time_flat)
            labels_dict[labeler.name] = anchor_label_output
            self._record_metadata(f"Applied labeler: {labeler}", ds)

        # Process remaining labelers in order
        for labeler in remaining_labelers:
            logger.info(f"Applying labeler: {labeler}")
            labels = labeler.apply(ds)
            labels_dict[labeler.name] = labels
            self._record_metadata(f"Applied labeler: {labeler}", ds)
        return labels_dict, anchor_times

    def apply_competing_risk_censoring(
        self, labels_dict: Dict[str, List[Any]], anchor_times: List[Any], ds: Dataset
    ):
        """
        For each patient, if a competing event occurs before an outcome, censor the outcome label:
        - boolean_value = False
        - prediction_time = time of first competing event - anchor_time
        Assumes outcome and competing event labels are present in labels_dict.
        """
        # Select outcome and competing labels based on label_type in the schema, not key name
        outcome_label_cols = [
            col
            for col, labels in labels_dict.items()
            if labels
            and isinstance(labels[0], list)
            and labels[0]
            and isinstance(labels[0][0], dict)
            and labels[0][0].get("label_type", None) == LabelType.OUTCOME
        ]
        if not outcome_label_cols:
            # No-op if no outcome labels
            return
        # For each patient, find earliest competing event time, censor outcomes if needed
        for i, anchor_time in enumerate(anchor_times):
            # Find earliest competing event time for this patient among all outcome labels with competing_event==True
            competing_times = []
            for outcome_col in outcome_label_cols:
                labels = labels_dict[outcome_col][i]
                if labels and isinstance(labels, list):
                    for label in labels:
                        if label.get("competing_event", False) and label.get(
                            "boolean_value", False
                        ):
                            comp_time = label.get("prediction_time", None)
                            if comp_time is not None:
                                competing_times.append(comp_time)
            if not competing_times:
                continue
            censor_time = min(competing_times)
            # Censor outcome labels if outcome occurs after competing event
            for outcome_col in outcome_label_cols:
                labels = labels_dict[outcome_col][i]
                for label in labels:
                    if label.get("prediction_time", float("inf")) > censor_time:
                        label["boolean_value"] = False
                        if anchor_time is not None:
                            # Handle datetime/date or numeric differences
                            try:
                                if isinstance(
                                    censor_time, (datetime.datetime, datetime.date)
                                ) and isinstance(
                                    anchor_time, (datetime.datetime, datetime.date)
                                ):
                                    delta = censor_time - anchor_time
                                    if self.date_diff_unit == "days":
                                        label["prediction_time"] = delta.days
                                    elif self.date_diff_unit == "hours":
                                        label["prediction_time"] = (
                                            delta.total_seconds() / 3600
                                        )
                                    elif self.date_diff_unit == "minutes":
                                        label["prediction_time"] = (
                                            delta.total_seconds() / 60
                                        )
                                    else:
                                        label["prediction_time"] = delta.total_seconds()
                                else:
                                    label["prediction_time"] = censor_time - anchor_time
                            except Exception:
                                label["prediction_time"] = censor_time - anchor_time
                        else:
                            label["prediction_time"] = censor_time
        self._record_metadata("Applied competing risk censoring", ds)

    def truncate_events_at_competing(
        self, labels_dict: Dict[str, List[Any]], anchor_times: List[Any], ds: Dataset
    ) -> Dataset:
        """
        Truncate all patient events at the earliest competing event time.
        Assumes 'events' column is present and competing event times are available.
        """
        if "events" not in ds.column_names:
            return ds
        # Select outcome labels with competing_event flag
        outcome_label_cols = [
            col
            for col, labels in labels_dict.items()
            if labels
            and isinstance(labels[0], list)
            and labels[0]
            and isinstance(labels[0][0], dict)
            and labels[0][0].get("competing_event", False)
        ]
        new_events = []
        for i, events in enumerate(ds["events"]):
            anchor_time = anchor_times[i]
            competing_times = []
            for outcome_col in outcome_label_cols:
                comp_labels = labels_dict[outcome_col][i]
                if comp_labels and isinstance(comp_labels, list):
                    for cl in comp_labels:
                        if cl.get("boolean_value", False):
                            comp_time = cl.get("prediction_time", None)
                            if comp_time is not None:
                                competing_times.append(comp_time)
            if competing_times:
                cutoff = min(competing_times)
                # Truncate events at cutoff
                if anchor_time is not None:
                    cutoff_time = cutoff - anchor_time
                else:
                    cutoff_time = cutoff
                truncated = [
                    e for e in events if e.get("time", float("inf")) <= cutoff_time
                ]
                new_events.append(truncated)
            else:
                new_events.append(events)
        ds = ds.remove_columns(["events"]).add_column("events", new_events)
        self._record_metadata("Truncated events at competing risk", ds)
        return ds

    def exclude_patients(
        self, labels_dict: Dict[str, List[Any]], anchor_times: List[Any], ds: Dataset
    ) -> Tuple[Dataset, Dict[str, List[Any]], List[Any]]:
        # Identify exclusion labels and patients to exclude
        exclusion_label_cols = [
            col
            for col, labels in labels_dict.items()
            if labels
            and isinstance(labels[0], list)
            and labels[0]
            and isinstance(labels[0][0], dict)
            and labels[0][0].get("label_type", None) == LabelType.EXCLUSION
        ]
        exclude_patient_indices = set()
        for i, _ in enumerate(ds):
            for exclusion_col in exclusion_label_cols:
                exclusion_labels = labels_dict[exclusion_col][i]
                if any(label.get("boolean_value", False) for label in exclusion_labels):
                    exclude_patient_indices.add(i)
                    break

        # Remove excluded patients from the dataset
        keep_indices = [i for i in range(len(ds)) if i not in exclude_patient_indices]
        ds, labels_dict, anchor_times = self._select_patients(
            ds, labels_dict, anchor_times, keep_indices
        )
        return ds, labels_dict, anchor_times

    def filter_patients_without_anchor(
        self, labels_dict: Dict[str, List[Any]], anchor_times: List[Any], ds: Dataset
    ) -> Tuple[Dataset, Dict[str, List[Any]], List[Any]]:
        """
        Remove patients who do not have an anchor event (i.e., anchor label's boolean_value == False).
        This reduces the dataset for all downstream tasks.
        """
        # Find anchor label columns
        anchor_label_cols = [
            col
            for col, labels in labels_dict.items()
            if labels
            and isinstance(labels[0], list)
            and labels[0]
            and isinstance(labels[0][0], dict)
            and labels[0][0].get("label_type", None) == LabelType.ANCHOR
        ]
        if not anchor_label_cols:
            # No anchor labelers found; return as is
            return ds, labels_dict, anchor_times
        anchor_col = anchor_label_cols[0]
        # Identify patients to keep (anchor label boolean_value == True)
        keep_indices = [
            i
            for i, anchor_labels in enumerate(labels_dict[anchor_col])
            if any(label.get("boolean_value", False) for label in anchor_labels)
        ]
        ds, labels_dict, anchor_times = self._select_patients(
            ds, labels_dict, anchor_times, keep_indices
        )
        self._record_metadata("Filtered patients without anchor event", ds)
        return ds, labels_dict, anchor_times

    def _select_patients(self, ds, labels_dict, anchor_times, keep_indices):
        """
        Utility to select/filter patients by indices across ds, labels_dict, and anchor_times.
        """
        ds = ds.select(keep_indices)
        labels_dict = {
            col: [labels[i] for i in keep_indices]
            for col, labels in labels_dict.items()
        }
        anchor_times = [anchor_times[i] for i in keep_indices]
        return ds, labels_dict, anchor_times

    def save_metadata(self, labels_dict, anchor_times):
        """
        Save cohort metadata (labels_dict, anchor_times, and provenance) to a JSON file in the output directory.
        The output directory is derived from self.processed_dir.
        """

        def make_serializable(obj):
            # Recursively convert objects to serializable forms
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif hasattr(obj, "isoformat"):
                return obj.isoformat()
            elif hasattr(obj, "__dict__"):
                return make_serializable(vars(obj))
            else:
                try:
                    json.dumps(obj)
                    return obj
                except Exception:
                    return str(obj)

        # Determine output directory
        meta = {
            "labels_dict": make_serializable(labels_dict),
            "anchor_times": make_serializable(anchor_times),
            "provenance": self.metadata,
        }
        meta_path = Path(self.processed_dir) / self.name / "cohort_metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def save_cohort_dataset(self, ds):
        """
        Save the cohort dataset to the output directory. Supports HuggingFace Dataset and pandas DataFrame.
        The output directory is derived from self.processed_dir.
        """
        cohort_path_parquet = Path(self.processed_dir) / self.name / "cohort.parquet"
        cohort_path_csv = Path(self.processed_dir) / self.name / "cohort.csv"
        cohort_path_parquet.parent.mkdir(parents=True, exist_ok=True)

        try:
            # HuggingFace Dataset
            ds.to_parquet(cohort_path_parquet)
        except Exception:
            try:
                # Try pandas DataFrame
                df = ds.to_pandas() if hasattr(ds, "to_pandas") else ds
                df.to_parquet(cohort_path_parquet)
            except Exception:
                try:
                    df.to_csv(cohort_path_csv, index=False)
                except Exception as e:
                    logger.warning(f"Could not save cohort dataset: {e}")

    def __call__(self):
        ds = self.load_data()
        ds = self.group_events(ds)
        labels_dict, anchor_times = self.apply_labelers(ds)
        ds, labels_dict, anchor_times = self.filter_patients_without_anchor(
            labels_dict, anchor_times, ds
        )
        ds, labels_dict, anchor_times = self.exclude_patients(
            labels_dict, anchor_times, ds
        )
        self.apply_competing_risk_censoring(labels_dict, anchor_times, ds)
        ds = self.truncate_events_at_competing(labels_dict, anchor_times, ds)
        self.save_metadata(labels_dict, anchor_times)
        self.save_cohort_dataset(ds)
