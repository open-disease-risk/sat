__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset

from sat.data.dataset.femr_extensions.schema import LabelType

logger = logging.getLogger(__name__)


# Custom JSON encoder to handle datetime objects and other non-serializable types
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects, NumPy arrays, and other non-serializable types."""

    def default(self, obj):
        # Handle datetime objects
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        # Handle NumPy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle NumPy scalar types
        elif np.isscalar(obj) and isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        # Handle other NumPy types
        elif isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        # Default
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


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
            # Include string_value in the list of columns to be grouped
            columns_to_keep = [self.time_field, "code", "numeric_value", "string_value"]
            # Filter to include only columns that exist in the dataframe
            existing_columns = [col for col in columns_to_keep if col in df.columns]

            grouped = (
                df.groupby(self.primary_key)
                .apply(lambda x: x[existing_columns].to_dict(orient="records"))
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
            anchor_time_flat = []
            for labels in anchor_label_output:
                anchor_time_flat.append(labels["prediction_time"] if labels else None)
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

    def truncate_events_at_anchor(
        self, anchor_times: List[Any], ds: Dataset
    ) -> Dataset:
        """
        Keep only events up to and including the anchor time.
        Assumes 'events' column is present and anchor times are available.
        Events with time values are sorted, and events without time values are appended at the end.
        """
        if "events" not in ds.column_names:
            return ds

        new_events = []
        for i, events in enumerate(ds["events"]):
            anchor_time = anchor_times[i] if i < len(anchor_times) else None

            if anchor_time is None:
                # No anchor time available, keep all events
                new_events.append(events)
                continue

            # For each event, check if it happened before or at the anchor time
            # We're using 0 as the reference point for the anchor
            events_with_time = []
            events_without_time = []

            for e in events:
                event_time = e.get("time", None)
                # Group events with no time information
                if event_time is None:
                    events_without_time.append(e)
                # Keep events with time <= 0 (at or before anchor)
                elif isinstance(event_time, (int, float)) and event_time <= 0:
                    events_with_time.append(e)
                # For datetime objects, compare with anchor time directly
                elif hasattr(event_time, "timestamp") and event_time <= anchor_time:
                    events_with_time.append(e)

            # Sort events with time values by their time
            sorted_events_with_time = sorted(
                events_with_time,
                key=lambda e: e.get("time", float("inf")),
                # Handle possible comparison errors
                # between different types by using a safe key function
            )

            # Combine sorted events with time first, then events without time
            truncated = sorted_events_with_time + events_without_time
            new_events.append(truncated)

        ds = ds.remove_columns(["events"]).add_column("events", new_events)
        self._record_metadata("Kept events up to anchor time, sorted by time", ds)
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

    def make_serializable(self, obj):
        """
        Recursively convert objects to JSON serializable forms.
        Handles datetime objects, nested structures, and custom objects.
        """
        if isinstance(obj, dict):
            # Ensure we're not dropping any keys in the dict
            return {k: self.make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(v) for v in obj]
        elif hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):  # custom objects
            return self.make_serializable(vars(obj))
        elif obj is None:
            # Explicitly handle None values to ensure they're not dropped
            return None
        # Handle NumPy arrays and pandas objects
        elif hasattr(obj, "shape") and hasattr(obj, "dtype"):
            # This is likely a NumPy array
            try:
                # If it's a scalar-like array (0-dim or 1-element), get the scalar value
                if obj.size == 1:
                    return obj.item()
                # Otherwise, convert to a list
                return obj.tolist()
            except:
                # If conversion fails, just stringify it
                return str(obj)
        elif hasattr(obj, "iloc") and hasattr(obj, "loc"):
            # This is likely a pandas Series or DataFrame
            try:
                return obj.to_dict()
            except:
                try:
                    return obj.tolist()
                except:
                    return str(obj)
        # Check for scalar NA values (safer than pd.isna() which can handle arrays)
        elif pd.api.types.is_scalar(obj) and pd.isna(obj):
            return None
        else:
            try:
                json.dumps(obj)
                return obj
            except Exception:
                return str(obj)

    def save_to_json_with_labels(
        self,
        ds: Union[Dataset, pd.DataFrame],
        labels_dict: Dict[str, List[Any]],
        anchor_times: List[Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Serialize the entire cohort dataset including labels to JSON for debugging purposes.

        Args:
            ds: Dataset or DataFrame to serialize
            labels_dict: Dictionary of labels for each patient
            anchor_times: List of anchor times for each patient
            output_path: Optional path to save the JSON file. If None, uses processed_dir/name/cohort_debug.json

        Returns:
            Path to the saved JSON file
        """
        if output_path is None:
            output_path = str(
                Path(self.processed_dir) / self.name / "cohort_debug.json"
            )

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert Dataset to pandas if needed
        if isinstance(ds, Dataset):
            df = ds.to_pandas()
        else:
            df = ds

        # Convert to list of dictionaries
        data_dicts = df.to_dict(orient="records")

        # Add labels and anchor times to each patient record
        for i, record in enumerate(data_dicts):
            if i < len(anchor_times):
                record["anchor_time"] = self.make_serializable(anchor_times[i])

            # Add labels for this patient
            record["labels"] = {}
            for label_name, label_values in labels_dict.items():
                if i < len(label_values):
                    record["labels"][label_name] = self.make_serializable(
                        label_values[i]
                    )

        # Save with custom JSON encoder
        with open(output_path, "w") as f:
            json.dump(data_dicts, f, indent=2, cls=CustomJSONEncoder)

        logger.info(
            f"Saved cohort data with labels to JSON for debugging: {output_path}"
        )

        return output_path

    def save_metadata(self, labels_dict, anchor_times):
        """
        Save cohort metadata (labels_dict, anchor_times, and provenance) to a JSON file in the output directory.
        The output directory is derived from self.processed_dir.
        """

        # Determine output directory
        meta = {
            "labels_dict": self.make_serializable(labels_dict),
            "anchor_times": self.make_serializable(anchor_times),
            "provenance": self.metadata,
        }
        meta_path = Path(self.processed_dir) / self.name / "cohort_metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def save_cohort_dataset(self, ds, labels_dict=None, anchor_times=None):
        """
        Save the cohort dataset to the output directory. Supports HuggingFace Dataset and pandas DataFrame.
        Optionally embeds labels_dict and anchor_times as columns in the dataset.
        The output directory is derived from self.processed_dir.
        """
        cohort_path_parquet = Path(self.processed_dir) / self.name / "cohort.parquet"
        cohort_path_csv = Path(self.processed_dir) / self.name / "cohort.csv"
        cohort_path_parquet.parent.mkdir(parents=True, exist_ok=True)

        # If labels_dict and anchor_times are provided, add them as columns
        if labels_dict is not None and anchor_times is not None:
            # Convert to pandas for easier manipulation
            if hasattr(ds, "to_pandas"):
                df = ds.to_pandas()
            else:
                df = ds if hasattr(ds, "loc") else pd.DataFrame(ds)

            # Add anchor_times as a column
            df["anchor_time"] = anchor_times

            # Add each label type as a separate column
            for label_name, label_values in labels_dict.items():
                # Ensure the label values are serializable
                serializable_labels = [
                    self.make_serializable(labels) for labels in label_values
                ]
                df[f"labels_{label_name}"] = serializable_labels

            # Convert back to dataset if needed
            if hasattr(ds, "from_pandas"):
                ds = Dataset.from_pandas(df)
            else:
                ds = df

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
        ds = self.truncate_events_at_anchor(anchor_times, ds)

        # Save metadata and dataset (with labels embedded)
        self.save_metadata(labels_dict, anchor_times)
        self.save_cohort_dataset(ds, labels_dict, anchor_times)

        # Save JSON for debugging (with labels)
        self.save_to_json_with_labels(ds, labels_dict, anchor_times)

        return ds
