"""Process OMOP (Omopical Event Data Standard) data.

1. Read the cohort Parquet file created by cohort_omop.py
2. Transform to SAT's survival analysis format with:
   - x: space-delimited codes
   - modality: list indicating if code has numeric value (1) or not (0)
   - numerics: list of numeric values (actual value if available, 1.0 otherwise)
3. Save as HuggingFace Dataset

This processor assumes the cohort has already been created and labeled by cohort_omop.py.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from dataclasses import dataclass
from functools import partial
from logging import DEBUG, ERROR
from pathlib import Path
from typing import Dict, Any

import numpy as np
from datasets import Dataset, load_dataset
from logdecorator import log_on_end, log_on_error, log_on_start

from sat.utils import logging

logger = logging.get_default_logger()


@dataclass
class omop:
    source: str  # Path to cohort parquet file created by cohort_omop.py
    processed_dir: str  # Output directory
    name: str  # Dataset name
    time_field: str = "time"  # Field to use for time measurements in events
    primary_key: str = "patient_id"  # Primary key for patients
    scale_numerics: bool = True  # Whether to scale numeric features
    scale_method: str = "standard"  # Scaling method to use: 'min_max' or 'standard'
    min_scale_numerics: float = 1.0  # Minimum value for min-max scaling
    batch_size: int = 100  # Batch size for streaming processing

    @log_on_start(DEBUG, "Creating SAT data representation from OMOP cohort...")
    @log_on_error(
        ERROR,
        "Error creating SAT data: {e!r}",
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!")
    def __call__(self) -> Dataset:
        # Prepare output directory
        out_dir = Path(f"{self.processed_dir}/{self.name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Configure code statistics calculation if scaling is enabled
        if self.scale_numerics:
            logger.info("Computing code statistics for scaling...")
            code_stats = compute_code_statistics(self.source, self.time_field)
        else:
            code_stats = {}

        # Define the map function for transforming each example
        map_fn = partial(
            transform_to_sat_format,
            primary_key=self.primary_key,
            time_field=self.time_field,
            code_stats=code_stats,
            scale_numerics=self.scale_numerics,
            scale_method=self.scale_method,
            min_scale_numerics=self.min_scale_numerics,
        )

        # Stream and process the dataset
        logger.info(f"Streaming and processing cohort data from {self.source}")
        ds = load_dataset("parquet", data_files={"train": self.source}, streaming=True)[
            "train"
        ]

        # Map the transformation function
        ds = ds.map(map_fn, batched=False)

        # Remove examples that don't have events
        ds = ds.filter(lambda x: x is not None and "x" in x and x["x"] != "")

        # Materialize the dataset and save to disk
        logger.info("Materializing and saving the dataset...")
        materialized_ds = Dataset.from_generator(lambda: iter(ds.take(ds.n_shards)))
        materialized_ds.save_to_disk(out_dir)
        logger.info(f"Saved processed dataset to {out_dir}")

        return materialized_ds


def compute_code_statistics(
    source_path: str, time_field: str
) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics for each code to use in scaling numeric values.
    Uses streaming to efficiently process large datasets.

    Args:
        source_path: Path to the cohort parquet file
        time_field: Name of the time field in events

    Returns:
        Dictionary mapping codes to their statistics (mean, std, min, max)
    """
    # Initialize statistics containers
    code_values = {}

    # Stream the dataset
    ds = load_dataset("parquet", data_files={"train": source_path}, streaming=True)[
        "train"
    ]

    # Process each example to collect values for each code
    for example in ds:
        if "events" not in example:
            continue

        events = example["events"]
        if not events:
            continue

        # Process each event in this patient's record
        for event in events:
            code = event.get("code", "")
            value = event.get("numeric_value")

            if code and value is not None:
                try:
                    # Convert datetime to timestamp if needed
                    if hasattr(value, "timestamp"):
                        numeric_value = float(value.timestamp())
                    else:
                        numeric_value = float(value)

                    if code not in code_values:
                        code_values[code] = []
                    code_values[code].append(numeric_value)
                except (ValueError, TypeError):
                    # Skip values that can't be converted to float
                    pass

    # Calculate statistics for each code
    code_stats = {}
    for code, values in code_values.items():
        if not values:
            continue

        values_array = np.array(values)
        stats = {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
        }

        # Avoid division by zero in scaling
        if stats["std"] == 0 or np.isnan(stats["std"]):
            stats["std"] = 1.0

        # Ensure min != max for min-max scaling
        if stats["min"] == stats["max"]:
            stats["max"] = stats["min"] + 1.0

        code_stats[code] = stats

    return code_stats


def transform_to_sat_format(
    example: Dict[str, Any],
    primary_key: str,
    time_field: str,
    code_stats: Dict[str, Dict[str, float]],
    scale_numerics: bool,
    scale_method: str,
    min_scale_numerics: float,
) -> Dict[str, Any]:
    """
    Transform a single patient example into SAT format with space-delimited codes,
    modality list, and numerics list.

    Args:
        example: Patient example with events list
        primary_key: Column name containing patient identifier
        time_field: Field name for time in events
        code_stats: Statistics for scaling numeric values
        scale_numerics: Whether to scale numeric values
        scale_method: Scaling method ('standard' or 'min_max')
        min_scale_numerics: Minimum value for min-max scaling

    Returns:
        Patient data in SAT format or None if no valid events
    """
    # Extract patient ID and events
    patient_id = example.get(primary_key)
    events = example.get("events", [])

    # Skip if no events or no patient ID
    if not events or not patient_id:
        return None

    # Sort events by time, handling None/datetime comparison issues
    def safe_sort_key(event):
        time_value = event.get(time_field)
        # Handle potential None values or datetime objects
        if time_value is None:
            return float("-inf")  # Place None values at the beginning
        # Handle datetime objects by converting to timestamp if needed
        if hasattr(time_value, "timestamp"):
            return time_value.timestamp()
        return time_value

    events = sorted(events, key=safe_sort_key)

    # Extract codes and prepare modality/numerics lists
    codes = []
    modalities = []
    numerics = []
    event_times = []

    for event in events:
        code = event.get("code", "")
        if not code:
            continue

        # Extract time and numeric value
        event_time = event.get(time_field, 0)
        # Convert datetime time to float timestamp if needed
        if hasattr(event_time, "timestamp"):
            event_time = float(event_time.timestamp())

        value = event.get("numeric_value")

        # Add code to the list
        codes.append(code)
        event_times.append(event_time)

        # Determine modality and numeric value
        try:
            if value is not None:
                # Convert datetime to float timestamp if needed
                if hasattr(value, "timestamp"):
                    float_value = float(value.timestamp())
                else:
                    float_value = float(value)

                modality = 1
                if scale_numerics and code in code_stats:
                    stats = code_stats[code]
                    if scale_method == "standard":
                        # Standard scaling (z-score)
                        mean, std = stats["mean"], stats["std"]
                        numeric_value = (float_value - mean) / std
                    elif scale_method == "min_max":
                        # Min-max scaling
                        min_val, max_val = stats["min"], stats["max"]
                        range_val = max_val - min_val
                        numeric_value = (
                            min_scale_numerics + (float_value - min_val) / range_val
                        )
                    else:
                        numeric_value = float_value
                else:
                    numeric_value = float_value
            else:
                modality = 0
                numeric_value = 1.0
        except (ValueError, TypeError):
            # If any conversion fails, treat as non-numeric
            modality = 0
            numeric_value = 1.0

        modalities.append(modality)
        numerics.append(numeric_value)

    # Skip if no valid codes
    if not codes:
        return None

    # Create the SAT format output
    result = {
        primary_key: patient_id,
        "x": " ".join(codes),
        "modality": modalities,
        "numerics": numerics,
        "time": event_times,
    }

    # Copy any outcome/target variables that might be present
    # (already processed by cohort_omop.py)
    for field in ["duration", "event", "event_type", "competing_event"]:
        if field in example:
            result[field] = example[field]

    return result
