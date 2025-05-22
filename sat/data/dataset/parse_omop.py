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

import datetime
import json
import logging
import os
from dataclasses import dataclass
from functools import partial
from logging import DEBUG, ERROR
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from datasets import Dataset, Features, Sequence, Value, load_dataset
from datasets.arrow_dataset import ArrowWriter
from logdecorator import log_on_end, log_on_error, log_on_start

from sat.utils import logging

logger = logging.get_default_logger()


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
        ds = None  # Initialize ds to handle potential UnboundLocalError
        # Prepare output directory
        out_dir = Path(f"{self.processed_dir}/{self.name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = str(out_dir)
        temp_output_dir = Path(os.path.join(final_output_path, "temp_arrow_processing"))
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        temp_arrow_file_path = os.path.join(str(temp_output_dir), "data.arrow")

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

        # Load the source dataset for streaming
        logger.info(f"Streaming and processing cohort data from {self.source}")
        try:
            source_ds = load_dataset(
                "parquet", data_files=self.source, split="train", streaming=True
            )
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        # Define the features for the TRANSFORMED data
        transformed_data_schema = {
            self.primary_key: Value("int64"),
            "x": Value("string"),
            "modality": Sequence(Value("int8")),
            "numerics": Sequence(Value("float32")),
            "time": Sequence(Value("int32")),
            "string_values": Sequence(Value("string")),
        }

        transformed_features = Features(transformed_data_schema)
        logger.debug(f"ArrowWriter will use features: {transformed_features}")

        # Process and save the dataset to a temporary Arrow file
        logger.info(
            f"Processing data and writing to temporary Arrow file: {temp_arrow_file_path}"
        )
        processed_count = 0
        with ArrowWriter(
            path=temp_arrow_file_path, features=transformed_features
        ) as writer:
            for i, example in enumerate(source_ds):
                if i < 5:  # Log details for the first few source examples
                    logger.debug(
                        f"Source example {i} keys: {list(example.keys()) if isinstance(example, dict) else 'Not a dict'}"
                    )

                transformed_example = map_fn(example)

                if transformed_example:
                    current_patient_id = transformed_example.get(
                        self.primary_key, "Unknown_ID"
                    )
                    if (
                        processed_count == 0
                    ):  # Log details for the first successfully transformed example
                        logger.info(
                            f"First non-None transformed_example (patient {current_patient_id}) for ArrowWriter "
                            f"has keys: {list(transformed_example.keys())}"
                        )
                    writer.write(transformed_example)
                    processed_count += 1
                else:
                    source_patient_id = (
                        example.get(self.primary_key, "Unknown_ID_Source")
                        if isinstance(example, dict)
                        else "Unknown_ID_Source_NonDict"
                    )
                    source_keys = (
                        list(example.keys()) if isinstance(example, dict) else "N/A"
                    )
                    logger.debug(
                        f"map_fn for patient {source_patient_id} (source keys: {source_keys}) produced None."
                    )

        logger.info(f"Processed {processed_count} examples into {temp_arrow_file_path}")

        # Load the dataset from the Arrow file
        logger.info(
            f"Attempting to load dataset from Arrow file: {temp_arrow_file_path}"
        )
        logger.debug(f"Using features for loading: {transformed_features}")

        loaded_arrow_ds = None  # Initialize a temporary variable for the loaded dataset
        try:
            # Ensure transformed_features is defined and available here
            loaded_arrow_ds = Dataset.from_file(str(temp_arrow_file_path))
        except Exception as e_load:
            logger.error(
                f"EXCEPTION during Dataset.from_file('{temp_arrow_file_path}'): {type(e_load).__name__}: {e_load}",
                exc_info=True,  # This will include the traceback for the loading error
            )
            # ds will remain its initial None value due to this exception.

        if loaded_arrow_ds is None:
            # This block will be hit if Dataset.from_file returned None OR if it excepted (and loaded_arrow_ds wasn't assigned)
            logger.warning(
                f"Dataset.from_file('{temp_arrow_file_path}') resulted in a None dataset. "
                f"This could be because it returned None directly or an exception occurred during loading."
            )
            # ds remains its initial None value if it wasn't updated by a successful load.
        else:
            ds = loaded_arrow_ds  # Assign to the main ds variable only if loading was successful
            logger.info(
                f"Successfully loaded dataset from {temp_arrow_file_path}. It has {len(ds)} examples."
            )

        # At this point, ds is either a valid Dataset object or None.
        # If ds is None, the next line will cause an AttributeError.
        logger.debug(
            f"Preparing to filter dataset. Current ds is {'a Dataset object' if ds is not None else 'None'}."
        )

        def filter_with_logging(x):
            # Just check for None, allow everything else through
            return x is not None

        ds = ds.filter(filter_with_logging)

        logger.info("Processing and saving the dataset...")

        try:
            first_example = next(iter(ds))
            logger.info(
                f"First example keys: {list(first_example.keys()) if isinstance(first_example, dict) else 'not a dict'}"
            )
        except Exception as e:
            logger.warning(f"Unable to examine first example: {e}")

        out_dir = Path(f"{self.processed_dir}/{self.name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            # First materialize to temporary directory in case of failure
            temp_dir = Path(f"{self.processed_dir}/{self.name}_temp")
            temp_dir.mkdir(parents=True, exist_ok=True)

            # The 'ds' variable is already a map-style Dataset at this point.
            # We can directly save it. For consistency with subsequent code, we'll use 'materialized_ds'.
            materialized_ds = ds
            logger.info(
                f"Saving materialized dataset with {len(materialized_ds)} examples to temp_dir..."
            )
            materialized_ds.save_to_disk(temp_dir)

            # Move to final location (or copy if moving fails)
            import shutil

            try:
                shutil.move(str(temp_dir), str(out_dir))
            except:
                shutil.copytree(str(temp_dir), str(out_dir), dirs_exist_ok=True)
                shutil.rmtree(str(temp_dir), ignore_errors=True)

            # The dataset is already in 'materialized_ds' from line 253.
            # No need to reload from disk for verification and JSON serialization.
            logger.info(
                f"Saved dataset to {out_dir} with {len(materialized_ds)} examples"
            )
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            return None

        # Create a small sample for JSON debugging (first 100 examples max)
        sample_size = min(100, len(materialized_ds))
        sample_ds = materialized_ds.select(range(sample_size))

        # Save to JSON for debugging (using only the sample for memory efficiency)
        if sample_size == 0:
            logger.warning("No parsed examples available to save to parsed_debug.json!")
        else:
            logger.info(f"Saving {sample_size} sample records to JSON for debugging.")
        self.save_to_json(sample_ds)

        return materialized_ds

    def save_to_json(self, ds: Dataset, output_path: Optional[str] = None) -> str:
        """
        Serialize the parsed OMOP dataset to JSON for debugging purposes.

        Args:
            ds: Dataset to serialize
            output_path: Optional path to save the JSON file. If None, uses processed_dir/name/parsed_debug.json

        Returns:
            Path to the saved JSON file
        """
        if output_path is None:
            output_path = str(
                Path(self.processed_dir) / self.name / "parsed_debug.json"
            )

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert HuggingFace Dataset to list of dicts directly (avoid pandas)
        data_dicts = [dict(ds[i]) for i in range(len(ds))]

        if not data_dicts:
            logger.warning(
                f"No parsed content to write to {output_path}! Writing diagnostic message."
            )
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "error": "No parsed examples available. Check transformation and filtering logic."
                    },
                    f,
                    indent=2,
                )
            return output_path

        # Diagnostic logging for the first example
        if data_dicts:
            first_example = data_dicts[0]
            logger.info(f"First example keys for JSON: {list(first_example.keys())}")
            for key, value in first_example.items():
                logger.info(f"  Key: '{key}', Type: {type(value)}")
                if isinstance(value, list) and value:
                    logger.info(
                        f"    First element type in list '{key}': {type(value[0])}"
                    )

        logger.info(f"Saving {len(data_dicts)} parsed records to JSON for debugging.")
        with open(output_path, "w") as f:
            json.dump(data_dicts, f, indent=2, cls=CustomJSONEncoder)
        logger.info(f"Saved parsed OMOP data to JSON for debugging: {output_path}")
        return output_path


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
        logger.debug(
            f"transform_to_sat_format: Returning None because events or patient_id is missing for patient {patient_id}"
        )
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
    string_values = []  # Added to preserve string_value data

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
        str_value = event.get("string_value")  # Get string value from event

        # Add code to the list
        codes.append(code)
        event_times.append(event_time)
        # Store string_value (None if not present)
        string_values.append(str_value)

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
        logger.debug(
            f"transform_to_sat_format: Returning None because no codes were generated for patient {patient_id}"
        )
        return None

    # Create the SAT format output
    result = {
        primary_key: patient_id,
        "x": " ".join(codes),
        "modality": modalities,
        "numerics": numerics,
        "time": event_times,
        "string_values": string_values,  # Add string_values to the result
    }

    # Copy any outcome/target variables that might be present
    # (already processed by cohort_omop.py)
    for field in ["duration", "event", "event_type", "competing_event"]:
        if field in example:
            result[field] = example[field]

    logger.debug(
        f"transform_to_sat_format: For patient {patient_id}, returning result with keys: {list(result.keys())}"
    )
    return result
