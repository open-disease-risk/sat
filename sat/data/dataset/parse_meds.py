"""Process MEDS (Medical Event Data Standard) data.

1. Read the Parquet file in MEDS format using official schema
2. Process events and timelines using medical labelers
3. Transform to SAT's survival analysis format with events and durations lists
4. Split into train/val/test
5. Save as pandas dataframes

The Medical Event Data Standard (MEDS) schema comes from the meds package:
https://github.com/Medical-Event-Data-Standard/meds
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import multiprocessing
from dataclasses import dataclass
from logging import DEBUG, ERROR
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from datasets import load_dataset
from logdecorator import log_on_end, log_on_error, log_on_start
from sklearn.model_selection import KFold
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
)

# Import MEDS schema
# Import custom medical labelers
from sat.data.dataset.medical_labelers import (
    CompetingRiskLabeler,
    CustomEventLabeler,
    MortalityLabeler,
    ProcessingMode,
    RiskFactorLabeler,
)
from sat.utils import logging
from sat.utils.data import train_val_test

logger = logging.get_default_logger()


def tokens(row, modalities):
    """Extract tokens based on modality vector.

    For categorical/discrete features (modality=0), use the actual token.
    For numeric features (modality=1), use the feature name.
    """
    (idx,) = np.where(np.array(modalities) == 0)
    toks = list(row.index[0 : len(modalities)])
    for i in idx:
        toks[i] = row.iloc[i]
    return list(toks)


def numerics(row, modalities):
    """Extract numeric values based on modality vector.

    For categorical/discrete features (modality=0), use 1.0.
    For numeric features (modality=1), use the actual value.
    """
    (idx,) = np.where(np.array(modalities) == 1)
    nums = [1.0] * len(modalities)
    for i in idx:
        nums[i] = row.iloc[i]
    return nums


@dataclass
class meds:
    source: str  # Path to MEDS format parquet file
    processed_dir: str  # Output directory
    train_ratio: float  # Ratio for training set
    validation_ratio: float  # Ratio for validation set
    test_ratio: float  # Ratio for test set
    n_bins: int  # Number of bins for discretization
    encode: str  # Encoding method
    strategy: str  # Discretization strategy
    name: str  # Dataset name
    time_field: str = "days"  # Field to use for time measurements
    label_definitions: Optional[List[Dict[str, Any]]] = None  # FEMR label definitions (deprecated)
    event_fields: Optional[List[str]] = (
        None  # Optional list of specific events to include
    )
    kfold: int = 0  # Number of folds for cross-validation
    scale_numerics: bool = True  # Whether to scale numeric features
    scale_method: str = "standard"  # Scaling method to use: 'min_max' or 'standard'
    min_scale_numerics: float = 1.0  # Minimum value for min-max scaling
    labelers: Optional[List[Dict[str, Any]]] = (
        None  # Custom labelers for event identification
    )
    risk_factors: Optional[List[str]] = None  # Risk factors to identify

    @log_on_start(DEBUG, "Create MEDS data representation...")
    @log_on_error(
        ERROR,
        "Error creating MEDS data: {e!r}",
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!")
    def prepare(self) -> None:
        """Process MEDS data using FEMR and HuggingFace datasets."""
        return self._prepare_with_femr()

    def _prepare_with_femr(self) -> None:
        """Process MEDS data using FEMR and HuggingFace datasets for generating labels."""
        # 1. Read MEDS-format data
        logger.debug(f"Reading MEDS data from {self.source} using HuggingFace datasets")

        # Load the MEDS data using HuggingFace datasets
        # This can be a directory of parquet files or a single parquet file
        try:
            # Check if source exists
            source_path = Path(self.source)
            if not source_path.exists():
                raise FileNotFoundError(f"MEDS data source not found: {self.source}")

            # Check if it's a file and has content
            if source_path.is_file() and source_path.stat().st_size == 0:
                raise ValueError(f"MEDS data file is empty (0 bytes): {self.source}")

            # If it's a directory, check for parquet files
            if source_path.is_dir():
                parquet_files = list(source_path.glob("*.parquet"))
                if not parquet_files:
                    raise FileNotFoundError(
                        f"No parquet files found in directory: {self.source}"
                    )

                # Check if any parquet files are empty
                empty_files = [f for f in parquet_files if f.stat().st_size == 0]
                if empty_files:
                    raise ValueError(
                        f"Found empty parquet files: {', '.join(str(f) for f in empty_files)}"
                    )

                logger.debug(
                    f"Found {len(parquet_files)} parquet files in {self.source}"
                )

            # Determine how to load the dataset based on source format
            logger.debug(f"Loading dataset from {self.source}")

            if source_path.is_dir():
                # For directory, directly load each parquet file separately and combine manually
                parquet_files = list(source_path.glob("*.parquet"))
                logger.info(
                    f"Found {len(parquet_files)} parquet files in {source_path}"
                )

                if not parquet_files:
                    raise ValueError(
                        f"No parquet files found in directory {source_path}"
                    )

                # Create a dictionary mapping table names to their dataframes
                tables = {}

                # Process each parquet file individually
                for file_path in parquet_files:
                    try:
                        # Extract table name from filename
                        parts = file_path.stem.split("_")
                        if len(parts) > 1:
                            table_name = parts[-1]  # Last part as table name
                        else:
                            table_name = file_path.stem

                        logger.info(f"Loading {file_path.name} as table '{table_name}'")

                        # Read the parquet file directly using pandas
                        df = pd.read_parquet(file_path)
                        tables[table_name] = df
                        logger.debug(f"Loaded table {table_name} with shape {df.shape}")

                    except Exception as e:
                        logger.warning(f"Error loading file {file_path}: {e}")

                if not tables:
                    raise ValueError(
                        f"Could not load any valid parquet files from {source_path}"
                    )

                logger.info(
                    f"Successfully loaded {len(tables)} tables: {', '.join(tables.keys())}"
                )

                # Create a dataset-like structure
                class MedsDataset:
                    def __init__(self, tables):
                        self.tables = tables
                        self.column_names = list(tables.keys())

                    def __getitem__(self, key):
                        return self.tables.get(key)

                dataset = MedsDataset(tables)
            else:
                # For a single file
                dataset = load_dataset(
                    "parquet", data_files={"data": str(self.source)}, split="train"
                )

        except Exception as e:
            logger.error(f"Failed to load MEDS data: {e}")
            raise ValueError(f"Error loading MEDS data from {self.source}: {e}") from e

        # Extract all patient data
        try:
            logger.debug("Extracting patient data")

            # For our custom MedsDataset from directory
            if hasattr(dataset, "tables"):
                logger.debug(f"Available tables: {list(dataset.tables.keys())}")

                # Try to find patients table
                if "patients" in dataset.tables:
                    logger.debug("Found 'patients' table")
                    patient_data = dataset.tables["patients"]
                else:
                    # Look for tables with 'patient' in the name
                    patient_tables = [
                        name
                        for name in dataset.tables.keys()
                        if "patient" in name.lower()
                    ]
                    if patient_tables:
                        logger.debug(f"Using '{patient_tables[0]}' as patient data")
                        patient_data = dataset.tables[patient_tables[0]]
                    else:
                        # Use the first table that isn't an event table as fallback
                        event_tables = [
                            "mortality",
                            "hospitalizations",
                            "medications",
                            "diagnoses",
                            "labs",
                        ]
                        for table_name in dataset.tables:
                            if table_name.lower() not in [
                                t.lower() for t in event_tables
                            ]:
                                logger.debug(
                                    f"Using '{table_name}' as patient data (default)"
                                )
                                patient_data = dataset.tables[table_name]
                                break
                        else:
                            # No suitable table found
                            raise ValueError(
                                f"No patient data table found in: {list(dataset.tables.keys())}"
                            )

            # For standard HuggingFace dataset
            elif hasattr(dataset, "column_names"):
                logger.debug(f"Dataset column names: {dataset.column_names}")

                if "patients" in dataset.column_names:
                    logger.debug("Found 'patients' column in dataset")
                    patient_data = pd.DataFrame(dataset["patients"])
                else:
                    # Look for columns with 'patient' in the name
                    patient_cols = [
                        name
                        for name in dataset.column_names
                        if "patient" in name.lower()
                    ]
                    if patient_cols:
                        logger.debug(f"Using '{patient_cols[0]}' as patient data")
                        patient_data = pd.DataFrame(dataset[patient_cols[0]])
                    else:
                        # Use first column as fallback
                        logger.warning(
                            f"Using first column '{dataset.column_names[0]}' as patient data"
                        )
                        patient_data = pd.DataFrame(dataset[dataset.column_names[0]])

            # Other dataset formats
            else:
                logger.warning(
                    "Unknown dataset structure, attempting direct conversion"
                )
                patient_data = pd.DataFrame(dataset)

            # Ensure we have a DataFrame
            if not isinstance(patient_data, pd.DataFrame):
                logger.debug("Converting patient data to DataFrame")
                patient_data = pd.DataFrame(patient_data)

            # Verify we have patient data
            if patient_data.empty:
                raise ValueError("Extracted patient data is empty")

            logger.info(f"Extracted patient data with shape: {patient_data.shape}")

        except Exception as e:
            logger.error(f"Error extracting patient data: {e}")
            raise ValueError(f"Failed to extract patient data: {e}") from e

        # Process each label definition to extract events
        logger.debug("Processing event definitions")
        event_data = []
        event_types = []

        # Check if we have any event specifications
        if not self.label_definitions and not self.labelers:
            logger.warning("Neither label_definitions nor labelers specified. Using default approach to find events.")
        
        # Process label definitions if provided (for backward compatibility)
        if self.label_definitions:
            for label_def in self.label_definitions:
                logger.debug(f"Processing label definition: {label_def}")
                event_name = label_def.get("name", "unknown_event")
                event_types.append(event_name)
    
                # Get event table name from definition
                table_name = label_def.get("table_name", event_name)
                time_field = label_def.get("time_field", "days")

                # Try to get the corresponding event table
                if hasattr(dataset, "tables") and table_name in dataset.tables:
                    # Our custom MedsDataset structure
                    logger.debug(f"Found table '{table_name}' in the dataset")
                    event_table = dataset.tables[table_name]

                    # Process the events
                    if not event_table.empty and "patient_id" in event_table.columns:
                        logger.debug(
                            f"Processing {len(event_table)} events from '{table_name}'"
                        )

                        # Extract time values and standardize columns
                        df_event = event_table.copy()

                        # Ensure we have the necessary time field
                        if time_field in df_event.columns:
                            # Add event type column if not present
                            if "event_type" not in df_event.columns:
                                df_event["event_type"] = event_name

                            # Rename time field to standard name if needed
                            if time_field != "time":
                                df_event = df_event.rename(columns={time_field: "time"})

                            # Add to collected events
                            event_data.append(df_event)
                            logger.info(
                                f"Added {len(df_event)} events of type '{event_name}'"
                            )
                        else:
                            logger.warning(
                                f"Time field '{time_field}' not found in table '{table_name}'"
                            )
                    else:
                        logger.warning(
                            f"Table '{table_name}' has no patient_id column or is empty"
                        )
                else:
                    logger.warning(f"Event table '{table_name}' not found in dataset")

                # Try alternate approach for standard HuggingFace dataset
                if (
                    hasattr(dataset, "column_names")
                    and table_name in dataset.column_names
                ):
                    logger.debug(f"Found column '{table_name}' in the dataset")
                    try:
                        events = dataset[table_name]
                        df_event = pd.DataFrame(events)

                        if not df_event.empty and "patient_id" in df_event.columns:
                            # Add event type if needed
                            if "event_type" not in df_event.columns:
                                df_event["event_type"] = event_name

                            # Rename time field if needed
                            if time_field in df_event.columns and time_field != "time":
                                df_event = df_event.rename(columns={time_field: "time"})

                            event_data.append(df_event)
                            logger.info(
                                f"Added {len(df_event)} events of type '{event_name}'"
                            )
                        else:
                            logger.warning(
                                f"Column '{table_name}' has no patient_id or is empty"
                            )
                    except Exception as e:
                        logger.warning(f"Error processing column '{table_name}': {e}")
                else:
                    logger.warning(f"No event source found for '{event_name}'")

        # If no events were collected (either no label definitions provided or none found), try direct approach
        if not event_data and hasattr(dataset, "tables"):
            logger.warning(
                "No events found using existing approaches. Trying direct table approach."
            )

            # Check for standard event tables by common names
            standard_event_tables = ["mortality", "hospitalizations", "readmissions"]
            for table_name in standard_event_tables:
                if table_name in dataset.tables:
                    logger.debug(f"Found standard event table '{table_name}'")
                    event_table = dataset.tables[table_name]

                    if not event_table.empty and "patient_id" in event_table.columns:
                        # Add event type if not present
                        df_event = event_table.copy()
                        if "event_type" not in df_event.columns:
                            df_event["event_type"] = table_name

                        # Look for time field
                        time_fields = [
                            f
                            for f in df_event.columns
                            if "day" in f.lower() or "time" in f.lower()
                        ]
                        if time_fields:
                            time_field = time_fields[0]
                            if time_field != "time":
                                df_event = df_event.rename(columns={time_field: "time"})

                            event_types.append(table_name)
                            event_data.append(df_event)
                            logger.info(
                                f"Added {len(df_event)} events of type '{table_name}'"
                            )
                        else:
                            logger.warning(f"No time field found in '{table_name}'")
                    else:
                        logger.warning(
                            f"Table '{table_name}' has no patient_id or is empty"
                        )

        # 2. Transform data to survival analysis format
        logger.debug("Transforming MEDS data to survival analysis format")

        # If no events were found and we don't have custom labelers, raise an error
        if not event_data and not self.labelers:
            # If label definitions were provided, mention them in the error
            if self.label_definitions:
                raise ValueError("No events found using the provided label definitions")
            else:
                raise ValueError("No events found in the dataset. Please specify labelers or label_definitions.")

        # Combine all event data
        df_events = pd.concat(event_data, axis=0)

        # Standardize column names
        if "time" not in df_events.columns and self.time_field in df_events.columns:
            df_events["time"] = df_events[self.time_field]

        # Create a mapping of event types to integer codes
        event_type_map = {event: idx + 1 for idx, event in enumerate(event_types)}

        # Map event types to integers (0 = censored, 1+ = event types)
        # In SAT, 0 usually indicates censored data
        df_events["event"] = (
            df_events["event_type"].map(event_type_map).fillna(0).astype(int)
        )

        # Filter to only include patients with events
        patient_ids = df_events["patient_id"].unique()
        patient_data = patient_data[patient_data["patient_id"].isin(patient_ids)]

        # Sort events by patient_id and time
        df_events = df_events.sort_values(by=["patient_id", "time"])

        # Group events by patient to capture the full history
        patient_histories = {}
        for patient_id, group in df_events.groupby("patient_id"):
            # Collect all events for this patient
            events = []
            for _, row in group.iterrows():
                event_info = {
                    "event_type": (
                        row["event_type"] if "event_type" in row else row["event"]
                    ),
                    "time": row["time"],
                    "event_code": row["event"] if "event" in row else 1,
                }
                # Add other event-specific columns
                for col in row.index:
                    if col not in [
                        "patient_id",
                        "time",
                        "event",
                        "event_type",
                    ] and not pd.isna(row[col]):
                        event_info[col] = row[col]
                events.append(event_info)

            # Store the patient's history
            patient_histories[patient_id] = events

        # For outcome analysis, still need a row with first/any event to determine censoring
        df_first_events = df_events.drop_duplicates(subset=["patient_id"], keep="first")

        # Merge patient features with first event data for convenience
        df_combined = pd.merge(
            df_first_events[["patient_id", "time", "event"]],
            patient_data,
            on="patient_id",
            how="inner",
        )

        # Add history length as a feature
        df_combined["history_length"] = df_combined["patient_id"].apply(
            lambda pid: len(patient_histories.get(pid, []))
        )

        # Add event count for each patient as a feature
        df_combined["event_count"] = df_combined["patient_id"].apply(
            lambda pid: len(patient_histories.get(pid, []))
        )

        # Add event pattern features - presence of specific events
        event_types = set()
        for histories in patient_histories.values():
            for event in histories:
                event_types.add(event["event_type"])

        # Create binary features for presence of each event type
        for event_type in event_types:
            df_combined[f"has_{event_type}"] = df_combined["patient_id"].apply(
                lambda pid: any(
                    event["event_type"] == event_type
                    for event in patient_histories.get(pid, [])
                )
            )

        # Store the full histories for later reference
        df_combined["full_history"] = df_combined["patient_id"].apply(
            lambda pid: patient_histories.get(pid, [])
        )

        # Create features that capture event sequences (first 3 events)
        logger.debug("Creating event sequence features")
        df_combined["event_sequence"] = df_combined["patient_id"].apply(
            lambda pid: "_".join(
                [
                    str(event["event_type"])
                    for event in sorted(
                        patient_histories.get(pid, []), key=lambda x: x["time"]
                    )[:3]
                ]
            )
        )

        # Create time between events features
        df_combined["time_between_events"] = df_combined["patient_id"].apply(
            lambda pid: self._calculate_time_between_events(
                patient_histories.get(pid, [])
            )
        )

        # Create feature for event density (events per time unit)
        df_combined["event_density"] = df_combined.apply(
            lambda row: len(patient_histories.get(row["patient_id"], []))
            / max(1, row["time"]),
            axis=1,
        )

        # Continue with feature processing and data splitting
        self._process_features_and_split(df_combined, event_type_map, len(df_events))

    def _calculate_time_between_events(self, events):
        """Calculate the average time between consecutive events."""
        if len(events) < 2:
            return 0.0

        # Sort events by time
        sorted_events = sorted(events, key=lambda x: x["time"])

        # Calculate time differences between consecutive events
        time_diffs = [
            sorted_events[i + 1]["time"] - sorted_events[i]["time"]
            for i in range(len(sorted_events) - 1)
        ]

        # Return average time difference
        return sum(time_diffs) / len(time_diffs)

    def _create_custom_labelers(self, events_df):
        """Create and initialize custom medical labelers based on configuration.

        Args:
            events_df: DataFrame with all events in MEDS format

        Returns:
            List of initialized labeler instances
        """
        # Initialize list for labelers
        initialized_labelers = []

        # If no custom labelers specified, use default competing risk labeler
        if not self.labelers:
            logger.debug(
                "No custom labelers specified, using default CompetingRiskLabeler"
            )

            # Create default event codes
            event_codes = {
                "death": [MEDS_DEATH_CODE],
                "hospitalization": ["ENC_INPATIENT"]
            }
            
            # Add additional codes from label_definitions if they exist (for backward compatibility)
            if self.label_definitions:
                for label_def in self.label_definitions:
                    event_name = label_def.get("name", "unknown_event")
                    if event_name not in event_codes:
                        event_codes[event_name] = [
                            event_name.upper()
                        ]  # Default code is uppercase event name

            competing_risk_labeler = CompetingRiskLabeler(
                name="default_competing_risk_labeler",
                event_codes=event_codes,
                max_followup_days=1095,  # 3 years default follow-up
            )
            initialized_labelers.append(competing_risk_labeler)

            # Add default mortality labeler
            mortality_labeler = MortalityLabeler(
                name="default_mortality_labeler", max_followup_days=1095
            )
            initialized_labelers.append(mortality_labeler)

            return initialized_labelers

        # Custom labelers were specified in config
        logger.debug(f"Initializing {len(self.labelers)} custom labelers")

        for labeler_config in self.labelers:
            labeler_type = labeler_config.get("type", "").lower()
            labeler_name = labeler_config.get("name", f"{labeler_type}_labeler")

            # Get common parameters
            max_followup_days = labeler_config.get("max_followup_days", 1095)
            enrollment_codes = labeler_config.get("enrollment_codes", ["ENROLLMENT"])

            if labeler_type == "risk_factor":
                # Initialize risk factor labeler
                logger.debug(f"Creating RiskFactorLabeler: {labeler_name}")
                # Get custom codes in new format
                custom_codes = labeler_config.get("custom_codes", {})

                labeler = RiskFactorLabeler(
                    name=labeler_name,
                    custom_codes=custom_codes
                )

                initialized_labelers.append(labeler)

            elif labeler_type == "mortality":
                # Initialize mortality labeler
                logger.debug(f"Creating MortalityLabeler: {labeler_name}")
                death_codes = labeler_config.get("death_codes", ["MEDS_DEATH"])

                labeler = MortalityLabeler(
                    name=labeler_name,
                    max_followup_days=max_followup_days,
                    death_codes=death_codes,
                    enrollment_codes=enrollment_codes
                )
                initialized_labelers.append(labeler)

            elif labeler_type == "competing_risk":
                # Initialize competing risk labeler
                logger.debug(f"Creating CompetingRiskLabeler: {labeler_name}")
                event_codes = labeler_config.get("event_codes", {})

                labeler = CompetingRiskLabeler(
                    name=labeler_name,
                    event_codes=event_codes,
                    max_followup_days=max_followup_days,
                    enrollment_codes=enrollment_codes
                )
                initialized_labelers.append(labeler)

            elif labeler_type == "custom_event":
                # Initialize custom event labeler
                logger.debug(f"Creating CustomEventLabeler: {labeler_name}")
                event_definition = labeler_config.get("event_definition", {})

                if not event_definition:
                    logger.warning(
                        f"No event definition provided for CustomEventLabeler: {labeler_name}"
                    )
                    continue

                labeler = CustomEventLabeler(
                    name=labeler_name,
                    event_definition=event_definition,
                    max_followup_days=max_followup_days,
                    enrollment_codes=enrollment_codes
                )
                initialized_labelers.append(labeler)

            else:
                logger.warning(f"Unknown labeler type: {labeler_type}, skipping")

        return initialized_labelers

    def _apply_labelers(self, events_df, patient_histories):
        """Apply custom labelers to identify events and risk factors.

        Args:
            events_df: DataFrame with all events in MEDS format
            patient_histories: Dictionary mapping patient IDs to their event histories

        Returns:
            Dictionary with labeled events and risk factors for each patient
        """
        logger.debug("Applying custom labelers to identify events and risk factors")

        # Make sure events_df has the column "subject_id" for our MEDS-compatible labelers
        if "patient_id" in events_df.columns and "subject_id" not in events_df.columns:
            events_df = events_df.rename(columns={"patient_id": "subject_id"})

        # Make sure time field is properly set
        if "time_type" not in events_df.columns:
            # If missing, set static events (demographics) as static, all others as event
            if "code" in events_df.columns:
                demographic_codes = ["SEX", "RACE", "ETHNICITY"]
                events_df["time_type"] = "event"  # Default to event type
                for code in demographic_codes:
                    demographic_mask = events_df["code"].str.contains(code, na=False)
                    events_df.loc[demographic_mask, "time_type"] = "static"
            else:
                # No code column, default all to event type
                events_df["time_type"] = "event"

        # Create the labelers
        labelers = self._create_custom_labelers(events_df)
        
        # Store labeler order for later use
        labeler_names = [labeler.name for labeler in labelers]
        logger.debug(f"Labelers will be processed in this order: {labeler_names}")

        # Initialize results dictionary
        results = {}

        # Determine best processing mode based on dataset size
        num_patients = len(events_df["subject_id"].unique()) if "subject_id" in events_df.columns else 0

        if num_patients > 5000:
            # For large datasets, use multiprocessing
            processing_mode = ProcessingMode.MULTIPROCESSING
            # Use half of available CPUs or 4, whichever is smaller
            n_jobs = min(4, max(1, multiprocessing.cpu_count() // 2))
            logger.debug(f"Using multiprocessing with {n_jobs} workers for {num_patients} patients")
        else:
            # For smaller datasets, serial processing is fine
            processing_mode = ProcessingMode.SERIAL
            n_jobs = 1
            logger.debug(f"Using serial processing for {num_patients} patients")

        # First phase: Process all labelers and collect raw results
        raw_results = {}
        
        # Apply each labeler to the events, using parallel processing
        for labeler in labelers:
            logger.debug(f"Applying labeler: {labeler.name} with {processing_mode.value} processing")

            try:
                # Use parallel processing
                labeler_results = labeler.parallel_label(
                    events_df,
                    mode=processing_mode,
                    n_jobs=n_jobs,
                    batch_size=1000,
                    show_progress=True
                )
                
                # Store all labeler results in the raw results dictionary
                for patient_id, data in labeler_results.items():
                    # Handle potential string IDs (convert to int if possible)
                    try:
                        if isinstance(patient_id, str) and patient_id.isdigit():
                            patient_id = int(patient_id)
                    except (ValueError, TypeError):
                        # Keep as string if conversion fails
                        pass
                        
                    if patient_id not in raw_results:
                        raw_results[patient_id] = {}
                    
                    # Store all results for this labeler
                    raw_results[patient_id][labeler.name] = data
                
            except Exception as e:
                logger.error(f"Error applying labeler {labeler.name} with parallel processing: {e}")
                logger.error(f"Falling back to serial processing for {labeler.name}")

                # Fall back to the legacy method if parallel processing fails
                try:
                    labeler_results = labeler.label(events_df)
                    
                    # Store all labeler results in the raw results dictionary
                    for patient_id, data in labeler_results.items():
                        # Convert patient_id to int if it's a string (from older implementation)
                        # We preserve the original string-based ID if it's not convertible to int
                        try:
                            if isinstance(patient_id, str) and patient_id.isdigit():
                                patient_id = int(patient_id)
                        except (ValueError, TypeError):
                            # Keep as string if conversion fails
                            pass
                            
                        if patient_id not in raw_results:
                            raw_results[patient_id] = {}
                        
                        # Store all results for this labeler
                        raw_results[patient_id][labeler.name] = data
                        
                except Exception as e2:
                    logger.error(f"Fatal error with labeler {labeler.name}: {e2}")

        # Second phase: Convert raw results to ordered events/durations lists
        for patient_id in raw_results.keys():
            # Initialize ordered lists
            ordered_events = []
            ordered_durations = []
            ordered_event_types = []
            
            # Get max time for this patient (for censoring times)
            if patient_id in patient_histories:
                max_time = max([e["time"] for e in patient_histories[patient_id]], default=1095)
            else:
                max_time = 1095  # Default to 3 years if no history available
            
            # Process each labeler in the correct order
            for labeler_idx, labeler in enumerate(labelers):
                labeler_name = labeler.name
                
                # Check if we have results for this labeler and patient
                if labeler_name in raw_results.get(patient_id, {}):
                    patient_labeler_data = raw_results[patient_id][labeler_name]
                    
                    # Handle different labeler types
                    if isinstance(labeler, CompetingRiskLabeler):
                        # Competing Risk Labeler returns a list of events
                        if "events" in patient_labeler_data and "durations" in patient_labeler_data:
                            events_list = patient_labeler_data["events"]
                            durations_list = patient_labeler_data["durations"]
                            
                            # Since all events from this labeler share the same prefix, we can add them all
                            if "event_types" in patient_labeler_data:
                                event_types = patient_labeler_data["event_types"]
                                # Add one entry per event type
                                for i, event_type in enumerate(event_types):
                                    if i < len(events_list) and i < len(durations_list):
                                        ordered_events.append(events_list[i])
                                        ordered_durations.append(durations_list[i])
                                        ordered_event_types.append(f"{labeler_name}_{event_type}")
                            else:
                                # No event types provided, use generic event names
                                for i, (event, duration) in enumerate(zip(events_list, durations_list)):
                                    ordered_events.append(event)
                                    ordered_durations.append(duration)
                                    ordered_event_types.append(f"{labeler_name}_event{i+1}")
                        else:
                            # Add a censored event if we don't have proper data
                            ordered_events.append(0)  # Censored
                            ordered_durations.append(float(max_time))
                            ordered_event_types.append(labeler_name)
                            
                    elif isinstance(labeler, (MortalityLabeler, CustomEventLabeler)):
                        # These labelers return a single event/time pair
                        event_status = patient_labeler_data.get("event", 0)
                        event_time = patient_labeler_data.get("time", max_time)
                        
                        ordered_events.append(event_status)
                        ordered_durations.append(float(event_time))
                        ordered_event_types.append(labeler_name)
                        
                    elif isinstance(labeler, RiskFactorLabeler):
                        # Risk factors don't necessarily generate events
                        if "risk_factors" in patient_labeler_data and len(patient_labeler_data["risk_factors"]) > 0:
                            # If risk factors exist, add a censored event entry
                            ordered_events.append(0)  # Censored
                            ordered_durations.append(float(max_time))
                            ordered_event_types.append(labeler_name)
                            
                            # Also store risk factors separately
                            if patient_id not in results:
                                results[patient_id] = {}
                            results[patient_id]["risk_factors"] = patient_labeler_data["risk_factors"]
                else:
                    # No results for this labeler, add a censored event unless it's a risk factor labeler
                    if not isinstance(labeler, RiskFactorLabeler):
                        ordered_events.append(0)  # Censored
                        ordered_durations.append(float(max_time))
                        ordered_event_types.append(labeler_name)
            
            # If we don't have any labelers' events (rare case), provide a default
            if len(ordered_events) == 0:
                ordered_events = [0]  # Default to censored
                ordered_durations = [float(max_time)]
                ordered_event_types = ["default"]
            
            # Store the final ordered results for this patient
            if patient_id not in results:
                results[patient_id] = {}
                
            results[patient_id]["events"] = ordered_events
            results[patient_id]["durations"] = ordered_durations
            results[patient_id]["event_types"] = ordered_event_types
            
            # Log some patients for debugging
            # Use int() conversion to ensure we can use the modulo operator
            # For string IDs that can't be converted to int, use a hash-based approach
            try:
                log_condition = int(patient_id) % 100 == 0
            except (ValueError, TypeError):
                # For non-integer IDs, use a hash-based approach to log some patients
                log_condition = hash(str(patient_id)) % 100 == 0
                
            if log_condition:  # Only log some patients to avoid spam
                logger.debug(
                    f"Patient {patient_id} has {len(ordered_events)} events: "
                    f"types={ordered_event_types}, "
                    f"statuses={ordered_events}, "
                    f"durations={ordered_durations}"
                )
        
        # Ensure all patients from the histories are included
        for patient_id in patient_histories:
            if patient_id not in results:
                # Patient wasn't processed, give default values
                max_time = max([e["time"] for e in patient_histories[patient_id]], default=1095)
                
                # Create default ordered lists based on labelers
                ordered_events = []
                ordered_durations = []
                ordered_event_types = []
                
                for labeler in labelers:
                    if not isinstance(labeler, RiskFactorLabeler):
                        ordered_events.append(0)  # Censored
                        ordered_durations.append(float(max_time))
                        ordered_event_types.append(labeler.name)
                
                # Store with defaults
                results[patient_id] = {
                    "events": ordered_events if ordered_events else [0],
                    "durations": ordered_durations if ordered_durations else [float(max_time)],
                    "event_types": ordered_event_types if ordered_event_types else ["default"]
                }
                
                # Log some patients for debugging
                try:
                    log_condition = int(patient_id) % 100 == 0
                except (ValueError, TypeError):
                    # For non-integer IDs, use a hash-based approach to log some patients
                    log_condition = hash(str(patient_id)) % 100 == 0
                    
                if log_condition:
                    logger.debug(
                        f"Added default events for patient {patient_id}: "
                        f"types={ordered_event_types}, "
                        f"statuses={ordered_events}, "
                        f"durations={ordered_durations}"
                    )

        return results

    def _process_features_and_split(self, df_combined, event_type_map, total_events):
        """Process features and split data into train/val/test sets."""
        # Get feature columns (exclude metadata columns)
        meta_cols = ["patient_id", "time", "event", "full_history"]
        date_cols = [col for col in df_combined.columns if "date" in col.lower()]
        meta_cols.extend(date_cols)
        feature_cols = [col for col in df_combined.columns if col not in meta_cols]

        # Log the features that include event history information
        history_features = [
            col
            for col in feature_cols
            if col.startswith("has_") or col in ["history_length", "event_count"]
        ]
        logger.debug(
            f"Including {len(history_features)} history-derived features: {history_features}"
        )

        # Apply labelers if present to extract events from medical histories
        if self.labelers or self.risk_factors:
            logger.debug(
                "Using custom medical labelers to identify events and risk factors"
            )

            # Extract full history for processing
            patient_histories = {}
            for _, row in df_combined.iterrows():
                patient_id = row["patient_id"]
                history = row["full_history"]
                patient_histories[patient_id] = history

            # Create a proper MEDS format DataFrame from all events for labelers
            all_events = []
            for patient_id, events in patient_histories.items():
                for event in events:
                    event_data = {
                        "subject_id": patient_id,
                        "time": event.get("time", None),
                        "code": event.get("event_type", event.get("code", "UNKNOWN")),
                        "numeric_value": event.get("numeric_value", None),
                        "string_value": event.get("string_value", None),
                    }
                    all_events.append(event_data)

            events_df = pd.DataFrame(all_events)

            # Apply labelers to get labeled events and risk factors
            labeled_data = self._apply_labelers(events_df, patient_histories)

            # Add labeled data to features
            for index, row in df_combined.iterrows():
                patient_id = row["patient_id"]

                if patient_id in labeled_data:
                    patient_data = labeled_data[patient_id]

                    # Add risk factor features if available
                    for key, value in patient_data.items():
                        if key.startswith("has_") and key not in df_combined.columns:
                            df_combined.at[index, key] = value

                    # Use labeled events/durations if available
                    if "events" in patient_data and "durations" in patient_data:
                        # Create the columns if they don't exist
                        if "labeled_events" not in df_combined.columns:
                            df_combined["labeled_events"] = None
                        if "labeled_durations" not in df_combined.columns:
                            df_combined["labeled_durations"] = None
                        if "event_types" not in df_combined.columns and "event_types" in patient_data:
                            df_combined["event_types"] = None
                            
                        # Set values
                        df_combined.at[index, "labeled_events"] = patient_data["events"]
                        df_combined.at[index, "labeled_durations"] = patient_data["durations"]
                        if "event_types" in patient_data:
                            df_combined.at[index, "event_types"] = patient_data["event_types"]

            # Update feature columns to include new risk factor features
            feature_cols = [
                col
                for col in df_combined.columns
                if col
                not in meta_cols
                + ["labeled_events", "labeled_durations", "event_types"]
            ]

        # 4. Preprocess features
        logger.debug("Preprocessing features")
        df_features = df_combined[feature_cols]

        # Handle categorical and numerical features differently
        categorical_features = df_features.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        numerical_features = df_features.select_dtypes(
            include=["number"]
        ).columns.tolist()

        # Create modality vector (0 for categorical, 1 for numerical)
        modality = [0] * len(categorical_features) + [1] * len(numerical_features)
        logger.debug(f"Modality vector: {modality}")

        # Scale numerical features if needed
        df_numerical = df_features[numerical_features].fillna(
            df_features[numerical_features].median()
        )

        if self.scale_numerics and len(numerical_features) > 0:
            logger.debug(f"Scaling numerical features using {self.scale_method} method")
            if self.scale_method == "min_max":
                scaler = MinMaxScaler()
                df_numerical_scaled = pd.DataFrame(
                    scaler.fit_transform(df_numerical) + self.min_scale_numerics,
                    columns=numerical_features,
                    index=df_numerical.index,
                )
            elif self.scale_method == "standard":
                scaler = StandardScaler()
                df_numerical_scaled = pd.DataFrame(
                    scaler.fit_transform(df_numerical),
                    columns=numerical_features,
                    index=df_numerical.index,
                )
            else:
                raise ValueError(
                    f"scale_method {self.scale_method} not supported. Use 'min_max' or 'standard'"
                )
        else:
            df_numerical_scaled = df_numerical

        # Process numerical features for tokenization
        if len(numerical_features) > 0:
            numerical_discretized = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode=self.encode,
                strategy=self.strategy,
            ).fit_transform(df_numerical.fillna(df_numerical.median()))
            df_numerical_disc = pd.DataFrame(
                numerical_discretized,
                columns=numerical_features,
                index=df_numerical.index,
            )
        else:
            df_numerical_disc = pd.DataFrame(index=df_features.index)

        # Process categorical features
        df_categorical = df_features[categorical_features].fillna("unknown")

        # Combine processed features for tokenization
        df_processed_features = pd.concat([df_categorical, df_numerical_disc], axis=1)

        # Initialize vocabulary counter
        vocab_size = 0

        # Convert all features to the format required by SAT
        for col in df_processed_features.columns:
            # Apply label encoding to create token indices
            df_processed_features[col] = (
                LabelEncoder()
                .fit_transform(df_processed_features[col].astype(str))
                .astype(int)
                + vocab_size
            )
            vocab_size = df_processed_features[col].max() + 1

            # Convert to token format: column_value
            df_processed_features[col] = df_processed_features[col].apply(
                lambda x: f"{col}_{x}"
            )

        # Add the processed numerical features back for numerics representation
        df_orig_features = pd.concat([df_categorical, df_numerical_scaled], axis=1)
        logger.debug(f"Combined feature columns: {df_orig_features.columns.tolist()}")

        # Add modality and numerics vectors
        df_processed_features.loc[:, "x"] = ""
        df_processed_features.loc[:, "x"] = df_processed_features.loc[:, "x"].astype(
            "object"
        )

        # Add modality and numerics as columns to processed features
        df_processed_features.loc[:, "modality"] = ""
        df_processed_features.loc[:, "modality"] = df_processed_features.loc[
            :, "modality"
        ].astype("object")

        df_processed_features.loc[:, "numerics"] = ""
        df_processed_features.loc[:, "numerics"] = df_processed_features.loc[
            :, "numerics"
        ].astype("object")

        # Generate tokens, modality, and numerics for each row
        for index, row in df_processed_features.iterrows():
            orig_row = df_orig_features.loc[index]
            df_processed_features.at[index, "x"] = " ".join(tokens(row, modality))
            df_processed_features.at[index, "modality"] = modality
            df_processed_features.at[index, "numerics"] = numerics(orig_row, modality)

        # 5. Create train/val/test split
        logger.debug("Creating train/val/test split")

        # Process the targets for multi-event framework
        # Use custom labeled events if available, otherwise use defaults
        df_targets = pd.DataFrame(index=df_combined.index)

        if (
            "labeled_events" in df_combined.columns
            and "labeled_durations" in df_combined.columns
        ):
            logger.debug("Using custom labeled events/durations")
            
            # Make sure all entries have lists for events and durations
            # Some might be None or individual values
            for idx in df_combined.index:
                events = df_combined.at[idx, "labeled_events"]
                durations = df_combined.at[idx, "labeled_durations"]
                event_types = df_combined.at[idx, "event_types"] if "event_types" in df_combined.columns else None
                
                # Convert None or individual values to lists
                if events is None or not isinstance(events, list):
                    df_combined.at[idx, "labeled_events"] = [0]  # Default to censored
                if durations is None or not isinstance(durations, list):
                    df_combined.at[idx, "labeled_durations"] = [1095.0]  # Default duration
                    
                # Log some examples for debugging
                if idx % 100 == 0:  # Only log every 100th row
                    patient_id = df_combined.at[idx, "patient_id"] if "patient_id" in df_combined.columns else idx
                    logger.debug(
                        f"Patient {patient_id} labeled events: {df_combined.at[idx, 'labeled_events']}, "
                        f"durations: {df_combined.at[idx, 'labeled_durations']}, "
                        f"types: {event_types}"
                    )
            
            # Now set the target values
            df_targets["events"] = df_combined["labeled_events"]
            df_targets["durations"] = df_combined["labeled_durations"]
            
            # Log the first few target rows
            logger.debug(f"Target events/durations shape: {df_targets.shape}")
            logger.debug(f"First 3 event lists: {df_targets['events'].head(3).tolist()}")
            logger.debug(f"First 3 duration lists: {df_targets['durations'].head(3).tolist()}")
        else:
            # Process the targets for multi-event framework
            # df_combined has ['time', 'event'] columns
            # We need to convert to ['events', 'durations'] lists
            logger.debug("Using default event/time processing")

            # Count the number of unique event types (excluding 0/censored)
            num_events = len([k for k, v in event_type_map.items() if v > 0])
            logger.debug(f"Number of event types: {num_events}")

            # Create events and durations as lists
            if num_events > 1:
                # For multi-event case, create lists
                df_targets["events"] = [
                    [0] * num_events for _ in range(len(df_combined))
                ]
                df_targets["durations"] = [
                    [0] * num_events for _ in range(len(df_combined))
                ]

                # Fill in the events and durations
                for i, (idx, row) in enumerate(df_combined.iterrows()):
                    event_val = int(row["event"])
                    time_val = float(row["time"])

                    # For each patient, mark the appropriate event
                    if event_val > 0:  # Not censored
                        event_idx = event_val - 1  # Convert to 0-based index
                        df_targets.at[idx, "events"][event_idx] = 1
                        df_targets.at[idx, "durations"][event_idx] = time_val
                    else:
                        # Censored - all events are 0, all durations are the censoring time
                        for j in range(num_events):
                            df_targets.at[idx, "durations"][j] = time_val
            else:
                # For single event case, use original format but as lists
                df_targets["events"] = df_combined["event"].apply(lambda x: [int(x)])
                df_targets["durations"] = df_combined["time"].apply(
                    lambda x: [float(x)]
                )

        # Prepare data for split
        X = df_processed_features
        y = df_targets

        # Create the splits
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test(
            X=X,
            y=y,
            train_ratio=self.train_ratio,
            test_ratio=self.test_ratio,
            validation_ratio=self.validation_ratio,
        )

        # 6. Save data frames
        logger.debug("Saving data to disk")

        # Format the data according to SAT requirements
        train_data = pd.DataFrame(
            {
                "x": X_train["x"],
                "modality": X_train["modality"],
                "numerics": X_train["numerics"],
                "events": y_train["events"],
                "durations": y_train["durations"],
                "split": "train",
            },
            index=X_train.index,
        )

        val_data = pd.DataFrame(
            {
                "x": X_val["x"],
                "modality": X_val["modality"],
                "numerics": X_val["numerics"],
                "events": y_val["events"],
                "durations": y_val["durations"],
                "split": "valid",
            },
            index=X_val.index,
        )

        test_data = pd.DataFrame(
            {
                "x": X_test["x"],
                "modality": X_test["modality"],
                "numerics": X_test["numerics"],
                "events": y_test["events"],
                "durations": y_test["durations"],
                "split": "test",
            },
            index=X_test.index,
        )

        # Create K-fold splits if requested
        if self.kfold > 1:
            logger.debug(f"Creating {self.kfold} cross-validation folds")
            df_train_data = pd.concat([X_train, X_val])
            df_y_train_data = pd.concat([y_train, y_val])

            kf = KFold(n_splits=self.kfold, shuffle=True)
            for i, (train_index, test_index) in enumerate(kf.split(df_train_data)):
                X_train_kf, X_test_kf = (
                    df_train_data.iloc[train_index],
                    df_train_data.iloc[test_index],
                )
                y_train_kf, y_test_kf = (
                    df_y_train_data.iloc[train_index],
                    df_y_train_data.iloc[test_index],
                )

                train_data_kf = pd.DataFrame(
                    data={
                        "x": X_train_kf["x"],
                        "modality": X_train_kf["modality"],
                        "numerics": X_train_kf["numerics"],
                        "events": y_train_kf["events"],
                        "durations": y_train_kf["durations"],
                        "split": "train",
                    },
                    index=X_train_kf.index,
                )

                test_data_kf = pd.DataFrame(
                    data={
                        "x": X_test_kf["x"],
                        "modality": X_test_kf["modality"],
                        "numerics": X_test_kf["numerics"],
                        "events": y_test_kf["events"],
                        "durations": y_test_kf["durations"],
                        "split": "valid",
                    },
                    index=X_test_kf.index,
                )

                # Create output directory and save the fold
                outDir = Path(f"{self.processed_dir}/{self.name}")
                outDir.mkdir(parents=True, exist_ok=True)
                data_kf = pd.concat(
                    [train_data_kf, test_data_kf, test_data]
                ).reset_index(drop=True)
                data_kf.to_json(
                    Path(f"{outDir}/{i}_{self.name}.json"),
                    orient="records",
                    lines=True,
                )

        # Save the complete dataset
        outDir = Path(f"{self.processed_dir}/{self.name}")
        outDir.mkdir(parents=True, exist_ok=True)

        data = pd.concat([train_data, val_data, test_data]).reset_index(drop=True)
        data.to_json(Path(f"{outDir}/{self.name}.json"), orient="records", lines=True)

        # Include event times in the output data
        if "labeled_events" in df_combined.columns and "labeled_durations" in df_combined.columns:
            # Add event times to the final dataset
            train_data["event_times"] = X_train.index.map(lambda idx: df_combined.loc[idx, "labeled_durations"] if idx in df_combined.index else [])
            val_data["event_times"] = X_val.index.map(lambda idx: df_combined.loc[idx, "labeled_durations"] if idx in df_combined.index else [])
            test_data["event_times"] = X_test.index.map(lambda idx: df_combined.loc[idx, "labeled_durations"] if idx in df_combined.index else [])
            
            # Also add the event type names for easier interpretation
            if "event_types" in df_combined.columns:
                train_data["event_type_names"] = X_train.index.map(lambda idx: df_combined.loc[idx, "event_types"] if idx in df_combined.index else [])
                val_data["event_type_names"] = X_val.index.map(lambda idx: df_combined.loc[idx, "event_types"] if idx in df_combined.index else [])
                test_data["event_type_names"] = X_test.index.map(lambda idx: df_combined.loc[idx, "event_types"] if idx in df_combined.index else [])
        
        # Save a metadata file with information about the events
        # Include both original event types and labeler-generated event types
        
        # Get all unique event types from the labeled data
        all_event_types = set()
        if "event_types" in df_combined.columns:
            # Extract all event types from the labeled data
            for event_type_list in df_combined["event_types"].dropna():
                if isinstance(event_type_list, list):
                    all_event_types.update(event_type_list)
        
        # Create an updated event_type_map including all labeler events
        updated_event_type_map = {}
        for idx, event_type in enumerate(sorted(all_event_types)):
            updated_event_type_map[event_type] = idx + 1
        
        # Use the original event_type_map as fallback if no labeler events are found
        final_event_type_map = updated_event_type_map if updated_event_type_map else event_type_map
        
        metadata_output = {
            "event_types": final_event_type_map,
            "feature_count": len(feature_cols),
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
            "total_patients": len(df_combined["patient_id"].unique()),
            "total_events": total_events,
            "labeler_events": list(all_event_types),  # Add explicit list of labeler event types
            "includes_event_times": True,              # Flag that event times are included in output
            "includes_risk_factor_timing": True        # Flag that risk factors include timing information
        }

        pd.DataFrame([metadata_output]).to_json(
            Path(f"{outDir}/{self.name}_metadata.json"), orient="records"
        )
