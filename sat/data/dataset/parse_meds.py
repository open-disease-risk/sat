"""Process MEDS (Medical Event Data Standard) data.

1. Read the Parquet file in MEDS format
2. Use FEMR to generate events and timelines
3. Transform to SAT's survival analysis format with events and durations lists
4. Split into train/val/test
5. Save as pandas dataframes

"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from dataclasses import dataclass
from logging import DEBUG, ERROR
from pathlib import Path
import os
import json
from typing import List, Optional, Dict, Any, Tuple, Union

import pandas as pd
import numpy as np
from logdecorator import log_on_end, log_on_error, log_on_start
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler, MinMaxScaler
from datasets import load_dataset

# Import FEMR - required for MEDS processing
import femr
from femr.labelers import Labeler

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
    source: str                   # Path to MEDS format parquet file
    processed_dir: str            # Output directory
    train_ratio: float            # Ratio for training set
    validation_ratio: float       # Ratio for validation set
    test_ratio: float             # Ratio for test set
    n_bins: int                   # Number of bins for discretization
    encode: str                   # Encoding method
    strategy: str                 # Discretization strategy
    name: str                     # Dataset name
    label_definitions: List[Dict[str, Any]]  # FEMR label definitions
    time_field: str = "days"      # Field to use for time measurements
    event_fields: Optional[List[str]] = None  # Optional list of specific events to include
    kfold: int = 0                # Number of folds for cross-validation
    scale_numerics: bool = True   # Whether to scale numeric features
    scale_method: str = "standard"  # Scaling method to use: 'min_max' or 'standard'
    min_scale_numerics: float = 1.0  # Minimum value for min-max scaling
    
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
                    raise FileNotFoundError(f"No parquet files found in directory: {self.source}")
                
                # Check if any parquet files are empty
                empty_files = [f for f in parquet_files if f.stat().st_size == 0]
                if empty_files:
                    raise ValueError(f"Found empty parquet files: {', '.join(str(f) for f in empty_files)}")
                    
                logger.debug(f"Found {len(parquet_files)} parquet files in {self.source}")
                
            # Determine how to load the dataset based on source format
            logger.debug(f"Loading dataset from {self.source}")
            
            if source_path.is_dir():
                # For directory, directly load each parquet file separately and combine manually
                parquet_files = list(source_path.glob("*.parquet"))
                logger.info(f"Found {len(parquet_files)} parquet files in {source_path}")
                
                if not parquet_files:
                    raise ValueError(f"No parquet files found in directory {source_path}")
                
                # Create a dictionary mapping table names to their dataframes
                tables = {}
                
                # Process each parquet file individually
                for file_path in parquet_files:
                    try:
                        # Extract table name from filename
                        parts = file_path.stem.split('_')
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
                    raise ValueError(f"Could not load any valid parquet files from {source_path}")
                
                logger.info(f"Successfully loaded {len(tables)} tables: {', '.join(tables.keys())}")
                
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
                dataset = load_dataset('parquet', 
                                      data_files={'data': str(self.source)}, 
                                      split='train')
                                  
        except Exception as e:
            logger.error(f"Failed to load MEDS data: {e}")
            raise ValueError(f"Error loading MEDS data from {self.source}: {e}") from e
        
        # Extract all patient data
        try:
            logger.debug("Extracting patient data")
            
            # For our custom MedsDataset from directory
            if hasattr(dataset, 'tables'):
                logger.debug(f"Available tables: {list(dataset.tables.keys())}")
                
                # Try to find patients table
                if 'patients' in dataset.tables:
                    logger.debug("Found 'patients' table")
                    patient_data = dataset.tables['patients']
                else:
                    # Look for tables with 'patient' in the name
                    patient_tables = [name for name in dataset.tables.keys() if 'patient' in name.lower()]
                    if patient_tables:
                        logger.debug(f"Using '{patient_tables[0]}' as patient data")
                        patient_data = dataset.tables[patient_tables[0]]
                    else:
                        # Use the first table that isn't an event table as fallback
                        event_tables = ['mortality', 'hospitalizations', 'medications', 'diagnoses', 'labs']
                        for table_name in dataset.tables:
                            if table_name.lower() not in [t.lower() for t in event_tables]:
                                logger.debug(f"Using '{table_name}' as patient data (default)")
                                patient_data = dataset.tables[table_name]
                                break
                        else:
                            # No suitable table found
                            raise ValueError(f"No patient data table found in: {list(dataset.tables.keys())}")
            
            # For standard HuggingFace dataset
            elif hasattr(dataset, 'column_names'):
                logger.debug(f"Dataset column names: {dataset.column_names}")
                
                if 'patients' in dataset.column_names:
                    logger.debug("Found 'patients' column in dataset")
                    patient_data = pd.DataFrame(dataset['patients'])
                else:
                    # Look for columns with 'patient' in the name
                    patient_cols = [name for name in dataset.column_names if 'patient' in name.lower()]
                    if patient_cols:
                        logger.debug(f"Using '{patient_cols[0]}' as patient data")
                        patient_data = pd.DataFrame(dataset[patient_cols[0]])
                    else:
                        # Use first column as fallback
                        logger.warning(f"Using first column '{dataset.column_names[0]}' as patient data")
                        patient_data = pd.DataFrame(dataset[dataset.column_names[0]])
            
            # Other dataset formats
            else:
                logger.warning("Unknown dataset structure, attempting direct conversion")
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
        
        # Process each event type using the label definitions
        for label_def in self.label_definitions:
            logger.debug(f"Processing label definition: {label_def}")
            event_name = label_def.get('name', 'unknown_event')
            event_types.append(event_name)
            
            # Get event table name from definition
            table_name = label_def.get('table_name', event_name)
            time_field = label_def.get('time_field', 'days')
            
            # Try to get the corresponding event table
            if hasattr(dataset, 'tables') and table_name in dataset.tables:
                # Our custom MedsDataset structure
                logger.debug(f"Found table '{table_name}' in the dataset")
                event_table = dataset.tables[table_name]
                
                # Process the events
                if not event_table.empty and 'patient_id' in event_table.columns:
                    logger.debug(f"Processing {len(event_table)} events from '{table_name}'")
                    
                    # Extract time values and standardize columns
                    df_event = event_table.copy()
                    
                    # Ensure we have the necessary time field
                    if time_field in df_event.columns:
                        # Add event type column if not present
                        if 'event_type' not in df_event.columns:
                            df_event['event_type'] = event_name
                        
                        # Rename time field to standard name if needed
                        if time_field != 'time':
                            df_event = df_event.rename(columns={time_field: 'time'})
                        
                        # Add to collected events
                        event_data.append(df_event)
                        logger.info(f"Added {len(df_event)} events of type '{event_name}'")
                    else:
                        logger.warning(f"Time field '{time_field}' not found in table '{table_name}'")
                else:
                    logger.warning(f"Table '{table_name}' has no patient_id column or is empty")
            else:
                logger.warning(f"Event table '{table_name}' not found in dataset")
                
                # Try alternate approach for standard HuggingFace dataset
                if hasattr(dataset, 'column_names') and table_name in dataset.column_names:
                    logger.debug(f"Found column '{table_name}' in the dataset")
                    try:
                        events = dataset[table_name]
                        df_event = pd.DataFrame(events)
                        
                        if not df_event.empty and 'patient_id' in df_event.columns:
                            # Add event type if needed
                            if 'event_type' not in df_event.columns:
                                df_event['event_type'] = event_name
                                
                            # Rename time field if needed
                            if time_field in df_event.columns and time_field != 'time':
                                df_event = df_event.rename(columns={time_field: 'time'})
                                
                            event_data.append(df_event)
                            logger.info(f"Added {len(df_event)} events of type '{event_name}'")
                        else:
                            logger.warning(f"Column '{table_name}' has no patient_id or is empty")
                    except Exception as e:
                        logger.warning(f"Error processing column '{table_name}': {e}")
                else:
                    logger.warning(f"No event source found for '{event_name}'")
        
        # If no events were collected with label definitions, try direct approach
        if not event_data and hasattr(dataset, 'tables'):
            logger.warning("No events found using label definitions. Trying direct approach.")
            
            # Check for standard event tables by common names
            standard_event_tables = ['mortality', 'hospitalizations', 'readmissions']
            for table_name in standard_event_tables:
                if table_name in dataset.tables:
                    logger.debug(f"Found standard event table '{table_name}'")
                    event_table = dataset.tables[table_name]
                    
                    if not event_table.empty and 'patient_id' in event_table.columns:
                        # Add event type if not present
                        df_event = event_table.copy()
                        if 'event_type' not in df_event.columns:
                            df_event['event_type'] = table_name
                            
                        # Look for time field
                        time_fields = [f for f in df_event.columns if 'day' in f.lower() or 'time' in f.lower()]
                        if time_fields:
                            time_field = time_fields[0]
                            if time_field != 'time':
                                df_event = df_event.rename(columns={time_field: 'time'})
                                
                            event_types.append(table_name)
                            event_data.append(df_event)
                            logger.info(f"Added {len(df_event)} events of type '{table_name}'")
                        else:
                            logger.warning(f"No time field found in '{table_name}'")
                    else:
                        logger.warning(f"Table '{table_name}' has no patient_id or is empty")
        
        # 2. Transform data to survival analysis format
        logger.debug("Transforming MEDS data to survival analysis format")
        
        # If no events were found, raise an error
        if not event_data:
            raise ValueError("No events found using the provided label definitions")
            
        # Combine all event data
        df_events = pd.concat(event_data, axis=0)
        
        # Standardize column names
        if 'time' not in df_events.columns and self.time_field in df_events.columns:
            df_events["time"] = df_events[self.time_field]
        
        # Create a mapping of event types to integer codes
        event_type_map = {event: idx+1 for idx, event in enumerate(event_types)}
        
        # Map event types to integers (0 = censored, 1+ = event types)
        # In SAT, 0 usually indicates censored data
        df_events["event"] = df_events["event_type"].map(event_type_map).fillna(0).astype(int)
        
        # Filter to only include patients with events
        patient_ids = df_events["patient_id"].unique()
        patient_data = patient_data[patient_data["patient_id"].isin(patient_ids)]
        
        # Merge features with event data - one row per patient with the earliest event
        df_events = df_events.sort_values(by=["patient_id", "time"])
        df_first_events = df_events.drop_duplicates(subset=["patient_id"], keep="first")
        
        # Merge patient features with event data
        df_combined = pd.merge(
            df_first_events[["patient_id", "time", "event"]],
            patient_data,
            on="patient_id",
            how="inner"
        )
        
        # Continue with feature processing and data splitting
        self._process_features_and_split(df_combined, event_type_map, len(df_events))
    
    
    def _process_features_and_split(self, df_combined, event_type_map, total_events):
        """Process features and split data into train/val/test sets."""
        # Get feature columns (exclude metadata columns)
        meta_cols = ["patient_id", "time", "event"]
        date_cols = [col for col in df_combined.columns if 'date' in col.lower()]
        meta_cols.extend(date_cols)
        feature_cols = [col for col in df_combined.columns if col not in meta_cols]
        
        # 4. Preprocess features
        logger.debug("Preprocessing features")
        df_features = df_combined[feature_cols]
        
        # Handle categorical and numerical features differently
        categorical_features = df_features.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_features = df_features.select_dtypes(include=["number"]).columns.tolist()
        
        # Create modality vector (0 for categorical, 1 for numerical)
        modality = [0] * len(categorical_features) + [1] * len(numerical_features)
        logger.debug(f"Modality vector: {modality}")
        
        # Scale numerical features if needed
        df_numerical = df_features[numerical_features].fillna(df_features[numerical_features].median())
        
        if self.scale_numerics and len(numerical_features) > 0:
            logger.debug(f"Scaling numerical features using {self.scale_method} method")
            if self.scale_method == "min_max":
                scaler = MinMaxScaler()
                df_numerical_scaled = pd.DataFrame(
                    scaler.fit_transform(df_numerical) + self.min_scale_numerics,
                    columns=numerical_features,
                    index=df_numerical.index
                )
            elif self.scale_method == "standard":
                scaler = StandardScaler()
                df_numerical_scaled = pd.DataFrame(
                    scaler.fit_transform(df_numerical),
                    columns=numerical_features,
                    index=df_numerical.index
                )
            else:
                raise ValueError(f"scale_method {self.scale_method} not supported. Use 'min_max' or 'standard'")
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
                index=df_numerical.index
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
                LabelEncoder().fit_transform(df_processed_features[col].astype(str)).astype(int) + vocab_size
            )
            vocab_size = df_processed_features[col].max() + 1
            
            # Convert to token format: column_value
            df_processed_features[col] = df_processed_features[col].apply(lambda x: f"{col}_{x}")
        
        # Add the processed numerical features back for numerics representation
        df_orig_features = pd.concat([df_categorical, df_numerical_scaled], axis=1)
        logger.debug(f"Combined feature columns: {df_orig_features.columns.tolist()}")
        
        # Add modality and numerics vectors
        df_processed_features.loc[:, "x"] = ""
        df_processed_features.loc[:, "x"] = df_processed_features.loc[:, "x"].astype("object")
        
        # Add modality and numerics as columns to processed features
        df_processed_features.loc[:, "modality"] = ""
        df_processed_features.loc[:, "modality"] = df_processed_features.loc[:, "modality"].astype("object")
        
        df_processed_features.loc[:, "numerics"] = ""
        df_processed_features.loc[:, "numerics"] = df_processed_features.loc[:, "numerics"].astype("object")
        
        # Generate tokens, modality, and numerics for each row
        for index, row in df_processed_features.iterrows():
            orig_row = df_orig_features.loc[index]
            df_processed_features.at[index, "x"] = " ".join(tokens(row, modality))
            df_processed_features.at[index, "modality"] = modality
            df_processed_features.at[index, "numerics"] = numerics(orig_row, modality)
            
        # 5. Create train/val/test split
        logger.debug("Creating train/val/test split")
        
        # Process the targets for multi-event framework
        # df_combined has ['time', 'event'] columns
        # We need to convert to ['events', 'durations'] lists
        df_targets = pd.DataFrame(index=df_combined.index)
        
        # Count the number of unique event types (excluding 0/censored)
        num_events = len([k for k, v in event_type_map.items() if v > 0])
        logger.debug(f"Number of event types: {num_events}")
        
        # Create events and durations as lists
        if num_events > 1:
            # For multi-event case, create lists
            df_targets["events"] = [[0] * num_events for _ in range(len(df_combined))]
            df_targets["durations"] = [[0] * num_events for _ in range(len(df_combined))]
            
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
            df_targets["durations"] = df_combined["time"].apply(lambda x: [float(x)])
        
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
                data_kf = pd.concat([train_data_kf, test_data_kf, test_data]).reset_index(drop=True)
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
        
        # Save a metadata file with information about the events
        metadata_output = {
            "event_types": event_type_map,
            "feature_count": len(feature_cols),
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
            "total_patients": len(df_combined["patient_id"].unique()),
            "total_events": total_events,
        }
        
        pd.DataFrame([metadata_output]).to_json(
            Path(f"{outDir}/{self.name}_metadata.json"),
            orient="records"
        )