"""Adapter for converting FEMR datasets to SAT format.

This module provides utilities to convert FEMR dataset results
into the format expected by SAT models for survival analysis.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"


import logging
from typing import Dict, List, Any, Optional
import pandas as pd

# Create a compatible Dataset class since femr.datasets might not exist
class FEMRDataset:
    """FEMR-compatible Dataset class."""
    
    def __init__(self, name="default"):
        self.name = name
        self.patients = []
        self.features = {}
        self.labels = {}
        self.tables = {}
        
    def add_patient(self, patient):
        self.patients.append(patient)

logger = logging.getLogger(__name__)


class FEMRAdapter:
    """Adapter to convert FEMR results to SAT format.

    This class provides utilities to:
    - Convert FEMR datasets to pandas DataFrames
    - Add SAT-specific columns (modality, numerics, tokens)
    - Format data for SAT survival models
    """

    @staticmethod
    def convert_to_dataframe(
        femr_dataset: FEMRDataset,
        labelers: Optional[List[Any]] = None,
        featurizers: Optional[List[Any]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Convert FEMR dataset with labels and features to SAT dataframes.

        Args:
            femr_dataset: FEMR dataset with patients, events, and processed results
            labelers: List of labelers that were applied
            featurizers: List of featurizers that were applied

        Returns:
            Dictionary mapping event types to dataframes with features and labels
        """
        # Get all patient IDs
        patient_ids = [patient.patient_id for patient in femr_dataset.patients]

        # Create base dataframe with patient IDs
        base_df = pd.DataFrame({"subject_id": patient_ids})

        # Add features for each featurizer
        if featurizers:
            for featurizer in featurizers:
                # Get feature data for all patients
                feature_data = []
                for patient in femr_dataset.patients:
                    # Get this patient's features
                    if patient.patient_id in femr_dataset.features:
                        patient_features = femr_dataset.features[
                            patient.patient_id
                        ].get(featurizer.name, {})
                        feature_data.append(
                            {"subject_id": patient.patient_id, **patient_features}
                        )
                    else:
                        # Handle missing features
                        feature_data.append({"subject_id": patient.patient_id})

                # Create features dataframe and merge with base
                features_df = pd.DataFrame(feature_data)
                base_df = pd.merge(base_df, features_df, on="subject_id", how="left")

        result_dfs = {}

        # Create separate dataframes for each labeler
        if labelers:
            for labeler in labelers:
                label_data = []
                for patient in femr_dataset.patients:
                    # Get this patient's labels
                    if patient.patient_id in femr_dataset.labels:
                        patient_labels = femr_dataset.labels[patient.patient_id].get(
                            labeler.name, {}
                        )

                        # Handle competing risks (multiple events per patient)
                        if isinstance(patient_labels.get("events", None), list):
                            events = patient_labels.get("events", [])
                            durations = patient_labels.get("durations", [])
                            event_types = patient_labels.get("event_types", [])

                            # Create a row for each event
                            for i, (event, duration, event_type) in enumerate(
                                zip(events, durations, event_types)
                            ):
                                label_data.append(
                                    {
                                        "subject_id": patient.patient_id,
                                        "event": event,
                                        "time": duration,
                                        "event_type": event_type,
                                    }
                                )
                        else:
                            # Single event per patient
                            label_data.append(
                                {
                                    "subject_id": patient.patient_id,
                                    "event": patient_labels.get("event", 0),
                                    "time": patient_labels.get("duration", 0),
                                    "event_type": labeler.name,
                                }
                            )
                    else:
                        # Handle missing labels
                        label_data.append(
                            {
                                "subject_id": patient.patient_id,
                                "event": 0,
                                "time": 0,
                                "event_type": labeler.name,
                            }
                        )

                # Create labels dataframe and merge with base
                labels_df = pd.DataFrame(label_data)

                # Group labels by event type and create a separate dataframe for each
                for event_type, group_df in labels_df.groupby("event_type"):
                    # Create a copy of the base dataframe for this event type
                    event_df = base_df.copy()

                    # Merge with the grouped labels
                    result_df = pd.merge(
                        event_df, group_df, on="subject_id", how="left"
                    )

                    # Store in result dictionary by event type
                    result_dfs[event_type] = result_df
        else:
            # If no labelers, just return the base dataframe with features
            result_dfs["default"] = base_df

        return result_dfs

    @staticmethod
    def convert_to_sat_format(
        df: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        prefix_tokens: bool = True,
    ) -> pd.DataFrame:
        """Convert processed dataframe to SAT-specific format.

        Args:
            df: DataFrame with features and labels
            categorical_cols: Optional list of categorical columns
            numerical_cols: Optional list of numerical columns
            prefix_tokens: Whether to prefix tokens with column names

        Returns:
            DataFrame formatted for SAT models
        """
        result_df = df.copy()

        # Standard metadata columns to exclude from features
        metadata_cols = ["subject_id", "time", "event", "event_type"]

        # Identify categorical and numerical columns if not provided
        if categorical_cols is None and numerical_cols is None:
            # Get all feature columns (excluding metadata)
            feature_cols = [c for c in df.columns if c not in metadata_cols]

            # Guess column types based on dtype
            numerical_cols = (
                df[feature_cols]
                .select_dtypes(include=["int", "float"])
                .columns.tolist()
            )
            categorical_cols = [c for c in feature_cols if c not in numerical_cols]

        # Ensure categorical_cols and numerical_cols are lists
        categorical_cols = categorical_cols or []
        numerical_cols = numerical_cols or []

        # Create modality list (0=categorical, 1=numeric)
        feature_cols = categorical_cols + numerical_cols
        modality = [0] * len(categorical_cols) + [1] * len(numerical_cols)

        # Add 'x' column with tokenized categorical features
        token_lists = []
        for _, row in df.iterrows():
            tokens = []
            for col in categorical_cols:
                if pd.notna(row.get(col)):
                    if prefix_tokens:
                        # Add column name as prefix
                        token = f"{col}_{row[col]}"
                    else:
                        # Use raw value
                        token = str(row[col])
                    tokens.append(token)
            token_lists.append(" ".join(tokens))

        result_df["x"] = token_lists

        # Add modality column
        result_df["modality"] = [modality] * len(df)

        # Add numerics column
        numeric_arrays = []
        for _, row in df.iterrows():
            numerics = []
            for col in categorical_cols:
                numerics.append(1.0)  # Default for categorical

            for col in numerical_cols:
                if pd.notna(row.get(col)):
                    numerics.append(float(row[col]))
                else:
                    numerics.append(0.0)  # Default for missing

            numeric_arrays.append(numerics)

        result_df["numerics"] = numeric_arrays

        # Add token_times column (with default value of 0.0 for all tokens)
        result_df["token_times"] = result_df["x"].apply(
            lambda x: [0.0] * len(x.split()) if x else []
        )

        # Rename columns to match SAT expectations if needed
        if "time" in result_df.columns and "duration" not in result_df.columns:
            result_df = result_df.rename(columns={"time": "duration"})

        return result_df

    @staticmethod
    def create_femr_dataset(
        df: pd.DataFrame, id_col: str = "subject_id"
    ) -> FEMRDataset:
        """Create a FEMR dataset from a pandas DataFrame.

        Args:
            df: DataFrame with patient data
            id_col: Column to use for patient IDs

        Returns:
            FEMR Dataset with patients and events
        """
        from femr.datasets import Patient, Event, Dataset

        # Create a new dataset
        dataset = Dataset(name="sat_dataset")

        # Group by patient ID
        for patient_id, group in df.groupby(id_col):
            # Create events for this patient
            events = []

            for _, row in group.iterrows():
                # Create event attributes dictionary
                event_attrs = {}

                # Add all non-ID columns as event attributes
                for col in group.columns:
                    if col != id_col and pd.notna(row[col]):
                        event_attrs[col] = row[col]

                # Create FEMR Event
                event = Event(**event_attrs)
                events.append(event)

            # Create FEMR Patient
            patient = Patient(patient_id=patient_id, events=events)

            # Add to dataset
            dataset.add_patient(patient)

        return dataset
