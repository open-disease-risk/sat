"""Domain-specific medical labelers for MEDS data.

This module provides specialized labelers for identifying medical conditions,
risk factors, and outcomes from MEDS-formatted medical data.

Includes parallel processing capabilities for efficient handling of large datasets
by processing patients in parallel batches.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import logging
import multiprocessing
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Import MEDS schema definitions
from meds.schema import (
    birth_code,
    death_code,
)
from tqdm.auto import tqdm

# Try to import backends for optimized processing
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


# Constants for MEDS schema
MEDS_BIRTH_CODE = birth_code
MEDS_DEATH_CODE = death_code

# Enum definitions for our processing
class TimeType(str, Enum):
    """Type of time for events in MEDS schema."""
    STATIC = "static"
    EVENT = "event"


class EventStatus(int, Enum):
    """Status of events in survival analysis."""
    CENSORED = 0
    OCCURRED = 1


class ProcessingMode(str, Enum):
    """Processing mode for labelers."""
    SERIAL = "serial"
    MULTIPROCESSING = "multiprocessing"
    RAY = "ray"


def validate_meds_schema(df: pd.DataFrame) -> None:
    """Validate that a dataframe follows the MEDS schema.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ["subject_id", "time", "code"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required MEDS columns: {missing}")


class MedicalLabeler:
    """Base class for medical labelers with parallel processing support."""

    def __init__(self, name: str):
        """Initialize the labeler.

        Args:
            name: The name of this labeler
        """
        self.name = name

    def label(self, events: pd.DataFrame) -> Dict[str, Any]:
        """Process events and return labels.

        Args:
            events: DataFrame with medical events in MEDS format
            
        Returns:
            Dictionary with label information
        """
        raise NotImplementedError("Subclasses must implement this method")

    def label_patient(self, patient_events: pd.DataFrame) -> Dict[str, Any]:
        """Process events for a single patient.
        
        This method should be implemented by subclasses for patient-level processing.
        
        Args:
            patient_events: DataFrame with medical events for a single patient
            
        Returns:
            Dictionary with label information for this patient
        """
        raise NotImplementedError("Subclasses must implement the label_patient method")

    def parallel_label(
        self,
        events: pd.DataFrame,
        mode: ProcessingMode = ProcessingMode.MULTIPROCESSING,
        n_jobs: int = -1,
        batch_size: int = 1000,
        show_progress: bool = True
    ) -> Dict[int, Dict[str, Any]]:
        """Process events in parallel by patient.
        
        Args:
            events: DataFrame with medical events in MEDS format
            mode: Processing mode to use (serial, multiprocessing, ray)
            n_jobs: Number of parallel jobs (-1 for all cores)
            batch_size: Number of patients to process in each batch
            show_progress: Whether to show a progress bar
            
        Returns:
            Dictionary mapping patient IDs to their label information
        """
        # Validate MEDS schema
        validate_meds_schema(events)

        # Get unique patient IDs
        patient_ids = events["subject_id"].unique()
        total_patients = len(patient_ids)

        if total_patients == 0:
            return {}

        # Determine the number of jobs to use
        if n_jobs <= 0:
            n_jobs = max(1, multiprocessing.cpu_count() + n_jobs + 1)
        n_jobs = min(n_jobs, total_patients)

        logger.info(f"Processing {total_patients} patients with {n_jobs} workers in {mode.value} mode")

        # Process according to selected mode
        if mode == ProcessingMode.SERIAL:
            return self._serial_process(events, patient_ids, show_progress)
        elif mode == ProcessingMode.MULTIPROCESSING:
            return self._mp_process(events, patient_ids, n_jobs, batch_size, show_progress)
        elif mode == ProcessingMode.RAY:
            if not RAY_AVAILABLE:
                logger.warning("Ray not available, falling back to multiprocessing")
                return self._mp_process(events, patient_ids, n_jobs, batch_size, show_progress)
            return self._ray_process(events, patient_ids, n_jobs, batch_size, show_progress)
        else:
            raise ValueError(f"Unsupported processing mode: {mode}")

    def _serial_process(
        self,
        events: pd.DataFrame,
        patient_ids: np.ndarray,
        show_progress: bool
    ) -> Dict[int, Dict[str, Any]]:
        """Process patients serially."""
        results = {}

        # Create progress iterator if requested
        if show_progress:
            patient_iter = tqdm(patient_ids, desc=f"Processing patients with {self.name}")
        else:
            patient_iter = patient_ids

        # Process each patient one by one
        for patient_id in patient_iter:
            patient_events = events[events["subject_id"] == patient_id]
            results[patient_id] = self.label_patient(patient_events)

        return results

    def _mp_process_batch(
        self,
        patient_batch: List[int],
        events: pd.DataFrame
    ) -> Dict[int, Dict[str, Any]]:
        """Process a batch of patients.
        
        This method is used by the multiprocessing executor.
        
        Args:
            patient_batch: List of patient IDs to process
            events: DataFrame with medical events in MEDS format
            
        Returns:
            Dictionary mapping patient IDs to their label information
        """
        batch_results = {}
        for patient_id in patient_batch:
            patient_events = events[events["subject_id"] == patient_id]
            try:
                batch_results[patient_id] = self.label_patient(patient_events)
            except Exception as e:
                logger.error(f"Error processing patient {patient_id}: {e}")
                # Include error information in the results
                batch_results[patient_id] = {"error": str(e)}

        return batch_results

    def _mp_process(
        self,
        events: pd.DataFrame,
        patient_ids: np.ndarray,
        n_jobs: int,
        batch_size: int,
        show_progress: bool
    ) -> Dict[int, Dict[str, Any]]:
        """Process patients using multiprocessing."""
        # Split patients into batches for better efficiency
        patient_batches = []
        for i in range(0, len(patient_ids), batch_size):
            batch = patient_ids[i:min(i+batch_size, len(patient_ids))]
            patient_batches.append(list(batch))

        logger.info(f"Split {len(patient_ids)} patients into {len(patient_batches)} batches")

        # Create a partial function with the events DataFrame
        process_fn = partial(self._mp_process_batch, events=events)

        # Process batches in parallel
        results = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_fn, batch) for batch in patient_batches]

            # Collect results as they complete
            if show_progress:
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing batches with {self.name}"):
                    batch_result = future.result()
                    results.update(batch_result)
            else:
                for future in as_completed(futures):
                    batch_result = future.result()
                    results.update(batch_result)

        return results

    def _ray_process(
        self,
        events: pd.DataFrame,
        patient_ids: np.ndarray,
        n_jobs: int,
        batch_size: int,
        show_progress: bool
    ) -> Dict[int, Dict[str, Any]]:
        """Process patients using Ray for distributed computing."""
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(num_cpus=n_jobs)

        # Split patients into batches
        patient_batches = []
        for i in range(0, len(patient_ids), batch_size):
            batch = patient_ids[i:min(i+batch_size, len(patient_ids))]
            patient_batches.append(list(batch))

        # Define remote function for processing a batch
        @ray.remote
        def process_batch(patient_batch, labeler, events_df):
            batch_results = {}
            for patient_id in patient_batch:
                patient_events = events_df[events_df["subject_id"] == patient_id]
                try:
                    batch_results[patient_id] = labeler.label_patient(patient_events)
                except Exception as e:
                    logger.error(f"Error processing patient {patient_id}: {e}")
                    batch_results[patient_id] = {"error": str(e)}
            return batch_results

        # Put large objects in shared memory
        events_ref = ray.put(events)
        labeler_ref = ray.put(self)

        # Submit tasks for all batches
        futures = [process_batch.remote(batch, labeler_ref, events_ref) for batch in patient_batches]

        # Collect results as they complete
        results = {}
        if show_progress:
            for batch_result in tqdm(ray.get(futures), total=len(futures), desc=f"Processing with Ray and {self.name}"):
                results.update(batch_result)
        else:
            for batch_result in ray.get(futures):
                results.update(batch_result)

        return results

    def get_time_to_event(
        self,
        patient_events: pd.DataFrame,
        event_codes: List[str],
        enrollment_time: datetime,
        max_followup_days: int
    ) -> Tuple[int, float]:
        """Calculate time to event or censoring.
        
        Args:
            patient_events: DataFrame with patient's events
            event_codes: List of codes that define the event
            enrollment_time: When the patient was enrolled
            max_followup_days: Maximum follow-up period
            
        Returns:
            Tuple of (event_status, time_to_event)
        """
        # Find events matching the codes
        matching_events = patient_events[patient_events["code"].isin(event_codes)]

        if matching_events.empty:
            # No event occurred, patient is censored
            return EventStatus.CENSORED.value, float(max_followup_days)

        # Find earliest matching event
        earliest_event = matching_events.sort_values("time").iloc[0]
        event_time = earliest_event["time"]

        # Calculate time from enrollment to event
        if pd.isna(enrollment_time):
            warnings.warn("Patient has no enrollment time. Using 0 days.")
            time_to_event = 0
        else:
            time_to_event = (event_time - enrollment_time).days

        # Cap at max follow-up
        time_to_event = min(float(time_to_event), float(max_followup_days))

        return EventStatus.OCCURRED.value, time_to_event

    def process_large_dataset(
        self,
        events_path: str,
        output_path: str,
        mode: ProcessingMode = ProcessingMode.MULTIPROCESSING,
        n_jobs: int = -1,
        batch_size: int = 1000,
        patient_chunk_size: int = 10000,
        show_progress: bool = True
    ) -> None:
        """Process a very large dataset that doesn't fit in memory.
        
        Reads the data in chunks by patient ID and processes each chunk separately.
        
        Args:
            events_path: Path to parquet file or directory with events
            output_path: Path to save the results
            mode: Processing mode (serial, multiprocessing, ray)
            n_jobs: Number of parallel jobs
            batch_size: Number of patients per batch within a chunk
            patient_chunk_size: Number of patients to load at once
            show_progress: Whether to show a progress bar
        """
        logger.info(f"Processing large dataset from {events_path} with {self.name}")

        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Get all unique patient IDs
        logger.info("Getting unique patient IDs from dataset")

        if Path(events_path).is_file():
            # Reading from a single parquet file
            df = pd.read_parquet(events_path, columns=["subject_id"])
            all_patient_ids = list(df["subject_id"].unique())
        else:
            # Reading from a directory with potentially multiple files
            all_patient_ids = set()
            for file in Path(events_path).glob("*.parquet"):
                df = pd.read_parquet(file, columns=["subject_id"])
                all_patient_ids.update(df["subject_id"].unique().tolist())

        all_patient_ids = sorted(all_patient_ids)
        total_patients = len(all_patient_ids)

        logger.info(f"Found {total_patients} patients, processing in chunks of {patient_chunk_size}")

        # Process patients in chunks
        for chunk_start in range(0, total_patients, patient_chunk_size):
            chunk_end = min(chunk_start + patient_chunk_size, total_patients)
            chunk_patient_ids = all_patient_ids[chunk_start:chunk_end]

            logger.info(f"Processing patient chunk {chunk_start+1}-{chunk_end} of {total_patients}")

            # Load events for just this chunk of patients
            if Path(events_path).is_file():
                # For Polars if available
                if POLARS_AVAILABLE:
                    chunk_events = pl.scan_parquet(events_path).filter(
                        pl.col("subject_id").is_in(chunk_patient_ids)
                    ).collect().to_pandas()
                else:
                    chunk_events = pd.read_parquet(
                        events_path,
                        filters=[("subject_id", "in", chunk_patient_ids)]
                    )
            else:
                # For directory, we need to concatenate from multiple files
                chunk_dfs = []
                for file in Path(events_path).glob("*.parquet"):
                    try:
                        if POLARS_AVAILABLE:
                            df = pl.scan_parquet(file).filter(
                                pl.col("subject_id").is_in(chunk_patient_ids)
                            ).collect()
                            if df.height > 0:
                                chunk_dfs.append(df.to_pandas())
                        else:
                            df = pd.read_parquet(
                                file,
                                filters=[("subject_id", "in", chunk_patient_ids)]
                            )
                            if not df.empty:
                                chunk_dfs.append(df)
                    except Exception as e:
                        logger.error(f"Error reading {file}: {e}")

                if not chunk_dfs:
                    logger.warning(f"No data found for patient chunk {chunk_start+1}-{chunk_end}")
                    continue

                chunk_events = pd.concat(chunk_dfs, ignore_index=True)

            # Process this chunk of patients
            chunk_results = self.parallel_label(
                chunk_events,
                mode=mode,
                n_jobs=n_jobs,
                batch_size=batch_size,
                show_progress=show_progress
            )

            # Save results for this chunk
            chunk_df = pd.DataFrame.from_dict(chunk_results, orient='index').reset_index()
            chunk_df.rename(columns={"index": "subject_id"}, inplace=True)

            chunk_filename = f"results_chunk_{chunk_start+1}_{chunk_end}.parquet"
            chunk_path = os.path.join(output_path, chunk_filename)
            chunk_df.to_parquet(chunk_path)

            logger.info(f"Saved results for chunk to {chunk_path}")

            # Clean up to free memory
            del chunk_events, chunk_results, chunk_df

        # Combine all chunk results
        logger.info("Combining all chunk results")
        result_files = list(Path(output_path).glob("results_chunk_*.parquet"))

        if not result_files:
            logger.warning("No result files found to combine")
            return

        result_dfs = []
        for file in result_files:
            try:
                df = pd.read_parquet(file)
                result_dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading result file {file}: {e}")

        if not result_dfs:
            logger.warning("No valid result files found to combine")
            return

        combined_results = pd.concat(result_dfs, ignore_index=True)
        combined_path = os.path.join(output_path, "combined_results.parquet")
        combined_results.to_parquet(combined_path)

        logger.info(f"Saved combined results to {combined_path}")


class RiskFactorLabeler(MedicalLabeler):
    """Labeler for identifying risk factors from medical codes."""

    # Common ICD-10 codes for conditions
    ICD10_CODES = {
        "hypertension": ["I10", "I11", "I12", "I13", "I15"],
        "diabetes": ["E10", "E11", "E13"],
        "copd": ["J44"],
        "heart_failure": ["I50"],
        "kidney_disease": ["N18"],
        "coronary_artery_disease": ["I25"],
        "atrial_fibrillation": ["I48"],
        "stroke": ["I63", "I64"],
        "alzheimers": ["G30"],
        "depression": ["F32", "F33"],
        "anxiety": ["F41"],
        "cancer": ["C"],
    }

    # Common medication codes (RxNorm) for conditions
    RXNORM_CODES = {
        "hypertension": ["C09", "C07", "C08", "C03"],
        "diabetes": ["A10"],
        "copd": ["R03"],
        "heart_failure": ["C03CA", "C09"],
        "depression": ["N06A"],
        "anxiety": ["N05B"],
        "pain": ["N02"],
    }

    # Common lab abnormalities for conditions
    LAB_ABNORMALITIES = {
        "diabetes": {
            "LOINC:2339-0": (126, float("inf")),  # Glucose > 126 mg/dL
            "LOINC:4548-4": (6.5, float("inf")),  # HbA1c > 6.5%
        },
        "kidney_disease": {
            "LOINC:2160-0": (1.5, float("inf")),  # Creatinine > 1.5 mg/dL
            "LOINC:3094-0": (25, float("inf")),  # BUN > 25 mg/dL
        },
        "hyperlipidemia": {
            "LOINC:2093-3": (240, float("inf")),  # Cholesterol > 240 mg/dL
            "LOINC:2089-1": (160, float("inf")),  # LDL > 160 mg/dL
        },
    }

    def __init__(self, name: str = "risk_factor_labeler", custom_codes: Dict[str, Dict[str, List[str]]] = None):
        """Initialize the risk factor labeler.

        Args:
            name: Name of this labeler
            custom_codes: Optional dictionary of custom codes to add to the default ones
                Format: {"icd10": {"condition": ["codes"]}, "rxnorm": {...}, "lab": {...}}
        """
        super().__init__(name)

        # Add custom codes if provided
        if custom_codes:
            if "icd10" in custom_codes:
                for condition, codes in custom_codes["icd10"].items():
                    if condition in self.ICD10_CODES:
                        self.ICD10_CODES[condition].extend(codes)
                    else:
                        self.ICD10_CODES[condition] = codes

            if "rxnorm" in custom_codes:
                for condition, codes in custom_codes["rxnorm"].items():
                    if condition in self.RXNORM_CODES:
                        self.RXNORM_CODES[condition].extend(codes)
                    else:
                        self.RXNORM_CODES[condition] = codes

            if "lab" in custom_codes:
                for condition, labs in custom_codes["lab"].items():
                    if condition in self.LAB_ABNORMALITIES:
                        self.LAB_ABNORMALITIES[condition].update(labs)
                    else:
                        self.LAB_ABNORMALITIES[condition] = labs

    def label(self, events: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Identify risk factors from medical codes in events.

        This method is maintained for backward compatibility.
        For new code, use parallel_label() instead.

        Args:
            events: DataFrame with medical events in MEDS format

        Returns:
            Dictionary mapping patient IDs to their risk factors
        """
        # Use the new parallel implementation with serial processing
        results = self.parallel_label(events, mode=ProcessingMode.SERIAL)

        # Format to match the old return structure
        return {str(patient_id): result for patient_id, result in results.items()}

    def label_patient(self, patient_events: pd.DataFrame) -> Dict[str, Any]:
        """Process events for a single patient to identify risk factors.
        
        Args:
            patient_events: DataFrame with medical events for a single patient
            
        Returns:
            Dictionary with risk factors for this patient
        """
        # Validate that we have a single patient
        if not patient_events.empty and len(patient_events["subject_id"].unique()) > 1:
            raise ValueError("patient_events contains data for multiple patients")

        # Initialize risk factors set
        risk_factors = set()

        # Process each event
        for _, event in patient_events.iterrows():
            code = event["code"]

            # Check for ICD-10 codes
            if isinstance(code, str) and code.startswith("ICD10:"):
                icd_code = code.split(":")[1]
                for condition, codes in self.ICD10_CODES.items():
                    if any(icd_code.startswith(c) for c in codes):
                        risk_factors.add(condition)

            # Check for medication codes
            elif isinstance(code, str) and code.startswith("RxNorm:"):
                rx_code = code.split(":")[1]
                for condition, codes in self.RXNORM_CODES.items():
                    if any(rx_code.startswith(c) for c in codes):
                        risk_factors.add(condition)

            # Check for lab abnormalities
            if code in [
                lab_code
                for condition_labs in self.LAB_ABNORMALITIES.values()
                for lab_code in condition_labs.keys()
            ]:
                numeric_value = event.get("numeric_value")
                if pd.notna(numeric_value):
                    for condition, labs in self.LAB_ABNORMALITIES.items():
                        if code in labs:
                            lower, upper = labs[code]
                            if lower <= numeric_value <= upper:
                                risk_factors.add(condition)

        # Return risk factors as a list
        return {"risk_factors": list(risk_factors)}


class MortalityLabeler(MedicalLabeler):
    """Labeler for identifying mortality events."""

    def __init__(self, name: str = "mortality_labeler", max_followup_days: int = 1095,
                 death_codes: List[str] = None, enrollment_codes: List[str] = None):
        """Initialize the mortality labeler.

        Args:
            name: Name of this labeler
            max_followup_days: Maximum followup period in days
            death_codes: List of codes that indicate death event (default: [MEDS_DEATH_CODE])
            enrollment_codes: List of codes for enrollment events (default: ["ENROLLMENT"])
        """
        super().__init__(name)
        self.max_followup_days = max_followup_days
        self.death_codes = death_codes or [MEDS_DEATH_CODE]
        self.enrollment_codes = enrollment_codes or ["ENROLLMENT"]

    def label(self, events: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Identify mortality events and create time-to-event labels.

        This method is maintained for backward compatibility.
        For new code, use parallel_label() instead.

        Args:
            events: DataFrame with medical events in MEDS format

        Returns:
            Dictionary mapping patient IDs to mortality status and time
        """
        # Use the new parallel implementation with serial processing
        return self.parallel_label(events, mode=ProcessingMode.SERIAL)

    def label_patient(self, patient_events: pd.DataFrame) -> Dict[str, Any]:
        """Process events for a single patient to identify mortality events.
        
        Args:
            patient_events: DataFrame with medical events for a single patient
            
        Returns:
            Dictionary with mortality event information for this patient
        """
        # Validate that we have a single patient
        if not patient_events.empty and len(patient_events["subject_id"].unique()) > 1:
            raise ValueError("patient_events contains data for multiple patients")

        if patient_events.empty:
            return {
                "event": EventStatus.CENSORED.value,
                "time": self.max_followup_days,
                "enrollment_date": None
            }

        # Get patient ID
        patient_id = patient_events["subject_id"].iloc[0]

        # Find enrollment date
        enrollment_events = patient_events[patient_events["code"].isin(self.enrollment_codes)]
        if enrollment_events.empty:
            # No enrollment date found, try to use the earliest event time
            if "time" in patient_events.columns and not patient_events["time"].isna().all():
                enrollment_date = patient_events["time"].min()
            else:
                # Cannot determine enrollment, return censored at max follow-up
                return {
                    "event": EventStatus.CENSORED.value,
                    "time": self.max_followup_days,
                    "enrollment_date": None
                }
        else:
            # Use the earliest enrollment date
            enrollment_date = enrollment_events["time"].min()

        # Check for death events
        death_events = patient_events[patient_events["code"].isin(self.death_codes)]

        if death_events.empty:
            # No death event, patient is censored
            return {
                "event": EventStatus.CENSORED.value,
                "time": float(self.max_followup_days),
                "enrollment_date": enrollment_date
            }
        else:
            # Patient died, find the earliest death event
            death_date = death_events["time"].min()
            time_to_event = (death_date - enrollment_date).days

            # Cap at max follow-up
            time_to_event = min(float(time_to_event), float(self.max_followup_days))

            return {
                "event": EventStatus.OCCURRED.value,
                "time": time_to_event,
                "enrollment_date": enrollment_date
            }


class CompetingRiskLabeler(MedicalLabeler):
    """Labeler for multiple competing events."""

    def __init__(
        self,
        name: str = "competing_risk_labeler",
        event_codes: Dict[str, List[str]] = None,
        max_followup_days: int = 1095,
        enrollment_codes: List[str] = None,
    ):
        """Initialize the competing risk labeler.

        Args:
            name: Name of this labeler
            event_codes: Dictionary mapping event names to codes that define them
            max_followup_days: Maximum followup period in days
            enrollment_codes: List of codes for enrollment events (default: ["ENROLLMENT"])
        """
        super().__init__(name)
        self.max_followup_days = max_followup_days
        self.enrollment_codes = enrollment_codes or ["ENROLLMENT"]

        # Default event codes if none provided
        if event_codes is None:
            self.event_codes = {
                "death": [MEDS_DEATH_CODE],
                "heart_failure": ["ICD10:I50"],
                "stroke": ["ICD10:I63", "ICD10:I64"],
                "kidney_failure": ["ICD10:N17", "ICD10:N18.5", "ICD10:N18.6"],
            }
        else:
            self.event_codes = event_codes

    def label(self, events: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Identify competing events and create time-to-event labels.

        This method is maintained for backward compatibility.
        For new code, use parallel_label() instead.

        Args:
            events: DataFrame with medical events in MEDS format

        Returns:
            Dictionary with competing events information
        """
        # Use the new parallel implementation with serial processing
        return self.parallel_label(events, mode=ProcessingMode.SERIAL)

    def label_patient(self, patient_events: pd.DataFrame) -> Dict[str, Any]:
        """Process events for a single patient to identify competing events.
        
        Args:
            patient_events: DataFrame with medical events for a single patient
            
        Returns:
            Dictionary with competing events information for this patient
        """
        # Validate that we have a single patient
        if not patient_events.empty and len(patient_events["subject_id"].unique()) > 1:
            raise ValueError("patient_events contains data for multiple patients")

        if patient_events.empty:
            # For empty data, return censored for all events
            events_list = [EventStatus.CENSORED.value] * len(self.event_codes)
            durations_list = [self.max_followup_days] * len(self.event_codes)
            return {
                "events": events_list,
                "durations": durations_list,
                "event_types": sorted(self.event_codes.keys()),
                "enrollment_date": None,
            }

        # Find enrollment date
        enrollment_events = patient_events[patient_events["code"].isin(self.enrollment_codes)]
        if enrollment_events.empty:
            # No enrollment date found, try to use the earliest event time
            if "time" in patient_events.columns and not patient_events["time"].isna().all():
                enrollment_date = patient_events["time"].min()
            else:
                # Cannot determine enrollment, return censored at max follow-up for all events
                events_list = [EventStatus.CENSORED.value] * len(self.event_codes)
                durations_list = [self.max_followup_days] * len(self.event_codes)
                return {
                    "events": events_list,
                    "durations": durations_list,
                    "event_types": sorted(self.event_codes.keys()),
                    "enrollment_date": None,
                }
        else:
            # Use the earliest enrollment date
            enrollment_date = enrollment_events["time"].min()

        # Initialize patient event data
        patient_events_data = {
            event_name: (EventStatus.CENSORED.value, float(self.max_followup_days))
            for event_name in self.event_codes.keys()
        }

        # Check for each event type
        for event_name, codes in self.event_codes.items():
            # Find relevant events
            relevant_events = patient_events[patient_events["code"].isin(codes)]

            if not relevant_events.empty:
                # Get earliest event
                earliest_event = relevant_events.sort_values("time").iloc[0]
                event_time = earliest_event["time"]
                time_to_event = (event_time - enrollment_date).days

                # Cap at max follow-up
                time_to_event = min(float(time_to_event), float(self.max_followup_days))

                # Store event info
                patient_events_data[event_name] = (EventStatus.OCCURRED.value, time_to_event)

        # Check for censoring due to earlier events
        min_time = min(time for _, time in patient_events_data.values())
        for event_name in patient_events_data:
            status, time = patient_events_data[event_name]
            if status == EventStatus.CENSORED.value and time > min_time:
                # Censor this event at the time of the earliest event
                patient_events_data[event_name] = (EventStatus.CENSORED.value, min_time)

        # Format results
        events_list = []
        durations_list = []
        for event_name in sorted(patient_events_data.keys()):
            status, time = patient_events_data[event_name]
            events_list.append(status)
            durations_list.append(time)

        return {
            "events": events_list,
            "durations": durations_list,
            "event_types": sorted(patient_events_data.keys()),
            "enrollment_date": enrollment_date,
        }


class CustomEventLabeler(MedicalLabeler):
    """Configurable labeler for custom event definitions."""

    def __init__(
        self,
        name: str,
        event_definition: Dict[str, Any],
        max_followup_days: int = 1095,
        enrollment_codes: List[str] = None
    ):
        """Initialize a custom event labeler.

        Args:
            name: Name of this labeler
            event_definition: Dictionary defining the event
            max_followup_days: Maximum followup period in days
            enrollment_codes: List of codes for enrollment events (default: ["ENROLLMENT"])
        """
        super().__init__(name)
        self.event_definition = event_definition
        self.max_followup_days = max_followup_days
        self.enrollment_codes = enrollment_codes or ["ENROLLMENT"]

    def label(self, events: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Apply custom event definition to label patients.

        This method is maintained for backward compatibility.
        For new code, use parallel_label() instead.

        Args:
            events: DataFrame with medical events in MEDS format

        Returns:
            Dictionary with custom event information
        """
        # Use the new parallel implementation with serial processing
        return self.parallel_label(events, mode=ProcessingMode.SERIAL)

    def label_patient(self, patient_events: pd.DataFrame) -> Dict[str, Any]:
        """Process events for a single patient to identify custom events.
        
        Args:
            patient_events: DataFrame with medical events for a single patient
            
        Returns:
            Dictionary with custom event information for this patient
        """
        # Validate that we have a single patient
        if not patient_events.empty and len(patient_events["subject_id"].unique()) > 1:
            raise ValueError("patient_events contains data for multiple patients")

        if patient_events.empty:
            return {
                "event": EventStatus.CENSORED.value,
                "time": float(self.max_followup_days),
                "enrollment_date": None
            }

        # Find enrollment date
        enrollment_events = patient_events[patient_events["code"].isin(self.enrollment_codes)]
        if enrollment_events.empty:
            # No enrollment date found, try to use the earliest event time
            if "time" in patient_events.columns and not patient_events["time"].isna().all():
                enrollment_date = patient_events["time"].min()
            else:
                # Cannot determine enrollment, return censored at max follow-up
                return {
                    "event": EventStatus.CENSORED.value,
                    "time": float(self.max_followup_days),
                    "enrollment_date": None
                }
        else:
            # Use the earliest enrollment date
            enrollment_date = enrollment_events["time"].min()

        # Apply event definition
        event_occurred, event_time = self._evaluate_event_definition(
            patient_events,
            self.event_definition,
            enrollment_date,
            self.max_followup_days,
        )

        return {
            "event": EventStatus.OCCURRED.value if event_occurred else EventStatus.CENSORED.value,
            "time": float(event_time),
            "enrollment_date": enrollment_date,
        }

    def _evaluate_event_definition(
        self,
        patient_data: pd.DataFrame,
        event_def: Dict[str, Any],
        enrollment_date: datetime,
        max_time: int,
    ) -> tuple:
        """Evaluate if a patient meets the event definition.

        Args:
            patient_data: DataFrame with this patient's events
            event_def: Dictionary defining the event
            enrollment_date: When the patient was enrolled
            max_time: Maximum follow-up time

        Returns:
            Tuple of (event_occurred, time_to_event)
        """
        # Handle multiple conditions with logical operators
        if "and" in event_def:
            # All conditions must be true
            all_occurred = True
            earliest_time = max_time

            for sub_def in event_def["and"]:
                occurred, time = self._evaluate_event_definition(
                    patient_data, sub_def, enrollment_date, max_time
                )
                all_occurred = all_occurred and occurred
                if occurred and time < earliest_time:
                    earliest_time = time

            return all_occurred, earliest_time if all_occurred else max_time

        elif "or" in event_def:
            # Any condition can be true
            any_occurred = False
            earliest_time = max_time

            for sub_def in event_def["or"]:
                occurred, time = self._evaluate_event_definition(
                    patient_data, sub_def, enrollment_date, max_time
                )
                if occurred and time < earliest_time:
                    any_occurred = True
                    earliest_time = time

            return any_occurred, earliest_time

        elif "sequence" in event_def:
            # Events must occur in specified sequence
            events = event_def["sequence"]
            prev_time = enrollment_date
            all_found = True

            for event_spec in events:
                # Find matching events after previous event
                matching_events = self._find_matching_events(
                    patient_data, event_spec, prev_time
                )

                if matching_events.empty:
                    all_found = False
                    break

                # Use earliest matching event
                next_event = matching_events.sort_values("time").iloc[0]
                prev_time = next_event["time"]

            if all_found:
                time_to_event = (prev_time - enrollment_date).days
                # Cap at max follow-up
                time_to_event = min(float(time_to_event), float(max_time))
                return True, time_to_event
            else:
                return False, max_time

        elif "codes" in event_def:
            # Simple code-based event
            codes = event_def["codes"]

            # Find events matching these codes
            matching_events = self._find_matching_events(
                patient_data, {"codes": codes}, enrollment_date
            )

            if not matching_events.empty:
                # Get earliest matching event
                earliest_event = matching_events.sort_values("time").iloc[0]
                time_to_event = (earliest_event["time"] - enrollment_date).days
                # Cap at max follow-up
                time_to_event = min(float(time_to_event), float(max_time))
                return True, time_to_event
            else:
                return False, max_time

        elif "lab" in event_def:
            # Lab-based event
            lab_code = event_def["lab"]["code"]
            comparison = event_def["lab"]["comparison"]
            threshold = event_def["lab"]["threshold"]

            # Find lab events
            lab_events = patient_data[patient_data["code"] == lab_code]

            if lab_events.empty:
                return False, max_time

            # Check values
            for _, lab in lab_events.iterrows():
                value = lab.get("numeric_value")
                if pd.isna(value):
                    continue

                meets_condition = False
                if comparison == ">":
                    meets_condition = value > threshold
                elif comparison == ">=":
                    meets_condition = value >= threshold
                elif comparison == "<":
                    meets_condition = value < threshold
                elif comparison == "<=":
                    meets_condition = value <= threshold
                elif comparison == "=":
                    meets_condition = value == threshold

                if meets_condition:
                    time_to_event = (lab["time"] - enrollment_date).days
                    # Cap at max follow-up
                    time_to_event = min(float(time_to_event), float(max_time))
                    return True, time_to_event

            return False, max_time

        else:
            # Unrecognized event definition
            return False, max_time

    def _find_matching_events(
        self,
        patient_data: pd.DataFrame,
        event_spec: Dict[str, Any],
        after_time: datetime,
    ) -> pd.DataFrame:
        """Find events matching a specification after a given time.

        Args:
            patient_data: DataFrame with patient events
            event_spec: Event specification to match
            after_time: Only include events after this time

        Returns:
            DataFrame with matching events
        """
        # Filter by time
        if "time" not in patient_data.columns:
            return pd.DataFrame()

        filtered_data = patient_data[patient_data["time"] > after_time]

        if "codes" in event_spec:
            # Filter by codes
            codes = event_spec["codes"]
            code_matches = filtered_data["code"].isin(codes)
            return filtered_data[code_matches]
        elif "code_pattern" in event_spec:
            # Filter by regex pattern on codes
            pattern = event_spec["code_pattern"]
            # Use vectorized string method where available
            if hasattr(filtered_data["code"], "str"):
                code_matches = filtered_data["code"].str.contains(pattern, regex=True, na=False)
                return filtered_data[code_matches]
            else:
                # Fallback for non-string columns
                code_matches = filtered_data["code"].apply(
                    lambda x: bool(re.search(pattern, str(x))) if not pd.isna(x) else False
                )
                return filtered_data[code_matches]
        else:
            # No valid filter criteria
            return pd.DataFrame()
