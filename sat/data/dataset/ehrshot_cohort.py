"""
EHRSHOT multi-event cohort extraction with true time-to-event calculation.

This module processes EHRSHOT benchmark data to create a faithful multi-event survival dataset
by extracting actual event times from the raw medical records.
"""

__authors__ = ["EHRSHOT Integration"]
__status__ = "Development"

import json
import yaml
from dataclasses import dataclass
from datetime import datetime, timedelta
from logging import DEBUG, ERROR
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from logdecorator import log_on_end, log_on_error, log_on_start

from sat.utils import logging

logger = logging.get_default_logger()


@dataclass
class EHRSHOTCohort:
    """
    EHRSHOT multi-event cohort extraction with true survival times.
    
    Extracts actual event times from raw medical records to compute
    faithful time-to-event values for survival analysis.
    """
    
    source: str  # Path to EHRSHOT_ASSETS directory
    condition_codes_path: str  # Path to condition_codes.yaml
    label_dirs: Dict[str, List[str]]  # Mapping of anchor types to task directories
    medical_history: str  # Path to ehrshot.csv with medical codes
    processed_dir: str  # Output directory
    name: str  # Dataset name
    primary_key: str = "patient_id"
    time_field: str = "time"
    max_events_per_patient: int = 10000  # Limit events to prevent memory issues
    validate_against_labels: bool = True  # Validate extracted events against EHRSHOT labels
    
    def __post_init__(self):
        """Load condition code mappings."""
        with open(self.condition_codes_path, 'r') as f:
            self.code_config = yaml.safe_load(f)
        
        self.condition_codes = self.code_config.get('conditions', {})
        self.lab_codes = self.code_config.get('lab_tasks', {})
    
    @log_on_start(DEBUG, "Creating EHRSHOT multi-event cohort with true event times...")
    @log_on_error(
        ERROR,
        "Error creating EHRSHOT cohort: {e!r}",
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!")
    def __call__(self) -> Dataset:
        """Create the multi-event cohort dataset with actual time-to-event values."""
        
        # Load label data for all tasks
        logger.info("Loading EHRSHOT label data...")
        admission_tasks = self._load_task_labels("admission_anchored")
        discharge_tasks = self._load_task_labels("discharge_anchored")
        
        # Load medical history
        logger.info("Loading medical history from ehrshot.csv...")
        medical_df = self._load_medical_history()
        
        # Process admission-anchored cohort
        logger.info("Processing admission-anchored tasks with true event extraction...")
        admission_cohort = self._create_cohort_with_true_events(
            admission_tasks, 
            medical_df, 
            anchor_type="admission"
        )
        
        # Process discharge-anchored cohort
        logger.info("Processing discharge-anchored tasks with true event extraction...")
        discharge_cohort = self._create_cohort_with_true_events(
            discharge_tasks, 
            medical_df, 
            anchor_type="discharge"
        )
        
        # Combine cohorts
        logger.info("Combining cohorts...")
        combined_cohort = self._combine_cohorts(admission_cohort, discharge_cohort)
        
        # Convert to HuggingFace Dataset
        logger.info("Converting to HuggingFace Dataset...")
        dataset = self._create_dataset(combined_cohort)
        
        # Save dataset
        output_path = Path(self.processed_dir) / self.name
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving dataset to {output_path}...")
        dataset.save_to_disk(str(output_path))
        
        # Save metadata
        metadata = {
            "created": datetime.now().isoformat(),
            "num_patients": len(dataset),
            "anchor_types": list(self.label_dirs.keys()),
            "tasks": {k: v for k, v in self.label_dirs.items()},
            "num_events": len(self.label_dirs.get("admission_anchored", [])) + 
                         len(self.label_dirs.get("discharge_anchored", [])),
            "extraction_method": "true_event_times",
            "condition_codes_used": self.condition_codes_path
        }
        
        with open(output_path / "_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created cohort with {len(dataset)} patient-anchor pairs")
        return dataset
    
    def _load_task_labels(self, anchor_type: str) -> Dict[str, pd.DataFrame]:
        """Load label data for tasks of a given anchor type."""
        tasks = {}
        task_dirs = self.label_dirs.get(anchor_type, [])
        
        for task in task_dirs:
            label_path = Path(self.source) / "benchmark" / task / "labeled_patients.csv"
            if label_path.exists():
                logger.debug(f"Loading labels from {label_path}")
                df = pd.read_csv(label_path)
                df['prediction_time'] = pd.to_datetime(df['prediction_time'])
                tasks[task] = df
            else:
                logger.warning(f"Label file not found: {label_path}")
        
        return tasks
    
    def _load_medical_history(self) -> pd.DataFrame:
        """Load and preprocess medical history data."""
        med_path = Path(self.medical_history)
        
        if not med_path.exists():
            raise FileNotFoundError(f"Medical history file not found: {med_path}")
        
        # Read in chunks for memory efficiency
        logger.info(f"Reading medical history from {med_path}")
        chunks = []
        for chunk in pd.read_csv(med_path, chunksize=100000):
            # Convert time columns
            chunk['start'] = pd.to_datetime(chunk['start'])
            chunk['end'] = pd.to_datetime(chunk['end'])
            chunks.append(chunk)
        
        medical_df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(medical_df)} medical events for {medical_df['patient_id'].nunique()} patients")
        
        return medical_df
    
    def _create_cohort_with_true_events(
        self, 
        tasks: Dict[str, pd.DataFrame], 
        medical_df: pd.DataFrame,
        anchor_type: str
    ) -> List[Dict[str, Any]]:
        """Create cohort with actual time-to-event values extracted from medical records."""
        
        if not tasks:
            return []
        
        cohort = []
        validation_results = []
        
        # Get unique patient-anchor combinations
        anchor_times = self._get_anchor_times(tasks)
        
        for (patient_id, anchor_time) in anchor_times:
            # Extract medical history up to anchor
            patient_events = self._extract_patient_events(
                patient_id, 
                anchor_time, 
                medical_df
            )
            
            if not patient_events:
                continue
            
            # Calculate TRUE time-to-event for each task
            outcomes = {}
            for task_name, task_df in tasks.items():
                # Get the EHRSHOT label for validation
                ehrshot_label = self._get_ehrshot_label(
                    patient_id, anchor_time, task_df
                )
                
                # Extract actual time-to-event from medical records
                duration, event = self._extract_true_time_to_event(
                    patient_id, 
                    anchor_time, 
                    task_name,
                    medical_df
                )
                
                # Validate against EHRSHOT label if requested
                if self.validate_against_labels and ehrshot_label is not None:
                    validation_results.append({
                        'patient_id': patient_id,
                        'task': task_name,
                        'ehrshot_label': ehrshot_label,
                        'extracted_event': event,
                        'duration': duration,
                        'match': (event == 1) == ehrshot_label
                    })
                
                # Store outcomes
                clean_name = task_name.replace("new_", "").replace("guo_", "")
                outcomes[f"duration_{clean_name}"] = duration
                outcomes[f"event_{clean_name}"] = event
            
            # Create cohort entry
            entry = {
                self.primary_key: int(patient_id),
                "anchor_time": anchor_time.isoformat(),
                "anchor_type": anchor_type,
                "events": patient_events,
                **outcomes
            }
            
            cohort.append(entry)
        
        # Report validation results
        if validation_results:
            self._report_validation_results(validation_results, anchor_type)
        
        logger.info(f"Created {len(cohort)} entries for {anchor_type} anchor")
        return cohort
    
    def _extract_true_time_to_event(
        self,
        patient_id: int,
        anchor_time: datetime,
        task_name: str,
        medical_df: pd.DataFrame
    ) -> Tuple[float, int]:
        """Extract actual time-to-event from medical records."""
        
        # Get condition codes for this task
        task_config = self._get_task_config(task_name)
        
        if not task_config:
            logger.warning(f"No configuration found for task {task_name}")
            return 365.0, 0  # Default censoring
        
        condition_codes = task_config.get('codes', [])
        horizon_days = task_config.get('horizon_days', 365)
        
        # Special handling for different task types
        if task_name == 'guo_los':
            # Length of stay is calculated from admission to discharge
            return self._calculate_length_of_stay(patient_id, anchor_time, medical_df)
        
        elif task_name == 'guo_readmission':
            # Readmission is next inpatient visit after discharge
            return self._calculate_readmission(patient_id, anchor_time, medical_df, horizon_days)
        
        elif task_name == 'guo_icu':
            # ICU transfer during current admission
            return self._calculate_icu_transfer(patient_id, anchor_time, medical_df)
        
        elif condition_codes:
            # Diagnosis tasks - look for first occurrence of condition codes
            patient_events = medical_df[
                (medical_df['patient_id'] == patient_id) &
                (medical_df['start'] > anchor_time) &
                (medical_df['code'].isin(condition_codes))
            ]
            
            if not patient_events.empty:
                # Sort by time and get first occurrence
                patient_events = patient_events.sort_values('start')
                first_event = patient_events.iloc[0]
                
                # Calculate time to event
                time_to_event = (first_event['start'] - anchor_time).total_seconds() / (24 * 3600)
                
                # Check if within horizon
                if time_to_event <= horizon_days:
                    logger.debug(f"Patient {patient_id} had {task_name} at day {time_to_event:.1f}")
                    return time_to_event, 1
                else:
                    # Event occurred but after horizon - censored
                    return horizon_days, 0
            else:
                # No event found - censored at horizon
                return horizon_days, 0
        
        else:
            # Lab tasks or unknown - use default
            return horizon_days, 0
    
    def _calculate_length_of_stay(
        self, 
        patient_id: int, 
        admission_time: datetime,
        medical_df: pd.DataFrame
    ) -> Tuple[float, int]:
        """Calculate actual length of stay from admission to discharge."""
        
        # Find discharge visit after this admission
        visits = medical_df[
            (medical_df['patient_id'] == patient_id) &
            (medical_df['start'] >= admission_time) &
            (medical_df['omop_table'] == 'visit_occurrence')
        ]
        
        if not visits.empty:
            # Find the end of the admission period
            visits = visits.sort_values('start')
            admission_visit = visits.iloc[0]
            
            if pd.notna(admission_visit['end']):
                los = (admission_visit['end'] - admission_time).total_seconds() / (24 * 3600)
                event = 1 if los >= 7 else 0  # Event if LOS >= 7 days
                return min(los, 7.0), event  # Cap at 7 days for the task
        
        return 7.0, 0  # Default censoring
    
    def _calculate_readmission(
        self, 
        patient_id: int,
        discharge_time: datetime,
        medical_df: pd.DataFrame,
        horizon_days: int
    ) -> Tuple[float, int]:
        """Calculate time to readmission after discharge."""
        
        # Look for next inpatient visit
        readmissions = medical_df[
            (medical_df['patient_id'] == patient_id) &
            (medical_df['start'] > discharge_time) &
            (medical_df['code'] == 'Visit/IP')
        ]
        
        if not readmissions.empty:
            readmissions = readmissions.sort_values('start')
            first_readmit = readmissions.iloc[0]
            
            time_to_readmit = (first_readmit['start'] - discharge_time).total_seconds() / (24 * 3600)
            
            if time_to_readmit <= horizon_days:
                return time_to_readmit, 1
        
        return horizon_days, 0
    
    def _calculate_icu_transfer(
        self,
        patient_id: int,
        admission_time: datetime,
        medical_df: pd.DataFrame
    ) -> Tuple[float, int]:
        """Calculate time to ICU transfer during admission."""
        
        # Look for ICU visit after admission (within reasonable window)
        icu_window = admission_time + timedelta(days=30)  # Max admission duration
        
        icu_events = medical_df[
            (medical_df['patient_id'] == patient_id) &
            (medical_df['start'] > admission_time) &
            (medical_df['start'] < icu_window) &
            (medical_df['code'].isin(['Visit/ICU', 'CPT/99291', 'CPT/99292']))
        ]
        
        if not icu_events.empty:
            icu_events = icu_events.sort_values('start')
            first_icu = icu_events.iloc[0]
            
            time_to_icu = (first_icu['start'] - admission_time).total_seconds() / (24 * 3600)
            
            if time_to_icu <= 7:  # Within admission window
                return time_to_icu, 1
        
        return 7.0, 0
    
    def _get_task_config(self, task_name: str) -> Dict[str, Any]:
        """Get configuration for a specific task."""
        
        # Remove prefixes
        clean_name = task_name.replace("new_", "").replace("guo_", "").replace("lab_", "")
        
        # Check diagnosis conditions
        if clean_name in self.condition_codes:
            return self.condition_codes[clean_name]
        
        # Check lab tasks
        if clean_name in self.lab_codes:
            return self.lab_codes[clean_name]
        
        # Check operational tasks
        for key in ['readmission', 'los', 'icu']:
            if key in clean_name:
                if key in self.condition_codes:
                    return self.condition_codes[key]
        
        return {}
    
    def _get_ehrshot_label(
        self,
        patient_id: int,
        anchor_time: datetime,
        task_df: pd.DataFrame
    ) -> Optional[bool]:
        """Get EHRSHOT label for validation."""
        
        patient_task = task_df[
            (task_df['patient_id'] == patient_id) &
            (task_df['prediction_time'] == anchor_time)
        ]
        
        if not patient_task.empty:
            row = patient_task.iloc[0]
            if row['label_type'] == 'boolean':
                return row['value'] in [True, 'True', 1, '1']
            elif row['label_type'] == 'categorical':
                return row['value'] not in [0, '0', False, 'False']
        
        return None
    
    def _report_validation_results(
        self,
        validation_results: List[Dict],
        anchor_type: str
    ):
        """Report validation results comparing extracted events to EHRSHOT labels."""
        
        if not validation_results:
            return
        
        df = pd.DataFrame(validation_results)
        
        # Calculate agreement by task
        agreement_by_task = df.groupby('task')['match'].mean() * 100
        
        logger.info(f"\nValidation Results for {anchor_type} anchor:")
        logger.info(f"Total comparisons: {len(df)}")
        logger.info(f"Overall agreement: {df['match'].mean() * 100:.1f}%")
        logger.info("\nAgreement by task:")
        for task, agreement in agreement_by_task.items():
            logger.info(f"  {task}: {agreement:.1f}%")
        
        # Log mismatches for investigation
        mismatches = df[~df['match']]
        if not mismatches.empty:
            logger.debug(f"\nFound {len(mismatches)} mismatches to investigate")
            for _, row in mismatches.head(10).iterrows():
                logger.debug(
                    f"  Patient {row['patient_id']}, {row['task']}: "
                    f"EHRSHOT={row['ehrshot_label']}, Extracted={row['extracted_event']}, "
                    f"Duration={row['duration']:.1f} days"
                )
    
    def _get_anchor_times(self, tasks: Dict[str, pd.DataFrame]) -> List[Tuple[int, datetime]]:
        """Get unique patient-anchor time combinations from tasks."""
        anchor_times = set()
        
        for task_df in tasks.values():
            for _, row in task_df.iterrows():
                anchor_times.add((row['patient_id'], row['prediction_time']))
        
        return sorted(list(anchor_times))
    
    def _extract_patient_events(
        self, 
        patient_id: int, 
        anchor_time: datetime,
        medical_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Extract medical events for a patient up to anchor time."""
        
        # Filter events for this patient before anchor
        patient_data = medical_df[
            (medical_df['patient_id'] == patient_id) & 
            (medical_df['start'] <= anchor_time)
        ]
        
        if patient_data.empty:
            return []
        
        # Sort by time
        patient_data = patient_data.sort_values('start')
        
        # Limit number of events
        if len(patient_data) > self.max_events_per_patient:
            logger.warning(f"Patient {patient_id} has {len(patient_data)} events, limiting to {self.max_events_per_patient}")
            patient_data = patient_data.tail(self.max_events_per_patient)
        
        # Convert to event list
        events = []
        for _, row in patient_data.iterrows():
            event = {
                self.time_field: row['start'].isoformat(),
                "code": str(row['code']),
                "numeric_value": float(row['value']) if pd.notna(row['value']) else None,
                "string_value": str(row['unit']) if pd.notna(row['unit']) else None,
                "omop_table": str(row['omop_table']) if 'omop_table' in row else None
            }
            events.append(event)
        
        return events
    
    def _combine_cohorts(
        self, 
        admission_cohort: List[Dict], 
        discharge_cohort: List[Dict]
    ) -> List[Dict]:
        """Combine admission and discharge anchored cohorts."""
        
        combined = admission_cohort + discharge_cohort
        logger.info(f"Combined cohort has {len(combined)} total entries")
        return combined
    
    def _create_dataset(self, cohort: List[Dict]) -> Dataset:
        """Convert cohort list to HuggingFace Dataset."""
        
        if not cohort:
            raise ValueError("Empty cohort - no data to convert")
        
        # Define features based on first entry
        sample = cohort[0]
        
        # Build feature schema
        features_dict = {
            self.primary_key: Value("int64"),
            "anchor_time": Value("string"),
            "anchor_type": Value("string"),
            "events": Sequence({
                self.time_field: Value("string"),
                "code": Value("string"),
                "numeric_value": Value("float32"),
                "string_value": Value("string"),
                "omop_table": Value("string")
            })
        }
        
        # Add duration and event columns
        for key in sample.keys():
            if key.startswith("duration_"):
                features_dict[key] = Value("float32")
            elif key.startswith("event_"):
                features_dict[key] = Value("int32")
        
        features = Features(features_dict)
        
        # Create dataset
        dataset = Dataset.from_list(cohort, features=features)
        
        return dataset