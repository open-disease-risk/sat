"""Generate synthetic MEDS format data for testing and demonstration.

This module creates realistic synthetic patient data in MEDS format with multiple
competing events for use in survival analysis.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import argparse
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import uuid
import json

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define event types for the synthetic data
EVENT_TYPES = [
    {"name": "mortality", "table": "mortality", "competing": True},
    {"name": "hospitalization", "table": "hospitalizations", "competing": True},
    {"name": "diagnosis", "table": "diagnoses", "competing": False},
    {"name": "medication", "table": "medications", "competing": False}
]

# Define covariate types for generating patient features
CATEGORICAL_COVARIATES = [
    {"name": "gender", "values": ["M", "F"], "probabilities": [0.48, 0.52]},
    {"name": "race", "values": ["White", "Black", "Asian", "Hispanic", "Other"], 
     "probabilities": [0.65, 0.15, 0.10, 0.07, 0.03]},
    {"name": "smoking", "values": ["Never", "Former", "Current"], 
     "probabilities": [0.55, 0.25, 0.20]},
    {"name": "insurance", "values": ["Private", "Medicare", "Medicaid", "None"], 
     "probabilities": [0.60, 0.20, 0.15, 0.05]}
]

NUMERICAL_COVARIATES = [
    {"name": "age", "min": 18, "max": 90, "distribution": "normal", "mean": 55, "std": 15},
    {"name": "bmi", "min": 15, "max": 45, "distribution": "normal", "mean": 28, "std": 5},
    {"name": "systolic_bp", "min": 90, "max": 200, "distribution": "normal", "mean": 120, "std": 15},
    {"name": "diastolic_bp", "min": 50, "max": 120, "distribution": "normal", "mean": 80, "std": 10},
    {"name": "heart_rate", "min": 40, "max": 120, "distribution": "normal", "mean": 75, "std": 12},
    {"name": "height", "min": 150, "max": 200, "distribution": "normal", "mean": 170, "std": 10},
    {"name": "weight", "min": 45, "max": 150, "distribution": "normal", "mean": 80, "std": 20},
    {"name": "glucose", "min": 70, "max": 200, "distribution": "skewed_normal", "mean": 100, "std": 25, "skew": 1.5},
    {"name": "creatinine", "min": 0.5, "max": 3.0, "distribution": "skewed_normal", "mean": 1.0, "std": 0.3, "skew": 2.0},
    {"name": "cholesterol", "min": 120, "max": 300, "distribution": "normal", "mean": 180, "std": 30}
]

# Define the diagnosis codes to use
ICD_CODES = [
    "I25.1", "I10", "E11.9", "J44.9", "F32.9", "M54.5", "K21.9", "G47.00", 
    "N18.3", "I50.9", "E78.5", "J45.909", "M17.9", "I48.91", "K57.30"
]

# Define medication codes to use
MEDICATION_CODES = [
    "C09AA02", "C10AA01", "A10BA02", "N02BE01", "C07AB07", "R03AC02", 
    "N06AB06", "M01AE01", "A02BC01", "C08CA01", "C03CA01", "N05BA06",
    "R03BB04", "B01AC06", "C09CA03"
]

def generate_patient_data(num_patients: int) -> pd.DataFrame:
    """Generate synthetic patient data.
    
    Args:
        num_patients: Number of patients to generate
        
    Returns:
        DataFrame with patient data
    """
    logger.info(f"Generating data for {num_patients} patients")
    
    # Generate patient IDs as strings, which is the standard for healthcare data
    patient_ids = [f"P{i:06d}" for i in range(1, num_patients + 1)]
    
    # Ensure IDs are strings
    patient_ids = [str(pid) for pid in patient_ids]
    
    # Initialize patient data dictionary
    patient_data = {
        "patient_id": patient_ids,
    }
    
    # Add categorical features
    for feature in CATEGORICAL_COVARIATES:
        patient_data[feature["name"]] = np.random.choice(
            feature["values"], 
            size=num_patients, 
            p=feature["probabilities"]
        )
    
    # Add numerical features
    for feature in NUMERICAL_COVARIATES:
        if feature["distribution"] == "normal":
            values = np.random.normal(feature["mean"], feature["std"], num_patients)
        elif feature["distribution"] == "skewed_normal":
            # Generate skewed normal distribution
            values = np.random.normal(feature["mean"], feature["std"], num_patients)
            # Apply skew transformation
            values = values + feature["skew"] * np.abs(np.random.normal(0, 1, num_patients))
        else:
            values = np.random.uniform(feature["min"], feature["max"], num_patients)
            
        # Clip values to specified range
        values = np.clip(values, feature["min"], feature["max"])
        patient_data[feature["name"]] = values
    
    # Create the patient DataFrame
    df_patients = pd.DataFrame(patient_data)
    
    # Add some derived features
    df_patients["has_hypertension"] = (df_patients["systolic_bp"] > 140) | (df_patients["diastolic_bp"] > 90)
    df_patients["has_diabetes"] = (df_patients["glucose"] > 126)
    df_patients["has_obesity"] = (df_patients["bmi"] > 30)
    df_patients["has_kidney_disease"] = (df_patients["creatinine"] > 1.5)
    
    # Add timestamp for enrollment
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2020, 12, 31)
    days_range = (end_date - start_date).days
    
    random_days = np.random.randint(0, days_range, num_patients)
    # Convert numpy.int64 to Python int to avoid type incompatibility
    enrollment_dates = [start_date + timedelta(days=int(days)) for days in random_days]
    
    df_patients["enrollment_date"] = enrollment_dates
    
    # Convert boolean columns to string for MEDS format compatibility
    bool_cols = df_patients.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_patients[col] = df_patients[col].map({True: "Yes", False: "No"})
    
    return df_patients

def generate_event_risk_scores(df_patients: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Generate risk scores for each event type based on patient covariates.
    
    Args:
        df_patients: DataFrame with patient data
        
    Returns:
        Dictionary mapping event types to risk scores for each patient
    """
    num_patients = len(df_patients)
    risk_scores = {}
    
    # Risk factors for mortality
    mortality_risk = (
        0.02 * df_patients["age"] +
        5 * (df_patients["has_hypertension"] == "Yes").astype(float) +
        8 * (df_patients["has_diabetes"] == "Yes").astype(float) +
        10 * (df_patients["has_kidney_disease"] == "Yes").astype(float) +
        3 * (df_patients["has_obesity"] == "Yes").astype(float) +
        7 * (df_patients["smoking"] == "Current").astype(float) +
        3 * (df_patients["smoking"] == "Former").astype(float) +
        np.random.normal(0, 20, num_patients)  # Random noise
    )
    risk_scores["mortality"] = mortality_risk
    
    # Risk factors for hospitalization
    hospitalization_risk = (
        0.01 * df_patients["age"] +
        10 * (df_patients["has_hypertension"] == "Yes").astype(float) +
        12 * (df_patients["has_diabetes"] == "Yes").astype(float) +
        15 * (df_patients["has_kidney_disease"] == "Yes").astype(float) +
        5 * (df_patients["has_obesity"] == "Yes").astype(float) +
        5 * (df_patients["smoking"] == "Current").astype(float) +
        2 * (df_patients["smoking"] == "Former").astype(float) +
        np.random.normal(0, 25, num_patients)  # Random noise
    )
    risk_scores["hospitalization"] = hospitalization_risk
    
    # Risk factors for diagnosis events
    diagnosis_risk = (
        0.005 * df_patients["age"] +
        7 * (df_patients["has_hypertension"] == "Yes").astype(float) +
        9 * (df_patients["has_diabetes"] == "Yes").astype(float) +
        6 * (df_patients["has_kidney_disease"] == "Yes").astype(float) +
        4 * (df_patients["has_obesity"] == "Yes").astype(float) +
        3 * (df_patients["smoking"] == "Current").astype(float) +
        1 * (df_patients["smoking"] == "Former").astype(float) +
        np.random.normal(0, 15, num_patients)  # Random noise
    )
    risk_scores["diagnosis"] = diagnosis_risk
    
    # Risk factors for medication events
    medication_risk = (
        0.003 * df_patients["age"] +
        9 * (df_patients["has_hypertension"] == "Yes").astype(float) +
        11 * (df_patients["has_diabetes"] == "Yes").astype(float) +
        7 * (df_patients["has_kidney_disease"] == "Yes").astype(float) +
        3 * (df_patients["has_obesity"] == "Yes").astype(float) +
        4 * (df_patients["smoking"] == "Current").astype(float) +
        2 * (df_patients["smoking"] == "Former").astype(float) +
        np.random.normal(0, 18, num_patients)  # Random noise
    )
    risk_scores["medication"] = medication_risk
    
    return risk_scores

def generate_event_times(
    risk_scores: Dict[str, np.ndarray], 
    df_patients: pd.DataFrame,
    censoring_time: int = 1095  # 3 years
) -> Dict[str, pd.DataFrame]:
    """Generate event times for each patient based on their risk scores.
    
    Args:
        risk_scores: Dictionary mapping event types to risk scores
        df_patients: DataFrame with patient data
        censoring_time: Maximum follow-up time in days
        
    Returns:
        Dictionary mapping event types to DataFrames with event data
    """
    num_patients = len(df_patients)
    # Make sure patient IDs are strings for consistent lookup
    patient_ids = [str(pid) for pid in df_patients["patient_id"].tolist()]
    
    event_data = {}
    
    # Helper function to generate event time based on risk score
    def generate_time(risk, scale=1000, shape=1.2):
        # Higher risk = lower time to event
        # Use Weibull distribution for realistic survival times
        lambda_param = np.exp(-risk / scale)
        time = np.random.weibull(shape) / lambda_param
        return time
    
    # Generate competing events (mortality and hospitalization)
    competing_events = [et for et in EVENT_TYPES if et["competing"]]
    non_competing_events = [et for et in EVENT_TYPES if not et["competing"]]
    
    # First generate times for competing events
    competing_event_times = {}
    for event_type in competing_events:
        event_name = event_type["name"]
        # Generate event times based on risk scores
        times = np.array([generate_time(risk) for risk in risk_scores[event_name]])
        competing_event_times[event_name] = times
    
    # Determine the earliest competing event for each patient
    earliest_event = []
    earliest_time = []
    
    for i in range(num_patients):
        min_time = float('inf')
        min_event = None
        
        for event_name, times in competing_event_times.items():
            if times[i] < min_time:
                min_time = times[i]
                min_event = event_name
        
        # Apply censoring
        if min_time > censoring_time:
            min_time = censoring_time
            min_event = "censored"
        
        earliest_event.append(min_event)
        earliest_time.append(min_time)
    
    # Create event dataframes for competing events
    for event_type in competing_events:
        event_name = event_type["name"]
        event_times = []
        event_patient_ids = []
        
        for i in range(num_patients):
            if earliest_event[i] == event_name:
                event_times.append(earliest_time[i])
                event_patient_ids.append(patient_ids[i])
        
        if event_patient_ids:
            event_data[event_name] = pd.DataFrame({
                "patient_id": event_patient_ids,
                "days": event_times,
                "event_type": event_name
            })
        else:
            event_data[event_name] = pd.DataFrame(columns=["patient_id", "days", "event_type"])
    
    # Generate non-competing events (diagnoses and medications)
    for event_type in non_competing_events:
        event_name = event_type["name"]
        all_patient_ids = []
        all_event_times = []
        all_event_codes = []
        
        codes_list = ICD_CODES if event_name == "diagnosis" else MEDICATION_CODES
        
        for i in range(num_patients):
            # Determine if this patient has this event type before competing event/censoring
            patient_max_time = earliest_time[i]
            
            # Generate number of events for this patient (more for higher risk)
            risk = risk_scores[event_name][i]
            num_events = np.random.poisson(max(1, risk / 20))
            
            # Generate each event
            for _ in range(num_events):
                event_time = np.random.uniform(0, patient_max_time)
                code = np.random.choice(codes_list)
                
                all_patient_ids.append(patient_ids[i])
                all_event_times.append(event_time)
                all_event_codes.append(code)
        
        if all_patient_ids:
            if event_name == "diagnosis":
                event_data[event_name] = pd.DataFrame({
                    "patient_id": all_patient_ids,
                    "days": all_event_times,
                    "icd_code": all_event_codes
                })
            else:  # medication
                event_data[event_name] = pd.DataFrame({
                    "patient_id": all_patient_ids,
                    "days": all_event_times,
                    "med_code": all_event_codes
                })
        else:
            cols = ["patient_id", "days"]
            if event_name == "diagnosis":
                cols.append("icd_code")
            else:
                cols.append("med_code")
            event_data[event_name] = pd.DataFrame(columns=cols)
    
    return event_data

def create_meds_tables(
    df_patients: pd.DataFrame, 
    event_data: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """Create MEDS-formatted tables from patient and event data.
    
    Args:
        df_patients: DataFrame with patient data
        event_data: Dictionary mapping event types to event DataFrames
        
    Returns:
        Dictionary mapping table names to DataFrames
    """
    meds_tables = {}
    
    # Patients table stays the same
    meds_tables["patients"] = df_patients
    
    # Create a lookup copy of patients with patient_id as index
    patients_lookup = df_patients.copy()
    patients_lookup.set_index('patient_id', inplace=True)
    
    # Format mortality events
    if "mortality" in event_data and not event_data["mortality"].empty:
        df_mortality = event_data["mortality"].copy()
        # Get enrollment dates for each patient using lookup with error handling
        try:
            enrollment_dates = df_mortality["patient_id"].map(lambda pid: patients_lookup.loc[pid, "enrollment_date"])
            df_mortality["mortality_date"] = enrollment_dates + pd.to_timedelta(df_mortality["days"], unit='d')
        except KeyError as e:
            # Log the error and missing patient IDs
            logger.error(f"Missing patient IDs in lookup: {e}")
            # Try an alternative approach
            for idx, row in df_mortality.iterrows():
                try:
                    enrollment_date = patients_lookup.loc[row["patient_id"], "enrollment_date"]
                    df_mortality.at[idx, "mortality_date"] = enrollment_date + pd.Timedelta(days=row["days"])
                except KeyError:
                    logger.warning(f"Could not find enrollment date for patient {row['patient_id']}")
                    # Use a default date as fallback
                    df_mortality.at[idx, "mortality_date"] = datetime(2019, 1, 1) + pd.Timedelta(days=row["days"])
        df_mortality["positive"] = True
        meds_tables["mortality"] = df_mortality
    else:
        meds_tables["mortality"] = pd.DataFrame(columns=["patient_id", "days", "mortality_date", "positive"])
    
    # Format hospitalization events
    if "hospitalization" in event_data and not event_data["hospitalization"].empty:
        df_hosp = event_data["hospitalization"].copy()
        # Get enrollment dates for each patient using lookup with error handling
        try:
            enrollment_dates = df_hosp["patient_id"].map(lambda pid: patients_lookup.loc[pid, "enrollment_date"])
            df_hosp["admission_date"] = enrollment_dates + pd.to_timedelta(df_hosp["days"], unit='d')
        except KeyError as e:
            # Log the error and missing patient IDs
            logger.error(f"Missing patient IDs in lookup for hospitalizations: {e}")
            # Try an alternative approach
            for idx, row in df_hosp.iterrows():
                try:
                    enrollment_date = patients_lookup.loc[row["patient_id"], "enrollment_date"]
                    df_hosp.at[idx, "admission_date"] = enrollment_date + pd.Timedelta(days=row["days"])
                except KeyError:
                    logger.warning(f"Could not find enrollment date for patient {row['patient_id']}")
                    # Use a default date as fallback
                    df_hosp.at[idx, "admission_date"] = datetime(2019, 1, 1) + pd.Timedelta(days=row["days"])
        # Add discharge date (random duration between 1-14 days)
        stay_duration = np.random.randint(1, 15, len(df_hosp))
        # Convert numpy array to list of Python ints
        stay_duration = [int(d) for d in stay_duration]
        df_hosp["discharge_date"] = df_hosp["admission_date"] + pd.to_timedelta(stay_duration, unit='d')
        df_hosp["positive"] = True
        meds_tables["hospitalizations"] = df_hosp
    else:
        meds_tables["hospitalizations"] = pd.DataFrame(
            columns=["patient_id", "days", "admission_date", "discharge_date", "positive"])
    
    # Format diagnosis events
    if "diagnosis" in event_data and not event_data["diagnosis"].empty:
        df_diag = event_data["diagnosis"].copy()
        # Get enrollment dates for each patient using lookup with error handling
        try:
            enrollment_dates = df_diag["patient_id"].map(lambda pid: patients_lookup.loc[pid, "enrollment_date"])
            df_diag["diagnosis_date"] = enrollment_dates + pd.to_timedelta(df_diag["days"], unit='d')
        except KeyError as e:
            # Log the error and missing patient IDs
            logger.error(f"Missing patient IDs in lookup for diagnoses: {e}")
            # Try an alternative approach
            for idx, row in df_diag.iterrows():
                try:
                    enrollment_date = patients_lookup.loc[row["patient_id"], "enrollment_date"]
                    df_diag.at[idx, "diagnosis_date"] = enrollment_date + pd.Timedelta(days=row["days"])
                except KeyError:
                    logger.warning(f"Could not find enrollment date for patient {row['patient_id']}")
                    # Use a default date as fallback
                    df_diag.at[idx, "diagnosis_date"] = datetime(2019, 1, 1) + pd.Timedelta(days=row["days"])
        meds_tables["diagnoses"] = df_diag
    else:
        meds_tables["diagnoses"] = pd.DataFrame(columns=["patient_id", "days", "icd_code", "diagnosis_date"])
    
    # Format medication events
    if "medication" in event_data and not event_data["medication"].empty:
        df_med = event_data["medication"].copy()
        # Get enrollment dates for each patient using lookup with error handling
        try:
            enrollment_dates = df_med["patient_id"].map(lambda pid: patients_lookup.loc[pid, "enrollment_date"])
            df_med["prescription_date"] = enrollment_dates + pd.to_timedelta(df_med["days"], unit='d')
        except KeyError as e:
            # Log the error and missing patient IDs
            logger.error(f"Missing patient IDs in lookup for medications: {e}")
            # Try an alternative approach
            for idx, row in df_med.iterrows():
                try:
                    enrollment_date = patients_lookup.loc[row["patient_id"], "enrollment_date"]
                    df_med.at[idx, "prescription_date"] = enrollment_date + pd.Timedelta(days=row["days"])
                except KeyError:
                    logger.warning(f"Could not find enrollment date for patient {row['patient_id']}")
                    # Use a default date as fallback
                    df_med.at[idx, "prescription_date"] = datetime(2019, 1, 1) + pd.Timedelta(days=row["days"])
        # Add random duration in days (30, 60, or 90 days)
        duration = np.random.choice([30, 60, 90], len(df_med))
        # Convert numpy array to list of Python ints
        duration = [int(d) for d in duration]
        df_med["end_date"] = df_med["prescription_date"] + pd.to_timedelta(duration, unit='d')
        meds_tables["medications"] = df_med
    else:
        meds_tables["medications"] = pd.DataFrame(
            columns=["patient_id", "days", "med_code", "prescription_date", "end_date"])
    
    return meds_tables

def save_meds_to_parquet(
    meds_tables: Dict[str, pd.DataFrame], 
    output_path: str
) -> None:
    """Save MEDS tables to a Parquet file.
    
    Args:
        meds_tables: Dictionary mapping table names to DataFrames
        output_path: Path to save the Parquet file
    """
    logger.info(f"Saving MEDS format data to {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save each table to its own parquet file
    parquet_files = {}
    
    for table_name, df in meds_tables.items():
        # Get the directory and base filename
        dir_name = os.path.dirname(output_path)
        base_name = os.path.basename(output_path).split('.')[0]
        
        # Create a filename for this table
        table_path = os.path.join(dir_name, f"{base_name}_{table_name}.parquet")
        
        # Save the table
        df.to_parquet(table_path, index=False)
        parquet_files[table_name] = table_path
        logger.info(f"Saved {table_name} table to {table_path}")
    
    # Also save a metadata file to help with loading
    metadata_path = os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path).split('.')[0]}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump({
            'tables': list(parquet_files.keys()),
            'paths': parquet_files,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    logger.info(f"MEDS data saved successfully to {os.path.dirname(output_path)}")

def generate_synthetic_meds(
    num_patients: int, 
    output_path: str,
    seed: Optional[int] = None
) -> None:
    """Generate a complete synthetic MEDS dataset.
    
    Args:
        num_patients: Number of patients to generate
        output_path: Path to save the Parquet file
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Generate patient data
    df_patients = generate_patient_data(num_patients)
    
    # Generate risk scores for each event type
    risk_scores = generate_event_risk_scores(df_patients)
    
    # Generate event times
    event_data = generate_event_times(risk_scores, df_patients)
    
    # Create MEDS tables
    meds_tables = create_meds_tables(df_patients, event_data)
    
    # Save to Parquet
    save_meds_to_parquet(meds_tables, output_path)
    
    # Print summary statistics
    logger.info(f"Generated synthetic MEDS dataset with {num_patients} patients")
    for event_type in EVENT_TYPES:
        event_name = event_type["name"]
        if event_name in event_data:
            if event_type["competing"]:
                n_events = len(event_data[event_name])
                logger.info(f"  - {event_name}: {n_events} events ({n_events/num_patients:.1%} of patients)")
            else:
                n_events = len(event_data[event_name])
                n_patients = len(event_data[event_name]["patient_id"].unique())
                avg_per_patient = n_events / max(1, n_patients)
                logger.info(f"  - {event_name}: {n_events} events across {n_patients} patients (avg {avg_per_patient:.1f} per patient)")

def main():
    """Command-line interface for generating synthetic MEDS data."""
    parser = argparse.ArgumentParser(description="Generate synthetic MEDS format data")
    parser.add_argument("--num_patients", type=int, default=10000, help="Number of patients to generate")
    parser.add_argument("--output", type=str, default="synthetic_meds.parquet", help="Output file path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    generate_synthetic_meds(args.num_patients, args.output, args.seed)

if __name__ == "__main__":
    main()