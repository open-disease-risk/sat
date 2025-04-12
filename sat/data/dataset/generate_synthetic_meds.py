"""Generate synthetic MEDS format data for testing and demonstration.

This module creates realistic synthetic patient data following the official
MEDS (Medical Event Data Standard) schema with proper medical histories and codes.

MEDS Schema: https://github.com/Medical-Event-Data-Standard/meds
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import argparse
import json
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define event types for the synthetic data (MEDS compliant)
EVENT_TYPES = [
    {"name": "MEDS_DEATH", "code": "MEDS_DEATH", "competing": True},
    {"name": "HOSPITALIZATION", "code": "ENC_INPATIENT", "competing": False},
    {"name": "DIAGNOSIS", "code": "DIAGNOSIS", "competing": False},
    {"name": "MEDICATION", "code": "MEDICATION", "competing": False},
    {"name": "LAB_RESULT", "code": "LAB", "competing": False},
]

# Define covariate types for generating patient features (static)
CATEGORICAL_COVARIATES = [
    {
        "name": "gender",
        "code": "GENDER",
        "values": ["M", "F"],
        "probabilities": [0.48, 0.52],
    },
    {
        "name": "race",
        "code": "RACE",
        "values": ["White", "Black", "Asian", "Hispanic", "Other"],
        "probabilities": [0.65, 0.15, 0.10, 0.07, 0.03],
    },
    {
        "name": "smoking",
        "code": "SMOKING",
        "values": ["Never", "Former", "Current"],
        "probabilities": [0.55, 0.25, 0.20],
    },
    {
        "name": "insurance",
        "code": "INSURANCE",
        "values": ["Private", "Medicare", "Medicaid", "None"],
        "probabilities": [0.60, 0.20, 0.15, 0.05],
    },
]

# Define numerical covariates (some static, some time-dependent)
NUMERICAL_COVARIATES = [
    {
        "name": "age",
        "code": "AGE",
        "time_dependent": False,
        "min": 18,
        "max": 90,
        "distribution": "normal",
        "mean": 55,
        "std": 15,
    },
    {
        "name": "bmi",
        "code": "BMI",
        "time_dependent": True,
        "min": 15,
        "max": 45,
        "distribution": "normal",
        "mean": 28,
        "std": 5,
    },
    {
        "name": "systolic_bp",
        "code": "SYSTOLIC",
        "time_dependent": True,
        "min": 90,
        "max": 200,
        "distribution": "normal",
        "mean": 120,
        "std": 15,
    },
    {
        "name": "diastolic_bp",
        "code": "DIASTOLIC",
        "time_dependent": True,
        "min": 50,
        "max": 120,
        "distribution": "normal",
        "mean": 80,
        "std": 10,
    },
    {
        "name": "heart_rate",
        "code": "HEART_RATE",
        "time_dependent": True,
        "min": 40,
        "max": 120,
        "distribution": "normal",
        "mean": 75,
        "std": 12,
    },
    {
        "name": "height",
        "code": "HEIGHT",
        "time_dependent": False,
        "min": 150,
        "max": 200,
        "distribution": "normal",
        "mean": 170,
        "std": 10,
    },
    {
        "name": "weight",
        "code": "WEIGHT",
        "time_dependent": True,
        "min": 45,
        "max": 150,
        "distribution": "normal",
        "mean": 80,
        "std": 20,
    },
    {
        "name": "glucose",
        "code": "GLUCOSE",
        "time_dependent": True,
        "min": 70,
        "max": 200,
        "distribution": "skewed_normal",
        "mean": 100,
        "std": 25,
        "skew": 1.5,
    },
    {
        "name": "creatinine",
        "code": "CREATININE",
        "time_dependent": True,
        "min": 0.5,
        "max": 3.0,
        "distribution": "skewed_normal",
        "mean": 1.0,
        "std": 0.3,
        "skew": 2.0,
    },
    {
        "name": "cholesterol",
        "code": "CHOLESTEROL",
        "time_dependent": True,
        "min": 120,
        "max": 300,
        "distribution": "normal",
        "mean": 180,
        "std": 30,
    },
]

# Define the diagnosis codes to use (ICD-10 codes with descriptions)
ICD_CODES = [
    {"code": "ICD10:I25.1", "description": "Atherosclerotic heart disease"},
    {"code": "ICD10:I10", "description": "Essential (primary) hypertension"},
    {
        "code": "ICD10:E11.9",
        "description": "Type 2 diabetes mellitus without complications",
    },
    {
        "code": "ICD10:J44.9",
        "description": "Chronic obstructive pulmonary disease, unspecified",
    },
    {
        "code": "ICD10:F32.9",
        "description": "Major depressive disorder, single episode, unspecified",
    },
    {"code": "ICD10:M54.5", "description": "Low back pain"},
    {
        "code": "ICD10:K21.9",
        "description": "Gastro-esophageal reflux disease without esophagitis",
    },
    {"code": "ICD10:G47.00", "description": "Insomnia, unspecified"},
    {"code": "ICD10:N18.3", "description": "Chronic kidney disease, stage 3"},
    {"code": "ICD10:I50.9", "description": "Heart failure, unspecified"},
    {"code": "ICD10:E78.5", "description": "Hyperlipidemia, unspecified"},
    {"code": "ICD10:J45.909", "description": "Unspecified asthma, uncomplicated"},
    {"code": "ICD10:M17.9", "description": "Osteoarthritis of knee, unspecified"},
    {"code": "ICD10:I48.91", "description": "Unspecified atrial fibrillation"},
    {
        "code": "ICD10:K57.30",
        "description": "Diverticulosis of large intestine without perforation or abscess",
    },
]

# Define medication codes to use (RxNorm codes with descriptions)
MEDICATION_CODES = [
    {"code": "RxNorm:C09AA02", "description": "Lisinopril"},
    {"code": "RxNorm:C10AA01", "description": "Simvastatin"},
    {"code": "RxNorm:A10BA02", "description": "Metformin"},
    {"code": "RxNorm:N02BE01", "description": "Acetaminophen"},
    {"code": "RxNorm:C07AB07", "description": "Bisoprolol"},
    {"code": "RxNorm:R03AC02", "description": "Salbutamol"},
    {"code": "RxNorm:N06AB06", "description": "Sertraline"},
    {"code": "RxNorm:M01AE01", "description": "Ibuprofen"},
    {"code": "RxNorm:A02BC01", "description": "Omeprazole"},
    {"code": "RxNorm:C08CA01", "description": "Amlodipine"},
    {"code": "RxNorm:C03CA01", "description": "Furosemide"},
    {"code": "RxNorm:N05BA06", "description": "Lorazepam"},
    {"code": "RxNorm:R03BB04", "description": "Tiotropium"},
    {"code": "RxNorm:B01AC06", "description": "Aspirin"},
    {"code": "RxNorm:C09CA03", "description": "Valsartan"},
]

# Define lab result codes
LAB_CODES = [
    {"code": "LOINC:2093-3", "description": "Cholesterol", "unit": "mg/dL"},
    {"code": "LOINC:2571-8", "description": "Triglycerides", "unit": "mg/dL"},
    {"code": "LOINC:2085-9", "description": "HDL Cholesterol", "unit": "mg/dL"},
    {"code": "LOINC:2089-1", "description": "LDL Cholesterol", "unit": "mg/dL"},
    {"code": "LOINC:2339-0", "description": "Glucose", "unit": "mg/dL"},
    {"code": "LOINC:4548-4", "description": "Hemoglobin A1c", "unit": "%"},
    {"code": "LOINC:2160-0", "description": "Creatinine", "unit": "mg/dL"},
    {"code": "LOINC:3094-0", "description": "BUN", "unit": "mg/dL"},
    {"code": "LOINC:2951-2", "description": "Sodium", "unit": "mmol/L"},
    {"code": "LOINC:2823-3", "description": "Potassium", "unit": "mmol/L"},
]


def generate_patient_data(num_patients: int) -> pd.DataFrame:
    """Generate synthetic patient data following MEDS schema.

    Args:
        num_patients: Number of patients to generate

    Returns:
        DataFrame with patient data in MEDS format
    """
    logger.info(f"Generating data for {num_patients} patients")

    # Generate patient IDs as integers, per MEDS schema
    patient_ids = list(range(1, num_patients + 1))

    # Initialize MEDS format data
    # MEDS requires: subject_id, time, code, numeric_value
    meds_data = []

    # Add birth date and enrollment date for each patient
    start_date = datetime(1940, 1, 1)
    end_date = datetime(2000, 12, 31)
    enrollment_start = datetime(2018, 1, 1)
    enrollment_end = datetime(2020, 12, 31)

    # Generate base data for each patient
    patient_birth_dates = {}
    patient_enrollment_dates = {}
    patient_death_dates = {}
    patient_derived_features = {}

    for patient_id in patient_ids:
        # Generate birth date
        birth_days_range = (end_date - start_date).days
        birth_offset = np.random.randint(0, birth_days_range)
        birth_date = start_date + timedelta(days=int(birth_offset))
        patient_birth_dates[patient_id] = birth_date

        # Generate enrollment date
        enrollment_days_range = (enrollment_end - enrollment_start).days
        enrollment_offset = np.random.randint(0, enrollment_days_range)
        enrollment_date = enrollment_start + timedelta(days=int(enrollment_offset))
        patient_enrollment_dates[patient_id] = enrollment_date

        # Initialize patient data structure
        patient_derived_features[patient_id] = {}

        # Add birth event (static, so timestamp is null)
        meds_data.append(
            {
                "subject_id": patient_id,
                "time": None,  # Static event has null timestamp
                "code": "MEDS_BIRTH",
                "numeric_value": None,
                "string_value": birth_date.strftime("%Y-%m-%d"),
            }
        )

        # Add enrollment event
        meds_data.append(
            {
                "subject_id": patient_id,
                "time": enrollment_date,
                "code": "ENROLLMENT",
                "numeric_value": None,
                "string_value": None,
            }
        )

    # Add static categorical features
    for feature in CATEGORICAL_COVARIATES:
        values = np.random.choice(
            feature["values"], size=num_patients, p=feature["probabilities"]
        )

        for i, patient_id in enumerate(patient_ids):
            meds_data.append(
                {
                    "subject_id": patient_id,
                    "time": None,  # Static feature has null timestamp
                    "code": feature["code"],
                    "numeric_value": None,
                    "string_value": values[i],
                }
            )

    # Add static numerical features
    for feature in NUMERICAL_COVARIATES:
        if not feature["time_dependent"]:
            if feature["distribution"] == "normal":
                values = np.random.normal(feature["mean"], feature["std"], num_patients)
            elif feature["distribution"] == "skewed_normal":
                values = np.random.normal(feature["mean"], feature["std"], num_patients)
                values = values + feature["skew"] * np.abs(
                    np.random.normal(0, 1, num_patients)
                )
            else:
                values = np.random.uniform(feature["min"], feature["max"], num_patients)

            # Clip values to specified range
            values = np.clip(values, feature["min"], feature["max"])

            for i, patient_id in enumerate(patient_ids):
                # Store age for later use
                if feature["name"] == "age":
                    patient_derived_features[patient_id]["age"] = values[i]

                meds_data.append(
                    {
                        "subject_id": patient_id,
                        "time": None,  # Static feature has null timestamp
                        "code": feature["code"],
                        "numeric_value": float(values[i]),
                        "string_value": None,
                    }
                )

    # Create DataFrame in MEDS format
    df_meds = pd.DataFrame(meds_data)

    # Initialize basic patient demographic features
    for patient_id in patient_ids:
        # We'll randomly assign risk scores later based on demographics,
        # but won't explicitly track risk factors
        pass

    return (
        df_meds,
        patient_birth_dates,
        patient_enrollment_dates,
        patient_derived_features,
    )


def generate_event_risk_scores(
    patient_derived_features: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, float]]:
    """Generate risk scores for each event type based on patient demographics.

    Args:
        patient_derived_features: Dictionary with patient demographic features

    Returns:
        Dictionary mapping patient IDs to their event type risk scores
    """
    patient_risk_scores = {}

    for patient_id, features in patient_derived_features.items():
        risk_scores = {}

        # Extract age as the primary demographic factor
        age = features.get("age", 50)

        # Base risk based on age (increases with age)
        age_modifier = 0.02 * age

        # Randomly sample other risk factors instead of explicit tracking
        random_health_factor = np.random.normal(0, 1)  # Random health status

        # Risk factors for death (MEDS_DEATH)
        mortality_risk = (
            age_modifier * 1.5  # Age has strongest effect on mortality
            + random_health_factor * 5
            + np.random.normal(0, 15)  # Random noise
        )
        risk_scores["MEDS_DEATH"] = max(0, mortality_risk)

        # Risk factors for hospitalization
        hospitalization_risk = (
            age_modifier
            + random_health_factor * 8
            + np.random.normal(0, 20)  # Random noise
        )
        risk_scores["HOSPITALIZATION"] = max(0, hospitalization_risk)

        # Risk factors for diagnoses
        diagnosis_risk = (
            0.6 * age_modifier
            + random_health_factor * 6
            + np.random.normal(0, 15)  # Random noise
        )
        risk_scores["DIAGNOSIS"] = max(0, diagnosis_risk)

        # Risk factors for medications
        medication_risk = (
            0.7 * age_modifier
            + random_health_factor * 7
            + np.random.normal(0, 12)  # Random noise
        )
        risk_scores["MEDICATION"] = max(0, medication_risk)

        # Risk factors for lab tests
        lab_risk = (
            0.5 * age_modifier
            + random_health_factor * 4
            + np.random.normal(0, 10)  # Random noise
        )
        risk_scores["LAB_RESULT"] = max(0, lab_risk)

        patient_risk_scores[patient_id] = risk_scores

    return patient_risk_scores


def generate_event_times(
    patient_risk_scores: Dict[int, Dict[str, float]],
    patient_enrollment_dates: Dict[int, datetime],
    patient_derived_features: Dict[int, Dict[str, Any]],
    censoring_time: int = 1095,  # 3 years (max follow-up in days)
) -> List[Dict[str, Any]]:
    """Generate medical events for each patient in MEDS format.

    Args:
        patient_risk_scores: Dictionary mapping patient IDs to event risk scores
        patient_enrollment_dates: Dictionary mapping patient IDs to enrollment dates
        patient_derived_features: Dictionary with patient risk factors and features
        censoring_time: Maximum follow-up time in days

    Returns:
        List of MEDS-formatted events
    """
    meds_events = []
    patient_death_times = {}

    # Helper function to generate event time based on risk score
    def generate_time(risk, scale=1000, shape=1.2):
        # Higher risk = lower time to event
        # Use Weibull distribution for realistic survival times
        lambda_param = np.exp(-risk / scale)
        time = np.random.weibull(shape) / lambda_param
        return time

    # Get competing and non-competing event types
    competing_events = [et for et in EVENT_TYPES if et["competing"]]
    non_competing_events = [et for et in EVENT_TYPES if not et["competing"]]

    # Process each patient
    for patient_id, enrollment_date in patient_enrollment_dates.items():
        # Generate death time (if it occurs)
        death_risk = patient_risk_scores[patient_id]["MEDS_DEATH"]
        death_time_days = generate_time(death_risk)

        # If death occurs within follow-up period
        if death_time_days <= censoring_time:
            death_date = enrollment_date + timedelta(days=int(death_time_days))
            patient_death_times[patient_id] = death_date

            # Record death event in MEDS format
            meds_events.append(
                {
                    "subject_id": patient_id,
                    "time": death_date,
                    "code": "MEDS_DEATH",
                    "numeric_value": None,
                    "string_value": None,
                }
            )

            # For this patient, other events can only occur before death
            patient_max_time = death_time_days
            patient_end_reason = "death"
        else:
            # Patient survived the entire follow-up period
            patient_max_time = censoring_time
            patient_end_reason = "censored"

        # Generate time-dependent measurements and events

        # 1. Generate hospitalizations
        hospitalization_risk = patient_risk_scores[patient_id]["HOSPITALIZATION"]
        # Number of hospitalizations depends on risk and max time
        num_hospitalizations = np.random.poisson(
            max(0.5, hospitalization_risk / 50 * (patient_max_time / 365))
        )

        for _ in range(num_hospitalizations):
            # Generate hospitalization time (days from enrollment)
            hosp_time_days = np.random.uniform(0, patient_max_time)
            hosp_date = enrollment_date + timedelta(days=int(hosp_time_days))

            # Generate length of stay (1-14 days)
            length_of_stay = np.random.randint(1, 15)
            discharge_date = hosp_date + timedelta(days=length_of_stay)

            # Add hospitalization event
            meds_events.append(
                {
                    "subject_id": patient_id,
                    "time": hosp_date,
                    "code": "ENC_INPATIENT",
                    "numeric_value": float(length_of_stay),
                    "string_value": None,
                }
            )

        # 2. Generate diagnoses (ICD codes)
        diagnosis_risk = patient_risk_scores[patient_id]["DIAGNOSIS"]
        num_diagnoses = np.random.poisson(
            max(1, diagnosis_risk / 20 * (patient_max_time / 365))
        )

        # Randomly select diagnosis codes
        all_codes = random.sample(
            ICD_CODES, min(len(ICD_CODES), 5 + int(diagnosis_risk / 10))
        )

        for _ in range(num_diagnoses):
            # Generate diagnosis time
            diag_time_days = np.random.uniform(0, patient_max_time)
            diag_date = enrollment_date + timedelta(days=int(diag_time_days))

            # Select a diagnosis code randomly
            icd_info = random.choice(ICD_CODES)

            # Add diagnosis event
            meds_events.append(
                {
                    "subject_id": patient_id,
                    "time": diag_date,
                    "code": icd_info["code"],
                    "numeric_value": None,
                    "string_value": icd_info["description"],
                }
            )

        # 3. Generate medications
        medication_risk = patient_risk_scores[patient_id]["MEDICATION"]
        num_medications = np.random.poisson(
            max(1, medication_risk / 15 * (patient_max_time / 365))
        )

        # Randomly select medications
        all_meds = random.sample(
            MEDICATION_CODES, min(len(MEDICATION_CODES), 3 + int(medication_risk / 10))
        )

        for _ in range(num_medications):
            # Generate medication time
            med_time_days = np.random.uniform(0, patient_max_time)
            med_date = enrollment_date + timedelta(days=int(med_time_days))

            # Select a medication randomly
            med_info = random.choice(MEDICATION_CODES)

            # Add medication event
            meds_events.append(
                {
                    "subject_id": patient_id,
                    "time": med_date,
                    "code": med_info["code"],
                    "numeric_value": None,
                    "string_value": med_info["description"],
                }
            )

        # 4. Generate lab results
        lab_risk = patient_risk_scores[patient_id]["LAB_RESULT"]
        num_labs = np.random.poisson(max(2, lab_risk / 10 * (patient_max_time / 365)))

        # Randomly select lab tests
        all_labs = random.sample(LAB_CODES, min(len(LAB_CODES), 3 + int(lab_risk / 8)))

        for _ in range(num_labs):
            # Generate lab time
            lab_time_days = np.random.uniform(0, patient_max_time)
            lab_date = enrollment_date + timedelta(days=int(lab_time_days))

            # Select a lab test randomly
            lab_info = random.choice(LAB_CODES)

            # Generate a value with some random variation
            # Randomly determine if value should be normal or abnormal
            if np.random.random() < 0.3:  # 30% chance of abnormal
                base_value = get_abnormal_lab_value(lab_info["code"])
                variation = np.random.normal(0, base_value * 0.1)  # 10% variation
            else:
                base_value = get_normal_lab_value(lab_info["code"])
                variation = np.random.normal(0, base_value * 0.05)  # 5% variation

            lab_value = max(0, base_value + variation)

            # Add lab result event
            meds_events.append(
                {
                    "subject_id": patient_id,
                    "time": lab_date,
                    "code": lab_info["code"],
                    "numeric_value": float(lab_value),
                    "string_value": lab_info["unit"],
                }
            )

        # 5. Generate time-dependent vitals and other numerical features
        for feature in NUMERICAL_COVARIATES:
            if feature["time_dependent"]:
                # Number of measurements based on follow-up time (roughly quarterly)
                num_measurements = max(
                    1, int(patient_max_time / 90) + np.random.randint(0, 3)
                )

                # Generate each measurement
                for _ in range(num_measurements):
                    # Generate measurement time
                    measure_time_days = np.random.uniform(0, patient_max_time)
                    measure_date = enrollment_date + timedelta(
                        days=int(measure_time_days)
                    )

                    # Generate value based on feature properties
                    if feature["distribution"] == "normal":
                        value = np.random.normal(feature["mean"], feature["std"])
                    elif feature["distribution"] == "skewed_normal":
                        value = np.random.normal(feature["mean"], feature["std"])
                        value = value + feature["skew"] * np.abs(np.random.normal(0, 1))
                    else:
                        value = np.random.uniform(feature["min"], feature["max"])

                    # Randomly add some variation to vitals
                    # We'll use a random factor to simulate some patients having
                    # higher or lower measurements without explicitly tracking conditions
                    patient_factor = patient_derived_features[patient_id].get(
                        "random_health_factor", np.random.normal(0, 1)
                    )

                    # Store this factor for consistency in future measures
                    if (
                        "random_health_factor"
                        not in patient_derived_features[patient_id]
                    ):
                        patient_derived_features[patient_id][
                            "random_health_factor"
                        ] = patient_factor

                    # Apply random adjustment based on feature
                    if (
                        feature["name"] in ["systolic_bp", "diastolic_bp"]
                        and patient_factor > 0.7
                    ):
                        value *= 1.15  # Higher BP for some patients
                    elif feature["name"] == "glucose" and patient_factor > 0.8:
                        value *= 1.2  # Higher glucose for some patients
                    elif feature["name"] == "bmi" and patient_factor > 0.6:
                        value *= 1.1  # Higher BMI for some patients
                    elif feature["name"] == "creatinine" and patient_factor > 0.85:
                        value *= 1.25  # Higher creatinine for some patients

                    # Clip value to specified range
                    value = np.clip(value, feature["min"], feature["max"])

                    # Add measurement event
                    meds_events.append(
                        {
                            "subject_id": patient_id,
                            "time": measure_date,
                            "code": feature["code"],
                            "numeric_value": float(value),
                            "string_value": None,
                        }
                    )

    return meds_events


def get_normal_lab_value(lab_code):
    """Return a normal value for a given lab code."""
    lab_normals = {
        "LOINC:2093-3": 180,  # Cholesterol (mg/dL)
        "LOINC:2571-8": 120,  # Triglycerides (mg/dL)
        "LOINC:2085-9": 55,  # HDL Cholesterol (mg/dL)
        "LOINC:2089-1": 100,  # LDL Cholesterol (mg/dL)
        "LOINC:2339-0": 95,  # Glucose (mg/dL)
        "LOINC:4548-4": 5.5,  # Hemoglobin A1c (%)
        "LOINC:2160-0": 0.9,  # Creatinine (mg/dL)
        "LOINC:3094-0": 15,  # BUN (mg/dL)
        "LOINC:2951-2": 140,  # Sodium (mmol/L)
        "LOINC:2823-3": 4.2,  # Potassium (mmol/L)
    }

    return lab_normals.get(lab_code, 100)  # Default if code not found


def get_abnormal_lab_value(lab_code):
    """Return an abnormal value for a given lab code."""
    lab_abnormals = {
        "LOINC:2093-3": 240,  # Cholesterol - high (mg/dL)
        "LOINC:2571-8": 200,  # Triglycerides - high (mg/dL)
        "LOINC:2085-9": 35,  # HDL Cholesterol - low (mg/dL)
        "LOINC:2089-1": 160,  # LDL Cholesterol - high (mg/dL)
        "LOINC:2339-0": 150,  # Glucose - high (mg/dL)
        "LOINC:4548-4": 7.5,  # Hemoglobin A1c - high (%)
        "LOINC:2160-0": 1.8,  # Creatinine - high (mg/dL)
        "LOINC:3094-0": 25,  # BUN - high (mg/dL)
        "LOINC:2951-2": 150,  # Sodium - high (mmol/L)
        "LOINC:2823-3": 5.5,  # Potassium - high (mmol/L)
    }

    return lab_abnormals.get(lab_code, 150)  # Default if code not found


def create_meds_tables(meds_events: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a single MEDS-formatted table from generated events.

    Args:
        meds_events: List of MEDS-formatted events

    Returns:
        DataFrame with events in MEDS format
    """
    # Convert list of event dictionaries to DataFrame
    df_meds = pd.DataFrame(meds_events)

    # Sort events by subject_id and time (requirements of MEDS format)
    df_meds = df_meds.sort_values(by=["subject_id", "time"])

    # Handle null times (for static events)
    df_meds["time_type"] = df_meds["time"].apply(
        lambda x: "static" if pd.isna(x) else "event"
    )

    # Ensure proper column orders according to MEDS schema
    column_order = [
        "subject_id",
        "time",
        "time_type",
        "code",
        "numeric_value",
        "string_value",
    ]
    df_meds = df_meds[column_order]

    return df_meds


def save_meds_to_parquet(
    df_meds: pd.DataFrame, output_path: str, metadata: Dict[str, Any] = None
) -> None:
    """Save MEDS format data to a Parquet file following MEDS schema.

    Args:
        df_meds: DataFrame with MEDS format data
        output_path: Path to save the Parquet file
        metadata: Optional metadata to save with the data
    """
    logger.info(f"Saving MEDS format data to {output_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create PyArrow schema conforming to MEDS requirements
    pa_schema = pa.schema(
        [
            ("subject_id", pa.int64()),
            ("time", pa.timestamp("us")),  # microsecond precision
            ("time_type", pa.string()),
            ("code", pa.string()),
            ("numeric_value", pa.float32()),
            ("string_value", pa.string()),
        ]
    )

    # Convert time column to proper timestamp format (static events have null timestamps)
    df_meds_copy = df_meds.copy()

    # Handle data types for PyArrow schema
    for subject_id in df_meds_copy["subject_id"].unique():
        # For each patient, get their data in correct order
        patient_data = df_meds_copy[
            df_meds_copy["subject_id"] == subject_id
        ].sort_values("time")

        # Make sure nulls are properly set
        patient_data.loc[patient_data["time_type"] == "static", "time"] = None
        patient_data.loc[patient_data["numeric_value"].isna(), "numeric_value"] = None
        patient_data.loc[patient_data["string_value"].isna(), "string_value"] = None

    # Get the directory and base filename
    dir_name = os.path.dirname(output_path)
    base_name = os.path.basename(output_path).split(".")[0]

    # Save data to a single parquet file (per MEDS schema)
    parquet_path = os.path.join(dir_name, f"{base_name}.parquet")

    # Use PyArrow to create table with correct schema
    try:
        # Convert DataFrame to PyArrow Table with the schema
        table = pa.Table.from_pandas(
            df_meds_copy, schema=pa_schema, preserve_index=False
        )

        # Write table to Parquet file
        pq.write_table(table, parquet_path, compression="snappy")
        logger.info(f"Saved MEDS data to {parquet_path}")
    except Exception as e:
        logger.error(f"Error saving to Parquet: {e}")
        # Fallback to pandas if PyArrow conversion fails
        df_meds_copy.to_parquet(parquet_path, index=False)
        logger.info(f"Saved MEDS data using pandas to {parquet_path}")

    # Save metadata (code descriptions, dataset info)
    if metadata:
        metadata_path = os.path.join(dir_name, f"{base_name}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")

    # Create code metadata file (required by MEDS)
    code_metadata = {
        "code_systems": {
            "ICD10": "International Classification of Diseases, 10th Revision",
            "RxNorm": "RxNorm Medication Codes",
            "LOINC": "Logical Observation Identifiers Names and Codes",
            "MEDS": "Medical Event Data Standard special codes",
        },
        "codes": {},
    }

    # Add ICD codes
    for icd in ICD_CODES:
        code_metadata["codes"][icd["code"]] = {
            "description": icd["description"],
            "system": "ICD10",
        }

    # Add medication codes
    for med in MEDICATION_CODES:
        code_metadata["codes"][med["code"]] = {
            "description": med["description"],
            "system": "RxNorm",
        }

    # Add lab codes
    for lab in LAB_CODES:
        code_metadata["codes"][lab["code"]] = {
            "description": lab["description"],
            "system": "LOINC",
            "unit": lab["unit"],
        }

    # Add special MEDS codes
    code_metadata["codes"]["MEDS_BIRTH"] = {
        "description": "Date of birth",
        "system": "MEDS",
    }
    code_metadata["codes"]["MEDS_DEATH"] = {
        "description": "Date of death",
        "system": "MEDS",
    }
    code_metadata["codes"]["ENROLLMENT"] = {
        "description": "Date of enrollment",
        "system": "MEDS",
    }
    code_metadata["codes"]["ENC_INPATIENT"] = {
        "description": "Inpatient hospitalization encounter",
        "system": "MEDS",
    }

    # Save code metadata
    codes_path = os.path.join(dir_name, f"{base_name}_codes.json")
    with open(codes_path, "w") as f:
        json.dump(code_metadata, f, indent=2)
        logger.info(f"Saved code metadata to {codes_path}")

    # Instead of creating random train/val/test splits,
    # we'll let the parsing script handle this based on configuration
    logger.info("Skipping subject splits - will be created during parsing")

    logger.info(f"MEDS data saved successfully to {dir_name}")


def generate_synthetic_meds(
    num_patients: int, output_path: str, seed: Optional[int] = None
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

    logger.info(f"Generating synthetic MEDS dataset with {num_patients} patients")

    # 1. Generate patient data and static features
    (
        df_patient_data,
        patient_birth_dates,
        patient_enrollment_dates,
        patient_derived_features,
    ) = generate_patient_data(num_patients)
    logger.info(f"Generated patient base data with {len(df_patient_data)} records")

    # 2. Generate risk scores for each event type based on patient characteristics
    patient_risk_scores = generate_event_risk_scores(patient_derived_features)
    logger.info(f"Generated risk scores for {len(patient_risk_scores)} patients")

    # 3. Generate medical events for each patient
    meds_events = generate_event_times(
        patient_risk_scores, patient_enrollment_dates, patient_derived_features
    )
    logger.info(f"Generated {len(meds_events)} medical events")

    # 4. Create single MEDS table from all events
    df_meds = create_meds_tables(meds_events)

    # 5. Save to Parquet in MEDS format
    dataset_metadata = {
        "dataset_name": "synthetic_meds",
        "description": "Synthetic MEDS-format dataset for survival analysis",
        "num_patients": num_patients,
        "num_events": len(meds_events),
        "date_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0",
    }

    save_meds_to_parquet(df_meds, output_path, dataset_metadata)

    # 6. Print summary statistics
    event_types = {}
    for event in meds_events:
        code = event["code"]
        if code not in event_types:
            event_types[code] = 0
        event_types[code] += 1

    logger.info("Event type statistics:")
    for code, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
        n_patients = len(set(e["subject_id"] for e in meds_events if e["code"] == code))
        if n_patients > 0:
            avg_per_patient = count / n_patients
            logger.info(
                f"  - {code}: {count} events across {n_patients} patients (avg {avg_per_patient:.1f} per patient)"
            )

    # Count patients with death events
    death_patients = set(
        e["subject_id"] for e in meds_events if e["code"] == "MEDS_DEATH"
    )
    logger.info(
        f"Patients with death events: {len(death_patients)} ({len(death_patients)/num_patients:.1%} of total)"
    )

    # Count variable length features by patient
    patient_event_counts = {}
    for event in meds_events:
        # Check if event has a time field that is not None (these are time-dependent events)
        if event.get("time") is not None:
            subject_id = event["subject_id"]
            if subject_id not in patient_event_counts:
                patient_event_counts[subject_id] = 0
            patient_event_counts[subject_id] += 1

    # Calculate statistics on variable-length histories
    if patient_event_counts:
        min_events = min(patient_event_counts.values())
        max_events = max(patient_event_counts.values())
        avg_events = sum(patient_event_counts.values()) / len(patient_event_counts)
        logger.info("Patient history statistics:")
        logger.info(f"  - Minimum events per patient: {min_events}")
        logger.info(f"  - Maximum events per patient: {max_events}")
        logger.info(f"  - Average events per patient: {avg_events:.1f}")

    logger.info(f"MEDS dataset generation complete: {output_path}")


def main():
    """Command-line interface for generating synthetic MEDS data."""
    parser = argparse.ArgumentParser(description="Generate synthetic MEDS format data")
    parser.add_argument(
        "--num_patients", type=int, default=10000, help="Number of patients to generate"
    )
    parser.add_argument(
        "--output", type=str, default="synthetic_meds.parquet", help="Output file path"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    generate_synthetic_meds(args.num_patients, args.output, args.seed)


if __name__ == "__main__":
    main()
