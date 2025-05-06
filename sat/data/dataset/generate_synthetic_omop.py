"""Generate synthetic OMOP CDM data for testing and demonstration.

This module creates realistic synthetic patient data following the OMOP Common Data Model (CDM) schema.

OMOP CDM Schema: https://ohdsi.github.io/CommonDataModel/
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

from .generate_patient_event_json import generate_patient_event_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- OMOP Covariate Definitions ---
CATEGORICAL_COVARIATES = [
    {
        "code": "OMOP_RACE",
        "values": ["White", "Black", "Asian", "Other"],
        "probabilities": [0.7, 0.15, 0.1, 0.05],
    },
    {
        "code": "OMOP_SEX",
        "values": ["Male", "Female"],
        "probabilities": [0.5, 0.5],
    },
    # Add more categorical covariates as needed
]

NUMERICAL_COVARIATES = [
    {
        "code": "OMOP_AGE",
        "name": "age",
        "distribution": "normal",
        "mean": 60,
        "std": 10,
        "min": 18,  # Minimum allowed age
        "max": 90,  # Maximum allowed age
        "time_dependent": False,
    },
    # Add more numerical covariates as needed
]

# --- OMOP Event Types ---
EVENT_TYPES = [
    {
        "name": "DEATH",
        "code": "OMOP_DEATH",
        "competing": False,
        "description": "Death event",
    },
    {
        "name": "MI",
        "code": "OMOP_MI",
        "competing": True,
        "description": "Myocardial infarction",
    },
    {
        "name": "STROKE",
        "code": "OMOP_STROKE",
        "competing": True,
        "description": "Stroke",
    },
    # Add other event types as needed
]


def generate_person_table(num_patients):
    np.random.seed(42)
    return pd.DataFrame(
        [
            {
                "person_id": pid,
                "gender_concept_id": np.random.choice(
                    [8507, 8532]
                ),  # 8507=male, 8532=female
                "year_of_birth": np.random.randint(1940, 2000),
                "month_of_birth": np.random.randint(1, 13),
                "day_of_birth": np.random.randint(1, 29),
                "race_concept_id": 8527,  # example: white
                "ethnicity_concept_id": 38003563,  # Not Hispanic or Latino
            }
            for pid in range(1, num_patients + 1)
        ]
    )


def generate_visit_occurrence_table(person_df):
    visits = []
    for pid in person_df["person_id"]:
        n_visits = np.random.randint(1, 5)
        for v in range(n_visits):
            start_date = datetime(2019, 1, 1) + timedelta(
                days=np.random.randint(0, 365)
            )
            visits.append(
                {
                    "visit_occurrence_id": int(f"{pid}{v}"),
                    "person_id": pid,
                    "visit_concept_id": 9201,  # Inpatient Visit
                    "visit_start_date": start_date.date(),
                    "visit_end_date": (
                        start_date + timedelta(days=np.random.randint(1, 10))
                    ).date(),
                }
            )
    return pd.DataFrame(visits)


def generate_condition_occurrence_table(person_df):
    conditions = []
    for pid in person_df["person_id"]:
        n_conditions = np.random.randint(1, 4)
        for c in range(n_conditions):
            cond_date = datetime(2019, 1, 1) + timedelta(days=np.random.randint(0, 365))
            conditions.append(
                {
                    "condition_occurrence_id": int(f"{pid}{c}"),
                    "person_id": pid,
                    "condition_concept_id": np.random.choice(
                        [321661, 201826, 319835]
                    ),  # MI, Diabetes, Stroke
                    "condition_start_date": cond_date.date(),
                    "condition_end_date": (
                        cond_date + timedelta(days=np.random.randint(1, 30))
                    ).date(),
                }
            )
    return pd.DataFrame(conditions)


def generate_drug_exposure_table(person_df):
    drugs = []
    for pid in person_df["person_id"]:
        n_drugs = np.random.randint(1, 4)
        for d in range(n_drugs):
            drug_date = datetime(2019, 1, 1) + timedelta(days=np.random.randint(0, 365))
            drugs.append(
                {
                    "drug_exposure_id": int(f"{pid}{d}"),
                    "person_id": pid,
                    "drug_concept_id": np.random.choice(
                        [1112807, 19019073, 1125315]
                    ),  # e.g., Metformin, Lisinopril, Atorvastatin
                    "drug_exposure_start_date": drug_date.date(),
                    "drug_exposure_end_date": (
                        drug_date + timedelta(days=np.random.randint(1, 90))
                    ).date(),
                }
            )
    return pd.DataFrame(drugs)


def generate_measurement_table(person_df):
    measurements = []
    for pid in person_df["person_id"]:
        n_meas = np.random.randint(2, 6)
        for m in range(n_meas):
            meas_date = datetime(2019, 1, 1) + timedelta(days=np.random.randint(0, 365))
            measurements.append(
                {
                    "measurement_id": int(f"{pid}{m}"),
                    "person_id": pid,
                    "measurement_concept_id": np.random.choice(
                        [3004249, 3016723, 3027114]
                    ),  # e.g., Systolic BP, Glucose, Cholesterol
                    "measurement_date": meas_date.date(),
                    "value_as_number": np.round(np.random.normal(120, 20), 1),
                    "unit_concept_id": 8582,  # mmHg, mg/dL etc.
                }
            )
    return pd.DataFrame(measurements)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OMOP CDM data")
    parser.add_argument(
        "--num_patients", type=int, default=10000, help="Number of patients to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="synthetic_omop",
        help="Output directory for OMOP tables",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Generating OMOP CDM synthetic data for {args.num_patients} patients")
    person_df = generate_person_table(args.num_patients)
    visit_df = generate_visit_occurrence_table(person_df)
    condition_df = generate_condition_occurrence_table(person_df)
    drug_df = generate_drug_exposure_table(person_df)
    meas_df = generate_measurement_table(person_df)

    person_df.to_parquet(os.path.join(args.output_dir, "person.parquet"))
    visit_df.to_parquet(os.path.join(args.output_dir, "visit_occurrence.parquet"))
    condition_df.to_parquet(
        os.path.join(args.output_dir, "condition_occurrence.parquet")
    )
    drug_df.to_parquet(os.path.join(args.output_dir, "drug_exposure.parquet"))
    meas_df.to_parquet(os.path.join(args.output_dir, "measurement.parquet"))

    logger.info(f"Saved OMOP tables to {args.output_dir}")
    logger.info(f"person: {len(person_df)} rows")
    logger.info(f"visit_occurrence: {len(visit_df)} rows")
    logger.info(f"condition_occurrence: {len(condition_df)} rows")
    logger.info(f"drug_exposure: {len(drug_df)} rows")
    logger.info(f"measurement: {len(meas_df)} rows")

    # Generate patient-centric JSON events for FEMR/MEDS-like structure
    generate_patient_event_json(args.output_dir)


if __name__ == "__main__":
    main()

# Define diagnosis codes to use (ICD codes with descriptions)
ICD_CODES = [
    {"code": "ICD10:I21", "description": "Acute myocardial infarction"},
    {"code": "ICD10:E11", "description": "Type 2 diabetes mellitus"},
    {"code": "ICD10:I63", "description": "Cerebral infarction (stroke)"},
    {"code": "ICD10:I10", "description": "Essential (primary) hypertension"},
    {"code": "ICD10:J44", "description": "Chronic obstructive pulmonary disease"},
    {"code": "ICD10:N18", "description": "Chronic kidney disease"},
    {"code": "ICD10:I50", "description": "Heart failure"},
    {"code": "ICD10:C34", "description": "Malignant neoplasm of bronchus and lung"},
    {"code": "ICD10:F32", "description": "Major depressive disorder, single episode"},
    {"code": "ICD10:M54", "description": "Dorsalgia (back pain)"},
]

# Define medication codes to use (RxNorm codes with descriptions)
OMOP_CODES = [
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
    """Generate synthetic patient data following OMOP schema.

    Args:
        num_patients: Number of patients to generate

    Returns:
        DataFrame with patient data in OMOP format
    """
    logger.info(f"Generating data for {num_patients} patients")

    # Generate patient IDs as integers, per OMOP schema
    patient_ids = list(range(1, num_patients + 1))

    # Initialize OMOP CDM format data
    # OMOP CDM requires: person_id, visit_occurrence_id, condition_occurrence_id, drug_exposure_id, measurement_id
    omop_data = []

    # Add birth date and enrollment date for each patient
    start_date = datetime(1940, 1, 1)
    end_date = datetime(2000, 12, 31)
    enrollment_start = datetime(2018, 1, 1)
    enrollment_end = datetime(2020, 12, 31)

    # Generate base data for each patient
    patient_birth_dates = {}
    patient_enrollment_dates = {}
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
        omop_data.append(
            {
                "patient_id": patient_id,
                "time": None,  # Static event has null timestamp
                "code": "OMOP_BIRTH",
                "numeric_value": None,
                "string_value": birth_date.strftime("%Y-%m-%d"),
            }
        )

        # Add enrollment event
        omop_data.append(
            {
                "patient_id": patient_id,
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
            omop_data.append(
                {
                    "patient_id": patient_id,
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

                omop_data.append(
                    {
                        "patient_id": patient_id,
                        "time": None,  # Static feature has null timestamp
                        "code": feature["code"],
                        "numeric_value": float(values[i]),
                        "string_value": None,
                    }
                )

    # Create DataFrame in OMOP format
    df_omop = pd.DataFrame(omop_data)

    # Risk scores will be assigned later based on demographics

    return (
        df_omop,
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

        # Risk factors for death (OMOP_DEATH)
        mortality_risk = (
            age_modifier * 1.5  # Age has strongest effect on mortality
            + random_health_factor * 5
            + np.random.normal(0, 15)  # Random noise
        )
        risk_scores["OMOP_DEATH"] = max(0, mortality_risk)

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
    """Generate medical events for each patient in OMOP format.

    Args:
        patient_risk_scores: Dictionary mapping patient IDs to event risk scores
        patient_enrollment_dates: Dictionary mapping patient IDs to enrollment dates
        patient_derived_features: Dictionary with patient risk factors and features
        censoring_time: Maximum follow-up time in days

    Returns:
        List of OMOP-formatted events
    """
    omop_events = []
    patient_death_times = {}

    # Helper function to generate event time based on risk score
    def generate_time(risk, scale=1000, shape=1.2):
        # Higher risk = lower time to event
        # Use Weibull distribution for realistic survival times
        lambda_param = np.exp(-risk / scale)
        time = np.random.weibull(shape) / lambda_param
        return time

    # Events are processed based on patient risk factors

    # Process each patient
    for patient_id, enrollment_date in patient_enrollment_dates.items():
        # Generate death time (if it occurs)
        death_risk = patient_risk_scores[patient_id]["OMOP_DEATH"]
        death_time_days = generate_time(death_risk)

        # If death occurs within follow-up period
        if death_time_days <= censoring_time:
            death_date = enrollment_date + timedelta(days=int(death_time_days))
            patient_death_times[patient_id] = death_date

            # Record death event in OMOP format
            omop_events.append(
                {
                    "patient_id": patient_id,
                    "time": death_date,
                    "code": "OMOP_DEATH",
                    "numeric_value": None,
                    "string_value": None,
                }
            )

            # For this patient, other events can only occur before death
            patient_max_time = death_time_days
        else:
            # Patient survived the entire follow-up period
            patient_max_time = censoring_time

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

            # Add hospitalization event
            omop_events.append(
                {
                    "patient_id": patient_id,
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

        # Sample diagnosis codes directly when needed

        for _ in range(num_diagnoses):
            # Generate diagnosis time
            diag_time_days = np.random.uniform(0, patient_max_time)
            diag_date = enrollment_date + timedelta(days=int(diag_time_days))

            # Select a diagnosis code randomly
            icd_info = random.choice(ICD_CODES)

            # Add diagnosis event
            omop_events.append(
                {
                    "patient_id": patient_id,
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

        # Sample medications directly when needed

        for _ in range(num_medications):
            # Generate medication time
            med_time_days = np.random.uniform(0, patient_max_time)
            med_date = enrollment_date + timedelta(days=int(med_time_days))

            # Select a medication randomly
            med_info = random.choice(OMOP_CODES)

            # Add medication event
            omop_events.append(
                {
                    "patient_id": patient_id,
                    "time": med_date,
                    "code": med_info["code"],
                    "numeric_value": None,
                    "string_value": med_info["description"],
                }
            )

        # 4. Generate lab results
        lab_risk = patient_risk_scores[patient_id]["LAB_RESULT"]
        num_labs = np.random.poisson(max(2, lab_risk / 10 * (patient_max_time / 365)))

        # Sample lab tests directly when needed

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
            omop_events.append(
                {
                    "patient_id": patient_id,
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
                    omop_events.append(
                        {
                            "patient_id": patient_id,
                            "time": measure_date,
                            "code": feature["code"],
                            "numeric_value": float(value),
                            "string_value": None,
                        }
                    )

    return omop_events


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


def create_omop_tables(omop_events: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a single OMOP-formatted table from generated events.

    Args:
        omop_events: List of OMOP-formatted events

    Returns:
        DataFrame with events in OMOP format
    """
    # Convert list of event dictionaries to DataFrame
    df_omop = pd.DataFrame(omop_events)

    # Sort events by patient_id and time (requirements of OMOP format)
    df_omop = df_omop.sort_values(by=["patient_id", "time"])

    # Handle null times (for static events)
    df_omop["time_type"] = df_omop["time"].apply(
        lambda x: "static" if pd.isna(x) else "event"
    )

    # Ensure proper column orders according to OMOP schema
    column_order = [
        "patient_id",
        "time",
        "time_type",
        "code",
        "numeric_value",
        "string_value",
    ]
    df_omop = df_omop[column_order]

    return df_omop


def save_omop_to_parquet(
    df_omop: pd.DataFrame, output_path: str, metadata: Dict[str, Any] = None
) -> None:
    """Save OMOP format data to a Parquet file following the required format:

    {fold_}{dataset_name}/
        data/
            data.parquet (subject_id, time, code, numeric_value)
        codes.parquet (code, description, parent_codes)
        subjects_splits.parquet (subject_id, split)

    Args:
        df_omop: DataFrame with OMOP format data
        output_path: Path to save the Parquet file
        metadata: Optional metadata to save with the data
    """
    logger.info(f"Saving OMOP format data to {output_path}")

    # Get base directory and dataset name
    dir_name = os.path.dirname(output_path)
    base_name = os.path.basename(output_path).split(".")[0]

    # Create the dataset folder structure
    dataset_folder = os.path.join(dir_name, base_name)
    data_folder = os.path.join(dataset_folder, "data")
    os.makedirs(data_folder, exist_ok=True)

    logger.info(f"Creating dataset directory structure at {dataset_folder}")

    # Create a copy of the data and prepare it for saving
    df_omop_copy = df_omop.copy()

    # Handle data types and NULL values
    for patient_id in df_omop_copy["patient_id"].unique():
        patient_data = df_omop_copy[
            df_omop_copy["patient_id"] == patient_id
        ].sort_values("time")

        # Make sure nulls are properly set
        patient_data.loc[patient_data["time_type"] == "static", "time"] = None
        patient_data.loc[patient_data["numeric_value"].isna(), "numeric_value"] = None
        patient_data.loc[patient_data["string_value"].isna(), "string_value"] = None

    # 1. Save the main data.parquet file with required columns
    data_df = df_omop_copy[["patient_id", "time", "code", "numeric_value"]].copy()
    data_path = os.path.join(data_folder, "data.parquet")

    # Use PyArrow for proper timestamp handling
    try:
        # Create schema for the data file
        data_schema = pa.schema(
            [
                ("patient_id", pa.int64()),
                ("time", pa.timestamp("us")),  # microsecond precision
                ("code", pa.string()),
                ("numeric_value", pa.float32()),
            ]
        )

        # Convert and save
        data_table = pa.Table.from_pandas(
            data_df, schema=data_schema, preserve_index=False
        )
        pq.write_table(data_table, data_path, compression="snappy")
        logger.info(f"Saved data.parquet with {len(data_df)} records")
    except Exception as e:
        logger.error(f"Error saving data.parquet: {e}")
        # Fallback to pandas
        data_df.to_parquet(data_path, index=False)
        logger.info(
            f"Saved data.parquet using pandas fallback with {len(data_df)} records"
        )

    # 2. Create and save codes.parquet
    codes_data = []

    # Add all codes from ICD, RxNorm, LOINC, and OMOP

    # Add ICD codes
    for icd in ICD_CODES:
        codes_data.append(
            {
                "code": icd["code"],
                "description": icd["description"],
                "parent_code": "ICD10",  # Use ICD10 as parent
            }
        )

    # Add medication codes
    for med in OMOP_CODES:
        codes_data.append(
            {
                "code": med["code"],
                "description": med["description"],
                "parent_code": "RxNorm",  # Use RxNorm as parent
            }
        )

    # Add lab codes
    for lab in LAB_CODES:
        codes_data.append(
            {
                "code": lab["code"],
                "description": lab["description"],
                "parent_code": "LOINC",  # Use LOINC as parent
            }
        )

    # Add special OMOP codes
    omop_special_codes = [
        {"code": "OMOP_BIRTH", "description": "Date of birth"},
        {"code": "OMOP_DEATH", "description": "Date of death"},
        {"code": "ENROLLMENT", "description": "Date of enrollment"},
        {"code": "ENC_INPATIENT", "description": "Inpatient hospitalization encounter"},
        {"code": "GENDER", "description": "Patient gender"},
        {"code": "AGE", "description": "Patient age"},
    ]

    for special_code in omop_special_codes:
        codes_data.append(
            {
                "code": special_code["code"],
                "description": special_code["description"],
                "parent_code": "OMOP",  # Use OMOP as parent
            }
        )

    # Create and save codes dataframe
    codes_df = pd.DataFrame(codes_data)
    codes_path = os.path.join(dataset_folder, "codes.parquet")

    try:
        codes_df.to_parquet(codes_path, index=False)
        logger.info(f"Saved codes.parquet with {len(codes_df)} codes")
    except Exception as e:
        logger.error(f"Error saving codes.parquet: {e}")

    # Save original metadata for reference
    if metadata:
        metadata_path = os.path.join(dataset_folder, f"{base_name}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")

    logger.info(f"OMOP data saved successfully in the format: {dataset_folder}/")
    logger.info(f"  - {data_path}")
    logger.info(f"  - {codes_path}")


def generate_synthetic_omop(
    num_patients: int, output_path: str, seed: Optional[int] = None
) -> None:
    """Generate a complete synthetic OMOP dataset.

    Args:
        num_patients: Number of patients to generate
        output_path: Path to save the Parquet file
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    logger.info(f"Generating synthetic OMOP dataset with {num_patients} patients")

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
    omop_events = generate_event_times(
        patient_risk_scores, patient_enrollment_dates, patient_derived_features
    )
    logger.info(f"Generated {len(omop_events)} medical events")

    # 4. Create single OMOP table from all events
    df_omop = create_omop_tables(omop_events)

    # 5. Save to Parquet in OMOP format
    dataset_metadata = {
        "dataset_name": "synthetic_omop",
        "description": "Synthetic OMOP-format dataset for survival analysis",
        "num_patients": num_patients,
        "num_events": len(omop_events),
        "date_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0",
    }

    save_omop_to_parquet(df_omop, output_path, dataset_metadata)

    # 6. Print summary statistics
    event_types = {}
    for event in omop_events:
        code = event["code"]
        if code not in event_types:
            event_types[code] = 0
        event_types[code] += 1

    logger.info("Event type statistics:")
    for code, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
        n_patients = len(set(e["patient_id"] for e in omop_events if e["code"] == code))
        if n_patients > 0:
            avg_per_patient = count / n_patients
            logger.info(
                f"  - {code}: {count} events across {n_patients} patients (avg {avg_per_patient:.1f} per patient)"
            )

    # Count patients with death events
    death_patients = set(
        e["patient_id"] for e in omop_events if e["code"] == "OMOP_DEATH"
    )
    logger.info(
        f"Patients with death events: {len(death_patients)} ({len(death_patients)/num_patients:.1%} of total)"
    )

    # Count variable length features by patient
    patient_event_counts = {}
    for event in omop_events:
        # Check if event has a time field that is not None (these are time-dependent events)
        if event.get("time") is not None:
            patient_id = event["patient_id"]
            if patient_id not in patient_event_counts:
                patient_event_counts[patient_id] = 0
            patient_event_counts[patient_id] += 1

    # Calculate statistics on variable-length histories
    if patient_event_counts:
        min_events = min(patient_event_counts.values())
        max_events = max(patient_event_counts.values())
        avg_events = sum(patient_event_counts.values()) / len(patient_event_counts)
        logger.info("Patient history statistics:")
        logger.info(f"  - Minimum events per patient: {min_events}")
        logger.info(f"  - Maximum events per patient: {max_events}")
        logger.info(f"  - Average events per patient: {avg_events:.1f}")

    logger.info(f"OMOP dataset generation complete: {output_path}")


def main():
    """Command-line interface for generating synthetic OMOP data."""
    parser = argparse.ArgumentParser(description="Generate synthetic OMOP format data")
    parser.add_argument(
        "--num_patients", type=int, default=10000, help="Number of patients to generate"
    )
    parser.add_argument(
        "--output", type=str, default="synthetic_omop.parquet", help="Output file path"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    generate_synthetic_omop(args.num_patients, args.output, args.seed)


if __name__ == "__main__":
    main()
