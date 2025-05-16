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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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


@dataclass
class SyntheticOmopGenerator:
    """Generator for synthetic OMOP data.

    This class creates a realistic synthetic dataset following the OMOP Common Data Model
    for use in testing and demonstration of survival analysis pipelines.
    """

    # Configuration parameters
    processed_dir: str = "./"
    num_patients: int = 10000
    name: str = "synthetic_omop"
    seed: Optional[int] = 42
    censoring_time: int = 1095  # Default 3 years (max follow-up in days)

    # Data definitions
    categorical_covariates: List[Dict[str, Any]] = field(
        default_factory=lambda: CATEGORICAL_COVARIATES
    )
    numerical_covariates: List[Dict[str, Any]] = field(
        default_factory=lambda: NUMERICAL_COVARIATES
    )
    event_types: List[Dict[str, Any]] = field(default_factory=lambda: EVENT_TYPES)
    icd_codes: List[Dict[str, str]] = field(default_factory=lambda: ICD_CODES)
    omop_codes: List[Dict[str, str]] = field(default_factory=lambda: OMOP_CODES)
    lab_codes: List[Dict[str, str]] = field(default_factory=lambda: LAB_CODES)

    # State variables (not part of initialization)
    _df_patient_data: Optional[pd.DataFrame] = field(
        default=None, init=False, repr=False
    )
    _patient_birth_dates: Dict[int, datetime] = field(
        default_factory=dict, init=False, repr=False
    )
    _patient_enrollment_dates: Dict[int, datetime] = field(
        default_factory=dict, init=False, repr=False
    )
    _patient_derived_features: Dict[int, Dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _patient_risk_scores: Dict[int, Dict[str, float]] = field(
        default_factory=dict, init=False, repr=False
    )
    _omop_events: List[Dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _df_omop: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize after instance creation."""
        self._initialize_random_seed()

    def _initialize_random_seed(self):
        """Set random seeds for reproducibility."""
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
            logger.info(f"Random seed set to {self.seed}")

    # Methods below are specific to the new implementation

    def generate_patient_data(self):
        """Generate synthetic patient data following OMOP schema.

        Returns:
            Tuple containing the patient dataframe and various patient metadata dictionaries
        """
        logger.info(f"Generating data for {self.num_patients} patients")

        # Generate patient IDs as integers, per OMOP schema
        patient_ids = list(range(1, self.num_patients + 1))

        # Initialize OMOP CDM format data
        omop_data = []

        # Add birth date and enrollment date for each patient
        start_date = datetime(1940, 1, 1)
        end_date = datetime(2000, 12, 31)
        enrollment_start = datetime(2018, 1, 1)
        enrollment_end = datetime(2020, 12, 31)

        # Reset state dictionaries
        self._patient_birth_dates = {}
        self._patient_enrollment_dates = {}
        self._patient_derived_features = {}

        for patient_id in patient_ids:
            # Generate birth date
            birth_days_range = (end_date - start_date).days
            birth_offset = np.random.randint(0, birth_days_range)
            birth_date = start_date + timedelta(days=int(birth_offset))
            self._patient_birth_dates[patient_id] = birth_date

            # Generate enrollment date
            enrollment_days_range = (enrollment_end - enrollment_start).days
            enrollment_offset = np.random.randint(0, enrollment_days_range)
            enrollment_date = enrollment_start + timedelta(days=int(enrollment_offset))
            self._patient_enrollment_dates[patient_id] = enrollment_date

            # Initialize patient data structure
            self._patient_derived_features[patient_id] = {}

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
        for feature in self.categorical_covariates:
            values = np.random.choice(
                feature["values"], size=self.num_patients, p=feature["probabilities"]
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
        for feature in self.numerical_covariates:
            if not feature["time_dependent"]:
                if feature["distribution"] == "normal":
                    values = np.random.normal(
                        feature["mean"], feature["std"], self.num_patients
                    )
                elif feature["distribution"] == "skewed_normal":
                    values = np.random.normal(
                        feature["mean"], feature["std"], self.num_patients
                    )
                    values = values + feature["skew"] * np.abs(
                        np.random.normal(0, 1, self.num_patients)
                    )
                else:
                    values = np.random.uniform(
                        feature["min"], feature["max"], self.num_patients
                    )

                # Clip values to specified range
                values = np.clip(values, feature["min"], feature["max"])

                for i, patient_id in enumerate(patient_ids):
                    # Store age for later use
                    if feature["name"] == "age":
                        self._patient_derived_features[patient_id]["age"] = values[i]

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
        self._df_patient_data = pd.DataFrame(omop_data)

        return (
            self._df_patient_data,
            self._patient_birth_dates,
            self._patient_enrollment_dates,
            self._patient_derived_features,
        )

    def generate_event_risk_scores(self):
        """Generate risk scores for each event type based on patient demographics.

        Returns:
            Dictionary mapping patient IDs to their event type risk scores
        """
        self._patient_risk_scores = {}

        for patient_id, features in self._patient_derived_features.items():
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

            self._patient_risk_scores[patient_id] = risk_scores

        return self._patient_risk_scores

    def get_normal_lab_value(self, lab_code):
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

    def get_abnormal_lab_value(self, lab_code):
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

    def generate_event_times(self):
        """Generate medical events for each patient in OMOP format.

        Returns:
            List of OMOP-formatted events
        """
        self._omop_events = []
        patient_death_times = {}

        # Helper function to generate event time based on risk score
        def generate_time(risk, scale=1000, shape=1.2):
            # Higher risk = lower time to event
            # Use Weibull distribution for realistic survival times
            lambda_param = np.exp(-risk / scale)
            time = np.random.weibull(shape) / lambda_param
            return time

        # Process each patient
        for patient_id, enrollment_date in self._patient_enrollment_dates.items():
            # Generate death time (if it occurs)
            death_risk = self._patient_risk_scores[patient_id]["OMOP_DEATH"]
            death_time_days = generate_time(death_risk)

            # If death occurs within follow-up period
            if death_time_days <= self.censoring_time:
                death_date = enrollment_date + timedelta(days=int(death_time_days))
                patient_death_times[patient_id] = death_date

                # Record death event in OMOP format
                self._omop_events.append(
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
                patient_max_time = self.censoring_time

            # Generate time-dependent measurements and events

            # 1. Generate hospitalizations
            hospitalization_risk = self._patient_risk_scores[patient_id][
                "HOSPITALIZATION"
            ]
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
                self._omop_events.append(
                    {
                        "patient_id": patient_id,
                        "time": hosp_date,
                        "code": "ENC_INPATIENT",
                        "numeric_value": float(length_of_stay),
                        "string_value": None,
                    }
                )

            # 2. Generate diagnoses (ICD codes)
            diagnosis_risk = self._patient_risk_scores[patient_id]["DIAGNOSIS"]
            num_diagnoses = np.random.poisson(
                max(1, diagnosis_risk / 20 * (patient_max_time / 365))
            )

            for _ in range(num_diagnoses):
                # Generate diagnosis time
                diag_time_days = np.random.uniform(0, patient_max_time)
                diag_date = enrollment_date + timedelta(days=int(diag_time_days))

                # Select a diagnosis code randomly
                icd_info = random.choice(self.icd_codes)

                # Add diagnosis event
                self._omop_events.append(
                    {
                        "patient_id": patient_id,
                        "time": diag_date,
                        "code": icd_info["code"],
                        "numeric_value": None,
                        "string_value": icd_info["description"],
                    }
                )

            # 3. Generate medications
            medication_risk = self._patient_risk_scores[patient_id]["MEDICATION"]
            num_medications = np.random.poisson(
                max(1, medication_risk / 15 * (patient_max_time / 365))
            )

            for _ in range(num_medications):
                # Generate medication time
                med_time_days = np.random.uniform(0, patient_max_time)
                med_date = enrollment_date + timedelta(days=int(med_time_days))

                # Select a medication randomly
                med_info = random.choice(self.omop_codes)

                # Add medication event
                self._omop_events.append(
                    {
                        "patient_id": patient_id,
                        "time": med_date,
                        "code": med_info["code"],
                        "numeric_value": None,
                        "string_value": med_info["description"],
                    }
                )

            # 4. Generate lab results
            lab_risk = self._patient_risk_scores[patient_id]["LAB_RESULT"]
            num_labs = np.random.poisson(
                max(2, lab_risk / 10 * (patient_max_time / 365))
            )

            for _ in range(num_labs):
                # Generate lab time
                lab_time_days = np.random.uniform(0, patient_max_time)
                lab_date = enrollment_date + timedelta(days=int(lab_time_days))

                # Select a lab test randomly
                lab_info = random.choice(self.lab_codes)

                # Generate a value with some random variation
                # Randomly determine if value should be normal or abnormal
                if np.random.random() < 0.3:  # 30% chance of abnormal
                    base_value = self.get_abnormal_lab_value(lab_info["code"])
                    variation = np.random.normal(0, base_value * 0.1)  # 10% variation
                else:
                    base_value = self.get_normal_lab_value(lab_info["code"])
                    variation = np.random.normal(0, base_value * 0.05)  # 5% variation

                lab_value = max(0, base_value + variation)

                # Add lab result event
                self._omop_events.append(
                    {
                        "patient_id": patient_id,
                        "time": lab_date,
                        "code": lab_info["code"],
                        "numeric_value": float(lab_value),
                        "string_value": lab_info["unit"],
                    }
                )

            # 5. Generate time-dependent vitals and other numerical features
            for feature in self.numerical_covariates:
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
                            value = value + feature["skew"] * np.abs(
                                np.random.normal(0, 1)
                            )
                        else:
                            value = np.random.uniform(feature["min"], feature["max"])

                        # Randomly add some variation to vitals
                        # We'll use a random factor to simulate some patients having
                        # higher or lower measurements without explicitly tracking conditions
                        patient_factor = self._patient_derived_features[patient_id].get(
                            "random_health_factor", np.random.normal(0, 1)
                        )

                        # Store this factor for consistency in future measures
                        if (
                            "random_health_factor"
                            not in self._patient_derived_features[patient_id]
                        ):
                            self._patient_derived_features[patient_id][
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
                        self._omop_events.append(
                            {
                                "patient_id": patient_id,
                                "time": measure_date,
                                "code": feature["code"],
                                "numeric_value": float(value),
                                "string_value": None,
                            }
                        )

        return self._omop_events

    def create_omop_tables(self):
        """Create a single OMOP-formatted table from generated events.

        Returns:
            DataFrame with events in OMOP format
        """
        # Convert list of event dictionaries to DataFrame
        self._df_omop = pd.DataFrame(self._omop_events)

        # Sort events by patient_id and time
        self._df_omop = self._df_omop.sort_values(by=["patient_id", "time"])

        # Handle null times (for static events)
        self._df_omop["time_type"] = self._df_omop["time"].apply(
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
        self._df_omop = self._df_omop[column_order]

        return self._df_omop

    def save_omop_to_parquet(self, metadata: Dict[str, Any] = None):
        """Save OMOP format data to a Parquet file following the required format:

        {fold_}{dataset_name}/
            data/
                data.parquet (subject_id, time, code, numeric_value)
            codes.parquet (code, description, parent_codes)
            subjects_splits.parquet (subject_id, split)

        Args:
            metadata: Optional metadata to save with the data
        """
        logger.info(f"Saving OMOP format data to {self.processed_dir}")

        dataset_folder = Path(f"{self.processed_dir}/{self.name}")
        data_folder = Path(f"{dataset_folder}/data")
        data_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating dataset directory structure at {dataset_folder}")

        # Create a copy of the data and prepare it for saving
        df_omop_copy = self._df_omop.copy()

        # Handle data types and NULL values
        for patient_id in df_omop_copy["patient_id"].unique():
            patient_data = df_omop_copy[
                df_omop_copy["patient_id"] == patient_id
            ].sort_values("time")

            # Make sure nulls are properly set
            patient_data.loc[patient_data["time_type"] == "static", "time"] = None
            patient_data.loc[patient_data["numeric_value"].isna(), "numeric_value"] = (
                None
            )
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
        for icd in self.icd_codes:
            codes_data.append(
                {
                    "code": icd["code"],
                    "description": icd["description"],
                    "parent_code": "ICD10",  # Use ICD10 as parent
                }
            )

        # Add medication codes
        for med in self.omop_codes:
            codes_data.append(
                {
                    "code": med["code"],
                    "description": med["description"],
                    "parent_code": "RxNorm",  # Use RxNorm as parent
                }
            )

        # Add lab codes
        for lab in self.lab_codes:
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
            {
                "code": "ENC_INPATIENT",
                "description": "Inpatient hospitalization encounter",
            },
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

        # Add subject splits for train/val/test (all train by default)
        # This can be enhanced in the future with more sophisticated splitting
        unique_patients = df_omop_copy["patient_id"].unique()
        subjects_splits = pd.DataFrame(
            {"subject_id": unique_patients, "split": ["train"] * len(unique_patients)}
        )
        subjects_splits_path = os.path.join(dataset_folder, "subjects_splits.parquet")
        subjects_splits.to_parquet(subjects_splits_path, index=False)
        logger.info(
            f"Saved subjects_splits.parquet with {len(subjects_splits)} subjects"
        )

        # Save original metadata for reference
        if metadata is None:
            metadata = {
                "dataset_name": self.name,
                "description": "Synthetic OMOP-format dataset for survival analysis",
                "num_patients": self.num_patients,
                "num_events": len(self._omop_events) if self._omop_events else 0,
                "date_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.0",
            }

        metadata_path = os.path.join(dataset_folder, "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")

        logger.info(f"OMOP data saved successfully in the format: {dataset_folder}/")
        logger.info(f"  - {data_path}")
        logger.info(f"  - {codes_path}")
        logger.info(f"  - {subjects_splits_path}")

        return dataset_folder

    def log_statistics(self):
        """Print summary statistics for the generated data."""
        if not self._omop_events:
            logger.warning("No events to analyze. Generate data first.")
            return

        # Count events by type
        event_types = {}
        for event in self._omop_events:
            code = event["code"]
            if code not in event_types:
                event_types[code] = 0
            event_types[code] += 1

        logger.info("Event type statistics:")
        for code, count in sorted(
            event_types.items(), key=lambda x: x[1], reverse=True
        ):
            n_patients = len(
                set(e["patient_id"] for e in self._omop_events if e["code"] == code)
            )
            if n_patients > 0:
                avg_per_patient = count / n_patients
                logger.info(
                    f"  - {code}: {count} events across {n_patients} patients (avg {avg_per_patient:.1f} per patient)"
                )

        # Count patients with death events
        death_patients = set(
            e["patient_id"] for e in self._omop_events if e["code"] == "OMOP_DEATH"
        )
        logger.info(
            f"Patients with death events: {len(death_patients)} ({len(death_patients)/self.num_patients:.1%} of total)"
        )

        # Count variable length features by patient
        patient_event_counts = {}
        for event in self._omop_events:
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

    def __call__(self):
        """Generate a complete synthetic OMOP dataset.

        This is the main method that orchestrates the full data generation process.
        """
        logger.info(
            f"Generating synthetic OMOP dataset with {self.num_patients} patients"
        )

        # 1. Generate patient data and static features
        self.generate_patient_data()
        logger.info(
            f"Generated patient base data with {len(self._df_patient_data)} records"
        )

        # 2. Generate risk scores for each event type based on patient characteristics
        self.generate_event_risk_scores()
        logger.info(
            f"Generated risk scores for {len(self._patient_risk_scores)} patients"
        )

        # 3. Generate medical events for each patient
        self.generate_event_times()
        logger.info(f"Generated {len(self._omop_events)} medical events")

        # 4. Create single OMOP table from all events
        self.create_omop_tables()

        # 5. Save to Parquet in OMOP format
        dataset_metadata = {
            "dataset_name": os.path.basename(self.processed_dir).split(".")[0],
            "description": "Synthetic OMOP-format dataset for survival analysis",
            "num_patients": self.num_patients,
            "num_events": len(self._omop_events),
            "date_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0",
            "censoring_time": self.censoring_time,
        }

        output_dir = self.save_omop_to_parquet(dataset_metadata)

        # 6. Print summary statistics
        self.log_statistics()

        logger.info(f"OMOP dataset generation complete: {output_dir}")

        return output_dir


def generate_omop_data(config):
    """Generate synthetic OMOP data from a configuration.

    This function is the entry point for Hydra configuration.

    Args:
        config: A configuration object with attributes matching SyntheticOmopGenerator parameters

    Returns:
        The path to the generated dataset
    """
    # Create the generator from config
    generator = SyntheticOmopGenerator(
        num_patients=getattr(config, "num_patients", 10000),
        processed_dir=getattr(config, "processed_dir", "synthetic_omop"),
        seed=getattr(config, "seed", 42),
        censoring_time=getattr(config, "censoring_time", 1095),
    )

    # Optional: use custom covariate definitions if provided
    if hasattr(config, "categorical_covariates"):
        generator.categorical_covariates = config.categorical_covariates

    if hasattr(config, "numerical_covariates"):
        generator.numerical_covariates = config.numerical_covariates

    if hasattr(config, "event_types"):
        generator.event_types = config.event_types

    if hasattr(config, "icd_codes"):
        generator.icd_codes = config.icd_codes

    if hasattr(config, "omop_codes"):
        generator.omop_codes = config.omop_codes

    if hasattr(config, "lab_codes"):
        generator.lab_codes = config.lab_codes

    # Generate the data
    output_dir = generator()

    # Return the dataset path for possible further processing
    return output_dir


def main():
    """Command-line interface for generating synthetic OMOP data."""
    parser = argparse.ArgumentParser(description="Generate synthetic OMOP format data")
    parser.add_argument(
        "--num_patients", type=int, default=10000, help="Number of patients to generate"
    )
    parser.add_argument(
        "--output", type=str, default="synthetic_omop", help="Output directory name"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--censoring_time",
        type=int,
        default=1095,
        help="Maximum follow-up time in days",
    )
    args = parser.parse_args()

    # Create a simple config object from args
    class Config:
        pass

    config = Config()
    config.num_patients = args.num_patients
    config.processed_dir = args.output
    config.seed = args.seed
    config.censoring_time = args.censoring_time

    # Use the same function as Hydra would use
    generate_omop_data(config)


if __name__ == "__main__":
    main()
