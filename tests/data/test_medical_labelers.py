"""Test suite for medical_labelers.py module.

This module tests the labeler classes for processing MEDS format data,
including their parallel processing capabilities.
"""

import datetime
import random
import unittest
from pathlib import Path

import pandas as pd

from sat.data.dataset.medical_labelers import (
    MEDS_DEATH_CODE,
    CompetingRiskLabeler,
    CustomEventLabeler,
    EventStatus,
    MedicalLabeler,
    MortalityLabeler,
    ProcessingMode,
    RiskFactorLabeler,
    TimeType,
)


def create_test_patient(
    patient_id: int,
    num_events: int = 10,
    has_death: bool = False,
    has_hospitalization: bool = False,
    has_diabetes: bool = False,
    has_heart_failure: bool = False,
) -> pd.DataFrame:
    """Create synthetic patient data for testing.
    
    Args:
        patient_id: Unique patient identifier
        num_events: Number of events to generate
        has_death: Whether to include a death event
        has_hospitalization: Whether to include a hospitalization event
        has_diabetes: Whether to include diabetes diagnosis
        has_heart_failure: Whether to include heart failure diagnosis
        
    Returns:
        DataFrame with patient events in MEDS format
    """
    # Generate base date and enrollment date
    base_date = datetime.datetime(2020, 1, 1)
    enrollment_date = base_date + datetime.timedelta(days=random.randint(0, 30))

    # Create enrollment event
    events = [{
        "subject_id": patient_id,
        "time": enrollment_date,
        "code": "ENROLLMENT",
        "time_type": TimeType.EVENT,
    }]

    # Generate random events
    event_codes = [
        "ENC_OUTPATIENT",
        "RxNorm:A10", "RxNorm:C07", "RxNorm:C09",
        "LOINC:2339-0", "LOINC:4548-4", "LOINC:2160-0",
    ]

    for i in range(num_events):
        event_time = enrollment_date + datetime.timedelta(days=random.randint(1, 365*3))
        code = random.choice(event_codes)

        event = {
            "subject_id": patient_id,
            "time": event_time,
            "code": code,
            "time_type": TimeType.EVENT,
        }

        # Add numeric value for lab results
        if code.startswith("LOINC"):
            event["numeric_value"] = random.uniform(50, 200)

        events.append(event)

    # Add specific conditions if requested
    if has_death:
        death_time = enrollment_date + datetime.timedelta(days=random.randint(180, 365*3))
        events.append({
            "subject_id": patient_id,
            "time": death_time,
            "code": MEDS_DEATH_CODE,
            "time_type": TimeType.EVENT,
        })

    if has_hospitalization:
        hosp_time = enrollment_date + datetime.timedelta(days=random.randint(30, 365*2))
        events.append({
            "subject_id": patient_id,
            "time": hosp_time,
            "code": "ENC_INPATIENT",
            "time_type": TimeType.EVENT,
        })

    if has_diabetes:
        diabetes_time = enrollment_date + datetime.timedelta(days=random.randint(10, 365))
        events.append({
            "subject_id": patient_id,
            "time": diabetes_time,
            "code": "ICD10:E11",
            "time_type": TimeType.EVENT,
        })

    if has_heart_failure:
        hf_time = enrollment_date + datetime.timedelta(days=random.randint(60, 365*2))
        events.append({
            "subject_id": patient_id,
            "time": hf_time,
            "code": "ICD10:I50",
            "time_type": TimeType.EVENT,
        })

    return pd.DataFrame(events)


def create_test_dataset(
    num_patients: int = 100,
    events_per_patient: int = 10,
    death_ratio: float = 0.2,
    hospitalization_ratio: float = 0.3,
    diabetes_ratio: float = 0.15,
    heart_failure_ratio: float = 0.1,
) -> pd.DataFrame:
    """Create a synthetic dataset with multiple patients.
    
    Args:
        num_patients: Number of patients to generate
        events_per_patient: Average number of events per patient
        death_ratio: Proportion of patients with death events
        hospitalization_ratio: Proportion of patients with hospitalization
        diabetes_ratio: Proportion of patients with diabetes
        heart_failure_ratio: Proportion of patients with heart failure
        
    Returns:
        DataFrame with events for all patients
    """
    all_patients = []

    for i in range(num_patients):
        # Determine if this patient has specific conditions
        has_death = random.random() < death_ratio
        has_hospitalization = random.random() < hospitalization_ratio
        has_diabetes = random.random() < diabetes_ratio
        has_heart_failure = random.random() < heart_failure_ratio

        # Create patient data with some randomness in event count
        actual_events = max(1, int(events_per_patient * random.uniform(0.5, 1.5)))
        patient_data = create_test_patient(
            patient_id=i,
            num_events=actual_events,
            has_death=has_death,
            has_hospitalization=has_hospitalization,
            has_diabetes=has_diabetes,
            has_heart_failure=has_heart_failure,
        )

        all_patients.append(patient_data)

    # Combine all patient data
    return pd.concat(all_patients, ignore_index=True)


class TestMedicalLabelers(unittest.TestCase):
    """Test cases for medical labelers."""

    @classmethod
    def setUpClass(cls):
        """Create test datasets for all tests."""
        # Small dataset for quick tests
        cls.small_dataset = create_test_dataset(
            num_patients=10,
            events_per_patient=8,
        )

        # Medium dataset for parallel processing tests
        cls.medium_dataset = create_test_dataset(
            num_patients=100,
            events_per_patient=15,
        )

        # Save dataset to temp file for large dataset tests
        cls.temp_dir = Path("tests/data/temp")
        cls.temp_dir.mkdir(parents=True, exist_ok=True)
        cls.dataset_path = cls.temp_dir / "test_events.parquet"
        cls.result_dir = cls.temp_dir / "results"
        cls.result_dir.mkdir(exist_ok=True)

        test_large = create_test_dataset(
            num_patients=500,
            events_per_patient=20,
        )
        test_large.to_parquet(cls.dataset_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        # Delete temp files
        if cls.dataset_path.exists():
            cls.dataset_path.unlink()

        # Delete result files
        for f in cls.result_dir.glob("*.parquet"):
            f.unlink()

        # Remove directories
        if cls.result_dir.exists():
            cls.result_dir.rmdir()
        if cls.temp_dir.exists():
            cls.temp_dir.rmdir()

    def test_validate_meds_schema(self):
        """Test MEDS schema validation."""
        from sat.data.dataset.medical_labelers import validate_meds_schema

        # Valid schema should pass
        valid_df = pd.DataFrame({
            "subject_id": [1, 1, 2],
            "time": [datetime.datetime.now()] * 3,
            "code": ["A", "B", "C"],
        })
        validate_meds_schema(valid_df)  # Should not raise an error

        # Invalid schema should raise ValueError
        invalid_df = pd.DataFrame({
            "subject_id": [1, 1, 2],
            "code": ["A", "B", "C"],
            # Missing time column
        })
        with self.assertRaises(ValueError):
            validate_meds_schema(invalid_df)

    def test_risk_factor_labeler_serial(self):
        """Test RiskFactorLabeler with serial processing."""
        # Initialize labeler
        labeler = RiskFactorLabeler("test_risk_labeler")

        # Test with small dataset
        results = labeler.parallel_label(
            self.small_dataset,
            mode=ProcessingMode.SERIAL,
            show_progress=False
        )

        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertGreaterEqual(len(results), 1)

        # Check that at least one patient has risk factors identified
        has_risk_factors = False
        for patient_data in results.values():
            if patient_data.get("risk_factors"):
                has_risk_factors = True
                break
        self.assertTrue(has_risk_factors)

        # Test with custom codes
        custom_labeler = RiskFactorLabeler(
            "custom_risk_labeler",
            custom_codes={
                "icd10": {
                    "test_condition": ["E11"],  # Match diabetes
                },
                "rxnorm": {
                    "test_medication": ["A10"],  # Match diabetes medication
                }
            }
        )

        custom_results = custom_labeler.parallel_label(
            self.small_dataset,
            mode=ProcessingMode.SERIAL,
            show_progress=False
        )

        # Check if custom conditions were detected
        found_test_condition = False
        found_test_medication = False
        for patient_data in custom_results.values():
            risk_factors = patient_data.get("risk_factors", [])
            if "test_condition" in risk_factors:
                found_test_condition = True
            if "test_medication" in risk_factors:
                found_test_medication = True

        self.assertTrue(found_test_condition or found_test_medication)

    def test_mortality_labeler_serial(self):
        """Test MortalityLabeler with serial processing."""
        # Initialize labeler
        labeler = MortalityLabeler(
            "test_mortality_labeler",
            max_followup_days=1095
        )

        # Test with small dataset
        results = labeler.parallel_label(
            self.small_dataset,
            mode=ProcessingMode.SERIAL,
            show_progress=False
        )

        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertGreaterEqual(len(results), 1)

        # Check that patients have event and time fields
        for patient_id, patient_data in results.items():
            self.assertIn("event", patient_data)
            self.assertIn("time", patient_data)

            # Event should be either 0 (censored) or 1 (occurred)
            self.assertIn(patient_data["event"], [EventStatus.CENSORED.value, EventStatus.OCCURRED.value])

            # Time should be a non-negative float
            self.assertIsInstance(patient_data["time"], (int, float))
            self.assertGreaterEqual(patient_data["time"], 0)

            # If event occurred, time should be <= max_followup_days
            if patient_data["event"] == EventStatus.OCCURRED.value:
                self.assertLessEqual(patient_data["time"], 1095)

    def test_competing_risk_labeler_serial(self):
        """Test CompetingRiskLabeler with serial processing."""
        # Initialize labeler
        labeler = CompetingRiskLabeler(
            "test_competing_risk_labeler",
            event_codes={
                "death": [MEDS_DEATH_CODE],
                "hospitalization": ["ENC_INPATIENT"],
            },
            max_followup_days=1095
        )

        # Test with small dataset
        results = labeler.parallel_label(
            self.small_dataset,
            mode=ProcessingMode.SERIAL,
            show_progress=False
        )

        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertGreaterEqual(len(results), 1)

        # Check that patients have events, durations and event_types fields
        for patient_id, patient_data in results.items():
            self.assertIn("events", patient_data)
            self.assertIn("durations", patient_data)
            self.assertIn("event_types", patient_data)

            # Should have two events (death and hospitalization)
            self.assertEqual(len(patient_data["events"]), 2)
            self.assertEqual(len(patient_data["durations"]), 2)
            self.assertEqual(len(patient_data["event_types"]), 2)

            # Events should be either 0 (censored) or 1 (occurred)
            for event in patient_data["events"]:
                self.assertIn(event, [EventStatus.CENSORED.value, EventStatus.OCCURRED.value])

            # Durations should be non-negative floats
            for duration in patient_data["durations"]:
                self.assertIsInstance(duration, (int, float))
                self.assertGreaterEqual(duration, 0)
                self.assertLessEqual(duration, 1095)

    def test_custom_event_labeler_serial(self):
        """Test CustomEventLabeler with serial processing."""
        # Define a custom event for diabetes with complications
        event_definition = {
            "and": [
                {"codes": ["ICD10:E11"]},  # Diabetes
                {"or": [
                    {"codes": ["ICD10:I50"]},  # Heart failure
                ]}
            ]
        }

        # Initialize labeler
        labeler = CustomEventLabeler(
            "test_custom_event_labeler",
            event_definition=event_definition,
            max_followup_days=1095
        )

        # Test with small dataset
        results = labeler.parallel_label(
            self.small_dataset,
            mode=ProcessingMode.SERIAL,
            show_progress=False
        )

        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertGreaterEqual(len(results), 1)

        # Check that patients have event and time fields
        for patient_id, patient_data in results.items():
            self.assertIn("event", patient_data)
            self.assertIn("time", patient_data)

            # Event should be either 0 (censored) or 1 (occurred)
            self.assertIn(patient_data["event"], [EventStatus.CENSORED.value, EventStatus.OCCURRED.value])

            # Time should be a non-negative float
            self.assertIsInstance(patient_data["time"], (int, float))
            self.assertGreaterEqual(patient_data["time"], 0)
            self.assertLessEqual(patient_data["time"], 1095)

    def test_parallel_processing(self):
        """Test parallel processing modes."""
        # Initialize labeler
        labeler = MortalityLabeler("parallel_test_labeler")

        # Test serial processing
        serial_results = labeler.parallel_label(
            self.medium_dataset,
            mode=ProcessingMode.SERIAL,
            show_progress=False
        )

        # Test multiprocessing
        mp_results = labeler.parallel_label(
            self.medium_dataset,
            mode=ProcessingMode.MULTIPROCESSING,
            n_jobs=2,
            batch_size=10,
            show_progress=False
        )

        # Compare results - should be identical
        self.assertEqual(len(serial_results), len(mp_results))

        # Events should match
        for patient_id in serial_results:
            self.assertEqual(
                serial_results[patient_id]["event"],
                mp_results[patient_id]["event"]
            )
            self.assertEqual(
                serial_results[patient_id]["time"],
                mp_results[patient_id]["time"]
            )

        # Test Ray if available
        try:
            import ray

            # Test Ray processing
            ray_results = labeler.parallel_label(
                self.medium_dataset,
                mode=ProcessingMode.RAY,
                n_jobs=2,
                batch_size=10,
                show_progress=False
            )

            # Compare results - should be identical
            self.assertEqual(len(serial_results), len(ray_results))

            # Events should match
            for patient_id in serial_results:
                self.assertEqual(
                    serial_results[patient_id]["event"],
                    ray_results[patient_id]["event"]
                )
                self.assertEqual(
                    serial_results[patient_id]["time"],
                    ray_results[patient_id]["time"]
                )

        except ImportError:
            # Ray not available, skip test
            pass

    def test_large_dataset_processing(self):
        """Test processing large datasets that don't fit in memory."""
        # Initialize labeler
        labeler = RiskFactorLabeler("large_dataset_risk_labeler")

        # Process large dataset
        labeler.process_large_dataset(
            events_path=str(self.dataset_path),
            output_path=str(self.result_dir),
            mode=ProcessingMode.MULTIPROCESSING,
            n_jobs=2,
            batch_size=50,
            patient_chunk_size=200,
            show_progress=False
        )

        # Check that output files were created
        combined_path = self.result_dir / "combined_results.parquet"
        self.assertTrue(combined_path.exists())

        # Load results and check structure
        results = pd.read_parquet(combined_path)
        self.assertGreater(len(results), 0)
        self.assertIn("subject_id", results.columns)
        self.assertIn("risk_factors", results.columns)

    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        # Create empty dataset
        empty_df = pd.DataFrame(columns=["subject_id", "time", "code", "time_type"])

        # Initialize labeler
        labeler = MortalityLabeler("empty_test_labeler")

        # Process empty dataset
        results = labeler.parallel_label(
            empty_df,
            mode=ProcessingMode.SERIAL,
            show_progress=False
        )

        # Should return empty dictionary
        self.assertEqual(len(results), 0)

    def test_error_handling(self):
        """Test error handling in labelers."""
        # Create invalid dataset (missing required columns)
        invalid_df = pd.DataFrame({
            "subject_id": [1, 2, 3],
            # Missing time and code columns
        })

        # Initialize labeler
        labeler = MortalityLabeler("error_test_labeler")

        # Should raise ValueError
        with self.assertRaises(ValueError):
            labeler.parallel_label(
                invalid_df,
                mode=ProcessingMode.SERIAL,
                show_progress=False
            )

    def test_batch_processing(self):
        """Test batch processing of patients."""
        # Initialize labeler
        labeler = MortalityLabeler("batch_test_labeler")

        # Process with small batch size
        results_small_batch = labeler.parallel_label(
            self.medium_dataset,
            mode=ProcessingMode.MULTIPROCESSING,
            n_jobs=2,
            batch_size=5,  # Small batches
            show_progress=False
        )

        # Process with large batch size
        results_large_batch = labeler.parallel_label(
            self.medium_dataset,
            mode=ProcessingMode.MULTIPROCESSING,
            n_jobs=2,
            batch_size=50,  # Larger batches
            show_progress=False
        )

        # Results should be identical regardless of batch size
        self.assertEqual(len(results_small_batch), len(results_large_batch))

        # Events should match
        for patient_id in results_small_batch:
            self.assertEqual(
                results_small_batch[patient_id]["event"],
                results_large_batch[patient_id]["event"]
            )
            self.assertEqual(
                results_small_batch[patient_id]["time"],
                results_large_batch[patient_id]["time"]
            )

    def test_get_time_to_event(self):
        """Test the get_time_to_event utility method."""
        # Create test patient with known events
        patient_id = 999
        base_date = datetime.datetime(2020, 1, 1)
        enrollment_date = base_date

        # Create events
        events = [
            {
                "subject_id": patient_id,
                "time": enrollment_date,
                "code": "ENROLLMENT",
                "time_type": TimeType.EVENT,
            },
            {
                "subject_id": patient_id,
                "time": enrollment_date + datetime.timedelta(days=100),
                "code": "TEST_EVENT",
                "time_type": TimeType.EVENT,
            },
            {
                "subject_id": patient_id,
                "time": enrollment_date + datetime.timedelta(days=200),
                "code": MEDS_DEATH_CODE,
                "time_type": TimeType.EVENT,
            }
        ]

        patient_df = pd.DataFrame(events)

        # Initialize labeler
        labeler = MedicalLabeler("time_to_event_test")

        # Test with event that exists
        status, time = labeler.get_time_to_event(
            patient_df,
            event_codes=[MEDS_DEATH_CODE],
            enrollment_time=enrollment_date,
            max_followup_days=1095
        )

        # Should find death at 200 days
        self.assertEqual(status, EventStatus.OCCURRED.value)
        self.assertEqual(time, 200.0)

        # Test with event that doesn't exist
        status, time = labeler.get_time_to_event(
            patient_df,
            event_codes=["NON_EXISTENT_EVENT"],
            enrollment_time=enrollment_date,
            max_followup_days=1095
        )

        # Should be censored at max follow-up
        self.assertEqual(status, EventStatus.CENSORED.value)
        self.assertEqual(time, 1095.0)

        # Test with event beyond max follow-up
        status, time = labeler.get_time_to_event(
            patient_df,
            event_codes=[MEDS_DEATH_CODE],
            enrollment_time=enrollment_date,
            max_followup_days=150
        )

        # Should be capped at max follow-up
        self.assertEqual(status, EventStatus.OCCURRED.value)
        self.assertEqual(time, 150.0)

if __name__ == "__main__":
    unittest.main()
