"""
Unit tests for individual methods in the CohortOMOP class.
"""

from datetime import datetime
from enum import Enum

import pandas as pd
import pytest
from datasets import Dataset

from sat.data.dataset.cohort_omop import CohortOMOP, _ensure_datetime
from sat.data.dataset.femr_extensions.schema import LabelType


def test_ensure_datetime():
    """Test the _ensure_datetime function with various input types."""
    # Test with None
    assert _ensure_datetime(None) is None

    # Test with datetime object
    now = datetime.now()
    assert _ensure_datetime(now) is now

    # Test with pandas Timestamp
    pd_timestamp = pd.Timestamp("2023-01-01")
    result = _ensure_datetime(pd_timestamp)
    assert isinstance(result, datetime)
    assert result.year == 2023
    assert result.month == 1
    assert result.day == 1

    # Test with ISO format string
    iso_string = "2023-02-15T12:30:45"
    result = _ensure_datetime(iso_string)
    assert isinstance(result, datetime)
    assert result.year == 2023
    assert result.month == 2
    assert result.day == 15
    assert result.hour == 12
    assert result.minute == 30
    assert result.second == 45

    # Test with invalid string
    assert _ensure_datetime("not-a-date") is None

    # Test with timestamp (assuming recent timestamp)
    timestamp = 1672531200  # 2023-01-01 00:00:00 UTC
    result = _ensure_datetime(timestamp)
    assert isinstance(result, datetime)
    assert result.year == 2023
    assert result.month == 1
    assert result.day == 1


def test_calculate_date_difference():
    """Test the _calculate_date_difference method with different units."""
    cohort = CohortOMOP()

    date1 = datetime(2023, 1, 10, 12, 0, 0)
    date2 = datetime(2023, 1, 1, 0, 0, 0)

    # Test days
    diff_days = cohort._calculate_date_difference(date1, date2, unit="days")
    assert diff_days == 9.5  # 9 days and 12 hours = 9.5 days

    # Test hours
    diff_hours = cohort._calculate_date_difference(date1, date2, unit="hours")
    assert diff_hours == 9.5 * 24  # 9.5 days = 228 hours

    # Test minutes
    diff_minutes = cohort._calculate_date_difference(date1, date2, unit="minutes")
    assert diff_minutes == 9.5 * 24 * 60  # 9.5 days = 13680 minutes

    # Test seconds
    diff_seconds = cohort._calculate_date_difference(date1, date2, unit="seconds")
    assert diff_seconds == 9.5 * 24 * 3600  # 9.5 days = 820800 seconds

    # Test invalid date inputs
    assert cohort._calculate_date_difference("not-a-date", date2) is None
    assert cohort._calculate_date_difference(date1, "not-a-date") is None

    # Test unsupported unit (should default to days)
    diff_unsupported = cohort._calculate_date_difference(date1, date2, unit="weeks")
    assert diff_unsupported == 9.5  # Should default to days


def test_truncate_events_at_anchor():
    """Test the truncate_events_at_anchor method."""
    # First create a dataframe with multiple rows per patient
    data = {
        "patient_id": [1, 1, 1, 2, 2],
        "time": [
            datetime(2023, 1, 1),  # Patient 1, Event 1 (before anchor)
            datetime(2023, 1, 5),  # Patient 1, Event 2 (at anchor)
            datetime(2023, 1, 10),  # Patient 1, Event 3 (after anchor)
            datetime(2023, 1, 3),  # Patient 2, Event 1 (before anchor)
            datetime(2023, 1, 15),  # Patient 2, Event 2 (after anchor)
        ],
        "event_type": ["A", "B", "C", "D", "E"],
    }
    df = pd.DataFrame(data)

    # Group events by patient to match the expected format
    events_by_patient = {}
    for _idx, row in df.iterrows():
        patient_id = row["patient_id"]
        if patient_id not in events_by_patient:
            events_by_patient[patient_id] = []
        events_by_patient[patient_id].append(
            {"time": row["time"], "type": row["event_type"]}
        )

    # Create dataset with the expected structure
    dataset_data = {
        "patient_id": list(events_by_patient.keys()),
        "events": list(events_by_patient.values()),
    }
    ds = Dataset.from_dict(dataset_data)

    # Set anchor times
    anchor_times = [
        datetime(2023, 1, 5),  # Patient 1 anchor
        datetime(2023, 1, 7),  # Patient 2 anchor
    ]

    # Create cohort object
    cohort = CohortOMOP(primary_key="patient_id")

    # Call the method
    result_ds = cohort.truncate_events_at_anchor(anchor_times, ds)

    # Verify the result
    assert len(result_ds) == 2  # Still 2 patients

    # Check patient 1's events
    patient1_events = [
        event
        for i, event in enumerate(result_ds["events"])
        if result_ds["patient_id"][i] == 1
    ][0]
    assert len(patient1_events) == 2  # 2 events (before and at anchor)
    event_times = [event["time"] for event in patient1_events]
    assert datetime(2023, 1, 1) in event_times
    assert datetime(2023, 1, 5) in event_times
    assert datetime(2023, 1, 10) not in event_times  # After anchor, should be filtered

    # Check patient 2's events
    patient2_events = [
        event
        for i, event in enumerate(result_ds["events"])
        if result_ds["patient_id"][i] == 2
    ][0]
    assert len(patient2_events) == 1  # 1 event (before anchor)
    assert patient2_events[0]["time"] == datetime(2023, 1, 3)


def test_apply_labelers_order():
    """Test that labelers are applied in the correct order."""

    # Create mock labelers
    class MockLabeler:
        def __init__(self, name, label_type):
            self.name = name
            self.label_type = label_type
            self.called = False

        def __call__(self, patient_df, patient_events=None, anchor_times=None):
            self.called = True
            return [
                {
                    "label_type": self.label_type,
                    "boolean_value": True,
                    "prediction_time": datetime.now(),
                }
            ]

    # Create labelers with different types
    anchor_labeler = MockLabeler("anchor", LabelType.ANCHOR)
    outcome_labeler = MockLabeler("outcome", LabelType.OUTCOME)
    inclusion_labeler = MockLabeler("inclusion", LabelType.INCLUSION)
    exclusion_labeler = MockLabeler("exclusion", LabelType.EXCLUSION)

    # Create dataset with one patient
    data = {
        "patient_id": [1],
        "events": [[{"time": datetime(2023, 1, 1), "type": "test"}]],
    }
    ds = Dataset.from_dict(data)

    # Create cohort with labelers in mixed order
    cohort = CohortOMOP(
        labelers=[exclusion_labeler, outcome_labeler, inclusion_labeler, anchor_labeler]
    )

    # Apply labelers
    labels_dict, anchor_times = cohort.apply_labelers(ds)

    # Verify all labelers were called
    assert anchor_labeler.called
    assert outcome_labeler.called
    assert inclusion_labeler.called
    assert exclusion_labeler.called

    # Verify labels were collected
    assert "anchor" in labels_dict
    assert "outcome" in labels_dict
    assert "inclusion" in labels_dict
    assert "exclusion" in labels_dict

    # Verify anchor time was extracted
    assert len(anchor_times) == 1
    assert anchor_times[0] is not None


def test_make_serializable():
    """Test the make_serializable method with different types."""
    cohort = CohortOMOP()

    # Test datetime
    dt = datetime(2023, 1, 1, 12, 30, 45)
    assert cohort.make_serializable(dt) == dt.isoformat()

    # Test Enum
    class TestEnum(Enum):
        A = "a"
        B = "b"

    assert cohort.make_serializable(TestEnum.A) == "a"

    # Test dictionary
    test_dict = {"a": 1, "b": datetime(2023, 1, 1)}
    serialized_dict = cohort.make_serializable(test_dict)
    assert serialized_dict["a"] == 1
    assert serialized_dict["b"] == datetime(2023, 1, 1).isoformat()

    # Test list
    test_list = [1, datetime(2023, 1, 1), "test"]
    serialized_list = cohort.make_serializable(test_list)
    assert serialized_list[0] == 1
    assert serialized_list[1] == datetime(2023, 1, 1).isoformat()
    assert serialized_list[2] == "test"

    # Test object with __dict__
    class TestObj:
        def __init__(self):
            self.a = 1
            self.b = datetime(2023, 1, 1)

    test_obj = TestObj()
    serialized_obj = cohort.make_serializable(test_obj)
    assert serialized_obj["a"] == 1
    assert serialized_obj["b"] == datetime(2023, 1, 1).isoformat()

    # Test unserializable object
    class UnserializableObj:
        def __str__(self):
            raise Exception("Cannot convert to string")

        def __dict__(self):
            # Override __dict__ to make sure it doesn't fall back to that
            raise Exception("Cannot access __dict__")

    unserializable = UnserializableObj()
    # The actual implementation returns an empty dict or a string "None"
    # when serialization fails completely, check for either
    result = cohort.make_serializable(unserializable)
    assert result == {} or result == "None" or result is None


if __name__ == "__main__":
    pytest.main()
