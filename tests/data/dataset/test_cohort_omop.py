import logging
from datetime import datetime, timedelta

import pandas as pd
import pytest
from datasets import Dataset

from sat.data.dataset.cohort_omop import CohortOMOP
from sat.data.dataset.femr_extensions.schema import LabelType

logger = logging.getLogger(__name__)


class DummyLabeler:
    def __init__(self, name, label_type, output_labels_for_all_patients):
        self.name = name
        self.label_type = label_type
        # Process output_labels_for_all_patients to include label_type in each dict
        self.processed_labels_for_all_patients = []
        if output_labels_for_all_patients:
            for patient_labels_list in output_labels_for_all_patients:
                new_patient_labels_list = []
                if patient_labels_list:
                    for label_dict in patient_labels_list:
                        if isinstance(label_dict, dict):
                            new_label_dict = label_dict.copy()
                            # Ensure label_type is correctly set from the instance's label_type
                            new_label_dict["label_type"] = self.label_type
                            new_patient_labels_list.append(new_label_dict)
                        else:
                            # This case should ideally not happen if inputs are structured ExtendedLabel dicts
                            new_patient_labels_list.append(label_dict)
                self.processed_labels_for_all_patients.append(new_patient_labels_list)

    def label(self, patient_group_df):
        # Get the patient ID from the dataframe to return the correct patient's labels
        if isinstance(patient_group_df, pd.DataFrame) and not patient_group_df.empty:
            # Look for primary_key or patient_id column
            id_column = next(
                (
                    col
                    for col in ["primary_key", "patient_id"]
                    if col in patient_group_df.columns
                ),
                None,
            )

            if id_column:
                patient_id = patient_group_df[id_column].iloc[0]
                # Convert to 0-based index (assuming patient IDs start from 1)
                patient_idx = int(patient_id) - 1

                if patient_idx < len(self.processed_labels_for_all_patients):
                    return self.processed_labels_for_all_patients[patient_idx]

        # Default: return empty list if no labels found
        return []

    def __repr__(self):
        return f"DummyLabeler({self.name}, type={self.label_type})"


def make_dummy_dataset(patient_event_data=None, force_integer_times=False):
    """Creates a HuggingFace Dataset for testing with the nested OMOP patient-events structure."""

    def default_events():
        if force_integer_times:
            # For legacy tests that expect integer times
            return [
                [
                    {
                        "time": 1,
                        "code": "OMOP_ENROLLMENT",
                        "numeric_value": None,
                        "string_value": None,
                    },
                    {
                        "time": 5,
                        "code": "OMOP_ENROLLMENT",
                        "numeric_value": None,
                        "string_value": None,
                    },
                    {
                        "time": 9,
                        "code": "OMOP_DEATH",
                        "numeric_value": None,
                        "string_value": None,
                    },
                ],
                [
                    {
                        "time": 2,
                        "code": "OMOP_ENROLLMENT",
                        "numeric_value": None,
                        "string_value": None,
                    },
                    {
                        "time": 4,
                        "code": "OMOP_ENROLLMENT",
                        "numeric_value": None,
                        "string_value": None,
                    },
                    {
                        "time": 10,
                        "code": "OMOP_DEATH",
                        "numeric_value": None,
                        "string_value": None,
                    },
                ],
            ]
        else:
            return [
                [
                    {
                        "time": "2020-01-01T00:00:00",
                        "code": "OMOP_ENROLLMENT",
                        "numeric_value": None,
                        "string_value": None,
                    },
                    {
                        "time": "2020-06-01T00:00:00",
                        "code": "OMOP_DEATH",
                        "numeric_value": None,
                        "string_value": None,
                    },
                ],
                [
                    {
                        "time": "2021-01-01T00:00:00",
                        "code": "OMOP_ENROLLMENT",
                        "numeric_value": None,
                        "string_value": None,
                    },
                ],
            ]

    if patient_event_data is None:
        data_dict = {
            "patient_id": [1, 2],
            "events": default_events(),
        }
    else:
        # Ensure all events have the required keys and correct types
        for patient_events in patient_event_data:
            for event in patient_events:
                if "time" not in event:
                    event["time"] = "1970-01-01T00:00:00"
                # Only convert to string if not forcing integer times
                if not force_integer_times and not isinstance(event["time"], str):
                    event["time"] = str(event["time"])
                if "code" not in event:
                    event["code"] = "OMOP_UNKNOWN"
                if "numeric_value" not in event:
                    event["numeric_value"] = None
                if "string_value" not in event:
                    event["string_value"] = None
        data_dict = {
            "patient_id": list(range(1, len(patient_event_data) + 1)),
            "events": patient_event_data,
        }
    return Dataset.from_dict(data_dict)


# def test_cohort_omop_end_to_end():
#     # Setup dummy labelers
#     anchor_labels = [ [ {'prediction_time': 0, 'competing_event': False} ], [ {'prediction_time': 0, 'competing_event': False} ] ]
#     outcome_labels = [ [ {'prediction_time': 8, 'boolean_value': True, 'competing_event': False} ], [ {'prediction_time': 9, 'boolean_value': True, 'competing_event': False} ] ]
#     competing_labels = [ [ {'prediction_time': 5, 'boolean_value': True, 'competing_event': True} ], [ {'prediction_time': 10, 'boolean_value': True, 'competing_event': True} ] ]
#
#     anchor_labeler = DummyLabeler('anchor', LabelType.ANCHOR, anchor_labels)
#     outcome_labeler = DummyLabeler('outcome_labels', LabelType.OUTCOME, outcome_labels)
#     competing_labeler = DummyLabeler('competing_labels', LabelType.OUTCOME, competing_labels)
#
#     cohort = CohortOMOP(
#         source=str(parquet_path), # Source is now the path to the parquet file
#         labelers=[anchor_labeler, outcome_labeler, competing_labeler],
#         filters=[],
#         featurizers=[],
#         date_diff_unit='days',
#         limit_num_patients=1, # Focus on the single patient in the dummy dataset
#         save_cohort_path=None, # Don't save during unit test
#     )
#
#     ds = make_dummy_dataset()
#     labels_dict, anchor_times = cohort.apply_labelers(ds)
#     cohort.apply_competing_risk_censoring(labels_dict, anchor_times)
#     # Check that outcome label for patient 0 is censored at time 5
#     print(labels_dict)
#     assert labels_dict['outcome_labels'][0][0]['boolean_value'] is False
#     assert labels_dict['outcome_labels'][0][0]['prediction_time'] == 5
#     # Check that outcome label for patient 1 is not censored (competing event at 10, outcome at 9)
#     assert labels_dict['outcome_labels'][1][0]['boolean_value'] is True
#     assert labels_dict['outcome_labels'][1][0]['prediction_time'] == 9
#     # Check anchor_time column is present
#     assert 'anchor_time' in ds.column_names
#     assert ds['anchor_time'] == [0, 0]

# Unit tests for individual methods and edge cases


def test_apply_labelers_anchor_only():
    ds = make_dummy_dataset()
    anchor_labels = [[{"prediction_time": 0}], [{"prediction_time": 1}]]
    anchor_labeler = DummyLabeler("anchor", LabelType.ANCHOR, anchor_labels)
    cohort = CohortOMOP(source=None, labelers=[anchor_labeler])
    labels_dict, anchor_times = cohort.apply_labelers(ds)
    assert "anchor" in labels_dict
    assert anchor_times == [0, 1]


def test_apply_labelers_no_anchor_raises():
    ds = make_dummy_dataset()
    outcome_labeler = DummyLabeler(
        "outcome_labels", LabelType.OUTCOME, [[{"prediction_time": 8}]]
    )
    cohort = CohortOMOP(source=None, labelers=[outcome_labeler])
    with pytest.raises(ValueError):
        cohort.apply_labelers(ds)


def test_competing_risk_censoring_censors():
    make_dummy_dataset()
    labels_dict = {
        "outcome_labels": [
            [
                {
                    "label_type": LabelType.OUTCOME,
                    "prediction_time": 8,
                    "boolean_value": True,
                    "competing_event": False,
                }
            ]
        ],
        "competing_labels": [
            [
                {
                    "label_type": LabelType.OUTCOME,
                    "prediction_time": 5,
                    "boolean_value": True,
                    "competing_event": True,
                }
            ]
        ],
    }
    anchor_times = [0]
    cohort = CohortOMOP(source=None, labelers=[])
    cohort.apply_competing_risk_censoring(labels_dict, anchor_times)

    assert labels_dict["outcome_labels"][0][0]["boolean_value"] is False
    assert labels_dict["outcome_labels"][0][0]["competing_event"] is False
    assert labels_dict["outcome_labels"][0][0]["prediction_time"] == 5


def test_competing_risk_censoring_no_competing():
    make_dummy_dataset()
    labels_dict = {
        "outcome_labels": [
            [
                {
                    "label_type": LabelType.OUTCOME,
                    "prediction_time": 8,
                    "boolean_value": True,
                    "competing_event": False,
                }
            ]
        ],
        "competing_labels": [
            [
                {
                    "label_type": LabelType.OUTCOME,
                    "prediction_time": 10,
                    "boolean_value": False,
                    "competing_event": True,
                }
            ]
        ],
    }
    anchor_times = [0]
    cohort = CohortOMOP(source=None, labelers=[])
    cohort.apply_competing_risk_censoring(labels_dict, anchor_times)
    assert labels_dict["outcome_labels"][0][0]["boolean_value"] is True
    assert labels_dict["outcome_labels"][0][0]["prediction_time"] == 8


def test_truncate_events_at_anchor_with_integer_times():
    # Using make_dummy_dataset with integer times and all required fields
    ds = make_dummy_dataset(force_integer_times=True)
    # Anchor times are 0 for both patients. truncate_events_at_anchor keeps events <= anchor_time.
    # Since event times are positive integers, no events should remain.
    anchor_times = [0, 0]
    cohort = CohortOMOP(source=None, labelers=[])
    ds_truncated = cohort.truncate_events_at_anchor(anchor_times, ds)
    assert (
        ds_truncated["events"][0] == []
    ), "Patient 0 events should be empty after truncating at anchor_time 0"
    assert (
        ds_truncated["events"][1] == []
    ), "Patient 1 events should be empty after truncating at anchor_time 0"

    # Test with anchor times that should keep some events
    ds = make_dummy_dataset(force_integer_times=True)
    anchor_times_p0_at_5 = [5, 10]  # P0 anchor at 5, P1 anchor at 10
    cohort = CohortOMOP(source=None, labelers=[])
    ds_truncated_some = cohort.truncate_events_at_anchor(anchor_times_p0_at_5, ds)
    # Patient 0: events at 1, 5, 9; anchor is 5, so 1, 5 remain (assuming inclusive)
    expected_events_p0 = [
        {
            "time": 1,
            "code": "OMOP_ENROLLMENT",
            "numeric_value": None,
            "string_value": None,
        },
        {
            "time": 5,
            "code": "OMOP_ENROLLMENT",
            "numeric_value": None,
            "string_value": None,
        },
    ]
    assert ds_truncated_some["events"][0] == expected_events_p0
    # Patient 1: events at 2, 4, 10; anchor is 10, so 2, 4, 10 remain
    expected_events_p1 = [
        {
            "time": 2,
            "code": "OMOP_ENROLLMENT",
            "numeric_value": None,
            "string_value": None,
        },
        {
            "time": 4,
            "code": "OMOP_ENROLLMENT",
            "numeric_value": None,
            "string_value": None,
        },
        {"time": 10, "code": "OMOP_DEATH", "numeric_value": None, "string_value": None},
    ]
    assert ds_truncated_some["events"][1] == expected_events_p1


def test_truncate_events_at_anchor_no_effect_if_anchor_late():
    ds = make_dummy_dataset(force_integer_times=True)
    # Anchor times are very late, so all events should remain.
    anchor_times = [100, 100]
    cohort = CohortOMOP(source=None, labelers=[])
    ds_truncated = cohort.truncate_events_at_anchor(anchor_times, ds)
    expected_events_p0 = [
        {
            "time": 1,
            "code": "OMOP_ENROLLMENT",
            "numeric_value": None,
            "string_value": None,
        },
        {
            "time": 5,
            "code": "OMOP_ENROLLMENT",
            "numeric_value": None,
            "string_value": None,
        },
        {"time": 9, "code": "OMOP_DEATH", "numeric_value": None, "string_value": None},
    ]
    expected_events_p1 = [
        {
            "time": 2,
            "code": "OMOP_ENROLLMENT",
            "numeric_value": None,
            "string_value": None,
        },
        {
            "time": 4,
            "code": "OMOP_ENROLLMENT",
            "numeric_value": None,
            "string_value": None,
        },
        {"time": 10, "code": "OMOP_DEATH", "numeric_value": None, "string_value": None},
    ]
    assert ds_truncated["events"][0] == expected_events_p0
    assert ds_truncated["events"][1] == expected_events_p1


def test_truncate_events_no_events_column():
    ds = make_dummy_dataset()
    # HuggingFace's remove_columns returns a new dataset, not modifies in place
    ds = ds.remove_columns(["events"])
    cohort = CohortOMOP(source=None, labelers=[])
    ds2 = cohort.truncate_events_at_anchor(anchor_times=[0], ds=ds)
    assert "events" not in ds2.column_names


def test_filter_patients_without_anchor_all_true():
    ds = make_dummy_dataset()
    # Both patients have anchor label True
    anchor_labels = [
        [{"boolean_value": True, "label_type": LabelType.ANCHOR}],
        [{"boolean_value": True, "label_type": LabelType.ANCHOR}],
    ]
    dummy_labels_dict = {"anchor": anchor_labels, "other": [[1], [2]]}
    anchor_times = [0, 1]
    cohort = CohortOMOP(source=None, labelers=[])
    ds2, labels_dict2, anchor_times2 = cohort.filter_patients_without_anchor(
        dummy_labels_dict, anchor_times, ds
    )
    assert len(ds2) == 2
    assert labels_dict2["anchor"] == anchor_labels
    assert anchor_times2 == [0, 1]


def test_filter_patients_without_anchor_some_false():
    ds = make_dummy_dataset()
    # First patient True, second False
    anchor_labels = [
        [{"boolean_value": True, "label_type": LabelType.ANCHOR}],
        [{"boolean_value": False, "label_type": LabelType.ANCHOR}],
    ]
    dummy_labels_dict = {"anchor": anchor_labels, "other": [[1], [2]]}
    anchor_times = [0, 1]
    cohort = CohortOMOP(source=None, labelers=[])
    ds2, labels_dict2, anchor_times2 = cohort.filter_patients_without_anchor(
        dummy_labels_dict, anchor_times, ds
    )
    assert len(ds2) == 1
    assert labels_dict2["anchor"] == [anchor_labels[0]]
    assert labels_dict2["other"] == [[1]]
    assert anchor_times2 == [0]


def test_filter_patients_without_anchor_all_false():
    ds = make_dummy_dataset()
    # Both patients False
    anchor_labels = [
        [{"boolean_value": False, "label_type": LabelType.ANCHOR}],
        [{"boolean_value": False, "label_type": LabelType.ANCHOR}],
    ]
    dummy_labels_dict = {"anchor": anchor_labels, "other": [[1], [2]]}
    anchor_times = [0, 1]
    cohort = CohortOMOP(source=None, labelers=[])
    ds2, labels_dict2, anchor_times2 = cohort.filter_patients_without_anchor(
        dummy_labels_dict, anchor_times, ds
    )
    assert len(ds2) == 0
    assert labels_dict2["anchor"] == []
    assert labels_dict2["other"] == []
    assert anchor_times2 == []


def test_filter_patients_without_anchor_no_anchor_labeler():
    ds = make_dummy_dataset()
    # No anchor labeler in labels_dict
    dummy_labels_dict = {"other": [[1], [2]]}
    anchor_times = [0, 1]
    cohort = CohortOMOP(source=None, labelers=[])
    ds2, labels_dict2, anchor_times2 = cohort.filter_patients_without_anchor(
        dummy_labels_dict, anchor_times, ds
    )
    assert len(ds2) == 2
    assert labels_dict2 == dummy_labels_dict
    assert anchor_times2 == anchor_times


def test_apply_labelers_empty_patients():
    class EmptyDataset:
        column_names = ["patient_id", "events"]

        def __getitem__(self, key):
            return []

        def add_column(self, name, values):
            return self

        def __len__(self):
            return 0

    ds = EmptyDataset()
    anchor_labeler = DummyLabeler("anchor", LabelType.ANCHOR, [])
    cohort = CohortOMOP(source=None, labelers=[anchor_labeler])
    labels_dict, anchor_times = cohort.apply_labelers(ds)
    assert labels_dict["anchor"] == []
    assert anchor_times == []


@pytest.mark.parametrize(
    (
        "scenario_name",
        "patient_events_timedeltas",
        "anchor_delta_days",
        "o1_delta_days",
        "o2_delta_days",
        "expected_o2_value",
        "expected_o2_time_delta_days",
        "expected_o2_competing_event_flag",
        "expected_event_count_at_anchor",
    ),
    [
        # Scenario 1: O2 after O1, no competing risk before O2
        (
            "O2 after O1, no competing",
            [timedelta(days=-10), timedelta(days=-5)],  # patient events
            0,  # anchor_delta_days from t_anchor_base
            5,  # o1_delta_days from anchor_time
            10,  # o2_delta_days from anchor_time
            False,  # expected_o2_value - censored by O1 at day 5
            5,  # expected_o2_time_delta_days - censoring time = competing event time
            False,  # expected_o2_competing_event_flag - should remain false (nature of event)
            2,  # expected_event_count_at_anchor (events at -10, -5, anchor at 0)
        ),
        # Scenario 2: O1 (competing) before O2, O2 should be censored
        (
            "O1 competing before O2",
            [timedelta(days=-10), timedelta(days=-5)],
            0,
            2,  # O1 (competing) at anchor + 2 days
            10,  # O2 at anchor + 10 days
            False,  # O2 is censored
            2,  # Censoring time is O1's time
            False,  # competing_event flag should remain False (nature of event)
            2,
        ),
        # Scenario 3: O2 before O1 (competing), O2 should occur
        (
            "O2 before O1 competing",
            [timedelta(days=-10), timedelta(days=-5)],
            0,
            10,  # O1 (competing) at anchor + 10 days
            2,  # O2 at anchor + 2 days
            True,  # O2 occurs
            2,  # O2 time
            False,  # competing_event flag should be False for O2
            2,
        ),
        # Scenario 4: O1 (competing) and O2 at the same time, O2 should be censored (competing wins ties)
        (
            "O1 competing same time as O2",
            [timedelta(days=-10), timedelta(days=-5)],
            0,
            5,  # O1 (competing) at anchor + 5 days
            5,  # O2 at anchor + 5 days
            False,  # O2 is censored
            5,  # Censoring time
            False,  # competing_event flag should remain False (nature of event)
            2,
        ),
        # Scenario 5: No outcome O1 (competing), O2 should occur
        (
            "No O1 competing, O2 occurs",
            [timedelta(days=-10), timedelta(days=-5)],
            0,
            None,  # No O1
            5,  # O2 at anchor + 5 days
            True,  # O2 occurs
            5,  # O2 time
            False,  # competing_event flag should be False for O2
            2,
        ),
        # Scenario 6: Anchor event is the only event
        (
            "Anchor event is only event",
            [],  # No prior events
            0,  # Anchor at t_anchor_base
            None,  # No O1
            5,  # O2 at anchor + 5 days
            True,  # O2 occurs
            5,  # O2 time
            False,  # competing_event flag should be False for O2
            0,  # No events at or before anchor if anchor is the first conceptual event
        ),
        # Scenario 7: All events after anchor, O2 should occur
        # (truncate_events_at_anchor should handle this by keeping no events if they are all after anchor)
        (
            "All events after anchor, O2 occurs",
            [timedelta(days=1), timedelta(days=2)],  # Events after t_anchor_base
            0,  # Anchor at t_anchor_base
            None,  # No O1
            5,  # O2 at anchor + 5 days
            True,  # O2 occurs
            5,  # O2 time
            False,  # competing_event flag should be False for O2
            0,  # No events at or before anchor
        ),
        # Scenario 8: O1 (competing) occurs, O2 is None (no O2 event defined)
        (
            "O1 competing, O2 is None",
            [timedelta(days=-10), timedelta(days=-5)],
            0,
            2,  # O1 (competing) at anchor + 2 days
            None,  # No O2 defined
            None,  # Expected O2 value (will check if label is missing or appropriately handled)
            None,  # Expected O2 time
            None,  # Expected O2 competing flag
            2,
        ),
        # Scenario 9: Anchor is far after events, O2 relative to new anchor
        (
            "Anchor after events, O2 relative to new anchor",
            [timedelta(days=-100), timedelta(days=-90)],
            0,  # Anchor at t_anchor_base
            5,  # O1 (competing) at t_anchor_base + 5 days
            10,  # O2 at t_anchor_base + 10 days
            False,  # O2 is censored by O1 (competing) which occurs at day 5
            5,  # O2 censoring time should match the competing event time (day 5)
            False,  # competing_event flag should remain False (nature of event)
            2,
        ),
        # Scenario 10: Events include exact anchor time
        (
            "Events include exact anchor time",
            [timedelta(days=-10), timedelta(days=0)],  # Event at t_anchor_base
            0,  # Anchor at t_anchor_base
            None,  # No O1
            5,  # O2 at anchor + 5 days
            True,  # O2 occurs
            5,  # O2 time
            False,  # competing_event flag should be False for O2
            2,  # Events at -10 and 0 (anchor)
        ),
    ],
)
def test_cohort_omop_datetime_processing_scenarios(
    tmp_path,  # Added tmp_path fixture
    scenario_name,
    patient_events_timedeltas,
    anchor_delta_days,
    o1_delta_days,
    o2_delta_days,
    expected_o2_value,
    expected_o2_time_delta_days,
    expected_o2_competing_event_flag,
    expected_event_count_at_anchor,
):
    t_anchor_base = datetime(2023, 1, 1, 12, 0, 0)

    # Create a dummy dataset for a single patient
    # Events are datetime objects
    patient_events = [
        {"time": t_anchor_base + delta} for delta in patient_events_timedeltas
    ]
    raw_ds = make_dummy_dataset(patient_event_data=[patient_events])

    # Save to temporary parquet file
    parquet_path = tmp_path / f"{scenario_name.replace(' ', '_')}_data.parquet"
    raw_ds.to_parquet(parquet_path)

    # Define anchor time based on t_anchor_base and anchor_delta_days
    anchor_time = t_anchor_base + timedelta(days=anchor_delta_days)

    # Define labelers
    labelers = []
    anchor_labeler = DummyLabeler(
        "anchor",
        LabelType.ANCHOR,
        [[{"prediction_time": anchor_time, "boolean_value": True}]],
    )
    labelers.append(anchor_labeler)

    outcome2_labeler_name = "outcome_target"
    if o2_delta_days is not None:
        outcome2_time = t_anchor_base + timedelta(days=o2_delta_days)
        outcome2_labeler = DummyLabeler(
            outcome2_labeler_name,
            LabelType.OUTCOME,
            [
                [
                    {
                        "prediction_time": outcome2_time,
                        "boolean_value": True,
                        "competing_event": False,
                    }
                ]
            ],
        )
        labelers.append(outcome2_labeler)
    elif (
        outcome2_labeler_name
    ):  # Ensure O2 labeler is added if name exists, even if no event time
        # This case might be for testing if a label type is present even if no events qualify
        # For now, we'll assume if o2_delta_days is None, the labeler isn't added unless specifically needed
        # for presence checks. The current parametrization expects None for values if o2_delta_days is None.
        pass

    if o1_delta_days is not None:
        outcome1_time = t_anchor_base + timedelta(days=o1_delta_days)
        outcome1_labeler = DummyLabeler(
            "outcome_competing",
            LabelType.OUTCOME,
            [
                [
                    {
                        "prediction_time": outcome1_time,
                        "boolean_value": True,
                        "competing_event": True,
                    }
                ]
            ],
        )
        labelers.append(outcome1_labeler)

    cohort = CohortOMOP(
        source=str(parquet_path),
        labelers=labelers,
        date_diff_unit="days",
        primary_key="patient_id",
    )

    # Run the full cohort generation process
    ds = cohort()

    # Assertions on the returned dataset
    assert isinstance(ds, Dataset), "CohortOMOP should return a Dataset"
    assert len(ds) == 1, "Expected one patient in the output"

    # If we expect a target outcome (O2), verify it's present
    if o2_delta_days is not None:
        # Verify the patient has the correct outcome label column
        assert (
            "labels_outcome_target" in ds.column_names
        ), f"Dataset must have 'labels_outcome_target' column, got: {ds.column_names}"

        # Get labels for the first (only) patient
        target_label_list = ds["labels_outcome_target"][0]
        assert (
            isinstance(target_label_list, list) and len(target_label_list) > 0
        ), "Target label should be a non-empty list"
    else:
        # If no target outcome is expected, verify the competing risk is present instead
        competing_column = "labels_outcome_competing"
        assert (
            competing_column in ds.column_names
        ), f"Dataset must have '{competing_column}' column when no target outcome, got: {ds.column_names}"

    # For scenarios where O2 is None, we don't expect to find the outcome2_labeler_name in the results
    # Skip subsequent checks for these scenarios
    if o2_delta_days is None:
        # Just verify we don't have unexpected O2 values when none should be present
        label_column_name = f"labels_{outcome2_labeler_name}"
        assert (
            label_column_name not in ds.column_names
        ), f"Target outcome '{outcome2_labeler_name}' found in dataset columns when it should be None"
        return

    # Check the target outcome label (O2) for all other scenarios
    outcome2_column = f"labels_{outcome2_labeler_name}"
    assert (
        outcome2_column in ds.column_names
    ), f"Dataset missing expected outcome column '{outcome2_column}'"

    # Get the outcome labels for the first patient
    target_label_list = ds[outcome2_column][0]
    assert isinstance(target_label_list, list), "Target label data should be a list"
    assert len(target_label_list) == 1, "Expected one instance for the target label"
    target_label = target_label_list[0]

    # Verify the label contents match expectations
    assert (
        target_label["boolean_value"] == expected_o2_value
    ), f"Scenario: {scenario_name} - O2 boolean_value failed. Got {target_label['boolean_value']}, expected {expected_o2_value}"
    assert (
        target_label["prediction_time"] == expected_o2_time_delta_days
    ), f"Scenario: {scenario_name} - O2 prediction_time (relative days) failed. Got {target_label['prediction_time']}, expected {expected_o2_time_delta_days}"
    assert (
        target_label["competing_event"] == expected_o2_competing_event_flag
    ), f"Scenario: {scenario_name} - O2 competing_event flag failed. Got {target_label['competing_event']}, expected {expected_o2_competing_event_flag}"


if __name__ == "__main__":
    pytest.main([__file__])
