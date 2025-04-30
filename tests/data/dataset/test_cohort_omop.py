import pytest

from sat.data.dataset.cohort_omop import CohortOMOP
from sat.data.dataset.femr_extensions.schema import LabelType


class DummyLabeler:
    def __init__(self, name, label_type, output):
        self.name = name
        self.label_type = label_type
        self.output = output

    def apply(self, ds):
        return self.output

    def __repr__(self):
        return f"DummyLabeler({self.name})"


def make_dummy_dataset():
    class DummyDataset:
        def __init__(self):
            self.column_names = ["patient_id", "events"]
            self.data = {
                "patient_id": [1, 2],
                "events": [
                    [{"time": 1}, {"time": 5}, {"time": 9}],
                    [{"time": 2}, {"time": 4}, {"time": 10}],
                ],
            }

        def __getitem__(self, key):
            return self.data[key]

        def add_column(self, name, values):
            self.data[name] = values
            self.column_names.append(name)
            return self

        def remove_columns(self, names):
            for name in names:
                self.column_names.remove(name)
                del self.data[name]
            return self

        def __len__(self):
            return len(self.data["patient_id"])

        def select(self, indices):
            # Create a new DummyDataset with only the selected indices
            new_ds = type(self)()
            for key in self.data:
                new_ds.data[key] = [self.data[key][i] for i in indices]
            return new_ds

    return DummyDataset()


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
#         source=None,
#         labelers=[anchor_labeler, outcome_labeler, competing_labeler],
#         filters=[],
#         featurizers=[],
#         date_diff_unit='days'
#     )
#
#     ds = make_dummy_dataset()
#     labels_dict, anchor_times = cohort.apply_labelers(ds)
#     cohort.apply_competing_risk_censoring(labels_dict, anchor_times, ds)
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
    ds = make_dummy_dataset()
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
    cohort.apply_competing_risk_censoring(labels_dict, anchor_times, ds)

    assert labels_dict["outcome_labels"][0][0]["boolean_value"] is False
    assert labels_dict["outcome_labels"][0][0]["prediction_time"] == 5


def test_competing_risk_censoring_no_competing():
    ds = make_dummy_dataset()
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
    cohort.apply_competing_risk_censoring(labels_dict, anchor_times, ds)
    assert labels_dict["outcome_labels"][0][0]["boolean_value"] is True
    assert labels_dict["outcome_labels"][0][0]["prediction_time"] == 8


def test_truncate_events_at_competing():
    ds = make_dummy_dataset()
    labels_dict = {
        "competing_labels": [
            [
                {
                    "label_type": LabelType.OUTCOME,
                    "prediction_time": 5,
                    "boolean_value": True,
                    "competing_event": True,
                }
            ],
            [
                {
                    "label_type": LabelType.OUTCOME,
                    "prediction_time": 10,
                    "boolean_value": True,
                    "competing_event": True,
                }
            ],
        ]
    }
    cohort = CohortOMOP(source=None, labelers=[])
    ds = cohort.truncate_events_at_competing(labels_dict, anchor_times=[0, 0], ds=ds)
    # Patient 0: events at 1, 5, 9; cutoff is 5, so only 1, 5 remain
    assert ds["events"][0] == [{"time": 1}, {"time": 5}]
    # Patient 1: cutoff is 10, so all events remain
    assert ds["events"][1] == [{"time": 2}, {"time": 4}, {"time": 10}]


def test_truncate_events_no_competing():
    ds = make_dummy_dataset()
    labels_dict = {
        "competing_labels": [
            [
                {
                    "label_type": LabelType.OUTCOME,
                    "prediction_time": 15,
                    "boolean_value": False,
                    "competing_event": False,
                }
            ],
            [
                {
                    "label_type": LabelType.OUTCOME,
                    "prediction_time": 20,
                    "boolean_value": False,
                    "competing_event": False,
                }
            ],
        ]
    }
    cohort = CohortOMOP(source=None, labelers=[])
    ds = cohort.truncate_events_at_competing(labels_dict, anchor_times=[0, 0], ds=ds)
    # No truncation should occur
    assert ds["events"][0] == [{"time": 1}, {"time": 5}, {"time": 9}]
    assert ds["events"][1] == [{"time": 2}, {"time": 4}, {"time": 10}]


def test_truncate_events_no_events_column():
    ds = make_dummy_dataset()
    ds.remove_columns(["events"])
    labels_dict = {
        "competing_labels": [
            [
                {
                    "label_type": LabelType.OUTCOME,
                    "prediction_time": 5,
                    "boolean_value": True,
                    "competing_event": True,
                }
            ]
        ]
    }
    cohort = CohortOMOP(source=None, labelers=[])
    ds2 = cohort.truncate_events_at_competing(labels_dict, anchor_times=[0], ds=ds)
    assert "events" not in ds2.column_names

def test_filter_patients_without_anchor_all_true():
    ds = make_dummy_dataset()
    # Both patients have anchor label True
    anchor_labels = [ [ {"boolean_value": True, "label_type": LabelType.ANCHOR} ],
                      [ {"boolean_value": True, "label_type": LabelType.ANCHOR} ] ]
    dummy_labels_dict = {"anchor": anchor_labels, "other": [[1],[2]]}
    anchor_times = [0, 1]
    cohort = CohortOMOP(source=None, labelers=[])
    ds2, labels_dict2, anchor_times2 = cohort.filter_patients_without_anchor(dummy_labels_dict, anchor_times, ds)
    assert len(ds2) == 2
    assert labels_dict2["anchor"] == anchor_labels
    assert anchor_times2 == [0, 1]


def test_filter_patients_without_anchor_some_false():
    ds = make_dummy_dataset()
    # First patient True, second False
    anchor_labels = [ [ {"boolean_value": True, "label_type": LabelType.ANCHOR} ],
                      [ {"boolean_value": False, "label_type": LabelType.ANCHOR} ] ]
    dummy_labels_dict = {"anchor": anchor_labels, "other": [[1],[2]]}
    anchor_times = [0, 1]
    cohort = CohortOMOP(source=None, labelers=[])
    ds2, labels_dict2, anchor_times2 = cohort.filter_patients_without_anchor(dummy_labels_dict, anchor_times, ds)
    assert len(ds2) == 1
    assert labels_dict2["anchor"] == [anchor_labels[0]]
    assert labels_dict2["other"] == [[1]]
    assert anchor_times2 == [0]


def test_filter_patients_without_anchor_all_false():
    ds = make_dummy_dataset()
    # Both patients False
    anchor_labels = [ [ {"boolean_value": False, "label_type": LabelType.ANCHOR} ],
                      [ {"boolean_value": False, "label_type": LabelType.ANCHOR} ] ]
    dummy_labels_dict = {"anchor": anchor_labels, "other": [[1],[2]]}
    anchor_times = [0, 1]
    cohort = CohortOMOP(source=None, labelers=[])
    ds2, labels_dict2, anchor_times2 = cohort.filter_patients_without_anchor(dummy_labels_dict, anchor_times, ds)
    assert len(ds2) == 0
    assert labels_dict2["anchor"] == []
    assert labels_dict2["other"] == []
    assert anchor_times2 == []


def test_filter_patients_without_anchor_no_anchor_labeler():
    ds = make_dummy_dataset()
    # No anchor labeler in labels_dict
    dummy_labels_dict = {"other": [[1],[2]]}
    anchor_times = [0, 1]
    cohort = CohortOMOP(source=None, labelers=[])
    ds2, labels_dict2, anchor_times2 = cohort.filter_patients_without_anchor(dummy_labels_dict, anchor_times, ds)
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


if __name__ == "__main__":
    pytest.main([__file__])
