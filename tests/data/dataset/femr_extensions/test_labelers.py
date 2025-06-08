import datetime

from sat.data.dataset.femr_extensions.labelers import CustomEventLabeler, LabelType


# Helper to create Event as a dict (TypedDict)
def make_event(code, time):
    return {"code": code, "time": time}


# Helper to create Patient as a dict (TypedDict)
def make_patient(patient_id, events):
    return {"patient_id": patient_id, "events": events}


def test_anchor_event():
    tmax = datetime.datetime(2025, 12, 31)
    labeler = CustomEventLabeler(
        "anchor", label_type=LabelType.ANCHOR, event_codes=["X"], max_time=tmax
    )
    t = datetime.datetime(2025, 4, 5)
    patient = make_patient("p0", [make_event("X", t)])
    labels = labeler.label(patient)
    assert (
        labels[0]["boolean_value"] is True
    ), f'boolean_value was {labels[0]["boolean_value"]}'
    assert (
        labels[0]["competing_event"] is False
    ), f'competing_event was {labels[0]["competing_event"]}'
    assert (
        labels[0]["prediction_time"] == t
    ), f'prediction_time was {labels[0]["prediction_time"]}'


def test_anchor_event_2():
    tmax = datetime.datetime(2025, 12, 31)
    labeler = CustomEventLabeler(
        "anchor", label_type=LabelType.ANCHOR, event_codes=["X"], max_time=tmax
    )
    t = datetime.datetime(2025, 4, 5)
    patient = make_patient("p0", [make_event("A", t)])
    labels = labeler.label(patient)
    assert (
        labels[0]["boolean_value"] is False
    ), f'boolean_value was {labels[0]["boolean_value"]}'
    assert (
        labels[0]["competing_event"] is False
    ), f'competing_event was {labels[0]["competing_event"]}'
    assert (
        labels[0]["prediction_time"] == t
    ), f'prediction_time was {labels[0]["prediction_time"]}'


def test_competing_risk_no_events():
    t = datetime.datetime(2025, 4, 10)
    labeler_a = CustomEventLabeler(
        "A", event_codes=["X"], competing_event=True, max_time=t
    )
    labeler_b = CustomEventLabeler(
        "B", event_codes=["Y"], competing_event=True, max_time=t
    )
    patient = make_patient("p0", [])
    a_label = labeler_a.label(patient)[0]
    b_label = labeler_b.label(patient)[0]
    assert a_label["competing_event"] is True
    assert a_label["boolean_value"] is False
    assert a_label["prediction_time"] == t
    assert b_label["competing_event"] is True
    assert b_label["boolean_value"] is False
    assert b_label["prediction_time"] == t


def test_competing_risk():
    # Competing events the labelling here does not take into account dependencies between labels
    tmax = datetime.datetime(2025, 12, 31)
    labeler_a = CustomEventLabeler(
        "A", event_codes=["X"], competing_event=True, max_time=tmax
    )
    labeler_b = CustomEventLabeler(
        "B", event_codes=["Y"], competing_event=True, max_time=tmax
    )
    t1 = datetime.datetime(2025, 4, 2)
    t2 = datetime.datetime(2025, 4, 5)
    patient = make_patient("p1", [make_event("Y", t1), make_event("X", t2)])
    a_label = labeler_a.label(patient)[0]
    b_label = labeler_b.label(patient)[0]
    assert a_label["competing_event"] is True, a_label
    assert a_label["boolean_value"] is True, a_label
    assert a_label["prediction_time"] == t2, a_label
    assert b_label["competing_event"] is True, b_label
    assert b_label["boolean_value"] is True, b_label
    assert b_label["prediction_time"] == t1, b_label


def test_competing_risk_no_matching_events():
    tmax = datetime.datetime(2025, 12, 31)
    labeler_a = CustomEventLabeler(
        "A", event_codes=["X"], competing_event=True, max_time=tmax
    )
    labeler_b = CustomEventLabeler(
        "B", event_codes=["Y"], competing_event=True, max_time=tmax
    )
    t1 = datetime.datetime(2025, 4, 3)
    t2 = datetime.datetime(2025, 4, 5)
    patient = make_patient("p2", [make_event("Z", t1), make_event("W", t2)])
    a_label = labeler_a.label(patient)[0]
    b_label = labeler_b.label(patient)[0]
    assert a_label["competing_event"] is True, a_label
    assert a_label["boolean_value"] is False, a_label
    assert a_label["prediction_time"] == t2, a_label
    assert b_label["competing_event"] is True, b_label
    assert b_label["boolean_value"] is False, b_label
    assert b_label["prediction_time"] == t2, b_label


# ---- Survival (with CustomEventLabeler) Tests ----
def test_survival_no_events():
    t = datetime.datetime(2025, 4, 10)
    labeler = CustomEventLabeler("surv", event_codes=["X"], max_time=t)
    patient = make_patient("p3", [])
    labels = labeler.label(patient)
    assert (
        labels[0]["boolean_value"] is False
    ), f'boolean_value was {labels[0]["boolean_value"]}'
    assert (
        labels[0]["competing_event"] is False
    ), f'competing_event was {labels[0]["competing_event"]}'
    assert (
        labels[0]["prediction_time"] == t
    ), f'prediction_time was {labels[0]["prediction_time"]}'


def test_survival_event_found():
    tmax = datetime.datetime(2025, 4, 10)
    t1 = datetime.datetime(2025, 4, 1)
    t2 = datetime.datetime(2025, 4, 3)
    t3 = datetime.datetime(2025, 4, 4)
    labeler = CustomEventLabeler("surv", event_codes=["X"], max_time=tmax)
    patient = make_patient(
        "p4", [make_event("A", t1), make_event("X", t2), make_event("X", t3)]
    )
    labels = labeler.label(patient)
    assert (
        labels[0]["boolean_value"] is True
    ), f'boolean_value was {labels[0]["boolean_value"]}'
    assert (
        labels[0]["competing_event"] is False
    ), f'competing_event was {labels[0]["competing_event"]}'
    assert (
        labels[0]["prediction_time"] == t2
    ), f'prediction_time was {labels[0]["prediction_time"]}'


def test_survival_event_not_found():
    tmax = datetime.datetime(2025, 4, 10)
    t1 = datetime.datetime(2025, 4, 1)
    t2 = datetime.datetime(2025, 4, 2)
    labeler = CustomEventLabeler("surv", event_codes=["X"], max_time=tmax)
    patient = make_patient("p5", [make_event("A", t1), make_event("B", t2)])
    labels = labeler.label(patient)
    assert (
        labels[0]["boolean_value"] is False
    ), f'boolean_value was {labels[0]["boolean_value"]}'
    assert (
        labels[0]["competing_event"] is False
    ), f'competing_event was {labels[0]["competing_event"]}'
    assert (
        labels[0]["prediction_time"] == t2
    ), f'prediction_time was {labels[0]["prediction_time"]}'


# ---- CustomEventLabeler Tests ----
def test_custom_event_exclusion():
    tmax = datetime.datetime(2025, 12, 31)
    labeler = CustomEventLabeler(
        "custom", label_type=LabelType.EXCLUSION, event_codes=["Y"], max_time=tmax
    )
    t1 = datetime.datetime(2025, 4, 1)
    t2 = datetime.datetime(2025, 4, 2)
    patient = make_patient("p6", [make_event("X", t1), make_event("Y", t2)])
    labels = labeler.label(patient)
    assert (
        labels[0]["boolean_value"] is True
    ), f'boolean_value was {labels[0]["boolean_value"]}'
    assert (
        labels[0]["competing_event"] is False
    ), f'competing_event was {labels[0]["competing_event"]}'
    assert (
        labels[0]["prediction_time"] == t2
    ), f'prediction_time was {labels[0]["prediction_time"]}'


def test_custom_event_primary_and_condition():
    tmax = datetime.datetime(2025, 12, 31)
    labeler = CustomEventLabeler(
        "custom", event_codes=["X"], condition_codes=["C"], max_time=tmax
    )
    t1 = datetime.datetime(2025, 4, 1)
    t2 = datetime.datetime(2025, 4, 2)
    patient = make_patient("p7", [make_event("X", t1), make_event("C", t2)])
    labels = labeler.label(patient)
    assert (
        labels[0]["boolean_value"] is True
    ), f'boolean_value was {labels[0]["boolean_value"]}'
    assert (
        labels[0]["competing_event"] is False
    ), f'competing_event was {labels[0]["competing_event"]}'
    assert (
        labels[0]["prediction_time"] == t1
    ), f'prediction_time was {labels[0]["prediction_time"]}'


def test_custom_event_primary_but_no_condition():
    labeler = CustomEventLabeler(
        "custom",
        event_codes=["X"],
        condition_codes=["C"],
        max_time=datetime.datetime(2025, 12, 31),
    )
    t1 = datetime.datetime(2025, 4, 1)
    t2 = datetime.datetime(2025, 4, 2)
    patient = make_patient("p8", [make_event("X", t1), make_event("A", t2)])
    labels = labeler.label(patient)
    assert (
        labels[0]["boolean_value"] is False
    ), f'boolean_value was {labels[0]["boolean_value"]}'
    assert (
        labels[0]["competing_event"] is False
    ), f'competing_event was {labels[0]["competing_event"]}'
    assert (
        labels[0]["prediction_time"] == t2
    ), f'prediction_time was {labels[0]["prediction_time"]}'


def test_custom_event_no_primary():
    labeler = CustomEventLabeler(
        "custom", event_codes=["X"], max_time=datetime.datetime(2025, 12, 31)
    )
    t1 = datetime.datetime(2025, 4, 1)
    patient = make_patient("p9", [make_event("A", t1)])
    labels = labeler.label(patient)
    assert (
        labels[0]["boolean_value"] is False
    ), f'boolean_value was {labels[0]["boolean_value"]}'
    assert (
        labels[0]["competing_event"] is False
    ), f'competing_event was {labels[0]["competing_event"]}'
    assert (
        labels[0]["prediction_time"] == t1
    ), f'prediction_time was {labels[0]["prediction_time"]}'


def test_custom_event_time_window():
    labeler = CustomEventLabeler(
        "custom",
        event_codes=["X"],
        condition_codes=["C"],
        time_window=1,
        max_time=datetime.datetime(2025, 12, 31),
    )
    t1 = datetime.datetime(2025, 4, 1)
    t2 = datetime.datetime(2025, 4, 5)
    patient = make_patient("p10", [make_event("X", t1), make_event("C", t2)])
    labels = labeler.label(patient)
    assert (
        labels[0]["boolean_value"] is False
    ), f'boolean_value was {labels[0]["boolean_value"]}'
    assert (
        labels[0]["competing_event"] is False
    ), f'competing_event was {labels[0]["competing_event"]}'
    assert (
        labels[0]["prediction_time"] == t2
    ), f'prediction_time was {labels[0]["prediction_time"]}'


def test_custom_event_time_window_2():
    labeler = CustomEventLabeler(
        "custom",
        event_codes=["X"],
        condition_codes=["C"],
        time_window=5,
        max_time=datetime.datetime(2025, 12, 31),
    )
    t1 = datetime.datetime(2025, 4, 1)
    t2 = datetime.datetime(2025, 4, 5)
    patient = make_patient("p10", [make_event("X", t1), make_event("C", t2)])
    labels = labeler.label(patient)
    assert (
        labels[0]["boolean_value"] is True
    ), f'boolean_value was {labels[0]["boolean_value"]}'
    assert (
        labels[0]["competing_event"] is False
    ), f'competing_event was {labels[0]["competing_event"]}'
    assert (
        labels[0]["prediction_time"] == t1
    ), f'prediction_time was {labels[0]["prediction_time"]}'


def test_custom_event_sequence_required():
    labeler = CustomEventLabeler(
        "custom",
        event_codes=["X"],
        condition_codes=["C"],
        sequence_required=True,
        max_time=datetime.datetime(2025, 12, 31),
    )
    t1 = datetime.datetime(2025, 4, 1)
    t2 = datetime.datetime(2025, 4, 5)
    patient = make_patient("p11", [make_event("C", t1), make_event("X", t2)])
    labels = labeler.label(patient)
    assert (
        labels[0]["boolean_value"] is False
    ), f'boolean_value was {labels[0]["boolean_value"]}'
    assert (
        labels[0]["competing_event"] is False
    ), f'competing_event was {labels[0]["competing_event"]}'
    assert (
        labels[0]["prediction_time"] == t2
    ), f'prediction_time was {labels[0]["prediction_time"]}'


def test_custom_event_sequence_required_2():
    labeler = CustomEventLabeler(
        "custom",
        event_codes=["X"],
        condition_codes=["C"],
        sequence_required=True,
        max_time=datetime.datetime(2025, 12, 31),
    )
    t1 = datetime.datetime(2025, 4, 3)
    t2 = datetime.datetime(2025, 4, 5)
    patient = make_patient("p11", [make_event("C", t2), make_event("X", t1)])
    labels = labeler.label(patient)
    assert (
        labels[0]["boolean_value"] is True
    ), f'boolean_value was {labels[0]["boolean_value"]}'
    assert (
        labels[0]["competing_event"] is False
    ), f'competing_event was {labels[0]["competing_event"]}'
    assert (
        labels[0]["prediction_time"] == t1
    ), f'prediction_time was {labels[0]["prediction_time"]}'
