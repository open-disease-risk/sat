from sat.data.dataset.femr_extensions.labelers import CustomEventLabeler, LabelType


# Helper to create Event as a dict (TypedDict)
def make_event(code, time):
    return {"code": code, "time": time}


# Helper to create Patient as a dict (TypedDict)
def make_patient(patient_id, events):
    return {"patient_id": patient_id, "events": events}


def test_anchor_event():
    labeler = CustomEventLabeler(
        "anchor", label_type=LabelType.ANCHOR, event_codes=["X"], max_time=10
    )
    patient = make_patient("p0", [make_event("X", 5)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is True
    assert labels[0]["competing_event"] is False
    assert labels[0]["prediction_time"] == 5


def test_anchor_event_2():
    labeler = CustomEventLabeler(
        "anchor", label_type=LabelType.ANCHOR, event_codes=["X"], max_time=10
    )
    patient = make_patient("p0", [make_event("A", 5)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is False
    assert labels[0]["competing_event"] is False
    assert labels[0]["prediction_time"] == 5


def test_competing_risk_no_events():
    labeler_a = CustomEventLabeler(
        "A", event_codes=["X"], competing_event=True, max_time=10
    )
    labeler_b = CustomEventLabeler(
        "B", event_codes=["Y"], competing_event=True, max_time=10
    )
    patient = make_patient("p0", [])
    a_label = labeler_a.label(patient)[0]
    b_label = labeler_b.label(patient)[0]
    assert a_label["competing_event"] is True
    assert a_label["boolean_value"] is False
    assert a_label["prediction_time"] == 10
    assert b_label["competing_event"] is True
    assert b_label["boolean_value"] is False
    assert b_label["prediction_time"] == 10


def test_competing_risk():
    # Competing events the labelling here does not take into account dependencies between labels
    labeler_a = CustomEventLabeler("A", event_codes=["X"], competing_event=True)
    labeler_b = CustomEventLabeler("B", event_codes=["Y"], competing_event=True)
    patient = make_patient("p1", [make_event("Y", 2), make_event("X", 5)])
    a_label = labeler_a.label(patient)[0]
    b_label = labeler_b.label(patient)[0]
    assert a_label["competing_event"] is True, a_label
    assert a_label["boolean_value"] is True, a_label
    assert a_label["prediction_time"] == 5, a_label
    assert b_label["competing_event"] is True, b_label
    assert b_label["boolean_value"] is True, b_label
    assert b_label["prediction_time"] == 2, b_label


def test_competing_risk_no_matching_events():
    labeler_a = CustomEventLabeler("A", event_codes=["X"], competing_event=True)
    labeler_b = CustomEventLabeler("B", event_codes=["Y"], competing_event=True)
    patient = make_patient("p2", [make_event("Z", 3), make_event("W", 5)])
    a_label = labeler_a.label(patient)[0]
    b_label = labeler_b.label(patient)[0]
    assert a_label["competing_event"] is True, a_label
    assert a_label["boolean_value"] is False, a_label
    assert a_label["prediction_time"] == 5, a_label
    assert b_label["competing_event"] is True, b_label
    assert b_label["boolean_value"] is False, b_label
    assert b_label["prediction_time"] == 5, b_label


# ---- Survival (with CustomEventLabeler) Tests ----
def test_survival_no_events():
    labeler = CustomEventLabeler("surv", event_codes=["X"], max_time=10)
    patient = make_patient("p3", [])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is False, labels[0]
    assert labels[0]["competing_event"] is False, labels[0]
    assert labels[0]["prediction_time"] == 10, labels[0]


def test_survival_event_found():
    labeler = CustomEventLabeler("surv", event_codes=["X"], max_time=10)
    patient = make_patient(
        "p4", [make_event("A", 1), make_event("X", 3), make_event("X", 4)]
    )
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is True, labels[0]
    assert labels[0]["competing_event"] is False, labels[0]
    assert labels[0]["prediction_time"] == 3, labels[0]


def test_survival_event_not_found():
    labeler = CustomEventLabeler("surv", event_codes=["X"], max_time=10)
    patient = make_patient("p5", [make_event("A", 1), make_event("B", 2)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is False, labels[0]
    assert labels[0]["competing_event"] is False, labels[0]
    assert labels[0]["prediction_time"] == 2, labels[0]


# ---- CustomEventLabeler Tests ----
def test_custom_event_exclusion():
    labeler = CustomEventLabeler(
        "custom", label_type=LabelType.EXCLUSION, event_codes=["Y"]
    )
    patient = make_patient("p6", [make_event("X", 1), make_event("Y", 2)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is True, labels[0]
    assert labels[0]["competing_event"] is False, labels[0]
    assert labels[0]["prediction_time"] == 2, labels[0]


def test_custom_event_primary_and_condition():
    labeler = CustomEventLabeler("custom", event_codes=["X"], condition_codes=["C"])
    patient = make_patient("p7", [make_event("X", 1), make_event("C", 2)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is True, labels[0]
    assert labels[0]["prediction_time"] == 1, labels[0]


def test_custom_event_primary_but_no_condition():
    labeler = CustomEventLabeler("custom", event_codes=["X"], condition_codes=["C"])
    patient = make_patient("p8", [make_event("X", 1), make_event("A", 2)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is False, labels[0]
    assert labels[0]["competing_event"] is False, labels[0]
    assert labels[0]["prediction_time"] == 2, labels[0]


def test_custom_event_no_primary():
    labeler = CustomEventLabeler("custom", event_codes=["X"])
    patient = make_patient("p9", [make_event("A", 1)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is False, labels[0]
    assert labels[0]["competing_event"] is False, labels[0]
    assert labels[0]["prediction_time"] == 1, labels[0]


def test_custom_event_time_window():
    labeler = CustomEventLabeler(
        "custom", event_codes=["X"], condition_codes=["C"], time_window=1
    )
    patient = make_patient("p10", [make_event("X", 1), make_event("C", 5)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is False, labels[0]
    assert labels[0]["competing_event"] is False, labels[0]
    assert labels[0]["prediction_time"] == 5, labels[0]


def test_custom_event_time_window_2():
    labeler = CustomEventLabeler(
        "custom", event_codes=["X"], condition_codes=["C"], time_window=5
    )
    patient = make_patient("p10", [make_event("X", 1), make_event("C", 5)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is True, labels[0]
    assert labels[0]["competing_event"] is False, labels[0]
    assert labels[0]["prediction_time"] == 1, labels[0]


def test_custom_event_sequence_required():
    labeler = CustomEventLabeler(
        "custom", event_codes=["X"], condition_codes=["C"], sequence_required=True
    )
    patient = make_patient("p11", [make_event("C", 1), make_event("X", 5)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is False, labels[0]
    assert labels[0]["competing_event"] is False, labels[0]
    assert labels[0]["prediction_time"] == 5, labels[0]


def test_custom_event_sequence_required_2():
    labeler = CustomEventLabeler(
        "custom", event_codes=["X"], condition_codes=["C"], sequence_required=True
    )
    patient = make_patient("p11", [make_event("C", 5), make_event("X", 3)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is True, labels[0]
    assert labels[0]["competing_event"] is False, labels[0]
    assert labels[0]["prediction_time"] == 3, labels[0]
