from sat.data.dataset.femr_extensions.labelers import (
    CompetingRiskLabeler,
    CustomEventLabeler,
    SurvivalLabeler,
)


# Helper to create Event as a dict (TypedDict)
def make_event(code, time):
    return {"code": code, "time": time}


# Helper to create Patient as a dict (TypedDict)
def make_patient(patient_id, events):
    return {"patient_id": patient_id, "events": events}


# ---- CompetingRiskLabeler Tests ----
def test_competing_risk_no_events():
    labeler = CompetingRiskLabeler("cr", {"A": ["X"], "B": ["Y"]}, max_time=10)
    patient = make_patient("p0", [])
    labels = labeler.label(patient)
    b_label = next(l for l in labels if l["event_category"] == "cr_B")
    a_label = next(l for l in labels if l["event_category"] == "cr_A")
    assert b_label["competing_event"] is True, f"Labels: {b_label}"
    assert b_label["boolean_value"] is False, f"Labels: {b_label}"
    assert b_label["prediction_time"] == 10, f"Labels: {b_label}"
    assert a_label["competing_event"] is True, f"Labels: {a_label}"
    assert a_label["boolean_value"] is False, f"Labels: {a_label}"
    assert a_label["prediction_time"] == 10, f"Labels: {a_label}"


def test_competing_risk_first_event():
    labeler = CompetingRiskLabeler("cr", {"A": ["X"], "B": ["Y"]})
    patient = make_patient("p1", [make_event("Y", 2), make_event("X", 5)])
    labels = labeler.label(patient)
    b_label = next(l for l in labels if l["event_category"] == "cr_B")
    a_label = next(l for l in labels if l["event_category"] == "cr_A")
    assert b_label["competing_event"] is True, f"Labels: {b_label}"
    assert b_label["boolean_value"] is True, f"Labels: {b_label}"
    assert b_label["prediction_time"] == 2, f"Labels: {b_label}"
    assert a_label["competing_event"] is True, f"Labels: {a_label}"
    assert a_label["boolean_value"] is False, f"Labels: {a_label}"
    assert a_label["prediction_time"] == 2, f"Labels: {a_label}"


def test_competing_risk_no_matching_events():
    labeler = CompetingRiskLabeler("cr", {"A": ["X"], "B": ["Y"]})
    patient = make_patient("p2", [make_event("Z", 3), make_event("W", 5)])
    labels = labeler.label(patient)
    b_label = next(l for l in labels if l["event_category"] == "cr_B")
    a_label = next(l for l in labels if l["event_category"] == "cr_A")
    assert b_label["competing_event"] is True, f"Labels: {b_label}"
    assert b_label["boolean_value"] is False, f"Labels: {b_label}"
    assert b_label["prediction_time"] == 5, f"Labels: {b_label}"
    assert a_label["competing_event"] is True, f"Labels: {a_label}"
    assert a_label["boolean_value"] is False, f"Labels: {a_label}"
    assert a_label["prediction_time"] == 5, f"Labels: {a_label}"


# ---- SurvivalLabeler Tests ----
def test_survival_no_events():
    labeler = SurvivalLabeler("surv", ["X"], max_time=10)
    patient = make_patient("p3", [])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is False, f"Labels: {labels[0]}"
    assert labels[0]["competing_event"] is False, f"Labels: {labels[0]}"
    assert labels[0]["prediction_time"] == 10, f"Labels: {labels[0]}"


def test_survival_event_found():
    labeler = SurvivalLabeler("surv", ["X"], max_time=10)
    patient = make_patient(
        "p4", [make_event("A", 1), make_event("X", 3), make_event("X", 4)]
    )
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is True, f"Labels: {labels[0]}"
    assert labels[0]["competing_event"] is False, f"Labels: {labels[0]}"
    assert labels[0]["prediction_time"] == 3, f"Labels: {labels[0]}"


def test_survival_event_not_found():
    labeler = SurvivalLabeler("surv", ["X"], max_time=10)
    patient = make_patient("p5", [make_event("A", 1), make_event("B", 2)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is False, f"Labels: {labels[0]}"
    assert labels[0]["competing_event"] is False, f"Labels: {labels[0]}"
    assert labels[0]["prediction_time"] == 2, f"Labels: {labels[0]}"


# ---- CustomEventLabeler Tests ----
def test_custom_event_exclusion():
    labeler = CustomEventLabeler("custom", primary_codes=["X"], exclusion_codes={"Y"})
    patient = make_patient("p6", [make_event("X", 1), make_event("Y", 2)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is False, f"Labels: {labels[0]}"
    assert labels[0]["competing_event"] is False, f"Labels: {labels[0]}"
    assert labels[0]["prediction_time"] == 2, f"Labels: {labels[0]}"


def test_custom_event_primary_and_condition():
    labeler = CustomEventLabeler("custom", primary_codes=["X"], condition_codes={"C"})
    patient = make_patient("p7", [make_event("X", 1), make_event("C", 2)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is True, f"Labels: {labels[0]}"
    assert labels[0]["prediction_time"] == 1, f"Labels: {labels[0]}"


def test_custom_event_primary_but_no_condition():
    labeler = CustomEventLabeler("custom", primary_codes=["X"], condition_codes={"C"})
    patient = make_patient("p8", [make_event("X", 1), make_event("A", 2)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is False, f"Labels: {labels[0]}"
    assert labels[0]["prediction_time"] == 2, f"Labels: {labels[0]}"


def test_custom_event_no_primary():
    labeler = CustomEventLabeler("custom", primary_codes=["X"])
    patient = make_patient("p9", [make_event("A", 1)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is False, f"Labels: {labels[0]}"
    assert labels[0]["competing_event"] is False, f"Labels: {labels[0]}"
    assert labels[0]["prediction_time"] == 1, f"Labels: {labels[0]}"


def test_custom_event_time_window():
    labeler = CustomEventLabeler(
        "custom", primary_codes=["X"], condition_codes={"C"}, time_window=1
    )
    patient = make_patient("p10", [make_event("X", 1), make_event("C", 5)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is False, f"Labels: {labels[0]}"
    assert labels[0]["prediction_time"] == 5, f"Labels: {labels[0]}"


def test_custom_event_sequence_required():
    labeler = CustomEventLabeler(
        "custom", primary_codes=["X"], condition_codes={"C"}, sequence_required=True
    )
    patient = make_patient("p11", [make_event("C", 1), make_event("X", 5)])
    labels = labeler.label(patient)
    assert labels[0]["boolean_value"] is False, f"Labels: {labels[0]}"
    assert labels[0]["prediction_time"] == 5, f"Labels: {labels[0]}"
