# FEMR Extensions for Survival Analysis

This guide provides comprehensive documentation on using FEMR extensions in the SAT project for survival analysis, with a focus on setting up anchor events, outcome events, and competing events (terminal events). It combines clinical knowledge with practical implementation details.

## Core Concepts in Survival Analysis

Before diving into implementation, let's understand the key concepts:

### Anchor Events

Anchor events define the starting point for time-to-event analysis. These events:
- Establish "time zero" for each patient
- Often represent diagnosis, treatment initiation, or study enrollment
- Are essential for proper alignment of patient trajectories
- Must occur before outcome or competing events to be valid

From a clinical perspective, anchor events should represent a clinically meaningful starting point that aligns with your research question. Examples include:
- Date of diabetes diagnosis (for diabetes complications study)
- Hospital admission (for readmission risk analysis)
- Start of chemotherapy (for cancer progression analysis)
- Surgical procedure (for post-surgical complication analysis)

### Outcome Events

Outcome events are the primary events of interest in your analysis. These events:
- Represent the primary endpoint being studied
- Are used to calculate the time-to-event
- Can be censored by competing events
- Should be clinically relevant to your research question

Clinical examples include:
- Disease progression or recurrence
- Development of a specific complication
- Hospital readmission
- Treatment response

### Competing Events (Terminal Events)

Competing events (also called terminal events) are events that:
- Prevent the primary outcome from being observed
- Terminate the observation period for a patient
- Lead to informative censoring that must be handled properly
- Require specific statistical approaches (competing risks analysis)

Common competing events in clinical studies:
- Death from any cause (when studying non-mortality outcomes)
- Death from other causes (when studying cause-specific mortality)
- Treatment discontinuation or switching
- Loss to follow-up for known reasons (e.g., transfer to hospice care)

## Implementation in SAT

The SAT project extends FEMR's capabilities with specialized labelers for survival analysis. Here's how to implement each event type:

### Setting Up Labelers for Different Event Types

The core component for implementing survival analysis is the `CustomEventLabeler` class, which can be configured for different event types:

```python
from sat.data.dataset.femr_extensions.labelers import CustomEventLabeler
from sat.data.dataset.femr_extensions.schema import LabelType
```

#### Anchor Event Labeler

```python
anchor_labeler = CustomEventLabeler(
    name="diabetes_diagnosis",  # Descriptive name for the event
    event_codes=["E11.0", "E11.9", "250.*"],  # ICD codes for diabetes
    competing_event=False,  # Not a competing event
    label_type=LabelType.ANCHOR,  # Marks this as an anchor event
    time_field="time",  # Name of time field in your data
    max_time=3650.0,  # Maximum follow-up time (10 years)
)
```

Clinical consideration: Choose an anchor event that is reliably recorded in your data and represents a clinically meaningful starting point. For chronic diseases, first diagnosis may be preferable; for acute conditions, the initial presentation or treatment may be more appropriate.

#### Primary Outcome Labeler

```python
outcome_labeler = CustomEventLabeler(
    name="kidney_disease",  # Name of the outcome
    event_codes=["N18.*", "585.*"],  # ICD codes for chronic kidney disease
    competing_event=False,  # Not a competing event
    label_type=LabelType.OUTCOME,  # Marks this as an outcome event
    time_field="time",
    max_time=3650.0,
    # Optional: require additional condition validation
    condition_codes=["E11.*"],  # Must also have diabetes diagnosis
    time_window=90,  # Within 90 days
    sequence_required=True,  # Condition must follow primary event
    time_unit="days",  # Time unit for window
)
```

Clinical consideration: Define outcomes that are specific enough to be captured accurately in your data. For example, "myocardial infarction" is more specific than "cardiovascular event". Ensure the codes you select have high positive predictive value for the outcome of interest.

#### Competing Event (Terminal Event) Labeler

```python
competing_labeler = CustomEventLabeler(
    name="death",  # Name of the competing event
    event_codes=["798", "R99", "DEATH"],  # Death codes
    competing_event=True,  # Mark this as a competing event
    label_type=LabelType.OUTCOME,  # Note: still uses OUTCOME type
    time_field="time",
    max_time=3650.0,
)
```

Clinical consideration: Competing events should be carefully selected based on your study design. Death is the most common competing event, but treatment discontinuation, major clinical events, or loss to follow-up may also be appropriate depending on your research question.

### Building a Cohort with Competing Risks

The `CohortOMOP` class provides a complete pipeline for building cohorts with proper handling of competing risks:

```python
from sat.data.dataset.cohort_omop import CohortOMOP
from sat.data.dataset.femr_extensions.schema import LabelType

# Create labelers
anchor_labeler = CustomEventLabeler(
    name="index_hospitalization",
    event_codes=["ENC_INPATIENT"],
    label_type=LabelType.ANCHOR,
    time_field="time"
)

outcome_labeler = CustomEventLabeler(
    name="readmission",
    event_codes=["ENC_INPATIENT"],
    label_type=LabelType.OUTCOME,
    time_field="time",
    max_time=30.0  # 30-day readmission
)

death_labeler = CustomEventLabeler(
    name="death",
    event_codes=["DEATH"],
    competing_event=True,  # This is a competing event
    label_type=LabelType.OUTCOME,
    time_field="time",
    max_time=30.0
)

exclusion_labeler = CustomEventLabeler(
    name="planned_readmission",
    event_codes=["PLANNED_ADMISSION"],
    label_type=LabelType.EXCLUSION,
    time_field="time"
)

# Create the cohort builder
cohort_builder = CohortOMOP(
    source="path/to/data.parquet",
    name="readmission_cohort",
    processed_dir="output/directory",
    labelers=[anchor_labeler, outcome_labeler, death_labeler, exclusion_labeler],
    primary_key="patient_id",
    time_field="time",
    date_diff_unit="days"
)

# Build the cohort with competing risk handling
cohort_builder()
```

## Key Implementation Details

### How Competing Events Are Handled

The `CohortOMOP` class implements several important steps for proper competing risk handling:

1. **Apply labelers in proper order**: Anchor events first, followed by outcomes and eligibility criteria
2. **Apply competing risk censoring**: If a competing event occurs before the primary outcome, the outcome is censored at the time of the competing event
3. **Truncate events**: Patient event timelines are truncated at the earliest competing event
4. **Filter patients without anchor events**: Removes patients who don't have the anchor event
5. **Apply exclusion criteria**: Removes patients meeting exclusion criteria

The competing risk censoring logic in `apply_competing_risk_censoring()` is particularly important:

```python
# For each patient, find the earliest competing event
competing_times = []
for outcome_col in outcome_label_cols:
    labels = labels_dict[outcome_col][i]
    for label in labels:
        if label.get("competing_event", False) and label.get("boolean_value", False):
            comp_time = label.get("prediction_time", None)
            if comp_time is not None:
                competing_times.append(comp_time)

if competing_times:
    # Find earliest competing event
    censor_time = min(competing_times)

    # Censor outcome labels if they occur after competing event
    for outcome_col in outcome_label_cols:
        labels = labels_dict[outcome_col][i]
        for label in labels:
            if label.get("prediction_time", float("inf")) > censor_time:
                # Set boolean_value to False (censored)
                label["boolean_value"] = False
                # Adjust prediction_time to time of competing event
                label["prediction_time"] = censor_time - anchor_time
```

### Clinical Considerations for Event Setup

When setting up your events, consider these clinical best practices:

1. **Anchor events**:
   - Choose clinically meaningful index dates
   - Ensure anchor events can be reliably identified in your data
   - Consider using a "clean" look-back period to ensure new-onset conditions

2. **Outcome events**:
   - Define specific, clinically relevant outcomes
   - Use validated code sets when available
   - Consider using condition_codes to improve specificity
   - For complex outcomes, consider creating composite outcomes with multiple labelers

3. **Competing events**:
   - Always include death as a competing event for non-mortality outcomes
   - Consider other events that preclude observing the primary outcome
   - Terminal states like hospice enrollment may be appropriate competing events
   - Treatment switches or discontinuations may be competing events in certain studies

4. **Time windows**:
   - Choose clinically appropriate time windows (e.g., 30 days, 90 days, 1 year)
   - Account for clinically plausible progression times
   - Consider domain knowledge about disease progression

## Advanced Usage: Complex Event Definitions

### Composite Outcomes

For complex outcomes that require multiple conditions, use the `condition_codes` parameter:

```python
# Myocardial infarction with elevated troponin
mi_labeler = CustomEventLabeler(
    name="myocardial_infarction",
    event_codes=["I21.*", "410.*"],  # MI diagnosis codes
    condition_codes=["TROPONIN_ELEVATED"],  # Lab result code
    time_window=3,  # Within 3 days
    time_unit="days"
)
```

### Sequential Events

For outcomes that require a specific sequence of events:

```python
# Recurrent infection after initial treatment
recurrent_infection = CustomEventLabeler(
    name="recurrent_infection",
    event_codes=["A41.*", "038.*"],  # Sepsis codes
    condition_codes=["ANTIBIOTIC_TREATMENT_COMPLETE"],  # Treatment completion
    sequence_required=True,  # Must occur after treatment
    time_window=30,  # Within 30 days after treatment
    time_unit="days"
)
```

### Multiple Competing Events

You can define multiple competing events to model complex scenarios:

```python
# Death competing event
death_labeler = CustomEventLabeler(
    name="death",
    event_codes=["DEATH"],
    competing_event=True,
    label_type=LabelType.OUTCOME
)

# Treatment discontinuation competing event
discontinuation_labeler = CustomEventLabeler(
    name="treatment_discontinuation",
    event_codes=["TREATMENT_STOPPED", "ADVERSE_EFFECT_SEVERE"],
    competing_event=True,
    label_type=LabelType.OUTCOME
)

# Disease progression competing event
progression_labeler = CustomEventLabeler(
    name="disease_progression",
    event_codes=["DISEASE_PROGRESSION", "TREATMENT_FAILURE"],
    competing_event=True,
    label_type=LabelType.OUTCOME
)
```

## Hydra Configuration Examples

Here's a complete example of setting up a competing risks analysis using Hydra configuration:

```yaml
# config.yaml
_target_: sat.data.dataset.parse_omop.parse_omop_cohort
source: ${paths.data_dir}/synthetic_omop
processed_dir: ${paths.outputs_dir}/cohorts
name: diabetes_complications

# Define labelers
cohort_builder:
  _target_: sat.data.dataset.cohort_omop.CohortOMOP
  source: ${paths.data_dir}/synthetic_omop/data.parquet
  name: ${name}
  processed_dir: ${processed_dir}
  primary_key: "patient_id"
  time_field: "time"
  date_diff_unit: "days"
  labelers:
    # Anchor event: diabetes diagnosis
    - _target_: sat.data.dataset.femr_extensions.labelers.CustomEventLabeler
      name: "diabetes_diagnosis"
      event_codes: ["E11.*", "250.*"]
      competing_event: False
      label_type: ${.....schema.LabelType.ANCHOR}
      time_field: "time"
      max_time: 3650.0

    # Primary outcome: kidney disease
    - _target_: sat.data.dataset.femr_extensions.labelers.CustomEventLabeler
      name: "kidney_disease"
      event_codes: ["N18.*", "585.*"]
      competing_event: False
      label_type: ${.....schema.LabelType.OUTCOME}
      time_field: "time"
      max_time: 3650.0

    # Competing event 1: death
    - _target_: sat.data.dataset.femr_extensions.labelers.CustomEventLabeler
      name: "death"
      event_codes: ["DEATH"]
      competing_event: True
      label_type: ${.....schema.LabelType.OUTCOME}
      time_field: "time"
      max_time: 3650.0

    # Competing event 2: end-stage renal disease requiring dialysis
    - _target_: sat.data.dataset.femr_extensions.labelers.CustomEventLabeler
      name: "esrd_dialysis"
      event_codes: ["Z99.2", "V45.11"]
      competing_event: True
      label_type: ${.....schema.LabelType.OUTCOME}
      time_field: "time"
      max_time: 3650.0

    # Exclusion: pre-existing kidney disease
    - _target_: sat.data.dataset.femr_extensions.labelers.CustomEventLabeler
      name: "preexisting_kidney_disease"
      event_codes: ["N18.*", "585.*"]
      label_type: ${.....schema.LabelType.EXCLUSION}
      time_field: "time"
```

## Conclusion

The FEMR extensions in the SAT project provide a powerful framework for implementing competing risks survival analysis with EHR data. By properly setting up anchor events, outcome events, and competing events, you can create cohorts that accurately model the complex reality of patient trajectories.

Remember these key principles:
1. Always define a clear anchor event that establishes "time zero"
2. Define specific, clinically relevant outcome events
3. Include appropriate competing events to avoid bias
4. Use exclusion criteria to ensure a clean cohort
5. Consider the clinical nuances of your specific research question

For further information on survival analysis methods implemented in SAT, refer to the [models.md](models.md) and [loss_functions.md](loss_functions.md) documentation.
