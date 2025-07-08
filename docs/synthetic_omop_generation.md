# Synthetic OMOP Data Generation Pipeline

This document describes the recommended, configuration-driven pipeline for generating, processing, and preparing synthetic OMOP (Observational Medical Outcomes Partnership) data for SAT survival analysis workflows.

## Pipeline Overview

The synthetic OMOP data preparation pipeline consists of three main stages, each controlled by a dedicated YAML configuration file:

1. **Generate OMOP Data**: Create synthetic patient event data in OMOP format.
2. **Create Cohort**: Build a labeled cohort suitable for survival analysis using event-based labelers.
3. **Parse into SAT Format**: Convert the cohort to the SAT format for model training and evaluation.

Each stage is invoked using a single command and a corresponding config file, ensuring reproducibility and modularity. See below for details on each step.

---

## 1. Generate Synthetic OMOP Data

**Command:**
```bash
poetry run python -m sat.prepare_data experiments=synthetic_omop/survival data/parse=synthetic_omop_generate
```

**Config:** [`conf/data/parse/synthetic_omop_generate.yaml`](../conf/data/parse/synthetic_omop_generate.yaml)

**Purpose:**
- Generates a realistic, large-scale synthetic OMOP CDM dataset with configurable demographics, event types, and observation windows.
- Output includes event data, code definitions, subject splits, and metadata.

**Key Config Parameters:**
- `num_patients`: Number of patients to generate (default: 10,000)
- `censoring_time`: Maximum follow-up time (days)
- `pre_enrollment_period`: Pre-enrollment history window (days)
- `mortality_rate`: Fraction of patients who die during observation
- `min_post_enrollment_obs`, `max_obs_window`: Control enrollment/event time logic

**Relevant Implementation:**
- [`generate_synthetic_omop.py`](../sat/data/dataset/generate_synthetic_omop.py): Implements the generator logic, event simulation, and OMOP-compatible output.

**Output Structure:**
```
synthetic_omop/
  ├── data/
  │   └── patients_with_events.parquet   # Main event data
  ├── codes.parquet                      # Code definitions
  ├── subject_splits.parquet             # Train/test splits
  └── _metadata.json                     # Dataset metadata
```

---

## 2. Create Cohort

**Command:**
```bash
poetry run python -m sat.prepare_data experiments=synthetic_omop/survival data/parse=synthetic_omop_cohort
```

**Config:** [`conf/data/parse/synthetic_omop_cohort.yaml`](../conf/data/parse/synthetic_omop_cohort.yaml)

**Purpose:**
- Applies FEMR-compatible labelers to the generated OMOP data to define anchor events, outcomes, and competing risks.
- Produces a labeled cohort suitable for survival analysis.

**Key Config Parameters:**
- `source`: Path to the generated OMOP event data (e.g., `synthetic_omop/patients_with_events.parquet`)
- `labelers`: List of event-based labelers (anchor, mortality, stroke, kidney failure, MI, etc.)
- `time_field`, `primary_key`, `date_diff_unit`: Control event timing and patient indexing

**Relevant Implementation:**
- [`cohort_omop.py`](../sat/data/dataset/cohort_omop.py): Handles cohort extraction, label assignment, and filtering based on anchor/outcome logic.

**Output Structure:**
```
cohort/<dataset>/
  └── cohort_dataset.arrow  # Labeled cohort in HuggingFace format
```

---

## 3. Parse into SAT Format

**Command:**
```bash
poetry run python -m sat.prepare_data experiments=synthetic_omop/survival
```

**Config:** [`conf/data/parse/synthetic_omop_parse.yaml`](../conf/data/parse/synthetic_omop_parse.yaml)

**Purpose:**
- Converts the labeled cohort to the SAT survival analysis format, including code sequence, modalities, numerics, and outcome/event structure.
- Supports feature scaling and batch processing for large datasets.

**Key Config Parameters:**
- `source`: Path to cohort directory
- `scale_method`: e.g., `standard` (z-score) or `min_max`
- `scale_numerics`: Whether to scale numeric features
- `batch_size`: For efficient streaming processing

**Relevant Implementation:**
- [`parse_omop.py`](../sat/data/dataset/parse_omop.py): Transforms cohort data to SAT format with all required fields for downstream modeling.

**Output Structure:**
```
modelhub/<dataset>/
  └── sat_dataset.arrow  # Final SAT-formatted dataset
```

---

## Configuration-Driven Workflow Summary

- **All stages are run via `sat.prepare_data` with the appropriate `data/parse` config.**
- **No direct CLI or Python scripting is needed for standard workflows.**
- Each YAML config is fully documented and can be customized for advanced use cases (e.g., changing event types, labelers, or scaling methods).
- For further details, see the docstrings in the corresponding Python modules:
  - `generate_synthetic_omop.py` (synthetic data generation)
  - `cohort_omop.py` (cohort/label extraction)
  - `parse_omop.py` (SAT format conversion)

## Configuring Labels with the Labeler Module

The SAT synthetic OMOP pipeline uses flexible, FEMR-compatible labelers (see [`labelers.py`](../sat/data/dataset/femr_extensions/labelers.py)) to define anchor, outcome, and competing risk events for cohort creation. Labelers are configured in your YAML (typically `synthetic_omop_cohort.yaml`) under the `labelers` section.

### Key Labeler Types

- **Anchor label**: Defines the cohort entry event (e.g., enrollment)
- **Outcome label**: Defines the primary event of interest (e.g., death, stroke)
- **Competing risk label**: Defines events that censor the outcome (e.g., death when studying non-fatal stroke)

### Example YAML Configuration

```yaml
labelers:
  # Anchor (cohort entry)
  - _target_: sat.data.dataset.femr_extensions.labelers.CustomEventLabeler
    name: enrollment
    event_codes: ["OMOP_ENROLLMENT"]
    label_type: anchor
    competing_event: false
    mode: first
    max_time: "2023-12-31T23:59:59"

  # Outcome (e.g., mortality)
  - _target_: sat.data.dataset.femr_extensions.labelers.CustomEventLabeler
    name: mortality
    event_codes: ["OMOP_DEATH"]
    label_type: outcome
    competing_event: true
    mode: first
    max_time: "2023-12-31T23:59:59"

  # Additional outcomes or competing risks
  - _target_: sat.data.dataset.femr_extensions.labelers.CustomEventLabeler
    name: stroke
    event_codes: ["OMOP_STROKE", "ICD10:I63"]
    label_type: outcome
    competing_event: false
    mode: all
    max_time: "2023-12-31T23:59:59"
```

### Important Parameters

- `name`: Unique label name (used as a column in the output dataset)
- `event_codes`: List of OMOP or custom codes to match for this label
- `label_type`: One of `anchor`, `outcome`, or `inclusion`/`exclusion` (see `LabelType`)
- `competing_event`: If `true`, this event censors other outcomes (competing risk)
- `mode`: `first` (only the first event per patient) or `all` (all matching events)
- `max_time`: Maximum follow-up time (ISO8601 string or days since epoch)
- `condition_codes`: (optional) Codes that must co-occur or follow the primary event
- `time_window`: (optional) Time window for `condition_codes` (interpreted using `time_unit`)
- `sequence_required`: (optional) If true, condition must occur after the primary event
- `time_unit`: (optional) Unit for `time_window` (e.g., `days`, `hours`)

### Advanced Example

```yaml
  # Composite label with condition and time window
  - _target_: sat.data.dataset.femr_extensions.labelers.CustomEventLabeler
    name: mi_with_metformin
    event_codes: ["ICD10:I21"]  # MI
    condition_codes: ["RxNorm:A10BA02"]  # Metformin
    time_window: 30
    time_unit: days
    sequence_required: true
    label_type: outcome
    competing_event: false
    mode: first
    max_time: "2023-12-31T23:59:59"
```

### Notes
- Instantiate one labeler per event type (including each outcome and each competing event).
- The anchor labeler is required for survival analysis pipelines.
- For details on available parameters and advanced logic, see the docstrings in [`labelers.py`](../sat/data/dataset/femr_extensions/labelers.py).

---

## Notes
- All outputs are versioned and reproducible via config.
- For advanced cohort definitions, modify the `labelers` section of `synthetic_omop_cohort.yaml`.
- For custom scaling or feature engineering, edit `synthetic_omop_parse.yaml`.
- For troubleshooting or debugging, enable debug logging in the configs or Python modules.

## Configuration Parameters

The following parameters can be configured:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_patients` | int | 10000 | Number of patients to generate |
| `output_path` | str | "synthetic_omop" | Output directory path |
| `seed` | int | 42 | Random seed for reproducibility |
| `censoring_time` | int | 1095 | Maximum follow-up time in days (3 years) |
| `enrollment_code` | str | "OMOP_ENROLLMENT" | Code used for enrollment events |

## Advanced Configuration

You can customize the data generation process by modifying the following parameters in your Hydra config:

- `categorical_covariates`: Categorical variables (race, sex, etc.)
- `numerical_covariates`: Numerical variables (age, BMI, etc.)
- `event_types`: Types of events to generate
- `icd_codes`: ICD-10 diagnosis codes to use
- `omop_codes`: RxNorm medication codes to use
- `lab_codes`: LOINC lab measurement codes to use

Example custom configuration:

```yaml
# Custom config example
_target_: sat.data.dataset.generate_synthetic_omop.SyntheticOmopGenerator
num_patients: 5000
output_path: ${data_source}/custom_synthetic_omop
seed: 123

# Custom age distribution
numerical_covariates:
  - code: "OMOP_AGE"
    name: "age"
    distribution: "normal"
    mean: 65  # Older population
    std: 12
    min: 18
    max: 95
    time_dependent: false
```

## Generated Data Structure

The generator creates a directory structure with the following files:

```
synthetic_omop/
  ├── data/
  │   └── data.parquet       # Main event data
  ├── codes.parquet          # Code definitions
  ├── subjects_splits.parquet # Subject train/test splits
  └── _metadata.json         # Dataset metadata
```

### Event Types

The generated data includes the following event types:

- **Enrollment**: Patient enrollment events with code `OMOP_ENROLLMENT`
- **Diagnoses**: ICD-10 diagnosis codes
- **Medications**: RxNorm drug codes
- **Lab measurements**: LOINC lab codes
- **Mortality**: Death events
- **Static attributes**: Age (normalized), BMI, sex, race

### Data Generation Process

The data generation follows these steps:

1. **Static attributes generation**: Creates patient demographics and baseline characteristics
2. **Medical events generation**: Samples diagnoses, medications, and lab measurements
3. **Survival events generation**: Generates event times based on Cox models
4. **Enrollment events**: Marks patient enrollment at time 0
5. **Data compilation**: Combines all data into OMOP-compatible format

The generator ensures that all event types are properly included in the final dataset and statistics reporting.

### Statistics and Metadata

The `_metadata.json` file includes comprehensive statistics about the generated dataset:

```json
{
  "dataset_name": "synthetic_omop",
  "num_subjects": 10000,
  "seed": 42,
  "vocab_size": {
    "synthetic_omop": <total unique codes>
  },
  "event_type_counts": {
    "OMOP_ENROLLMENT": <count>,
    "ICD10CM": <count>,
    "RxNorm": <count>,
    ...
  },
  "subject_stats": {
    "min_events": <min>,
    "max_events": <max>,
    "mean_events": <avg>,
    "std_events": <std>
  }
}
```

The statistics include all event types: enrollment, diagnoses, medications, labs, mortality, and survival events.

This structure is compatible with other SAT modules for survival analysis.

## Integration with FEMR/MEDS

The generated data is fully compatible with FEMR and MEDS packages for medical event data analysis. For implementation details on how to use the generated data with those packages, refer to the `parse_omop.py` module and its in-code documentation (docstrings).

## Cohort Building with Labelers

The generated synthetic OMOP data can be used for cohort building using FEMR labelers. The labelers allow you to:

1. Define anchor events based on specific codes (diagnoses, medications, labs)
2. Create time-based prediction windows
3. Extract labels for events within specified timeframes

Example cohort building:

```python
from femr.labelers import LabeledPatients, Label
from sat.data.dataset.femr_extensions.labelers import FEMRDatasetLabeler
from datetime import timedelta

# Define anchor event (e.g., diabetes diagnosis)
labeler = FEMRDatasetLabeler(
    synth_dist_labels=[
        Label(code="ICD10CM/E11", value=True),  # Type 2 diabetes
    ],
    time_horizon=timedelta(days=365),  # 1-year prediction window
    labeling_function=lambda patient: patient.events[0].time  # Anchor on first occurrence
)

# Apply labeler to dataset
labeled_patients = labeler.apply(dataset)
```

### Advanced Labeling Scenarios

The labeler supports complex cohort definitions:

```python
# Multiple anchor events
multi_event_labeler = FEMRDatasetLabeler(
    synth_dist_labels=[
        Label(code="ICD10CM/I21", value=True),   # MI
        Label(code="ICD10CM/I50", value=True),   # Heart failure
        Label(code="RxNorm/1037045", value=True), # Metformin
    ],
    time_horizon=timedelta(days=730),  # 2-year window
    prediction_offset=timedelta(days=30),  # 30-day gap after anchor
)

# Event-based outcomes
outcome_labeler = FEMRDatasetLabeler(
    synth_dist_labels=[
        Label(code="Event_1", value=True),  # Survival event 1
        Label(code="Event_2", value=True),  # Survival event 2
    ],
    time_horizon=timedelta(days=1095),  # 3-year follow-up
)
```

### Integration with Survival Analysis

The labeled cohorts can be directly used with SAT survival models:

```python
from sat.data.dataset.cohort_omop import CohortOmopDataset

# Create survival analysis dataset
dataset = CohortOmopDataset(
    data_directory="synthetic_omop",
    task_config=task_config,
    labeler=labeler,
    static_features=["OMOP_AGE", "OMOP_BMI", "OMOP_SEX"],
    temporal_features=["diagnoses", "medications", "labs"]
)
```

## Code Conventions

The generator follows OMOP CDM standards for code naming:

- Enrollment events: `OMOP_ENROLLMENT`
- Diagnoses: `ICD10CM/` prefix
- Medications: `RxNorm/` prefix
- Lab measurements: `LOINC/` prefix
- Survival events: `Event_<number>`

## Performance Optimization

For large-scale data generation:

- Batch processing is used for efficient memory management
- Parquet format enables compressed storage and fast I/O
- Parallel processing available for multi-core systems
