# Synthetic OMOP Data Generation

This document describes how to generate synthetic OMOP (Observational Medical Outcomes Partnership) data for testing and development of survival analysis models.

## Overview

The `SyntheticOmopGenerator` class creates realistic synthetic patient data following the OMOP Common Data Model (CDM) schema. This data can be used for:

- Testing survival analysis models
- Developing new algorithms
- Benchmarking performance
- Demonstrations and tutorials

## Usage

### Using Hydra Configuration

The generator is fully compatible with Hydra configuration. To generate data using Hydra:

```bash
# Use the default configuration
python -m sat.prepare_data experiments=synthetic_omop/survival dataset=synthetic_omop

# Specify custom parameters
python -m sat.prepare_data experiments=synthetic_omop/survival dataset=synthetic_omop data.parse.synthetic_omop.num_patients=5000 data.parse.synthetic_omop.seed=123
```

### Using Direct Python Import

You can also use the generator directly in your Python code:

```python
from sat.data.dataset.generate_synthetic_omop import SyntheticOmopGenerator

# Create generator with default settings
generator = SyntheticOmopGenerator(
    num_patients=10000,
    output_path="synthetic_omop",
    seed=42,
    censoring_time=1095  # 3 years
)

# Generate data
output_dir = generator.generate_all()
print(f"Data generated at: {output_dir}")
```

### Command Line Interface

For quick generation without Hydra, use the command-line interface:

```bash
python -m sat.data.dataset.generate_synthetic_omop --num_patients 5000 --output synthetic_omop --seed 123
```

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
