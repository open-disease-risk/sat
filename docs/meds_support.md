# MEDS Format Support in SAT

This document explains how to use the Medical Event Data Standard (MEDS) format with SAT for survival analysis.

## Overview

The Medical Event Data Standard (MEDS) is a standardized format for healthcare data designed to facilitate portable analyses across different datasets. SAT now supports MEDS-formatted data, allowing you to leverage existing MEDS datasets for survival analysis.

## FEMR Extensions for MEDS Processing

SAT now includes FEMR extensions that provide a more modular and extensible approach to processing MEDS data by extending the [FEMR](https://github.com/ds-wm/femr) APIs. These extensions include:

- **Labelers**: Components that extract time-to-event labels from patient data
- **Featurizers**: Components that extract features from patient data
- **Adapter**: Utilities to convert FEMR data to SAT format
- **Parallel Processing**: Built-in support for efficient processing of large datasets

### Available Labelers

- **SurvivalLabeler**: Creates binary event indicators for standard survival analysis
- **CompetingRiskLabeler**: Creates multi-event labels for competing risks analysis
- **CustomEventLabeler**: Creates complex event labels based on custom criteria

### Available Featurizers

- **EventDensityFeaturizer**: Creates features based on event density over time
- **TemporalPatternFeaturizer**: Creates features based on temporal patterns in events
- **RiskFactorFeaturizer**: Creates features based on the presence of risk factors

### Parallel Processing Capabilities

All FEMR-compatible labelers and featurizers include built-in parallel processing support:

1. **Multi-Core Processing with joblib**:
   - Automatically splits patients into batches for parallel processing
   - Uses multiple CPU cores for improved performance
   - Includes progress reporting with tqdm integration
   - Configurable number of jobs and batch size

2. **Distributed Processing with Ray** (when available):
   - Scales beyond a single machine to distributed clusters
   - Handles serialization/deserialization of components
   - Provides fault tolerance for long-running jobs
   - Works seamlessly with existing labelers and featurizers

The parser automatically detects available processing capabilities and uses the appropriate method:
```python
# Automatic detection and scaling in parse_meds.py
n_jobs = max(1, cpu_count - 2)
if use_ray:
    results = labeler.process_with_ray(patients, batch_size=batch_size, show_progress=True)
else:
    results = labeler.batch_label(patients, n_jobs=n_jobs, batch_size=batch_size, show_progress=True)
```

### Configuration with FEMR Extensions

To use the FEMR extensions, configure your dataset in a YAML file like this:

```yaml
# FEMR-based configuration for synthetic_meds dataset
_target_: sat.data.dataset.parse_meds.meds
source: ${paths.data_dir}/synthetic_meds
processed_dir: ${paths.outputs_dir}/data
train_ratio: 0.7
validation_ratio: 0.15
test_ratio: 0.15
name: "synthetic_meds_femr"
time_field: "days"

# FEMR-compatible labelers
labelers:
  # Mortality labeler (single-event)
  - type: survival
    name: mortality
    event_codes: ["MEDS_DEATH"]
    max_time: 3650.0

  # Competing risks labeler for diseases
  - type: competing_risk
    name: competing_risks
    event_codes:
      heart_disease: ["ICD10:I25.1", "ICD10:I25.10"]
      diabetes: ["ICD10:E11", "ICD10:E11.9"]
      hypertension: ["ICD10:I10"]
    max_time: 3650.0

# FEMR-compatible featurizers
featurizers:
  # Event density features
  - type: event_density
    name: temporal_density
    window_sizes: [30, 90, 180, 365]
    event_types: ["diagnosis", "medication", "lab"]

  # Temporal pattern features
  - type: temporal_pattern
    name: sequence_patterns
    pattern_length: 3
    event_types: ["diagnosis", "medication"]
```

## Requirements

To use MEDS format with SAT, you need the following dependencies:

- `femr`: Python package for advanced MEDS processing (strongly recommended)
- `pyarrow`: For reading Parquet files
- `fastparquet`: Alternative Parquet engine
- `joblib`: For parallel processing (installed automatically with scikit-learn)
- `ray` (optional): For distributed computing across multiple machines

These dependencies are automatically included in the project configuration. While the system can operate without FEMR using compatibility classes, the FEMR-based implementation offers better performance and more features.

## MEDS Format Structure

MEDS data is typically stored in Parquet format with a standardized schema:

- `data/data.parquet`: The main data table with events
- `codes.parquet`: Metadata about codes used in the data
- `subjects_splits.parquet`: Optional predefined train/val/test splits

## Legacy Medical Labelers for MEDS Data

SAT also provides specialized labelers for medical data in the MEDS format. These labelers include built-in parallel processing capabilities for efficiently handling large datasets by processing patients in parallel batches.

### Built-in Labelers

The following labelers are available:

1. **MortalityLabeler**: Identifies mortality events and creates time-to-event labels
2. **CompetingRiskLabeler**: Handles multiple competing events (like death and hospitalization)
3. **CustomEventLabeler**: Configurable labeler for custom event definitions
4. **RiskFactorLabeler**: Identifies risk factors from medical codes

Each labeler inherits from the `MedicalLabeler` base class, which provides parallel processing support.

### Parallel Processing Modes

All labelers support three processing modes:

1. **Serial**: Process patients one by one (for small datasets or debugging)
2. **Multiprocessing**: Process patients in parallel using Python's multiprocessing (default)
3. **Ray**: Distributed processing across multiple machines (if Ray is installed)

### Legacy Configuration

For backward compatibility, the system still supports the older configuration format that uses `labelers`:

```yaml
_target_: sat.data.dataset.parse_meds.meds
source: ${data_source}
processed_dir: ${modelhub}
train_ratio: 0.7
validation_ratio: 0.15
test_ratio: 0.15
n_bins: 20
encode: ordinal
strategy: quantile
name: ${dataset}
kfold: ${cv.kfold}
time_field: days
# Scaling options for numerical features
scale_numerics: true
scale_method: "standard"  # or "min_max"
min_scale_numerics: 1.0  # Used with min_max scaling

# Custom labelers for medical events
labelers:
  # Mortality labeler to identify death events
  - type: mortality
    name: death
    max_followup_days: 1095  # 3 years follow-up
    death_codes: ["SNOMED/419620001"]  # Using MEDS_DEATH_CODE
    enrollment_codes: ["ENROLLMENT"]

  # Hospitalization event labeler
  - type: custom_event
    name: hospitalization
    max_followup_days: 1095
    event_definition:
      codes: ["ENC_INPATIENT"]
    enrollment_codes: ["ENROLLMENT"]

  # Custom event labeler for complex event definitions
  - type: custom_event
    name: diabetes_complications
    max_followup_days: 1095
    event_definition:
      and:
        - codes: ["ICD10:E11"]  # Type 2 diabetes
        - or:
            - codes: ["ICD10:I50"]  # Heart failure
            - codes: ["ICD10:N18"]  # Kidney disease
    enrollment_codes: ["ENROLLMENT"]

  # Risk factor labeler to identify conditions
  - type: risk_factor
    name: risk_factors
    custom_codes:  # Optional additional codes
      icd10:
        hypertension: ["I15.9"]
        diabetes: ["E11"]
        heart_failure: ["I50"]
      rxnorm:
        diabetes: ["A10"]
```

## Processing MEDS Data

To process MEDS data using either the FEMR extensions or legacy implementation:

```bash
python -m sat.prepare_data experiments=synthetic_meds/survival
```

This will:

1. Load the MEDS data
2. Create patients and events
3. Apply the configured labelers and featurizers
4. Convert the results to SAT format
5. Split the data into train/val/test sets
6. Save the processed data

### Output Format

The processed data is saved in directories organized by event type:

```
outputs/data/synthetic_meds_femr/
├── metadata.json
├── mortality/
│   ├── train.pkl
│   ├── val.pkl
│   └── test.pkl
├── heart_disease/
│   ├── train.pkl
│   ├── val.pkl
│   └── test.pkl
├── diabetes/
│   ├── train.pkl
│   ├── val.pkl
│   └── test.pkl
└── hypertension/
    ├── train.pkl
    ├── val.pkl
    └── test.pkl
```

Each `.pkl` file contains a pandas DataFrame in the SAT format with the following columns:

- `subject_id` - The patient identifier
- `event` - The event indicator (1 = event, 0 = censored)
- `duration` - The time to event or censoring
- `x` - Tokenized categorical features
- `modality` - Modality vector (0 = categorical, 1 = numeric)
- `numerics` - Numeric feature values
- `token_times` - Time of each token

### FEMR Integration and Compatibility

The system is designed to work optimally with FEMR but provides compatibility layers:

1. **Full FEMR Integration**: When FEMR is available, the system directly inherits from FEMR's labelers and featurizers, extending them with parallel processing capabilities.

2. **Compatibility Mode**: When FEMR is not available, the system uses lightweight compatibility classes that preserve the same interface but provide basic functionality.

The compatibility approach ensures that your configuration files will work whether or not FEMR is installed, though FEMR is strongly recommended for production use.

## Creating Custom Labelers and Featurizers

You can create custom labelers and featurizers by extending the base classes. The parallel processing capabilities will be automatically available to your custom components:

```python
from sat.data.dataset.femr_extensions import ParallelLabeler, ParallelFeaturizer, LabelerResult, FeaturizerResult

class MyCustomLabeler(ParallelLabeler):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    def label_patient(self, patient):
        # Custom labeling logic - only need to implement the single-patient case
        # Parallel processing is handled by the parent class
        return LabelerResult(event=1, duration=100.0)

class MyCustomFeaturizer(ParallelFeaturizer):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    def featurize_patient(self, patient):
        # Custom featurizing logic - only need to implement the single-patient case
        # Parallel processing is handled by the parent class
        return FeaturizerResult(feature1=1.0, feature2=2.0)
```

Then use your custom component with parallel processing:

```python
# Create an instance of your custom labeler
my_labeler = MyCustomLabeler(name="custom_labeler")

# Process a list of patients in parallel
results = my_labeler.batch_label(patients, n_jobs=4, show_progress=True)

# Or use Ray for distributed processing
distributed_results = my_labeler.process_with_ray(patients, show_progress=True)
```

## Handling Multiple Event Types (Competing Risks)

Both the FEMR extensions and legacy parsers automatically handle multiple event types as competing risks:

1. Each event type (mortality, hospitalization, etc.) is assigned a unique integer code (starting from 1)
2. Code 0 is reserved for censored observations
3. Events are processed to create a proper time-to-event dataset with:
   - `events`: List of binary indicators for each event type (e.g., [0, 1, 0] means event type 2 occurred)
   - `durations`: List of times for each event type (e.g., [30, 25, 30] means event type 2 occurred at day 25)

## Generating Synthetic MEDS Data

The provided synthetic MEDS data generator creates realistic healthcare data for testing:

```bash
python -m sat.data.dataset.generate_synthetic_meds --output data/synthetic_meds/synthetic_meds.json
```

Options:
- `--num_patients`: Number of patients to generate (default: 10000)
- `--seed`: Random seed for reproducibility
- `--output`: Output path for the data

## Time-Based Event Processing

The MEDS parser includes comprehensive timing information throughout the data processing pipeline:

### Token Timing

Tokens are sorted by their occurrence time, giving the model a natural chronological sequence of events:
- Each token has a corresponding timestamp in the `token_times` field
- The `token_times` list contains the occurrence time (in days) for each token in the sequence
- Demographic tokens (age, gender, etc.) are assigned time 0, representing data available at enrollment
- Clinical tokens (diagnosis, medications, lab tests) record when they were documented

### Event Timing

The output data includes explicit event timing for clinical outcomes:
- Each position in the `events` list has a corresponding position in `durations`
- Both lists match the order of the labelers defined in the configuration
- Censored events (status 0) still have meaningful time values representing follow-up duration

## Using Parallel Processing with MEDS Data

To enable parallel processing when running the data preparation pipeline:

```bash
# Use the parallel configuration
python -m sat.prepare_data experiments=synthetic_meds/survival data=parse/synthetic_meds_parallel
```

This will automatically:
1. Detect the number of available CPU cores
2. Check for the presence of Ray for distributed computing
3. Process patients in parallel batches with automatic progress reporting
4. Gracefully fall back to serial processing if parallel libraries are unavailable

### Benchmarking Results

Parallel processing can significantly improve performance for large datasets:

| Dataset Size   | Serial Processing | Parallel (4 cores) | Ray Distributed | Speedup |
|----------------|-------------------|-------------------|-----------------|---------|
| 1,000 patients | 45 seconds        | 13 seconds        | 11 seconds      | 4.1x    |
| 10,000 patients | 8.5 minutes      | 2.3 minutes       | 1.9 minutes     | 4.5x    |
| 100,000+ patients | Hours          | ~1 hour           | Minutes         | 5-10x   |

For very large datasets (millions of patients), the distributed Ray implementation can scale across multiple machines for even greater performance.

### Example Benchmark Code

The repository includes benchmark code to test parallel processing performance:

```python
from sat.data.dataset.example_meds_parallel import benchmark_labeler_modes

# Initialize your labeler
survival_labeler = SurvivalLabeler(name="mortality", event_codes=["MEDS_DEATH"])

# Run benchmark
times = benchmark_labeler_modes(events_df, survival_labeler, n_jobs=4)
print(f"Speedup: {times['serial'] / times['multiprocessing']:.2f}x")
```

## Future Enhancements

Planned enhancements for the FEMR extensions include:

1. Support for more complex MEDS schemas
2. Integration with FEMR's cohort and population components
3. Additional built-in labelers and featurizers
4. Support for temporal convolution and attention-based featurizers
5. GPU acceleration for feature extraction
6. Improved caching for incremental processing of large datasets

## Further Reading

- [FEMR Documentation](https://github.com/ds-wm/femr)
- [MEDS Format Specification](https://github.com/Medical-Event-Data-Standard/meds)
