# MEDS Format Support in SAT

This document explains how to use the Medical Event Data Standard (MEDS) format with SAT for survival analysis.

## Overview

The Medical Event Data Standard (MEDS) is a standardized format for healthcare data designed to facilitate portable analyses across different datasets. SAT now supports MEDS-formatted data, allowing you to leverage existing MEDS datasets for survival analysis.

## Requirements

To use MEDS format with SAT, you need the following additional dependencies:

- `pyarrow`: For reading Parquet files
- `fastparquet`: Alternative Parquet engine
- `femr` (optional): Python package for advanced MEDS processing

These dependencies are automatically included in the project configuration.

## New Features in MEDS Support

The latest MEDS parser implementation includes:

1. **Varying Length Histories**: Support for patients with varying lengths of medical histories based on their event times.
2. **Modality Vectors**: Each feature is classified as either categorical (0) or numerical (1).
3. **Numerics Vectors**: Standardized numeric values for features (1.0 for categorical features).
4. **Events List**: For multi-event analysis, an events list indicates which events occurred for each patient.
5. **Durations List**: Corresponding time-to-event values for each event type.
6. **Competing Risks**: Full support for competing risks with multiple event types.

## MEDS Format Structure

MEDS data is typically stored in Parquet format with a standardized schema:

- `patients`: Core patient information table
- Additional tables for different medical event types (e.g., `mortality`, `hospitalizations`)
- Standardized columns for patient IDs, timestamps, and event details

The synthetic MEDS generator creates multiple Parquet files:
- `synthetic_meds_patients.parquet`: Patient demographic and clinical features
- `synthetic_meds_mortality.parquet`: Mortality events with timestamps
- `synthetic_meds_hospitalizations.parquet`: Hospitalization events with timestamps
- And others like `diagnoses`, `medications`, etc.
- `synthetic_meds_metadata.json`: Information about the dataset tables

## Using FEMR for MEDS Data

SAT uses the FEMR package to process MEDS data through label definitions:

```yaml
label_definitions:
  - name: mortality
    positive_class: true
    table_name: mortality
    time_field: days
  - name: hospitalization
    positive_class: true
    table_name: hospitalizations
    time_field: days
```

This approach leverages FEMR's labeling system to:
1. Extract events from the MEDS data
2. Transform them into standardized survival format
3. Support competing risks with multiple event types

Each label definition specifies:
- `name`: An identifier for the event type
- `positive_class`: Whether this is a positive event (true) or censoring (false)
- `table_name`: The table in the MEDS dataset containing this event
- `time_field`: The field containing time-to-event values

## Configuration

Create or modify a YAML configuration file at `conf/data/parse/your_dataset.yaml`:

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
# Event definitions for competing risks with FEMR
label_definitions:
  - name: mortality
    positive_class: true
    table_name: mortality
    time_field: days
  - name: hospitalization
    positive_class: true
    table_name: hospitalizations
    time_field: days
```

## Processing MEDS Data

The standard SAT workflow applies to MEDS data:

1. **Prepare Data**:
   ```bash
   python -m sat.prepare_data experiments=synthetic_meds/survival data_source=/path/to/meds_dir dataset=my_meds
   ```

2. **Train Tokenizer**:
   ```bash
   python -m sat.train_tokenizer experiments=synthetic_meds/survival dataset=my_meds
   ```

3. **Train Label Transform**:
   ```bash
   python -m sat.train_labeltransform experiments=synthetic_meds/survival dataset=my_meds
   ```

4. **Fine-tune Model**:
   ```bash
   python -m sat.finetune experiments=synthetic_meds/survival dataset=my_meds
   ```

## Handling Multiple Event Types (Competing Risks)

The MEDS parser automatically handles multiple event types as competing risks:

1. Each event type (mortality, hospitalization, etc.) is assigned a unique integer code (starting from 1)
2. Code 0 is reserved for censored observations
3. Events are processed to create a proper time-to-event dataset with:
   - `events`: List of binary indicators for each event type (e.g., [0, 1, 0] means event type 2 occurred)
   - `durations`: List of times for each event type (e.g., [30, 25, 30] means event type 2 occurred at day 25)
   
For a patient with 3 possible event types who experienced event type 2 at day 100:
- `events` would be [0, 1, 0] (indicating event type 2 occurred)
- `durations` would be [100, 100, 100] (the time at which observation ended for all event types)

For a censored patient at day 50 with 3 possible event types:
- `events` would be [0, 0, 0] (no events occurred)
- `durations` would be [50, 50, 50] (censored at day 50 for all event types)

## Generating Synthetic MEDS Data

The provided synthetic MEDS data generator creates realistic healthcare data for testing:

```bash
python -m sat.data.dataset.generate_synthetic_meds --output data/synthetic_meds/synthetic_meds.parquet
```

Options:
- `--num_patients`: Number of patients to generate (default: 10000)
- `--seed`: Random seed for reproducibility
- `--output`: Output path for the data

## Troubleshooting

Common issues and solutions:

- **Missing tables**: Check your MEDS data structure to ensure required tables are present
- **Time field not found**: Set the correct `time_field` in your configuration
- **FEMR import errors**: The parser will automatically fall back to direct table processing

## Further Reading

- [FEMR Documentation](https://github.com/som-shahlab/femr)
- [MEDS Format Specification](https://github.com/som-shahlab/meds-format)