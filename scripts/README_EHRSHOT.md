# EHRSHOT Data Preparation for SAT

This directory contains scripts for preparing the EHRSHOT dataset for use with the SAT (Survival Analysis Transformer) framework.

## Overview

The EHRSHOT dataset is a comprehensive electronic health record dataset for studying post-discharge complications. It contains:

- **14,251 patients** with survival labels
- **7 post-discharge complications** tracked for up to 365 days
- Medical event sequences formatted as space-delimited OMOP codes
- Compatible with both Motor-T and CLMBR models

## Post-Discharge Events

The dataset tracks these 7 complications that can occur after hospital discharge:

1. **Hypertension** - High blood pressure (17.8% of patients)
2. **Hyperlipidemia** - High cholesterol/blood fats (12.9% of patients)
3. **Acute Myocardial Infarction** - Heart attack (1.7% of patients)
4. **Pancreatic Cancer** - Cancer of the pancreas (0.3% of patients)
5. **Lupus** - Autoimmune disease (1.8% of patients)
6. **Celiac Disease** - Gluten intolerance/autoimmune digestive disorder (0.9% of patients)
7. **Length of Stay** - Extended hospital stay above threshold (1.0% of patients)

*Note: Some events serve as proxies for other conditions from the original EHRSHOT paper.*

## Prerequisites

### Required Data Files

You need the EHRSHOT_ASSETS directory with:
```
~/Downloads/EHRSHOT_ASSETS/
├── benchmark/
│   ├── multievent_dataset/
│   │   └── cohort.csv                # Patient cohort with survival labels
│   └── all_features.csv.gz           # Medical events data
└── prepare_ehrshot_sequences.py      # Data preparation script
```

### Python Environment

Ensure you have the required dependencies:
```bash
pip install pandas numpy tqdm pyarrow
```

## Usage

### Basic Data Preparation

```bash
cd ~/Downloads/EHRSHOT_ASSETS/benchmark
python prepare_ehrshot_sequences.py \
    --cohort multievent_dataset/cohort.csv \
    --medical all_features.csv.gz \
    --output_dir ehrshot_motor_prepared \
    --max_sequence_length 496
```

### Using the SAT Scripts Directory

From the SAT repository root:

```bash
# Copy script to current directory or run directly
python scripts/prepare_ehrshot_data.py \
    --cohort ~/Downloads/EHRSHOT_ASSETS/benchmark/multievent_dataset/cohort.csv \
    --medical ~/Downloads/EHRSHOT_ASSETS/benchmark/all_features.csv.gz \
    --output_dir data/ehrshot_motor \
    --max_sequence_length 496
```

### Full Parameter Options

```bash
python scripts/prepare_ehrshot_data.py \
    --cohort path/to/cohort.csv \
    --medical path/to/all_features.csv.gz \
    --output_dir ehrshot_motor \
    --max_sequence_length 496 \
    --train_ratio 0.6 \
    --valid_ratio 0.2 \
    --seed 42
```

## Parameters

- `--cohort`: Path to cohort CSV with patient IDs, anchor times, and survival outcomes
- `--medical`: Path to medical events CSV (can be gzipped)
- `--output_dir`: Output directory for processed parquet files
- `--max_sequence_length`: Maximum sequence length (496 for Motor-T compatibility)
- `--train_ratio`: Training set proportion (default: 0.6)
- `--valid_ratio`: Validation set proportion (default: 0.2)
- `--seed`: Random seed for reproducible splits (default: 42)

## Output Files

The script generates:

- `sequences.parquet`: Main dataset file with processed sequences and labels
- `sequences_sample.csv`: Sample of 10 rows for inspection
- Processing logs with dataset statistics

### Output Format

Each row contains:
- `patient_id`: Unique patient identifier
- `x`: Medical code sequence (space-delimited OMOP codes)
- `sequence_length`: Number of codes in sequence
- `durations`: Array of time-to-event for each of the 7 outcomes
- `events`: Binary array indicating event occurrence (1) or censoring (0)
- `anchor_time`: Reference time point (discharge)
- `split`: Data split assignment (train/valid/test)

## Important Column Mapping Fix

This script contains a **critical fix** for the original EHRSHOT data preparation. The original script expected column names like `hypertension_time`, but the actual data uses:

```python
# Correct mapping in this script:
event_mappings = [
    ('duration_hypertension', 'event_hypertension'),      # Hypertension
    ('duration_hyperlipidemia', 'event_hyperlipidemia'),  # Hyperlipidemia
    ('duration_acutemi', 'event_acutemi'),                # Acute MI
    ('duration_pancan', 'event_pancan'),                  # Pancreatic cancer
    ('duration_lupus', 'event_lupus'),                    # Lupus
    ('duration_celiac', 'event_celiac'),                  # Celiac disease
    ('duration_los', 'event_los'),                        # Length of stay
]
```

This fix ensures proper extraction of event indicators, resolving the issue where all samples appeared censored.

## Data Validation

The script provides comprehensive statistics:

```
Dataset Statistics:
Total patients: 14251
Split distribution:
  train: 8550 (60.0%)
  valid: 2850 (20.0%)
  test: 2851 (20.0%)

Sequence length statistics:
  Mean: 245.3
  Median: 198.0
  Max: 496
  Empty sequences: 156

Event distribution:
  hypertension: 2538 events (17.8%)
  hyperlipidemia: 1840 events (12.9%)
  acutemi: 240 events (1.7%)
  pancan: 42 events (0.3%)
  lupus: 257 events (1.8%)
  celiac: 123 events (0.9%)
  los: 142 events (1.0%)
```

## Integration with SAT

After preparation, use the data with SAT experiments:

### Motor-T Experiments
```bash
poetry run python -m sat.finetune experiments=ehrshot_motor/survival
poetry run python -m sat.finetune experiments=ehrshot_motor/mtl
```

### CLMBR Experiments
```bash
poetry run python -m sat.finetune experiments=ehrshot_clmbr/survival
poetry run python -m sat.finetune experiments=ehrshot_clmbr/mtl
```

## Troubleshooting

### Common Issues

1. **Missing cohort file**: Ensure `multievent_dataset/cohort.csv` exists
2. **Gzip compression**: Medical events file should be compressed (`.gz` extension)
3. **Memory issues**: Large dataset may require sufficient RAM (8GB+ recommended)
4. **Column mismatch**: This script fixes the column naming issue automatically

### Error: "All samples appear censored"

This was the original issue that led to zero importance weights. The current script fixes this by correctly mapping the event columns. If you still see this error, verify:
- The cohort file has the expected columns
- Medical events data is properly formatted
- Event mappings in the script match your data structure

## Notes

- Missing event data is handled by treating patients as censored (event=0)
- Sequences are truncated to `max_sequence_length` keeping the most recent events
- The script maintains temporal ordering of medical events
- Empty sequences are allowed and handled appropriately by the model