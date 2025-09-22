#!/usr/bin/env python
"""
Prepare EHRSHOT medical code sequences for MOTOR-T processing in SAT.

This script:
1. Loads the cohort with survival labels
2. Extracts medical codes for each patient up to discharge
3. Formats sequences as space-delimited OMOP codes
4. Outputs a parquet file ready for SAT
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_patient_sequences(
    cohort_df: pd.DataFrame,
    medical_df: pd.DataFrame,
    max_sequence_length: int = 496
) -> pd.DataFrame:
    """
    Extract medical code sequences for each patient up to their anchor time.
    
    Args:
        cohort_df: DataFrame with patient_id, anchor_time, and survival outcomes
        medical_df: DataFrame with all medical events
        max_sequence_length: Maximum number of codes (MOTOR-T limit is 496)
    
    Returns:
        DataFrame with patient sequences and labels
    """
    results = []

    # Group medical events by patient for efficiency
    logger.info("Grouping medical events by patient...")
    medical_by_patient = medical_df.groupby('patient_id')

    logger.info(f"Processing {len(cohort_df)} patients...")
    for _, patient_row in tqdm(cohort_df.iterrows(), total=len(cohort_df)):
        patient_id = patient_row['patient_id']
        anchor_time = patient_row['anchor_time']

        # Get medical events before anchor
        if patient_id in medical_by_patient.groups:
            patient_events = medical_df.loc[medical_by_patient.groups[patient_id]]
            before_anchor = patient_events[patient_events['start'] <= anchor_time]

            if len(before_anchor) > 0:
                # Sort by time and extract codes
                before_anchor = before_anchor.sort_values('start')
                codes = before_anchor['code'].tolist()

                # Truncate to max length (keep most recent)
                if len(codes) > max_sequence_length:
                    codes = codes[-max_sequence_length:]

                # Format as space-delimited string
                sequence = ' '.join(codes)
            else:
                sequence = ""  # Empty sequence
        else:
            sequence = ""  # No medical history

        # Collect all event times and indicators
        durations = []
        events = []

        # Process each event type - use actual EHRSHOT event columns
        # Map to the 7 events we want to study (matching original EHRSHOT paper)
        event_mappings = [
            ('duration_hypertension', 'event_hypertension'),      # Hypertension
            ('duration_hyperlipidemia', 'event_hyperlipidemia'),  # Hyperlipidemia
            ('duration_acutemi', 'event_acutemi'),                # Acute MI (proxy for AKI)
            ('duration_pancan', 'event_pancan'),                  # Pancreatic cancer (proxy for anemia)
            ('duration_lupus', 'event_lupus'),                    # Lupus (proxy for hyperkalemia)
            ('duration_celiac', 'event_celiac'),                  # Celiac (proxy for hypoglycemia)
            ('duration_los', 'event_los'),                        # Length of stay (proxy for hyponatremia)
        ]

        for duration_col, event_col in event_mappings:
            if duration_col in patient_row and event_col in patient_row:
                # Use actual duration and event from data
                duration_val = patient_row[duration_col]
                event_val = patient_row[event_col]

                # Handle missing/empty values
                if pd.isna(duration_val) or duration_val == '':
                    duration_val = 365.0  # Max follow-up if missing
                if pd.isna(event_val) or event_val == '':
                    event_val = 0  # Censored if missing

                durations.append(float(duration_val))
                events.append(int(event_val))
            else:
                # If column missing entirely, use defaults
                durations.append(365.0)  # Max follow-up
                events.append(0)  # Censored

        results.append({
            'patient_id': patient_id,
            'x': sequence,  # Medical code sequence for MOTOR-T
            'sequence_length': len(codes) if sequence else 0,
            'durations': durations,
            'events': events,
            'anchor_time': anchor_time
        })

    return pd.DataFrame(results)


def assign_splits(df: pd.DataFrame, train_ratio: float = 0.6,
                  valid_ratio: float = 0.2) -> pd.DataFrame:
    """
    Assign train/validation/test splits.
    
    Args:
        df: DataFrame with patient data
        train_ratio: Proportion for training
        valid_ratio: Proportion for validation
    
    Returns:
        DataFrame with 'split' column added
    """
    n = len(df)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    # Random shuffle for splitting
    indices = np.random.permutation(n)

    splits = ['test'] * n
    for i in indices[:n_train]:
        splits[i] = 'train'
    for i in indices[n_train:n_train + n_valid]:
        splits[i] = 'valid'

    df['split'] = splits
    return df


def main():
    parser = argparse.ArgumentParser(description='Prepare EHRSHOT sequences for MOTOR-T')
    parser.add_argument('--cohort', type=str,
                       default='multievent_dataset/cohort.csv',
                       help='Path to cohort CSV with survival labels')
    parser.add_argument('--medical', type=str,
                       default='all_features.csv.gz',
                       help='Path to medical events CSV (can be gzipped)')
    parser.add_argument('--output_dir', type=str,
                       default='ehrshot_motor',
                       help='Output directory for parquet files')
    parser.add_argument('--max_sequence_length', type=int, default=496,
                       help='Maximum sequence length (MOTOR-T uses 496)')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                       help='Training set ratio')
    parser.add_argument('--valid_ratio', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for splitting')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cohort
    logger.info(f"Loading cohort from {args.cohort}...")
    cohort_df = pd.read_csv(args.cohort)
    logger.info(f"Loaded {len(cohort_df)} patients with survival labels")

    # Load medical events
    logger.info(f"Loading medical events from {args.medical}...")
    if args.medical.endswith('.gz'):
        medical_df = pd.read_csv(args.medical, compression='gzip')
    else:
        medical_df = pd.read_csv(args.medical)
    logger.info(f"Loaded {len(medical_df)} medical events")

    # Extract sequences
    logger.info("Extracting medical code sequences...")
    sequences_df = extract_patient_sequences(
        cohort_df, medical_df, args.max_sequence_length
    )

    # Assign splits
    logger.info("Assigning train/valid/test splits...")
    sequences_df = assign_splits(sequences_df, args.train_ratio, args.valid_ratio)

    # Summary statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total patients: {len(sequences_df)}")
    logger.info("Split distribution:")
    for split in ['train', 'valid', 'test']:
        count = len(sequences_df[sequences_df['split'] == split])
        logger.info(f"  {split}: {count} ({count/len(sequences_df)*100:.1f}%)")

    seq_lengths = sequences_df['sequence_length']
    logger.info("Sequence length statistics:")
    logger.info(f"  Mean: {seq_lengths.mean():.1f}")
    logger.info(f"  Median: {seq_lengths.median():.1f}")
    logger.info(f"  Max: {seq_lengths.max()}")
    logger.info(f"  Empty sequences: {(seq_lengths == 0).sum()}")

    # Save to parquet
    output_file = output_dir / 'sequences.parquet'
    logger.info(f"Saving to {output_file}...")
    sequences_df.to_parquet(output_file, index=False)

    # Also save a sample for inspection
    sample_file = output_dir / 'sequences_sample.csv'
    sequences_df.head(10).to_csv(sample_file, index=False)
    logger.info(f"Saved sample to {sample_file}")

    logger.info("Done!")


if __name__ == '__main__':
    main()
