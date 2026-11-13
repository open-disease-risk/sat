#!/usr/bin/env python3
"""
Test script for generating synthetic MEDS data.
"""

import os

from sat.data.dataset.generate_synthetic_meds import generate_synthetic_meds

# Set parameters
num_patients = 500  # Number of synthetic patients
output_path = "data/synthetic_meds/synthetic_meds.parquet"  # Output path
random_seed = 42  # For reproducibility

print(f"Generating synthetic MEDS data for {num_patients} patients...")
print(f"Output path: {output_path}")

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Generate the data
generate_synthetic_meds(num_patients, output_path, random_seed)

print("Data generation complete!")
