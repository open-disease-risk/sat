"""Test MEDS parser functionality"""

import logging
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from sat.data.dataset.generate_synthetic_meds import generate_synthetic_meds
from sat.data.dataset.parse_meds import meds

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Mock FEMR and its behavior for testing
class MockDataset:
    def get_dataframe(self, tables):
        # Create a mock patients dataframe
        df = pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3", "p4", "p5"],
                "age": [65, 55, 70, 45, 60],
                "gender": ["M", "F", "M", "F", "M"],
                "condition": [
                    "diabetes",
                    "hypertension",
                    "diabetes",
                    "cancer",
                    "hypertension",
                ],
            }
        )
        return df


class MockLabeler:
    def get_labels(self, label_def):
        # Create mock labeled data based on the label definition
        if label_def["name"] == "mortality":
            return pd.DataFrame(
                {
                    "patient_id": ["p1", "p3", "p5"],
                    "days": [120, 90, 365],
                    "label_name": ["mortality", "mortality", "mortality"],
                    "label_value": [1, 1, 1],
                }
            )
        elif label_def["name"] == "readmission":
            return pd.DataFrame(
                {
                    "patient_id": ["p2", "p4"],
                    "days": [30, 60],
                    "label_name": ["readmission", "readmission"],
                    "label_value": [1, 1],
                }
            )
        return pd.DataFrame()


class MockBuilder:
    def build(self):
        return MockDataset()


# Mock the femr imports
pytest.importorskip("femr")


# Patch the femr modules for testing
def mock_femr_import(monkeypatch):
    # We're not using FEMR directly in the new implementation
    pass


def test_meds_parser_with_mock(monkeypatch):
    """Test that the MEDS parser correctly processes data using mocks."""
    # Mock FEMR imports
    mock_femr_import(monkeypatch)

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configure the MEDS parser
        parser = meds(
            source="dummy_path.parquet",  # Path doesn't matter for the mock
            processed_dir=temp_dir,
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2,
            n_bins=10,
            encode="ordinal",
            strategy="quantile",
            name="test_meds",
            scale_numerics=True,
            scale_method="standard",
            label_definitions=[
                {
                    "name": "mortality",
                    "positive_class": True,
                    "table_name": "mortality",
                    "time_field": "days",
                    "output_label_fields": ["days", "label_name"],
                },
                {
                    "name": "readmission",
                    "positive_class": True,
                    "table_name": "readmissions",
                    "time_field": "days",
                    "output_label_fields": ["days", "label_name"],
                },
            ],
            time_field="days",
        )

        # Run the parser
        parser.prepare()

        # Check that the output file exists
        output_file = Path(f"{temp_dir}/test_meds/test_meds.json")
        assert output_file.exists(), f"Output file {output_file} was not created"

        # Load and verify the output
        df = pd.read_json(output_file, lines=True)

        # Check basic properties of the processed data
        assert "x" in df.columns, "Features column 'x' missing"
        assert "split" in df.columns, "Split column missing"
        assert "modality" in df.columns, "Modality column missing"
        assert "numerics" in df.columns, "Numerics column missing"
        assert "events" in df.columns, "Events list column missing"
        assert "durations" in df.columns, "Durations list column missing"

        # Verify that we have events as lists
        if len(df) > 0:
            assert isinstance(df["events"].iloc[0], list), "Events should be a list"
            assert isinstance(
                df["durations"].iloc[0], list
            ), "Durations should be a list"

        # Verify split distribution
        split_counts = df["split"].value_counts(normalize=True)
        assert (
            abs(split_counts["train"] - 0.6) < 0.2
        ), "Train ratio not approximately correct"
        assert (
            abs(split_counts["valid"] - 0.2) < 0.1
        ), "Validation ratio not approximately correct"
        assert (
            abs(split_counts["test"] - 0.2) < 0.1
        ), "Test ratio not approximately correct"

        # Check the metadata file
        metadata_file = Path(f"{temp_dir}/test_meds/test_meds_metadata.json")
        assert metadata_file.exists(), f"Metadata file {metadata_file} was not created"

        metadata = pd.read_json(metadata_file)
        assert (
            "event_types" in metadata.columns
        ), "Metadata missing event types information"
        assert "feature_count" in metadata.columns, "Metadata missing feature count"
        assert (
            "categorical_features" in metadata.columns
        ), "Metadata missing categorical features"
        assert (
            "numerical_features" in metadata.columns
        ), "Metadata missing numerical features"


def test_meds_parser_with_synthetic_data():
    """Test the MEDS parser with generated synthetic data."""
    # Create temporary directories for testing
    with (
        tempfile.TemporaryDirectory() as data_dir,
        tempfile.TemporaryDirectory() as output_dir,
    ):
        data_path = Path(data_dir)
        output_path = Path(output_dir)

        # Generate synthetic MEDS data
        logger.info("Generating synthetic MEDS data for testing...")
        generate_synthetic_meds(
            num_patients=50,
            output_path=str(data_path / "synthetic_meds.parquet"),
            seed=42,
        )

        # Configure the MEDS parser
        parser = meds(
            source=str(data_path),
            processed_dir=str(output_path),
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15,
            n_bins=20,
            encode="ordinal",
            strategy="quantile",
            name="synthetic_test",
            scale_numerics=True,
            scale_method="standard",
            label_definitions=[
                {
                    "name": "mortality",
                    "positive_class": True,
                    "table_name": "mortality",
                    "time_field": "days",
                },
                {
                    "name": "hospitalization",
                    "positive_class": True,
                    "table_name": "hospitalizations",
                    "time_field": "days",
                },
            ],
        )

        # Parse the data
        logger.info("Running MEDS parser on synthetic data...")
        parser.prepare()

        # Check the output
        output_file = output_path / "synthetic_test" / "synthetic_test.json"
        assert output_file.exists(), f"Output file not created: {output_file}"

        # Load and validate the output
        df = pd.read_json(output_file, lines=True)

        # Check required columns
        required_columns = ["x", "modality", "numerics", "events", "durations", "split"]
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

        # Check modality and numerics format
        assert isinstance(df["modality"].iloc[0], list), "Modality should be a list"
        assert all(
            x in [0, 1] for x in df["modality"].iloc[0]
        ), "Modality should contain only 0s and 1s"

        assert isinstance(df["numerics"].iloc[0], list), "Numerics should be a list"
        assert all(
            isinstance(x, (int, float)) for x in df["numerics"].iloc[0]
        ), "Numerics should contain numeric values"

        # Check events format
        assert isinstance(df["events"].iloc[0], list), "Events should be a list"
        assert all(
            x in [0, 1] for x in df["events"].iloc[0]
        ), "Events should contain only 0s and 1s"

        # Check durations format
        assert isinstance(df["durations"].iloc[0], list), "Durations should be a list"
        assert all(
            isinstance(x, (int, float)) for x in df["durations"].iloc[0]
        ), "Durations should contain numeric values"

        # Check data shape requirements
        assert len(df["modality"].iloc[0]) == len(
            df["numerics"].iloc[0]
        ), "Modality and numerics should have the same length"
        assert len(df["events"].iloc[0]) == len(
            df["durations"].iloc[0]
        ), "Events and durations should have the same length"

        # Check split distribution
        assert set(df["split"].unique()) == {
            "train",
            "valid",
            "test",
        }, "Missing data splits"

        logger.info("MEDS parser test with synthetic data completed successfully")
