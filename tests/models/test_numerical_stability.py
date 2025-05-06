"""Tests for numerical stability of survival models."""

import os
import tempfile

import pandas as pd
import pytest
import torch

from sat.loss.survival.dsm import DSMLoss
from sat.loss.survival.mensa import MENSALoss
from sat.models.heads.dsm import DSMConfig, DSMTaskHead
from sat.models.heads.mensa import MENSAConfig, MENSATaskHead


# Create temporary files needed for tests
@pytest.fixture
def temp_csv_files():
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create duration cuts file
        duration_cuts_path = os.path.join(tmp_dir, "duration_cuts.csv")
        duration_cuts = pd.DataFrame([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        duration_cuts.to_csv(duration_cuts_path, index=False, header=False)

        # Create importance sampling weights file
        imp_weights_path = os.path.join(tmp_dir, "imp_sample.csv")
        imp_weights = pd.DataFrame(
            [1.0, 1.0, 1.0]
        )  # One weight for censoring, two for events
        imp_weights.to_csv(imp_weights_path, index=False, header=False)

        yield {"duration_cuts": duration_cuts_path, "imp_weights": imp_weights_path}


def test_dsm_numerical_stability(temp_csv_files):
    """Test DSM with extreme parameter values to check for numerical stability."""
    # Create model with config
    config = DSMConfig(
        num_features=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_mixtures=4,
        num_events=2,
        num_labels=20,
    )

    model = DSMTaskHead(config)

    # Create test inputs - normal, small, and extreme values
    batch_size = 8
    num_events = 2
    # num_mixtures = 4  # Unused variable

    # Create input features
    features = torch.randn(batch_size, 32)

    # Create reference labels with extreme durations
    references = torch.zeros(batch_size, 4 * num_events)
    references[:, num_events : 2 * num_events] = 1  # All samples have events

    # Set some extreme durations (very small and very large)
    references[0, 3 * num_events] = 1e-10  # Very small duration for event 0
    references[1, 3 * num_events + 1] = 1e10  # Very large duration for event 1
    references[2:, 3 * num_events :] = (
        torch.rand(batch_size - 2, num_events) * 10
    )  # Normal durations

    # Create loss
    loss_fn = DSMLoss(
        duration_cuts=temp_csv_files["duration_cuts"],
        importance_sample_weights=temp_csv_files["imp_weights"],
        num_events=2,
        distribution="weibull",
    )

    model.loss = loss_fn

    # Forward pass
    output = model(features, references)

    # Check for numerical stability issues
    assert torch.isfinite(output.loss), "Loss is not finite"
    assert not torch.isnan(output.loss), "Loss contains NaN values"
    assert not torch.isinf(output.loss), "Loss contains Inf values"

    # Check that survival values are valid probabilities
    assert torch.all(output.survival >= 0), "Survival contains negative values"
    assert torch.all(output.survival <= 1), "Survival contains values greater than 1"

    # Backward pass should not produce NaN gradients
    output.loss.backward()

    # Check parameters for NaN gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"


def test_mensa_numerical_stability(temp_csv_files):
    """Test MENSA with extreme parameter values to check for numerical stability."""
    # Create model with config
    config = MENSAConfig(
        num_features=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_mixtures=4,
        num_events=2,
        num_labels=20,
        event_dependency=True,
    )

    # Add loss configuration to avoid KeyError: 'survival'
    config.loss = {
        "survival": {
            "_target_": "sat.loss.survival.mensa.MENSALoss",
            "duration_cuts": temp_csv_files["duration_cuts"],
            "importance_sample_weights": temp_csv_files["imp_weights"],
            "num_events": 2,
            "distribution": "weibull",
            "dependency_regularization": 0.01,
        }
    }

    model = MENSATaskHead(config)

    # Create test inputs - use less extreme values for this test
    batch_size = 8
    num_events = 2
    # num_mixtures = 4  # Unused variable

    # Create input features
    features = torch.randn(batch_size, 32)

    # Create reference labels with less extreme durations
    references = torch.zeros(batch_size, 4 * num_events)
    references[:, num_events : 2 * num_events] = 1  # All samples have events

    # Set durations that are extreme but within reasonable limits
    references[0, 3 * num_events] = (
        1e-5  # Small but not too extreme duration for event 0
    )
    references[1, 3 * num_events + 1] = (
        1e5  # Large but not too extreme duration for event 1
    )
    references[2:, 3 * num_events :] = (
        torch.rand(batch_size - 2, num_events) * 10
    )  # Normal durations

    # Create loss
    loss_fn = MENSALoss(
        duration_cuts=temp_csv_files["duration_cuts"],
        importance_sample_weights=temp_csv_files["imp_weights"],
        num_events=2,
        distribution="weibull",
        dependency_regularization=0.01,
    )

    model.loss = loss_fn

    # Forward pass
    output = model(features, references)

    # Debug print to see what values we're getting
    print(f"Loss value: {output.loss.item()}")

    # Check for numerical stability issues
    assert torch.isfinite(output.loss), "Loss is not finite"
    assert not torch.isnan(output.loss), "Loss contains NaN values"
    assert not torch.isinf(output.loss), "Loss contains Inf values"

    # Check that survival values are valid probabilities
    assert torch.all(output.survival >= 0), "Survival contains negative values"
    assert torch.all(output.survival <= 1), "Survival contains values greater than 1"

    # Backward pass should not produce NaN gradients
    output.loss.backward()

    # Check parameters for NaN gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
