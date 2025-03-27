"""Tests for the DSMTaskHead and DSMLoss classes"""

import os
import tempfile
import pytest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from transformers import TrainingArguments

from sat.models.heads.dsm import DSMConfig, DSMTaskHead
from sat.models.parameter_nets import (
    ParamCauseSpecificNet,
    ParamCauseSpecificNetCompRisk,
)
from sat.loss.survival.dsm import DSMLoss
from sat.models.heads.output import SAOutput
from sat.transformers.trainer import SATTrainer, TrainingArgumentsWithMPSSupport


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


def test_parameter_nets():
    """Test the parameter networks directly"""
    # Test ParamCauseSpecificNet (single-event)
    single_net = ParamCauseSpecificNet(
        in_features=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_mixtures=4,
        num_events=1,
    )

    # Generate random input
    batch_size = 8
    input_tensor = torch.randn(batch_size, 32)

    # Forward pass
    shape, scale, logits_g = single_net(input_tensor)

    # Check shape
    assert shape.shape == (
        batch_size,
        1,
        4,
    ), f"Expected shape (8, 1, 4), got {shape.shape}"
    assert scale.shape == (
        batch_size,
        1,
        4,
    ), f"Expected shape (8, 1, 4), got {scale.shape}"
    assert logits_g.shape == (
        batch_size,
        1,
        4,
    ), f"Expected shape (8, 1, 4), got {logits_g.shape}"

    # Test ParamCauseSpecificNetCompRisk (multi-event)
    multi_net = ParamCauseSpecificNetCompRisk(
        in_features=32,
        shared_intermediate_size=64,
        shared_num_hidden_layers=2,
        indiv_intermediate_size=32,
        indiv_num_hidden_layers=1,
        num_mixtures=4,
        num_events=2,
    )

    # Forward pass
    shape, scale, logits_g = multi_net(input_tensor)

    # Check shape
    assert shape.shape == (
        batch_size,
        2,
        4,
    ), f"Expected shape (8, 2, 4), got {shape.shape}"
    assert scale.shape == (
        batch_size,
        2,
        4,
    ), f"Expected shape (8, 2, 4), got {scale.shape}"
    assert logits_g.shape == (
        batch_size,
        2,
        4,
    ), f"Expected shape (8, 2, 4), got {logits_g.shape}"


def test_dsm_task_head_single_event():
    """Test DSMTaskHead for single-event case"""
    # Create config
    config = DSMConfig(
        num_features=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_mixtures=4,
        num_events=1,
        num_labels=20,  # Number of time points for prediction
    )

    # Create model
    model = DSMTaskHead(config)

    # Generate random input
    batch_size = 8
    input_tensor = torch.randn(batch_size, 32)

    # Forward pass
    output = model(input_tensor)

    # Check output fields
    assert output.hazard is not None, "Hazard output is None"
    assert output.risk is not None, "Risk output is None"
    assert output.survival is not None, "Survival output is None"
    assert output.shape is not None, "Shape output is None"
    assert output.scale is not None, "Scale output is None"
    assert output.logits_g is not None, "Logits_g output is None"
    assert output.logits is not None, "Logits output is None"

    # Check output shapes
    assert (
        output.hazard.shape[0] == batch_size
    ), f"Batch dimension mismatch: {output.hazard.shape}"
    assert (
        output.hazard.shape[1] == 1
    ), f"Event dimension mismatch: {output.hazard.shape}"
    assert (
        output.survival.shape[1] == 1
    ), f"Event dimension mismatch: {output.survival.shape}"
    # Verify hazard and survival shapes match
    assert (
        output.hazard.shape[2] == output.survival.shape[2]
    ), f"Hazard and survival time dimensions don't match: {output.hazard.shape} vs {output.survival.shape}"
    assert output.shape.shape == (
        batch_size,
        1,
        4,
    ), f"Shape dimension mismatch: {output.shape.shape}"
    assert output.scale.shape == (
        batch_size,
        1,
        4,
    ), f"Scale dimension mismatch: {output.scale.shape}"
    assert output.logits_g.shape == (
        batch_size,
        1,
        4,
    ), f"Logits_g dimension mismatch: {output.logits_g.shape}"


def test_dsm_task_head_multi_event():
    """Test DSMTaskHead for multi-event case"""
    # Create config
    config = DSMConfig(
        num_features=32,
        intermediate_size=64,
        num_hidden_layers=2,
        indiv_intermediate_size=32,
        indiv_num_hidden_layers=1,
        num_mixtures=4,
        num_events=2,
        num_labels=20,  # Number of time points for prediction
    )

    # Create model
    model = DSMTaskHead(config)

    # Generate random input
    batch_size = 8
    input_tensor = torch.randn(batch_size, 32)

    # Forward pass
    output = model(input_tensor)

    # Check output fields
    assert output.hazard is not None, "Hazard output is None"
    assert output.risk is not None, "Risk output is None"
    assert output.survival is not None, "Survival output is None"
    assert output.shape is not None, "Shape output is None"
    assert output.scale is not None, "Scale output is None"
    assert output.logits_g is not None, "Logits_g output is None"
    assert output.logits is not None, "Logits output is None"

    # Check output shapes
    assert (
        output.hazard.shape[0] == batch_size
    ), f"Batch dimension mismatch: {output.hazard.shape}"
    assert (
        output.hazard.shape[1] == 2
    ), f"Event dimension mismatch: {output.hazard.shape}"
    assert (
        output.survival.shape[1] == 2
    ), f"Event dimension mismatch: {output.survival.shape}"
    # Verify hazard and survival shapes match
    assert (
        output.hazard.shape[2] == output.survival.shape[2]
    ), f"Hazard and survival time dimensions don't match: {output.hazard.shape} vs {output.survival.shape}"
    assert output.shape.shape == (
        batch_size,
        2,
        4,
    ), f"Shape dimension mismatch: {output.shape.shape}"
    assert output.scale.shape == (
        batch_size,
        2,
        4,
    ), f"Scale dimension mismatch: {output.scale.shape}"
    assert output.logits_g.shape == (
        batch_size,
        2,
        4,
    ), f"Logits_g dimension mismatch: {output.logits_g.shape}"


def test_dsm_loss_initialization(temp_csv_files):
    """Test DSMLoss initialization"""
    # Create loss
    loss = DSMLoss(
        duration_cuts=temp_csv_files["duration_cuts"],
        importance_sample_weights=temp_csv_files["imp_weights"],
        num_events=2,
        distribution="weibull",
        discount=1.0,
        elbo=False,
    )

    # Check parameters
    assert loss.num_events == 2, f"Expected num_events=2, got {loss.num_events}"
    assert (
        loss.distribution == "weibull"
    ), f"Expected distribution='weibull', got {loss.distribution}"
    assert loss.discount == 1.0, f"Expected discount=1.0, got {loss.discount}"
    assert loss.elbo is False, f"Expected elbo=False, got {loss.elbo}"
    assert hasattr(loss, "duration_cuts"), "Missing duration_cuts attribute"
    assert hasattr(loss, "weights"), "Missing weights attribute"
    assert len(loss.weights) == 3, f"Expected 3 weights, got {len(loss.weights)}"


def test_dsm_loss_computation(temp_csv_files):
    """Test DSMLoss forward computation"""
    # Create loss
    loss_fn = DSMLoss(
        duration_cuts=temp_csv_files["duration_cuts"],
        importance_sample_weights=temp_csv_files["imp_weights"],
        num_events=2,
        distribution="weibull",
        discount=1.0,
        elbo=False,
    )

    # Create mock SAOutput
    batch_size = 8
    num_events = 2
    num_mixtures = 4

    # Create a batch of references (event indicators, durations, etc.)
    # Format: [percentiles, event indicators, fractions, durations] * num_events
    references = torch.zeros(batch_size, 4 * num_events)

    # Set event indicators and durations
    # Half the batch has event 0, half has event 1
    references[: batch_size // 2, num_events + 0] = 1  # Event 0 indicators
    references[batch_size // 2 :, num_events + 1] = 1  # Event 1 indicators

    # Set durations - between 1 and 5
    references[:, 3 * num_events :] = torch.rand(batch_size, num_events) * 4 + 1

    # Create mock DSMTaskHead outputs
    shape = torch.ones(batch_size, num_events, num_mixtures) * 1.5  # Shape > 0
    scale = torch.ones(batch_size, num_events, num_mixtures) * 2.0  # Scale > 0
    logits_g = torch.zeros(
        batch_size, num_events, num_mixtures
    )  # Equal mixture weights

    # Time points
    time_points = (
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(0).expand(batch_size, -1)
    )

    # Compute survival and hazard functions
    survival = torch.zeros(batch_size, num_events, time_points.shape[1])
    # Simple deterministic survival function for testing
    for i in range(time_points.shape[1]):
        # Survival decreases over time
        survival[:, :, i] = 1.0 - i * 0.2

    # Create hazard with zero in first position to match survival dimensions
    hazard = torch.zeros(batch_size, num_events, time_points.shape[1])
    hazard[:, :, 1:] = (
        torch.ones(batch_size, num_events, time_points.shape[1] - 1) * 0.2
    )
    risk = 1.0 - survival

    # Create output container
    output = SAOutput(
        loss=None,
        logits=torch.ones_like(survival),
        hazard=hazard,
        risk=risk,
        survival=survival,
        hidden_states=None,
        shape=shape,
        scale=scale,
        logits_g=logits_g,
    )

    # Compute loss
    loss_value = loss_fn(output, references)

    # Check loss is a scalar tensor with gradient
    assert isinstance(
        loss_value, torch.Tensor
    ), f"Loss is not a tensor: {type(loss_value)}"
    assert loss_value.shape == torch.Size(
        []
    ), f"Loss is not a scalar: {loss_value.shape}"
    assert loss_value.requires_grad, "Loss does not have requires_grad=True"
    assert not torch.isnan(loss_value), "Loss is NaN"
    assert not torch.isinf(loss_value), "Loss is Inf"
    assert loss_value > 0, "Loss should be positive"


def test_end_to_end_dsm(temp_csv_files):
    """Test DSMTaskHead and DSMLoss together in an end-to-end scenario"""
    # Create model
    config = DSMConfig(
        num_features=32,
        intermediate_size=64,
        num_hidden_layers=2,
        indiv_intermediate_size=32,
        indiv_num_hidden_layers=1,
        num_mixtures=4,
        num_events=2,
        num_labels=20,  # Number of time points for prediction
    )

    model = DSMTaskHead(config)

    # Create loss function
    loss_fn = DSMLoss(
        duration_cuts=temp_csv_files["duration_cuts"],
        importance_sample_weights=temp_csv_files["imp_weights"],
        num_events=2,
        distribution="weibull",
        discount=1.0,
        elbo=False,
    )

    # Manually set the loss function to the model
    model.loss = loss_fn

    # Create random input and labels
    batch_size = 8
    num_events = 2
    features = torch.randn(batch_size, 32)

    # Create sample labels
    references = torch.zeros(batch_size, 4 * num_events)

    # Set event indicators and durations
    # Half the batch has event 0, half has event 1
    references[: batch_size // 2, num_events + 0] = 1  # Event 0 indicators
    references[batch_size // 2 :, num_events + 1] = 1  # Event 1 indicators

    # Set durations - between 1 and 5
    references[:, 3 * num_events :] = torch.rand(batch_size, num_events) * 4 + 1

    # Forward pass
    output = model(features, references)

    # Check output
    assert output.loss is not None, "Loss is None"
    assert not torch.isnan(output.loss), "Loss is NaN"
    assert not torch.isinf(output.loss), "Loss is Inf"
    assert output.loss.requires_grad, "Loss does not have requires_grad=True"

    # Check that loss computes correctly through backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    output.loss.backward()
    optimizer.step()

    # Check that model can run a second forward pass
    output2 = model(features, references)
    assert output2.loss is not None, "Loss is None after optimization step"
    assert (
        output2.loss < output.loss * 1.1
    ), "Loss increased significantly after optimization"


def test_dsm_with_sat_output():
    """Test that DSMTaskHead correctly creates SAOutput with proper shapes"""
    # Create model
    config = DSMConfig(
        num_features=32,
        intermediate_size=64,
        num_hidden_layers=2,
        indiv_intermediate_size=32,
        indiv_num_hidden_layers=1,
        num_mixtures=4,
        num_events=2,
        num_labels=20,  # Number of time points for prediction
    )

    # Create model with config.num_events set
    model = DSMTaskHead(config)

    # Generate random input
    batch_size = 8
    features = torch.randn(batch_size, 32)

    # Forward pass
    output = model(features)

    # Check that the output has all the fields expected by the SATTrainer prediction_step
    assert isinstance(output, SAOutput), "Output is not an SAOutput instance"
    assert output.hazard is not None, "Hazard output is None"
    assert output.risk is not None, "Risk output is None"
    assert output.survival is not None, "Survival output is None"
    assert output.shape is not None, "Shape parameter is None"
    assert output.scale is not None, "Scale parameter is None"
    assert output.logits_g is not None, "Mixture weights are None"

    # Check that hazard tensor has the expected format with padding
    # If the SATTrainer.prediction_step is fixed, it should handle this format correctly
    assert (
        output.hazard.shape[0] == batch_size
    ), f"Batch dimension mismatch: {output.hazard.shape[0]}"
    assert (
        output.hazard.shape[1] == config.num_events
    ), f"Event dimension mismatch: {output.hazard.shape[1]}"

    # The hazard tensor should be properly padded to match the survival tensor's time dimension
    assert output.hazard.shape[2] == output.survival.shape[2], (
        f"Hazard and survival time dimensions should match after padding: "
        f"{output.hazard.shape[2]} vs {output.survival.shape[2]}"
    )

    # The first column of hazard should be zeros (padding)
    assert torch.all(
        output.hazard[:, :, 0] == 0
    ), "First column of hazard should be zero padding"

    # Check parameter shapes
    assert output.shape.shape == (
        batch_size,
        config.num_events,
        config.num_mixtures,
    ), f"Shape parameter dimension mismatch: {output.shape.shape}"
    assert output.scale.shape == (
        batch_size,
        config.num_events,
        config.num_mixtures,
    ), f"Scale parameter dimension mismatch: {output.scale.shape}"
    assert output.logits_g.shape == (
        batch_size,
        config.num_events,
        config.num_mixtures,
    ), f"Mixture weights dimension mismatch: {output.logits_g.shape}"
