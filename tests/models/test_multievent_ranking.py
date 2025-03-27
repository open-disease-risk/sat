"""Tests for MultiEventRankingLoss behavior on HSA synthetic dataset."""

import os
import tempfile
import pandas as pd
import torch
import pytest
import numpy as np
from typing import Tuple, Dict

from sat.models.heads import SAOutput
from sat.loss.ranking.multievent import MultiEventRankingLoss
from sat.loss.ranking.sample import SampleRankingLoss


def create_fake_data(
    batch_size: int = 16, num_events: int = 2, num_cuts: int = 10
) -> Tuple[SAOutput, torch.Tensor]:
    """Create synthetic data for testing ranking losses."""
    # Create fake duration cuts
    duration_cuts = torch.linspace(1, 100, num_cuts)

    # Create fake hazard and survival tensors
    hazard = torch.rand(batch_size, num_events, num_cuts)

    # Ensure survival is decreasing for visualization
    survival_base = torch.cumsum(
        torch.nn.functional.softplus(torch.randn(batch_size, num_events, num_cuts)),
        dim=2,
    )
    # Scale to 0-1 range and flip to get decreasing values
    max_vals = survival_base.max(dim=2, keepdim=True)[0]
    survival_base = 1 - (survival_base / (max_vals + 1e-6))
    # Add survival at time 0 (always 1.0)
    ones = torch.ones(batch_size, num_events, 1)
    survival = torch.cat([ones, survival_base], dim=2)

    # Create logits (we don't need them for the test)
    logits = torch.zeros(batch_size, num_events, num_cuts)

    # Create targets
    # Each row is: [duration_percentile, event, fraction, duration] for each event
    # For num_events=2, shape will be [batch_size, 8]
    targets = torch.zeros(batch_size, 4 * num_events)

    # Set some events to 1
    for i in range(batch_size):
        event_type = i % num_events
        targets[i, num_events + event_type] = 1  # Set event indicator
        targets[i, 3 * num_events + event_type] = duration_cuts[
            i % num_cuts
        ]  # Set duration
        # Set duration index (percentile)
        targets[i, event_type] = i % num_cuts

    # Create fake predictions
    predictions = SAOutput(logits=logits, hazard=hazard, survival=survival)

    return predictions, targets


def create_duration_cuts_file(num_cuts: int = 10) -> str:
    """Create a temporary file with duration cuts."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        duration_cuts = np.linspace(1, 100, num_cuts)
        for cut in duration_cuts:
            f.write(f"{cut}\n")
        return f.name


def create_importance_weights_file(num_events: int = 2) -> str:
    """Create a temporary file with importance weights."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        weights = np.ones(num_events + 1)  # Add one for censoring weight
        for weight in weights:
            f.write(f"{weight}\n")
        return f.name


def test_tensor_orientations():
    """Test and visualize tensor orientations in MultiEventRankingLoss vs SampleRankingLoss."""
    batch_size = 8
    num_events = 2
    num_cuts = 5

    # Create test data
    predictions, targets = create_fake_data(batch_size, num_events, num_cuts)

    # Print shapes for debugging
    print("\nTensor shapes:")
    print(
        f"hazard shape: {predictions.hazard.shape}"
    )  # [batch_size, num_events, num_cuts]
    print(
        f"survival shape: {predictions.survival.shape}"
    )  # [batch_size, num_events, num_cuts+1]
    print(f"targets shape: {targets.shape}")  # [batch_size, 4*num_events]

    # Extract events and durations
    events = targets[:, num_events : 2 * num_events]
    durations = targets[:, 3 * num_events : 4 * num_events]

    print(f"\nEvents tensor: {events}")
    print(f"Events shape: {events.shape}")

    print(f"\nDurations tensor: {durations}")
    print(f"Durations shape: {durations.shape}")

    # Show tensor after MultiEventRankingLoss permutation (no permutation)
    multi_events = events.clone()
    multi_durations = durations.clone()

    print("\nMultiEventRankingLoss tensor orientation:")
    print(f"Events tensor: {multi_events}")
    print(f"Events shape: {multi_events.shape}")

    # Show tensor after SampleRankingLoss permutation
    sample_events = events.permute(1, 0)
    sample_durations = durations.permute(1, 0)

    print("\nSampleRankingLoss tensor orientation (after permute):")
    print(f"Events tensor: {sample_events}")
    print(f"Events shape: {sample_events.shape}")

    # Now check if the permutation is handled correctly in the weights expansion
    duration_cuts_file = create_duration_cuts_file(num_cuts)
    importance_weights_file = create_importance_weights_file(num_events)

    try:
        # Create loss instances
        multi_loss = MultiEventRankingLoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=importance_weights_file,
            num_events=num_events,
            sigma=0.1,
        )

        sample_loss = SampleRankingLoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=importance_weights_file,
            num_events=num_events,
            sigma=0.1,
        )

        # Print original weights
        print(f"\nOriginal weights: {multi_loss.weights}")

        # Check how the weights are expanded in MultiEventRankingLoss (line 47-51)
        n = batch_size
        e = num_events

        # MultiEventRankingLoss weights expansion
        multi_weights = (
            multi_loss.weights[1:]
            .unsqueeze(0)
            .unsqueeze(2)
            .repeat(1, 1, e)
            .expand(n, -1, -1)
        )
        print(f"\nMultiEventRankingLoss expanded weights shape: {multi_weights.shape}")
        print(f"MultiEventRankingLoss expanded weights: {multi_weights}")

        # SampleRankingLoss weights expansion
        sample_weights = (
            sample_loss.weights[1:].unsqueeze(1).unsqueeze(2).repeat(1, e, e)
        )
        print(f"\nSampleRankingLoss expanded weights shape: {sample_weights.shape}")
        print(f"SampleRankingLoss expanded weights: {sample_weights}")

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts_file)
        os.unlink(importance_weights_file)


def test_loss_calculation():
    """Test initialization and basic functionality of ranking losses."""
    batch_size = 8
    num_events = 2
    num_cuts = 10

    # Create files needed for loss initialization
    duration_cuts_file = create_duration_cuts_file(num_cuts)
    importance_weights_file = create_importance_weights_file(num_events)

    try:
        # Create loss instances with identical parameters
        multi_loss = MultiEventRankingLoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=importance_weights_file,
            num_events=num_events,
            sigma=0.1,
        )

        sample_loss = SampleRankingLoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=importance_weights_file,
            num_events=num_events,
            sigma=0.1,
        )

        # Check initialization parameters
        assert multi_loss.sigma == 0.1, "Sigma not properly initialized"
        assert multi_loss.margin == 0.0, "Default margin should be 0.0"
        assert (
            multi_loss.num_events == num_events
        ), "num_events not properly initialized"
        assert (
            len(multi_loss.duration_cuts) == num_cuts
        ), "Duration cuts not properly loaded"

        assert sample_loss.sigma == 0.1, "Sigma not properly initialized"
        assert sample_loss.margin == 0.0, "Default margin should be 0.0"
        assert (
            sample_loss.num_events == num_events
        ), "num_events not properly initialized"

        # Test with different sigma values
        sigma_values = [0.01, 0.1, 0.5, 1.0, 2.0]
        print("\nTesting with different sigma values:")

        for sigma in sigma_values:
            multi_loss = MultiEventRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                sigma=sigma,
            )

            assert multi_loss.sigma == sigma, f"Sigma not properly set to {sigma}"
            print(f"Sigma={sigma}: Initialized successfully")

        # Test with different margin values
        margin_values = [0.0, 0.01, 0.05, 0.1, 0.2]
        print("\nTesting with different margin values:")

        for margin in margin_values:
            multi_loss = MultiEventRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                sigma=0.1,
                margin=margin,
            )

            assert multi_loss.margin == margin, f"Margin not properly set to {margin}"
            print(f"Margin={margin}: Initialized successfully")

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts_file)
        os.unlink(importance_weights_file)


def test_hsa_synthetic_specific():
    """Test initialization with HSA synthetic dataset patterns."""
    # Setup test parameters
    batch_size = 16
    num_events = 2
    num_cuts = 10

    # Create files needed for loss initialization
    duration_cuts_file = create_duration_cuts_file(num_cuts)
    importance_weights_file = create_importance_weights_file(num_events)

    try:
        # Create loss with specific parameters for HSA synthetic
        multi_loss = MultiEventRankingLoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=importance_weights_file,
            num_events=num_events,
            sigma=0.1,
            margin=0.05,
        )

        # Verify initialization parameters
        assert multi_loss.sigma == 0.1, "Sigma parameter not set correctly"
        assert multi_loss.margin == 0.05, "Margin parameter not set correctly"
        assert multi_loss.num_events == num_events, "Number of events not set correctly"
        assert (
            len(multi_loss.duration_cuts) == num_cuts
        ), "Duration cuts not loaded correctly"

        # Test the events and duration extraction functions
        test_targets = torch.zeros(batch_size, 4 * num_events)

        # Set event indicators for testing
        for i in range(batch_size):
            event_type = i % num_events
            test_targets[i, num_events + event_type] = 1  # Set event indicator
            test_targets[i, 3 * num_events + event_type] = i + 1  # Set duration

        # Test for a few samples with multiple events (competing risks)
        for i in range(0, batch_size, 4):
            # Set both events to 1 for this sample
            test_targets[i, num_events : 2 * num_events] = 1.0
            # Set different durations for each event
            test_targets[i, 3 * num_events] = 30.0  # First event duration
            test_targets[i, 3 * num_events + 1] = 60.0  # Second event duration

        # Test events extraction
        events = multi_loss.events(test_targets)
        assert events.shape == (
            batch_size,
            num_events,
        ), f"Events shape incorrect: {events.shape}"

        # Test durations extraction
        durations = multi_loss.durations(test_targets)
        assert durations.shape == (
            batch_size,
            num_events,
        ), f"Durations shape incorrect: {durations.shape}"

        # Test that competing risks (multiple events per sample) are extracted correctly
        # The 0th, 4th, 8th, etc. samples should have both events active
        multi_event_samples = [0, 4, 8, 12]
        for i in multi_event_samples:
            if i < batch_size:
                assert (
                    events[i, 0] == 1 and events[i, 1] == 1
                ), f"Sample {i} should have both events"
                assert (
                    durations[i, 0] == 30.0
                ), f"Sample {i} should have duration 30.0 for event 0"
                assert (
                    durations[i, 1] == 60.0
                ), f"Sample {i} should have duration 60.0 for event 1"

        print("Event and duration extraction for multi-event samples verified.")

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts_file)
        os.unlink(importance_weights_file)


def test_ranking_loss_gradient():
    """Test basic gradient functionality of MultiEventRankingLoss."""
    batch_size = 8
    num_events = 2
    num_cuts = 10

    # Create files needed for loss initialization
    duration_cuts_file = create_duration_cuts_file(num_cuts)
    importance_weights_file = create_importance_weights_file(num_events)

    try:
        # Create simple test data
        # Create trainable parameters - simple test case with decreasing survival
        hazard = torch.rand(batch_size, num_events, num_cuts, requires_grad=True)
        ones = torch.ones(batch_size, num_events, 1)
        survival_base = (
            torch.linspace(0.9, 0.1, num_cuts)
            .view(1, 1, -1)
            .expand(batch_size, num_events, -1)
        )
        survival = torch.cat([ones, survival_base], dim=2)
        logits = torch.zeros_like(hazard)

        # Create targets with simple pattern - alternate between event types
        targets = torch.zeros(batch_size, 4 * num_events)
        for i in range(batch_size):
            event_idx = i % num_events
            targets[i, num_events + event_idx] = 1  # Event indicator
            targets[i, 3 * num_events + event_idx] = 10.0  # Duration

        # Create SAOutput object
        predictions = SAOutput(logits=logits, hazard=hazard, survival=survival)

        # Create loss instance
        multi_loss = MultiEventRankingLoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=importance_weights_file,
            num_events=num_events,
            sigma=0.1,
        )

        # Test gradient calculation
        # Calculate loss
        loss_val = multi_loss(predictions, targets)

        # Ensure loss is valid scalar value
        assert isinstance(loss_val, torch.Tensor), "Loss should be a tensor"
        assert loss_val.numel() == 1, "Loss should be a scalar tensor"
        assert not torch.isnan(loss_val), "Loss should not be NaN"
        assert not torch.isinf(loss_val), "Loss should not be infinite"

        # Test that gradients can be computed
        loss_val.backward()

        # Check gradient
        assert hazard.grad is not None, "No gradient computed for hazard"
        assert not torch.isnan(hazard.grad).any(), "Gradient contains NaN values"
        assert not torch.isinf(hazard.grad).any(), "Gradient contains infinite values"

        # Check that gradient has non-zero magnitude
        grad_magnitude = torch.norm(hazard.grad)
        assert grad_magnitude > 0, "Gradient should not be zero"

        print(f"\nGradient test for MultiEventRankingLoss:")
        print(f"Loss value: {loss_val.item()}")
        print(f"Gradient magnitude: {grad_magnitude.item()}")

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts_file)
        os.unlink(importance_weights_file)


if __name__ == "__main__":
    print("Testing tensor orientations:")
    test_tensor_orientations()

    print("\n\nTesting loss calculation:")
    test_loss_calculation()

    print("\n\nTesting HSA synthetic specific patterns:")
    test_hsa_synthetic_specific()

    print("\n\nTesting ranking loss gradients:")
    test_ranking_loss_gradient()
