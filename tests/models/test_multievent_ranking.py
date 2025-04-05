"""Tests for MultiEventRankingLoss behavior on HSA synthetic dataset."""

import os
import tempfile
from typing import Tuple

import numpy as np
import torch

from sat.loss.ranking.multievent import MultiEventRankingLoss
from sat.loss.ranking.sample import SampleRankingLoss
from sat.models.heads import SAOutput


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
    """Test actual loss calculation with controlled inputs."""
    batch_size = 8
    num_events = 2
    num_cuts = 10

    # Create test data
    predictions, targets = create_fake_data(batch_size, num_events, num_cuts)

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

        # Calculate losses
        multi_loss_val = multi_loss(predictions, targets)
        sample_loss_val = sample_loss(predictions, targets)

        print(f"\nMultiEventRankingLoss value: {multi_loss_val.item()}")
        print(f"SampleRankingLoss value: {sample_loss_val.item()}")

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

            sample_loss = SampleRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                sigma=sigma,
            )

            multi_loss_val = multi_loss(predictions, targets)
            sample_loss_val = sample_loss(predictions, targets)

            print(
                f"Sigma={sigma}: MultiEvent={multi_loss_val.item()}, Sample={sample_loss_val.item()}, Ratio={multi_loss_val.item()/sample_loss_val.item()}"
            )

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

            sample_loss = SampleRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                sigma=0.1,
                margin=margin,
            )

            multi_loss_val = multi_loss(predictions, targets)
            sample_loss_val = sample_loss(predictions, targets)

            print(
                f"Margin={margin}: MultiEvent={multi_loss_val.item()}, Sample={sample_loss_val.item()}, Ratio={multi_loss_val.item()/sample_loss_val.item()}"
            )

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts_file)
        os.unlink(importance_weights_file)


def test_hsa_synthetic_specific():
    """Test with data patterns similar to HSA synthetic dataset."""
    # Load specific test cases based on HSA dataset characteristics
    batch_size = 16
    num_events = 2
    num_cuts = 10

    # Create files needed for loss initialization
    duration_cuts_file = create_duration_cuts_file(num_cuts)
    importance_weights_file = create_importance_weights_file(num_events)

    try:
        # Create a specific test case where multiple events are active for same sample
        # This is common in HSA synthetic dataset (competing risks)
        predictions, targets = create_fake_data(batch_size, num_events, num_cuts)

        # Set multiple events active for same samples (one row = one sample)
        # Modify a few samples to have both events
        for i in range(0, batch_size, 4):
            # Set both events to 1 for this sample
            targets[i, num_events : 2 * num_events] = 1.0
            # Set different durations for each event
            targets[i, 3 * num_events] = 30.0  # First event duration
            targets[i, 3 * num_events + 1] = 60.0  # Second event duration

        # Create loss instances
        multi_loss = MultiEventRankingLoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=importance_weights_file,
            num_events=num_events,
            sigma=0.1,
            margin=0.05,
        )

        sample_loss = SampleRankingLoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=importance_weights_file,
            num_events=num_events,
            sigma=0.1,
            margin=0.05,
        )

        # Calculate losses
        multi_loss_val = multi_loss(predictions, targets)
        sample_loss_val = sample_loss(predictions, targets)

        print("\nHSA Synthetic-like test:")
        print(f"MultiEventRankingLoss value: {multi_loss_val.item()}")
        print(f"SampleRankingLoss value: {sample_loss_val.item()}")
        print(f"Loss ratio: {multi_loss_val.item() / sample_loss_val.item()}")

        # Let's examine how tensor shapes affect the ranking loss calculation
        # by looking at intermediate values in ranking_loss method

        # For MultiEventRankingLoss
        events_multi = multi_loss.events(targets)
        n_multi = events_multi.shape[0]
        e_multi = events_multi.shape[1]

        # Show the event masks
        I_multi = events_multi.to(bool)
        I_censored_multi = ~I_multi
        print(
            f"\nShape of event indicator tensor (MultiEventRankingLoss): {I_multi.shape}"
        )
        print(f"Number of events per sample: {I_multi.sum(dim=1)}")

        # For SampleRankingLoss
        events_sample = sample_loss.events(targets).permute(1, 0)
        n_sample = events_sample.shape[0]
        e_sample = events_sample.shape[1]

        # Show the event masks
        I_sample = events_sample.to(bool)
        I_censored_sample = ~I_sample
        print(
            f"\nShape of event indicator tensor (SampleRankingLoss): {I_sample.shape}"
        )
        print(f"Number of events per event type: {I_sample.sum(dim=1)}")

        # See how the orientation affects the ranking calculation
        # In MultiEventRankingLoss, we compare events within a sample
        # In SampleRankingLoss, we compare samples within an event type

        # Create a minimal test for sample with multiple events
        print("\nAnalyzing simplified test case with multi-event sample:")
        simple_targets = torch.zeros(1, 8)  # 1 sample, 2 events, 4 values per event
        # Set both events to 1 for this sample
        simple_targets[0, 2:4] = 1.0
        # Set different durations for each event
        simple_targets[0, 6] = 30.0  # First event duration
        simple_targets[0, 7] = 60.0  # Second event duration

        # Simple logits, hazards and survival
        simple_logits = torch.zeros(1, 2, num_cuts)
        simple_hazard = torch.rand(1, 2, num_cuts)
        # Make first event have higher risk than second event at all times
        simple_survival = torch.zeros(1, 2, num_cuts + 1)
        # First event: rapidly decreasing survival
        simple_survival[0, 0, 0] = 1.0
        simple_survival[0, 0, 1:] = torch.linspace(0.9, 0.1, num_cuts)
        # Second event: slowly decreasing survival
        simple_survival[0, 1, 0] = 1.0
        simple_survival[0, 1, 1:] = torch.linspace(0.95, 0.5, num_cuts)

        simple_predictions = SAOutput(
            logits=simple_logits, hazard=simple_hazard, survival=simple_survival
        )

        # Calculate losses
        multi_simple_loss = multi_loss(simple_predictions, simple_targets)
        sample_simple_loss = sample_loss(simple_predictions, simple_targets)

        print(f"Simple test MultiEventRankingLoss value: {multi_simple_loss.item()}")
        print(f"Simple test SampleRankingLoss value: {sample_simple_loss.item()}")

        # Extract key values from the ranking_loss calculation
        events_multi = multi_loss.events(simple_targets)
        durations_multi = multi_loss.durations(simple_targets)

        # Show survival values
        print(f"\nSurvival values for event 0: {simple_survival[0, 0, :]}")
        print(f"Survival values for event 1: {simple_survival[0, 1, :]}")

        # Compare tensor orientations
        print(f"\nEvents tensor (MultiEventRankingLoss): {events_multi}")
        print(f"Durations tensor (MultiEventRankingLoss): {durations_multi}")

        events_sample = sample_loss.events(simple_targets).permute(1, 0)
        durations_sample = sample_loss.durations(simple_targets).permute(1, 0)

        print(f"\nEvents tensor (SampleRankingLoss): {events_sample}")
        print(f"Durations tensor (SampleRankingLoss): {durations_sample}")

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts_file)
        os.unlink(importance_weights_file)


def test_ranking_loss_gradient():
    """Test gradients from MultiEventRankingLoss vs SampleRankingLoss."""
    batch_size = 8
    num_events = 2
    num_cuts = 10

    # Create test data
    predictions_data, targets = create_fake_data(batch_size, num_events, num_cuts)

    # Create files needed for loss initialization
    duration_cuts_file = create_duration_cuts_file(num_cuts)
    importance_weights_file = create_importance_weights_file(num_events)

    try:
        # Create trainable parameters
        hazard = torch.rand(batch_size, num_events, num_cuts, requires_grad=True)
        ones = torch.ones(batch_size, num_events, 1)
        survival_base = (
            1 - torch.cumsum(torch.nn.functional.softplus(hazard), dim=2) / num_cuts
        )
        survival = torch.cat([ones, survival_base], dim=2)
        logits = torch.zeros_like(hazard)

        predictions = SAOutput(logits=logits, hazard=hazard, survival=survival)

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

        # Calculate losses and gradients
        multi_loss_val = multi_loss(predictions, targets)
        multi_loss_val.backward(retain_graph=True)
        multi_grad = hazard.grad.clone()
        hazard.grad.zero_()

        sample_loss_val = sample_loss(predictions, targets)
        sample_loss_val.backward()
        sample_grad = hazard.grad.clone()

        print(f"\nGradient test:")
        print(f"MultiEventRankingLoss gradient norm: {torch.norm(multi_grad)}")
        print(f"SampleRankingLoss gradient norm: {torch.norm(sample_grad)}")

        # Check if gradients have the same sign
        sign_match = torch.sign(multi_grad) == torch.sign(sample_grad)
        sign_match_percentage = sign_match.float().mean().item() * 100

        print(f"Gradient sign match percentage: {sign_match_percentage:.2f}%")

        # Check correlation between gradients
        correlation = torch.corrcoef(
            torch.stack([multi_grad.flatten(), sample_grad.flatten()])
        )[0, 1].item()

        print(f"Gradient correlation: {correlation:.4f}")

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
