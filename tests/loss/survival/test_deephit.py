"""Tests for DeepHit loss components."""

import os

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from sat.loss.meta import MetaLoss
from sat.loss.ranking.sample import SampleRankingLoss
from sat.loss.survival.deephit import DeepHitCalibrationLoss, DeepHitLikelihoodLoss
from sat.models.heads import SAOutput


@pytest.fixture
def tmp_duration_cuts(tmpdir):
    """Create a temporary duration cuts file for testing."""
    cuts_file = os.path.join(tmpdir, "duration_cuts.csv")
    cuts = np.linspace(0, 10, 11)
    pd.DataFrame(cuts).to_csv(cuts_file, header=False, index=False)
    return cuts_file


@pytest.fixture
def mock_predictions(batch_size=10, num_events=2, num_time_bins=10):
    """Create mock model predictions for testing."""
    # Create random logits
    logits = torch.randn(batch_size, num_events, num_time_bins)

    # Calculate hazard rates using softplus
    hazard = F.softplus(logits)

    # Calculate survival probabilities
    cumulative_hazard = torch.cumsum(hazard, dim=2)
    survival_without_zero = torch.exp(-cumulative_hazard)

    # Add S(0) = 1 for each survival curve
    survival = torch.cat(
        [torch.ones(batch_size, num_events, 1), survival_without_zero], dim=2
    )

    # Create mock SAOutput object
    output = SAOutput(
        logits=logits,
        loss=None,
        hazard=hazard,
        survival=survival,
        risk=None,
        time_to_event=None,
        event=None,
        hidden_states=None,
        attentions=None,
    )

    return output


@pytest.fixture
def mock_references(batch_size=10, num_events=2):
    """Create mock reference data for testing."""
    # Structure: [duration_idx, events, fractions, durations]
    references = torch.zeros(batch_size, 4 * num_events)

    # Set random duration indices (time bin indices)
    references[:, 0:num_events] = torch.randint(0, 10, (batch_size, num_events))

    # Set event indicators (some events, some censored)
    for i in range(batch_size):
        if i % 3 == 0:  # Censored
            pass
        elif i % 3 == 1:  # Event type 0
            references[i, num_events] = 1
        else:  # Event type 1 (if multiple events)
            if num_events > 1:
                references[i, num_events + 1] = 1

    # Set random fractions within intervals
    references[:, 2 * num_events : 3 * num_events] = torch.rand(batch_size, num_events)

    # Set random durations
    references[:, 3 * num_events : 4 * num_events] = (
        torch.rand(batch_size, num_events) * 10
    )

    return references


def test_likelihood_loss_init(tmp_duration_cuts):
    """Test initialization of the DeepHitLikelihoodLoss class."""
    loss_fn = DeepHitLikelihoodLoss(
        num_events=2,
    )

    assert loss_fn.num_events == 2
    assert torch.allclose(loss_fn.weights, torch.ones(3))  # num_events + 1


def test_ranking_loss_init(tmp_duration_cuts):
    """Test initialization of the SampleRankingLoss class (replaces DeepHitRankingLoss)."""
    loss_fn = SampleRankingLoss(
        duration_cuts=tmp_duration_cuts,
        sigma=0.1,
        num_events=2,
    )

    assert loss_fn.sigma == 0.1
    assert loss_fn.num_events == 2
    assert loss_fn.num_time_bins == 11
    assert torch.allclose(loss_fn.duration_cuts, torch.linspace(0, 10, 11))
    assert torch.allclose(loss_fn.weights, torch.ones(3))  # num_events + 1


def test_calibration_loss_init(tmp_duration_cuts):
    """Test initialization of the DeepHitCalibrationLoss class."""
    loss_fn = DeepHitCalibrationLoss(
        duration_cuts=tmp_duration_cuts,
        num_events=2,
    )

    assert loss_fn.num_events == 2
    assert loss_fn.num_time_bins == 11
    assert torch.allclose(loss_fn.duration_cuts, torch.linspace(0, 10, 11))
    assert torch.allclose(loss_fn.weights, torch.ones(3))  # num_events + 1

    # Test with specific eval times
    eval_times = [2.5, 5.0, 7.5]
    loss_fn = DeepHitCalibrationLoss(
        duration_cuts=tmp_duration_cuts,
        eval_times=eval_times,
        num_events=2,
    )

    assert loss_fn.eval_times is not None
    assert len(loss_fn.eval_time_indices) == 3


def test_likelihood_loss_forward(mock_predictions, mock_references):
    """Test likelihood loss computation."""
    loss_fn = DeepHitLikelihoodLoss(num_events=2)

    # Forward pass directly with the component
    ll_loss = loss_fn(mock_predictions, mock_references)

    # Just check that the loss is positive and finite
    assert ll_loss > 0
    assert torch.isfinite(ll_loss)


def test_ranking_loss_forward(tmp_duration_cuts, mock_predictions, mock_references):
    """Test ranking loss computation using SampleRankingLoss."""
    loss_fn = SampleRankingLoss(
        duration_cuts=tmp_duration_cuts,
        sigma=0.1,
        num_events=2,
    )

    # Forward pass directly with the component
    rank_loss = loss_fn(mock_predictions, mock_references)

    # Just check that the loss is positive and finite
    assert rank_loss >= 0
    assert torch.isfinite(rank_loss)


def test_sample_ranking_loss_multi_event(tmp_duration_cuts):
    """
    Test that SampleRankingLoss correctly penalizes when subjects with earlier events
    have lower risk than subjects with later events (for the same event type) in multi-event setting.
    """
    batch_size = 4
    num_events = 2
    num_time_bins = 10

    # Create loss function
    loss_fn = SampleRankingLoss(
        duration_cuts=tmp_duration_cuts,
        sigma=0.1,
        num_events=num_events,
    )

    # Create controlled test data
    # Subjects with earlier event times should have higher risk
    # For event type 0:
    # - Subject 0 has event at t=3 (earlier)
    # - Subject 1 has event at t=7 (later)
    # For event type 1:
    # - Subject 2 has event at t=2 (earlier)
    # - Subject 3 has event at t=8 (later)

    dummy_logits = torch.zeros(batch_size, num_events, num_time_bins)

    # Create hazard rates that will produce correct & incorrect risk ordering
    hazard_correct = torch.zeros(batch_size, num_events, num_time_bins + 1)
    hazard_incorrect = torch.zeros(batch_size, num_events, num_time_bins + 1)

    # CORRECT ORDER: Earlier events (0, 2) have higher risk than later events (1, 3)

    # Event type 0
    # Subject 0 (early event) - high risk
    hazard_correct[0, 0, 1:4] = 0.3  # High cumulative hazard rate before t=3
    # Subject 1 (late event) - lower risk
    hazard_correct[1, 0, 1:8] = 0.1  # Lower cumulative hazard rate before t=7

    # Event type 1
    # Subject 2 (early event) - high risk
    hazard_correct[2, 1, 1:3] = 0.4  # High cumulative hazard rate before t=2
    # Subject 3 (late event) - lower risk
    hazard_correct[3, 1, 1:9] = 0.08  # Lower cumulative hazard rate before t=8

    # INCORRECT ORDER: Earlier events (0, 2) have lower risk than later events (1, 3)

    # Event type 0
    # Subject 0 (early event) - low risk (incorrect)
    hazard_incorrect[0, 0, 1:4] = 0.1  # Low cumulative hazard rate
    # Subject 1 (late event) - higher risk (incorrect)
    hazard_incorrect[1, 0, 1:8] = 0.3  # Higher cumulative hazard rate

    # Event type 1
    # Subject 2 (early event) - low risk (incorrect)
    hazard_incorrect[2, 1, 1:3] = 0.08  # Low cumulative hazard rate
    # Subject 3 (late event) - higher risk (incorrect)
    hazard_incorrect[3, 1, 1:9] = 0.4  # Higher cumulative hazard rate

    # Compute survival and risk
    survival_correct = torch.exp(-torch.cumsum(hazard_correct, dim=2))
    risk_correct = 1.0 - survival_correct

    survival_incorrect = torch.exp(-torch.cumsum(hazard_incorrect, dim=2))
    risk_incorrect = 1.0 - survival_incorrect

    # Create predictions
    predictions_correct = SAOutput(
        logits=dummy_logits,
        hazard=hazard_correct,
        survival=survival_correct,
        risk=risk_correct,
    )

    predictions_incorrect = SAOutput(
        logits=dummy_logits,
        hazard=hazard_incorrect,
        survival=survival_incorrect,
        risk=risk_incorrect,
    )

    # Create reference data
    references = torch.zeros(batch_size, 4 * num_events)

    # Set event indicators
    references[0, num_events] = 1  # Subject 0: Event 0 occurred
    references[1, num_events] = 1  # Subject 1: Event 0 occurred
    references[2, num_events + 1] = 1  # Subject 2: Event 1 occurred
    references[3, num_events + 1] = 1  # Subject 3: Event 1 occurred

    # Set percentile indices - not actually used in the test but included for completeness
    time_indices = torch.tensor([3, 7, 2, 8])
    for i in range(batch_size):
        event_idx = 0 if i < 2 else 1
        references[i, event_idx] = time_indices[i]

    # Set fractions - not used but included for completeness
    references[:, 2 * num_events : 3 * num_events] = 0.5

    # Set event times (durations)
    # For event type 0
    references[0, 3 * num_events] = 3.0  # Subject 0: Early event
    references[1, 3 * num_events] = 7.0  # Subject 1: Late event
    # For event type 1
    references[2, 3 * num_events + 1] = 2.0  # Subject 2: Early event
    references[3, 3 * num_events + 1] = 8.0  # Subject 3: Late event

    # Calculate loss for both scenarios
    loss_correct = loss_fn(predictions_correct, references)
    loss_incorrect = loss_fn(predictions_incorrect, references)

    # Correct ranking (earlier events have higher risk) should have lower loss
    assert loss_correct < loss_incorrect

    # The difference should be substantial
    assert (
        loss_incorrect > loss_correct * 1.5
    ), f"Expected significant difference, got {loss_correct} vs {loss_incorrect}"

    # Verify that SampleRankingLoss compares within each event type separately
    # It should compare subject 0 with subject 1 (both had event type 0)
    # And compare subject 2 with subject 3 (both had event type 1)
    # But it should NOT compare subject 0 with subject 2, or subject 1 with subject 3
    # (those are different event types and should be handled by MultiEventRankingLoss)


def test_sample_ranking_loss_single_event(tmp_duration_cuts):
    """
    Test that SampleRankingLoss correctly works for a single event setting.
    """
    batch_size = 4
    num_events = 1  # Single event
    num_time_bins = 10

    # Create loss function
    loss_fn = SampleRankingLoss(
        duration_cuts=tmp_duration_cuts,
        sigma=0.1,
        num_events=num_events,
    )

    # Create controlled test data
    # Subjects with earlier event times should have higher risk
    # - Subject 0 has event at t=2 (earliest)
    # - Subject 1 has event at t=4
    # - Subject 2 has event at t=6
    # - Subject 3 has event at t=8 (latest)

    dummy_logits = torch.zeros(batch_size, num_events, num_time_bins)

    # Create hazard rates that will produce correct & incorrect risk ordering
    hazard_correct = torch.zeros(batch_size, num_events, num_time_bins + 1)
    hazard_incorrect = torch.zeros(batch_size, num_events, num_time_bins + 1)

    # CORRECT ORDER: Risk decreases as time increases
    # Subject 0 (earliest event at t=2) - highest risk
    hazard_correct[0, 0, 1:3] = 0.5  # High cumulative hazard by t=2
    # Subject 1 (event at t=4) - high-medium risk
    hazard_correct[1, 0, 1:5] = 0.3  # Medium-high cumulative hazard by t=4
    # Subject 2 (event at t=6) - medium-low risk
    hazard_correct[2, 0, 1:7] = 0.2  # Medium-low cumulative hazard by t=6
    # Subject 3 (latest event at t=8) - lowest risk
    hazard_correct[3, 0, 1:9] = 0.1  # Low cumulative hazard by t=8

    # INCORRECT ORDER: Risk increases as time increases (wrong)
    # Subject 0 (earliest event at t=2) - lowest risk (incorrect)
    hazard_incorrect[0, 0, 1:3] = 0.1  # Low cumulative hazard
    # Subject 1 (event at t=4) - medium-low risk (incorrect)
    hazard_incorrect[1, 0, 1:5] = 0.2  # Medium-low cumulative hazard
    # Subject 2 (event at t=6) - medium-high risk (incorrect)
    hazard_incorrect[2, 0, 1:7] = 0.3  # Medium-high cumulative hazard
    # Subject 3 (latest event at t=8) - highest risk (incorrect)
    hazard_incorrect[3, 0, 1:9] = 0.5  # High cumulative hazard

    # Compute survival and risk
    survival_correct = torch.exp(-torch.cumsum(hazard_correct, dim=2))
    risk_correct = 1.0 - survival_correct

    survival_incorrect = torch.exp(-torch.cumsum(hazard_incorrect, dim=2))
    risk_incorrect = 1.0 - survival_incorrect

    # Create predictions
    predictions_correct = SAOutput(
        logits=dummy_logits,
        hazard=hazard_correct,
        survival=survival_correct,
        risk=risk_correct,
    )

    predictions_incorrect = SAOutput(
        logits=dummy_logits,
        hazard=hazard_incorrect,
        survival=survival_incorrect,
        risk=risk_incorrect,
    )

    # Create reference data
    references = torch.zeros(batch_size, 4 * num_events)

    # Set event indicators - all subjects have the same event type (0)
    references[:, num_events] = 1  # All subjects experienced event type 0

    # Set event times (durations)
    event_times = torch.tensor([2.0, 4.0, 6.0, 8.0])
    references[:, 3 * num_events] = event_times

    # Set duration indices (percentiles)
    duration_indices = torch.tensor([2, 4, 6, 8])
    references[:, 0] = duration_indices

    # Calculate loss for both scenarios
    loss_correct = loss_fn(predictions_correct, references)
    loss_incorrect = loss_fn(predictions_incorrect, references)

    # Correct ranking (earlier events have higher risk) should have lower loss
    assert loss_correct < loss_incorrect

    # The difference should be substantial
    assert (
        loss_incorrect > loss_correct * 1.5
    ), f"Expected significant difference, got {loss_correct} vs {loss_incorrect}"


def test_calibration_loss_forward(tmp_duration_cuts, mock_predictions, mock_references):
    """Test calibration loss computation."""
    loss_fn = DeepHitCalibrationLoss(duration_cuts=tmp_duration_cuts, num_events=2)

    # Forward pass directly with the component
    calib_loss = loss_fn(mock_predictions, mock_references)

    # Check that the loss is between 0 and 1 (since it's MSE on probabilities)
    assert 0 <= calib_loss <= 1
    assert torch.isfinite(calib_loss)


def test_meta_loss_combination(tmp_duration_cuts, mock_predictions, mock_references):
    """Test combining DeepHit components with MetaLoss."""
    # Create individual loss components
    likelihood_loss = DeepHitLikelihoodLoss(num_events=2)
    ranking_loss = SampleRankingLoss(
        duration_cuts=tmp_duration_cuts, sigma=0.1, num_events=2
    )
    calibration_loss = DeepHitCalibrationLoss(
        duration_cuts=tmp_duration_cuts, num_events=2
    )

    # Combine with MetaLoss
    meta_loss = MetaLoss(
        losses=[likelihood_loss, ranking_loss, calibration_loss],
        coeffs=[0.3, 0.5, 0.2],
        num_events=2,
    )

    # Forward pass
    total_loss = meta_loss(mock_predictions, mock_references)

    # Check that the loss is positive and finite
    assert total_loss > 0
    assert torch.isfinite(total_loss)


def test_with_balancing(tmp_duration_cuts, mock_predictions, mock_references):
    """Test DeepHit components with balancing strategy."""
    # Create individual loss components
    likelihood_loss = DeepHitLikelihoodLoss(num_events=2)
    ranking_loss = SampleRankingLoss(
        duration_cuts=tmp_duration_cuts, sigma=0.1, num_events=2
    )
    calibration_loss = DeepHitCalibrationLoss(
        duration_cuts=tmp_duration_cuts, num_events=2
    )

    # Combine with MetaLoss using scale balancing
    meta_loss = MetaLoss(
        losses=[likelihood_loss, ranking_loss, calibration_loss],
        coeffs=[0.3, 0.5, 0.2],
        balance_strategy="scale",
        num_events=2,
    )

    # Forward pass
    total_loss = meta_loss(mock_predictions, mock_references)

    # Check that the loss is positive and finite
    assert total_loss > 0
    assert torch.isfinite(total_loss)

    # Check that loss weights are accessible
    weights = meta_loss.get_loss_weights()
    assert len(weights) == 3  # likelihood, ranking, calibration components
