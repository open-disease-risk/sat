"""Tests for DeepHit loss function."""

import pytest
import os
import torch
import pandas as pd
import numpy as np

from sat.loss.survival.deephit import DeepHitLoss
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

    # Create mock SAOutput object
    output = SAOutput(
        logits=logits,
        loss=None,
        hazard=torch.nn.functional.softplus(logits),
        survival=None,
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


def test_deephit_init(tmp_duration_cuts):
    """Test initialization of the DeepHitLoss class."""
    loss_fn = DeepHitLoss(
        duration_cuts=tmp_duration_cuts,
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        sigma=0.1,
        num_events=2,
    )

    assert loss_fn.alpha == 0.5
    assert loss_fn.beta == 0.5
    assert loss_fn.gamma == 0.0
    assert loss_fn.sigma == 0.1
    assert loss_fn.num_events == 2
    assert loss_fn.num_time_bins == 11
    assert torch.allclose(loss_fn.duration_cuts, torch.linspace(0, 10, 11))
    assert torch.allclose(loss_fn.weights, torch.ones(3))  # num_events + 1


def test_get_survival_curves(tmp_duration_cuts, mock_predictions):
    """Test conversion of logits to survival curves."""
    loss_fn = DeepHitLoss(duration_cuts=tmp_duration_cuts, num_events=2)

    logits = mock_predictions.logits
    survival = loss_fn._get_survival_curves(logits)

    assert survival.shape == (10, 2, 11)  # batch_size, num_events, num_time_bins+1
    assert torch.allclose(survival[:, :, 0], torch.ones(10, 2))  # S(0) = 1

    # Test monotonicity of survival curves
    for i in range(10):
        for j in range(2):
            assert torch.all(survival[i, j, 1:] <= survival[i, j, :-1])


def test_likelihood_loss(tmp_duration_cuts, mock_predictions, mock_references):
    """Test likelihood loss computation."""
    loss_fn = DeepHitLoss(
        duration_cuts=tmp_duration_cuts, alpha=1.0, beta=0.0, gamma=0.0, num_events=2
    )

    logits = mock_predictions.logits
    hazards = torch.nn.functional.softplus(logits)
    survival = loss_fn._get_survival_curves(logits)

    ll_loss = loss_fn.likelihood_loss(survival, hazards, mock_references)

    # Just check that the loss is positive and finite
    assert ll_loss > 0
    assert torch.isfinite(ll_loss)


def test_ranking_loss(tmp_duration_cuts, mock_predictions, mock_references):
    """Test ranking loss computation."""
    loss_fn = DeepHitLoss(
        duration_cuts=tmp_duration_cuts,
        alpha=0.0,
        beta=1.0,
        gamma=0.0,
        sigma=0.1,
        num_events=2,
    )

    logits = mock_predictions.logits
    survival = loss_fn._get_survival_curves(logits)

    rank_loss = loss_fn.ranking_loss(survival, mock_references)

    # Just check that the loss is positive and finite
    assert rank_loss >= 0
    assert torch.isfinite(rank_loss)


def test_calibration_loss(tmp_duration_cuts, mock_predictions, mock_references):
    """Test calibration loss computation."""
    loss_fn = DeepHitLoss(
        duration_cuts=tmp_duration_cuts, alpha=0.0, beta=0.0, gamma=1.0, num_events=2
    )

    logits = mock_predictions.logits
    survival = loss_fn._get_survival_curves(logits)

    calib_loss = loss_fn.calibration_loss(survival, mock_references)

    # Check that the loss is between 0 and 1 (since it's MSE on probabilities)
    assert 0 <= calib_loss <= 1
    assert torch.isfinite(calib_loss)


def test_forward_pass(tmp_duration_cuts, mock_predictions, mock_references):
    """Test full forward pass of the loss function."""
    loss_fn = DeepHitLoss(
        duration_cuts=tmp_duration_cuts, alpha=0.3, beta=0.5, gamma=0.2, num_events=2
    )

    total_loss = loss_fn(mock_predictions, mock_references)

    # Check that the loss is positive and finite
    assert total_loss > 0
    assert torch.isfinite(total_loss)


def test_with_balancing(tmp_duration_cuts, mock_predictions, mock_references):
    """Test DeepHitLoss with balancing strategy."""
    loss_fn = DeepHitLoss(
        duration_cuts=tmp_duration_cuts,
        alpha=0.3,
        beta=0.5,
        gamma=0.2,
        num_events=2,
        balance_strategy="scale",
    )

    total_loss = loss_fn(mock_predictions, mock_references)

    # Check that the loss is positive and finite
    assert total_loss > 0
    assert torch.isfinite(total_loss)

    # Check that loss weights are accessible
    weights = loss_fn.get_loss_weights()
    assert len(weights) == 3  # alpha, beta, gamma components
