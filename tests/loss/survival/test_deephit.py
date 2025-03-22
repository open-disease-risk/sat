"""Tests for DeepHit loss components."""

import pytest
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from sat.loss.survival.deephit import (
    DeepHitLikelihoodLoss,
    DeepHitRankingLoss,
    DeepHitCalibrationLoss,
)
from sat.loss.meta import MetaLoss
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
    """Test initialization of the DeepHitRankingLoss class."""
    loss_fn = DeepHitRankingLoss(
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
    """Test ranking loss computation."""
    loss_fn = DeepHitRankingLoss(
        duration_cuts=tmp_duration_cuts,
        sigma=0.1,
        num_events=2,
    )

    # Forward pass directly with the component
    rank_loss = loss_fn(mock_predictions, mock_references)

    # Just check that the loss is positive and finite
    assert rank_loss >= 0
    assert torch.isfinite(rank_loss)


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
    ranking_loss = DeepHitRankingLoss(
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
    ranking_loss = DeepHitRankingLoss(
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
