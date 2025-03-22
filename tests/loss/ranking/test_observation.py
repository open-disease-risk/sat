"""Tests for the ObservationEventRankingLoss function"""

import os
import pytest
import torch
import pandas as pd
import numpy as np

from sat.loss import ObservationEventRankingLoss
from sat.models.heads import SAOutput


@pytest.fixture
def duration_cuts_file(tmp_path):
    """Create a temporary duration cuts file."""
    cuts_path = tmp_path / "duration_cuts.csv"
    cuts = pd.DataFrame({"cuts": np.linspace(0, 10, 11)})
    cuts.to_csv(cuts_path, index=False, header=False)
    return str(cuts_path)


@pytest.fixture
def importance_weights_file(tmp_path):
    """Create a temporary importance weights file."""
    weights_path = tmp_path / "weights.csv"
    # Weights for [censored, event1, event2]
    weights = pd.DataFrame({"weights": [0.5, 1.0, 1.0]})
    weights.to_csv(weights_path, index=False, header=False)
    return str(weights_path)


def test_observation_event_ranking_loss_initialization(
    duration_cuts_file, importance_weights_file
):
    """Test that ObservationEventRankingLoss initializes correctly."""
    loss_fn = ObservationEventRankingLoss(
        duration_cuts=duration_cuts_file,
        importance_sample_weights=importance_weights_file,
        sigma=0.1,
        num_events=2,
    )

    assert loss_fn.num_events == 2
    assert loss_fn.sigma == 0.1
    assert torch.allclose(
        loss_fn.weights, torch.tensor([0.5, 1.0, 1.0], dtype=torch.float32)
    )
    assert len(loss_fn.duration_cuts) == 11


def test_observation_event_ranking_loss_forward(duration_cuts_file):
    """Test the forward pass of ObservationEventRankingLoss."""
    batch_size = 4
    num_events = 2
    num_time_bins = 10

    # Create loss function
    loss_fn = ObservationEventRankingLoss(
        duration_cuts=duration_cuts_file,
        sigma=0.1,
        num_events=num_events,
    )

    # Create fake predictions
    logits = torch.randn(batch_size, num_events, num_time_bins)
    hazard = torch.nn.functional.softplus(logits)
    hazard = torch.cat(
        [torch.zeros(batch_size, num_events, 1), hazard], dim=2
    )  # Padding for time 0
    survival = torch.exp(-torch.cumsum(hazard, dim=2))
    risk = 1.0 - survival

    predictions = SAOutput(
        logits=logits,
        hazard=hazard,
        survival=survival,
        risk=risk,
    )

    # Create fake references - format matches SurvivalTaskHead.forward's expected format
    # Reference tensor structure: [percentiles, events, fractions, durations]
    # For 2 events: [2 percentile cols, 2 event cols, 2 fraction cols, 2 duration cols]
    references = torch.zeros(batch_size, 4 * num_events)

    # Set event status - first 2 samples have event 0, second 2 samples have event 1
    references[0:2, num_events] = 1  # Event 0 occurred
    references[2:4, num_events + 1] = 1  # Event 1 occurred

    # Set durations
    references[0:2, 3 * num_events] = 5.0  # Time for event 0
    references[2:4, 3 * num_events + 1] = 8.0  # Time for event 1

    # Call forward
    loss = loss_fn(predictions, references)

    # Basic checks
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
    assert loss >= 0.0  # Loss should be non-negative


def test_observation_event_ranking_loss_requires_multiple_events():
    """Test that ObservationEventRankingLoss requires at least 2 event types."""
    with pytest.raises(ValueError):
        ObservationEventRankingLoss(
            duration_cuts="dummy_path",
            sigma=0.1,
            num_events=1,  # Should raise error
        )


def test_observation_event_ranking_loss_censored_only(duration_cuts_file):
    """Test the forward pass of ObservationEventRankingLoss with only censored observations."""
    batch_size = 4
    num_events = 2
    num_time_bins = 10

    # Create loss function
    loss_fn = ObservationEventRankingLoss(
        duration_cuts=duration_cuts_file,
        sigma=0.1,
        num_events=num_events,
    )

    # Create fake predictions
    logits = torch.randn(batch_size, num_events, num_time_bins)
    hazard = torch.nn.functional.softplus(logits)
    hazard = torch.cat(
        [torch.zeros(batch_size, num_events, 1), hazard], dim=2
    )  # Padding for time 0
    survival = torch.exp(-torch.cumsum(hazard, dim=2))
    risk = 1.0 - survival

    predictions = SAOutput(
        logits=logits,
        hazard=hazard,
        survival=survival,
        risk=risk,
    )

    # Create references with all censored observations
    references = torch.zeros(batch_size, 4 * num_events)

    # Set durations (even though censored)
    references[:, 3 * num_events] = 5.0
    references[:, 3 * num_events + 1] = 8.0

    # Call forward - should return zero loss since no events to rank
    loss = loss_fn(predictions, references)

    assert loss.item() == 0.0
