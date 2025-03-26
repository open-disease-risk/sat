"""Tests for the MultiEventRankingLoss function."""

import os
import pytest
import torch
import pandas as pd
import numpy as np

from sat.loss import MultiEventRankingLoss, SampleRankingLoss
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
    # Weights for [censored, event1, event2, event3]
    weights = pd.DataFrame({"weights": [0.5, 1.0, 1.0, 1.0]})
    weights.to_csv(weights_path, index=False, header=False)
    return str(weights_path)


def test_multievent_ranking_loss_initialization(
    duration_cuts_file, importance_weights_file
):
    """Test that MultiEventRankingLoss initializes correctly."""
    loss_fn = MultiEventRankingLoss(
        duration_cuts=duration_cuts_file,
        importance_sample_weights=importance_weights_file,
        sigma=0.1,
        num_events=3,
    )

    assert loss_fn.num_events == 3
    assert loss_fn.sigma == 0.1
    assert torch.allclose(
        loss_fn.weights, torch.tensor([0.5, 1.0, 1.0, 1.0], dtype=torch.float32)
    )
    assert len(loss_fn.duration_cuts) == 11


def test_multievent_ranking_loss_forward(duration_cuts_file):
    """Test the forward pass of MultiEventRankingLoss."""
    batch_size = 4
    num_events = 3
    num_time_bins = 10

    # Create loss function
    loss_fn = MultiEventRankingLoss(
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
    # For 3 events: [3 percentile cols, 3 event cols, 3 fraction cols, 3 duration cols]
    references = torch.zeros(batch_size, 4 * num_events)

    # Set event status - distribute different event types across samples
    # Sample 0: all event types (multi-event)
    references[0, num_events : 2 * num_events] = 1
    # Sample 1: event type 0 only
    references[1, num_events] = 1
    # Sample 2: event type 1 only
    references[2, num_events + 1] = 1
    # Sample 3: no events (censored)

    # Set durations - Event type 0 occurs earlier than type 1, which occurs earlier than type 2
    references[0, 3 * num_events] = 2.0  # Sample 0: Event 0 at t=2
    references[0, 3 * num_events + 1] = 4.0  # Sample 0: Event 1 at t=4
    references[0, 3 * num_events + 2] = 6.0  # Sample 0: Event 2 at t=6
    references[1, 3 * num_events] = 3.0  # Sample 1: Event 0 at t=3
    references[2, 3 * num_events + 1] = 5.0  # Sample 2: Event 1 at t=5
    references[3, 3 * num_events : 4 * num_events] = 8.0  # Sample 3: Censored at t=8

    # Call forward
    loss = loss_fn(predictions, references)

    # Basic checks
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
    assert loss >= 0.0  # Loss should be non-negative


def test_multievent_ranking_loss_censored_only(duration_cuts_file):
    """Test the forward pass of MultiEventRankingLoss with only censored observations."""
    batch_size = 4
    num_events = 3
    num_time_bins = 10

    # Create loss function
    loss_fn = MultiEventRankingLoss(
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
    references[:, 3 * num_events : 4 * num_events] = 5.0

    # Call forward - should return zero loss since no events to rank
    loss = loss_fn(predictions, references)

    assert loss.item() == 0.0


def test_event_type_ranking_correct(duration_cuts_file):
    """
    Test that MultiEventRankingLoss correctly ranks different event types
    within the same observation.
    """
    batch_size = 2
    num_events = 3
    num_time_bins = 10

    # Create loss function
    loss_fn = MultiEventRankingLoss(
        duration_cuts=duration_cuts_file,
        sigma=0.1,
        num_events=num_events,
    )

    # Create two sets of predictions:
    # 1. Correctly ranked: Earlier event types have higher risk
    # 2. Incorrectly ranked: Earlier event types have lower risk

    dummy_logits = torch.zeros(batch_size, num_events, num_time_bins)

    # Create controlled test data with predictable event patterns
    # Sample 0: Has all three event types (multirisks) with event 0 earliest, then 1, then 2
    # Sample 1: Censored

    # Hazard rates for correct ranking (earlier events have higher risk)
    hazard_correct = torch.zeros(batch_size, num_events, num_time_bins + 1)

    # Sample 0 - Multi-event, correct order
    # Event type 0 - earliest at t=2 - highest risk
    hazard_correct[0, 0, 1:3] = 0.5  # High hazard
    # Event type 1 - middle at t=5 - medium risk
    hazard_correct[0, 1, 1:6] = 0.3  # Medium hazard
    # Event type 2 - latest at t=8 - lowest risk
    hazard_correct[0, 2, 1:9] = 0.1  # Low hazard

    # Sample 1 is censored - random hazards
    hazard_correct[1, :, 1:] = 0.2

    # Now create incorrect ordering - earlier events have lower risk
    hazard_incorrect = torch.zeros(batch_size, num_events, num_time_bins + 1)

    # Sample 0 - Multi-event, incorrect order
    # Event type 0 - earliest at t=2 - lowest risk (WRONG)
    hazard_incorrect[0, 0, 1:3] = 0.1  # Low hazard
    # Event type 1 - middle at t=5 - medium risk
    hazard_incorrect[0, 1, 1:6] = 0.3  # Medium hazard
    # Event type 2 - latest at t=8 - highest risk (WRONG)
    hazard_incorrect[0, 2, 1:9] = 0.5  # High hazard

    # Sample 1 is censored - random hazards
    hazard_incorrect[1, :, 1:] = 0.2

    # Compute survival and risk
    survival_correct = torch.exp(-torch.cumsum(hazard_correct, dim=2))
    risk_correct = 1.0 - survival_correct

    survival_incorrect = torch.exp(-torch.cumsum(hazard_incorrect, dim=2))
    risk_incorrect = 1.0 - survival_incorrect

    # Create predictions objects
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

    # Sample 0: All event types
    references[0, num_events : 2 * num_events] = 1

    # Sample 1: Censored

    # Set event durations
    # Sample 0: Events at increasing times
    references[0, 3 * num_events] = 2.0  # Event 0 (earliest)
    references[0, 3 * num_events + 1] = 5.0  # Event 1 (middle)
    references[0, 3 * num_events + 2] = 8.0  # Event 2 (latest)

    # Sample 1: Censoring time
    references[1, 3 * num_events : 4 * num_events] = 9.0

    # Calculate loss for both scenarios
    loss_correct = loss_fn(predictions_correct, references)
    loss_incorrect = loss_fn(predictions_incorrect, references)

    # The loss should be lower when earlier events have higher risk (correct case)
    assert (
        loss_correct < loss_incorrect
    ), f"Expected lower loss for correct ranking, got {loss_correct.item()} vs {loss_incorrect.item()}"

    # The difference should be substantial
    assert (
        loss_incorrect > loss_correct * 1.5
    ), f"Expected significant difference, got {loss_correct.item()} vs {loss_incorrect.item()}"


def test_multievent_vs_sample_ranking_different_focus(duration_cuts_file):
    """
    Test that MultiEventRankingLoss and SampleRankingLoss focus on different
    aspects of ranking:

    - MultiEventRankingLoss: Ranks event types within the same observation (cross-event)
    - SampleRankingLoss: Ranks different observations with the same event type (within-event)
    """
    batch_size = 4
    num_events = 2
    num_time_bins = 10

    # Create loss functions
    multi_event_loss = MultiEventRankingLoss(
        duration_cuts=duration_cuts_file,
        sigma=0.1,
        num_events=num_events,
    )

    sample_loss = SampleRankingLoss(
        duration_cuts=duration_cuts_file,
        sigma=0.1,
        num_events=num_events,
    )

    # Create data that satisfies one ranking criterion but not the other

    dummy_logits = torch.zeros(batch_size, num_events, num_time_bins)

    # === SCENARIO 1: Good cross-event ranking, bad within-event ranking ===

    hazard_1 = torch.zeros(batch_size, num_events, num_time_bins + 1)

    # All samples have both event types (simplification)
    # Event type 0 always occurs before event type 1 in all samples
    # Sample 0 has earliest events of both types
    # Sample 3 has latest events of both types

    # Cross-event ranking is GOOD (earlier event types have higher risk)
    # But within-event ranking is BAD (earlier samples don't have higher risk)

    # Sample 0: Events at t=2 and t=4
    hazard_1[0, 0, 1:3] = 0.1  # Event 0 at t=2 - LOW risk (bad within-event ranking)
    hazard_1[0, 1, 1:5] = 0.05  # Event 1 at t=4 - LOWER risk (good cross-event ranking)

    # Sample 1: Events at t=3 and t=5
    hazard_1[1, 0, 1:4] = 0.2  # Event 0 at t=3 - MEDIUM risk (bad within-event ranking)
    hazard_1[1, 1, 1:6] = 0.1  # Event 1 at t=5 - LOWER risk (good cross-event ranking)

    # Sample 2: Events at t=4 and t=6
    hazard_1[2, 0, 1:5] = 0.3  # Event 0 at t=4 - HIGH risk (bad within-event ranking)
    hazard_1[2, 1, 1:7] = 0.15  # Event 1 at t=6 - LOWER risk (good cross-event ranking)

    # Sample 3: Events at t=5 and t=7
    hazard_1[3, 0, 1:6] = (
        0.4  # Event 0 at t=5 - HIGHEST risk (bad within-event ranking)
    )
    hazard_1[3, 1, 1:8] = 0.2  # Event 1 at t=7 - LOWER risk (good cross-event ranking)

    # === SCENARIO 2: Bad cross-event ranking, good within-event ranking ===

    hazard_2 = torch.zeros(batch_size, num_events, num_time_bins + 1)

    # Sample 0: Events at t=2 and t=4
    hazard_2[0, 0, 1:3] = 0.4  # Event 0 at t=2 - HIGH risk (good within-event ranking)
    hazard_2[0, 1, 1:5] = 0.5  # Event 1 at t=4 - HIGHER risk (bad cross-event ranking)

    # Sample 1: Events at t=3 and t=5
    hazard_2[1, 0, 1:4] = (
        0.3  # Event 0 at t=3 - MEDIUM-HIGH risk (good within-event ranking)
    )
    hazard_2[1, 1, 1:6] = 0.4  # Event 1 at t=5 - HIGHER risk (bad cross-event ranking)

    # Sample 2: Events at t=4 and t=6
    hazard_2[2, 0, 1:5] = (
        0.2  # Event 0 at t=4 - MEDIUM-LOW risk (good within-event ranking)
    )
    hazard_2[2, 1, 1:7] = 0.3  # Event 1 at t=6 - HIGHER risk (bad cross-event ranking)

    # Sample 3: Events at t=5 and t=7
    hazard_2[3, 0, 1:6] = 0.1  # Event 0 at t=5 - LOW risk (good within-event ranking)
    hazard_2[3, 1, 1:8] = 0.2  # Event 1 at t=7 - HIGHER risk (bad cross-event ranking)

    # Compute survival and risk
    survival_1 = torch.exp(-torch.cumsum(hazard_1, dim=2))
    risk_1 = 1.0 - survival_1

    survival_2 = torch.exp(-torch.cumsum(hazard_2, dim=2))
    risk_2 = 1.0 - survival_2

    # Create prediction objects
    predictions_1 = SAOutput(
        logits=dummy_logits,
        hazard=hazard_1,
        survival=survival_1,
        risk=risk_1,
    )

    predictions_2 = SAOutput(
        logits=dummy_logits,
        hazard=hazard_2,
        survival=survival_2,
        risk=risk_2,
    )

    # Create reference data
    references = torch.zeros(batch_size, 4 * num_events)

    # All samples have both event types
    references[:, num_events : 2 * num_events] = 1

    # Set event times
    # For event type 0
    references[0, 3 * num_events] = 2.0  # Sample 0: Earliest event
    references[1, 3 * num_events] = 3.0
    references[2, 3 * num_events] = 4.0
    references[3, 3 * num_events] = 5.0  # Sample 3: Latest event

    # For event type 1
    references[0, 3 * num_events + 1] = 4.0  # Sample 0: Earliest event
    references[1, 3 * num_events + 1] = 5.0
    references[2, 3 * num_events + 1] = 6.0
    references[3, 3 * num_events + 1] = 7.0  # Sample 3: Latest event

    # MultiEventRankingLoss should prefer scenario 1 (good cross-event, bad within-event)
    multi_loss_1 = multi_event_loss(predictions_1, references)
    multi_loss_2 = multi_event_loss(predictions_2, references)

    assert (
        multi_loss_1 < multi_loss_2
    ), "MultiEventRankingLoss should prefer scenario 1 (good cross-event ranking)"

    # SampleRankingLoss should prefer scenario 2 (bad cross-event, good within-event)
    sample_loss_1 = sample_loss(predictions_1, references)
    sample_loss_2 = sample_loss(predictions_2, references)

    assert (
        sample_loss_2 < sample_loss_1
    ), "SampleRankingLoss should prefer scenario 2 (good within-event ranking)"

    # Print for debugging
    print("MultiEventRankingLoss - Cross-event focus:")
    print(
        f"  Scenario 1 (good cross-event, bad within-event): {multi_loss_1.item():.6f}"
    )
    print(
        f"  Scenario 2 (bad cross-event, good within-event): {multi_loss_2.item():.6f}"
    )

    print("SampleRankingLoss - Within-event focus:")
    print(
        f"  Scenario 1 (good cross-event, bad within-event): {sample_loss_1.item():.6f}"
    )
    print(
        f"  Scenario 2 (bad cross-event, good within-event): {sample_loss_2.item():.6f}"
    )
