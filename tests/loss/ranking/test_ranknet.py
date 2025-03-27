"""Tests for RankNet loss implementations."""

import os
import tempfile
import pandas as pd
import torch
from torch.testing import assert_close

from sat.models.heads import SAOutput
from sat.loss.ranking import RankNetLoss, SampleRankNetLoss, EventRankNetLoss


def create_temporary_csv(data):
    """Create a temporary CSV file from data."""
    fd, path = tempfile.mkstemp()
    try:
        df = pd.DataFrame(data)
        df.to_csv(path, header=False, index=False)
        return path
    finally:
        os.close(fd)


def test_ranknet_loss_initialization():
    """Test RankNet loss initialization."""
    # Create temporary files
    duration_cuts = create_temporary_csv([1, 2, 3, 4, 5])
    importance_weights = create_temporary_csv([1.0, 0.5, 0.5])

    try:
        # Create loss function
        loss = RankNetLoss(
            duration_cuts=duration_cuts,
            importance_sample_weights=importance_weights,
            num_events=2,
            sigma=1.0,
            sampling_ratio=0.3,
            use_adaptive_sampling=True,
        )

        # Verify attributes
        assert loss.num_events == 2
        assert loss.sigma == 1.0
        assert loss.sampling_ratio == 0.3
        assert loss.use_adaptive_sampling is True
        assert torch.allclose(
            loss.duration_cuts, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        assert torch.allclose(loss.weights, torch.tensor([1.0, 0.5, 0.5]))
    finally:
        # Clean up
        os.unlink(duration_cuts)
        os.unlink(importance_weights)


def test_sample_ranknet_loss_initialization():
    """Test SampleRankNetLoss initialization."""
    # Create temporary files
    duration_cuts = create_temporary_csv([1, 2, 3, 4, 5])
    importance_weights = create_temporary_csv([1.0, 0.5, 0.5])

    try:
        # Create loss function
        loss = SampleRankNetLoss(
            duration_cuts=duration_cuts,
            importance_sample_weights=importance_weights,
            num_events=2,
            sigma=2.0,
            sampling_ratio=0.5,
            use_adaptive_sampling=False,
        )

        # Verify attributes
        assert loss.num_events == 2
        assert loss.sigma == 2.0
        assert loss.sampling_ratio == 0.5
        assert loss.use_adaptive_sampling is False
        assert torch.allclose(
            loss.duration_cuts, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        assert torch.allclose(loss.weights, torch.tensor([1.0, 0.5, 0.5]))
    finally:
        # Clean up
        os.unlink(duration_cuts)
        os.unlink(importance_weights)


def test_event_ranknet_loss_initialization():
    """Test EventRankNetLoss initialization."""
    # Create temporary files
    duration_cuts = create_temporary_csv([1, 2, 3, 4, 5])
    importance_weights = create_temporary_csv([1.0, 0.5, 0.5])

    try:
        # Create loss function
        loss = EventRankNetLoss(
            duration_cuts=duration_cuts,
            importance_sample_weights=importance_weights,
            num_events=2,
            sigma=0.5,
            sampling_ratio=1.0,
            use_adaptive_sampling=True,
        )

        # Verify attributes
        assert loss.num_events == 2
        assert loss.sigma == 0.5
        assert loss.sampling_ratio == 1.0
        assert loss.use_adaptive_sampling is True
        assert torch.allclose(
            loss.duration_cuts, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        assert torch.allclose(loss.weights, torch.tensor([1.0, 0.5, 0.5]))
    finally:
        # Clean up
        os.unlink(duration_cuts)
        os.unlink(importance_weights)


def test_sample_ranknet_loss_forward():
    """Test SampleRankNetLoss forward pass."""
    # Create test data
    batch_size = 4
    num_events = 1
    num_time_bins = 5

    # Create test references tensor
    references = torch.zeros(batch_size, 4 * num_events)

    # Set event indicators (1 or 0)
    event_indicators = torch.tensor(
        [
            [1],  # Sample 0 has event
            [1],  # Sample 1 has event
            [1],  # Sample 2 has event
            [1],  # Sample 3 has event
        ]
    )

    # Set event durations
    event_durations = torch.tensor(
        [
            [2.0],  # Sample 0: event at t=2
            [3.0],  # Sample 1: event at t=3
            [1.0],  # Sample 2: event at t=1
            [4.0],  # Sample 3: event at t=4
        ]
    )

    # Fill references tensor
    references[:, num_events : 2 * num_events] = event_indicators
    references[:, 3 * num_events : 4 * num_events] = event_durations

    # Create model predictions
    survival = torch.ones(batch_size, num_events, num_time_bins + 1)
    # Create decreasing survival curves
    for i in range(batch_size):
        for t in range(num_time_bins + 1):
            survival[i, 0, t] = 1.0 - 0.1 * t

    hazard = torch.zeros(batch_size, num_events, num_time_bins)

    predictions = SAOutput(
        survival=survival,
        hazard=hazard,
    )

    # Create temporary files
    duration_cuts = create_temporary_csv([0, 1, 2, 3, 4, 5])

    try:
        # Create loss function with full sampling
        loss = SampleRankNetLoss(
            duration_cuts=duration_cuts,
            num_events=num_events,
            sigma=1.0,
            sampling_ratio=1.0,  # Use all pairs
            use_adaptive_sampling=False,
        )

        # Compute loss
        result = loss.forward(predictions, references)

        # Loss should be a single scalar
        assert result.dim() == 0
        assert result.item() >= 0

        # Create another version with different sigma
        loss_sharp = SampleRankNetLoss(
            duration_cuts=duration_cuts,
            num_events=num_events,
            sigma=10.0,  # More confident predictions
            sampling_ratio=1.0,
            use_adaptive_sampling=False,
        )

        # Compute loss
        result_sharp = loss_sharp.forward(predictions, references)

        # Sharper loss should be different
        assert result.item() != result_sharp.item()

    finally:
        # Clean up
        os.unlink(duration_cuts)


def test_event_ranknet_loss_forward():
    """Test EventRankNetLoss forward pass."""
    # Create test data
    batch_size = 3
    num_events = 2
    num_time_bins = 5

    # Create test references tensor
    references = torch.zeros(batch_size, 4 * num_events)

    # Set event indicators (1 or 0)
    event_indicators = torch.tensor(
        [
            [1, 1],  # Sample 0 has both events
            [1, 0],  # Sample 1 has only event 0
            [0, 1],  # Sample 2 has only event 1
        ]
    )

    # Set event durations
    event_durations = torch.tensor(
        [
            [2.0, 4.0],  # Sample 0: event 0 at t=2, event 1 at t=4
            [3.0, 0.0],  # Sample 1: event 0 at t=3, no event 1
            [0.0, 2.0],  # Sample 2: no event 0, event 1 at t=2
        ]
    )

    # Fill references tensor
    references[:, num_events : 2 * num_events] = event_indicators
    references[:, 3 * num_events : 4 * num_events] = event_durations

    # Create model predictions
    survival = torch.ones(batch_size, num_events, num_time_bins + 1)
    # Create decreasing survival curves
    for i in range(batch_size):
        for e in range(num_events):
            for t in range(num_time_bins + 1):
                # Different curve shapes for different event types
                if e == 0:
                    survival[i, e, t] = 1.0 - 0.1 * t
                else:
                    survival[i, e, t] = 1.0 - 0.15 * t

    hazard = torch.zeros(batch_size, num_events, num_time_bins)

    predictions = SAOutput(
        survival=survival,
        hazard=hazard,
    )

    # Create temporary files
    duration_cuts = create_temporary_csv([0, 1, 2, 3, 4, 5])

    try:
        # Create loss function
        loss = EventRankNetLoss(
            duration_cuts=duration_cuts,
            num_events=num_events,
            sigma=1.0,
            sampling_ratio=1.0,
            use_adaptive_sampling=False,
        )

        # Compute loss
        result = loss.forward(predictions, references)

        # Loss should be a single scalar
        assert result.dim() == 0
        assert result.item() >= 0

    finally:
        # Clean up
        os.unlink(duration_cuts)


def test_ranknet_compute_loss():
    """Test RankNet loss computation directly."""
    # Create temporary file
    duration_cuts = create_temporary_csv([0, 1, 2, 3, 4, 5])

    try:
        # Create loss function
        loss = RankNetLoss(
            duration_cuts=duration_cuts,
            num_events=1,
            sigma=1.0,
        )

        # Create test data
        risk_i = torch.tensor([0.8, 0.6, 0.7])
        risk_j = torch.tensor([0.6, 0.7, 0.3])
        targets = torch.tensor([1.0, 0.0, 1.0])

        # Compute loss
        result = loss.compute_ranknet_loss(risk_i, risk_j, targets)

        # Loss should be non-negative
        assert result.item() >= 0

        # Test with perfectly aligned predictions
        perfect_i = torch.tensor([0.9, 0.1, 0.8])
        perfect_j = torch.tensor([0.1, 0.9, 0.2])
        perfect_targets = torch.tensor([1.0, 0.0, 1.0])

        perfect_result = loss.compute_ranknet_loss(
            perfect_i, perfect_j, perfect_targets
        )

        # Perfect predictions should have lower loss
        assert perfect_result.item() < result.item()

    finally:
        # Clean up
        os.unlink(duration_cuts)


if __name__ == "__main__":
    test_ranknet_loss_initialization()
    test_sample_ranknet_loss_initialization()
    test_event_ranknet_loss_initialization()
    test_sample_ranknet_loss_forward()
    test_event_ranknet_loss_forward()
    test_ranknet_compute_loss()
    print("All tests passed!")
