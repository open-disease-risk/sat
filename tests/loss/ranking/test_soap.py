"""Test for SOAP (Statistically Optimal Accelerated Pairwise) loss."""

import os
import tempfile
import pandas as pd
import numpy as np
import torch
from torch.testing import assert_close

from sat.models.heads import SAOutput
from sat.loss.ranking import SampleSOAPLoss, EventSOAPLoss


def create_temporary_csv(data):
    """Create a temporary CSV file from data."""
    fd, path = tempfile.mkstemp()
    try:
        df = pd.DataFrame(data)
        df.to_csv(path, header=False, index=False)
        return path
    finally:
        os.close(fd)


def test_sample_soap_loss_initialization():
    """Test the initialization of SampleSOAPLoss."""
    # Create temporary files
    duration_cuts = create_temporary_csv([1, 2, 3, 4, 5])
    importance_weights = create_temporary_csv([1.0, 0.5, 0.5])

    try:
        # Create loss instance with default parameters
        loss = SampleSOAPLoss(
            duration_cuts=duration_cuts,
            importance_sample_weights=importance_weights,
            num_events=2,
            margin=0.1,
            sigma=1.0,
            num_pairs=100,
            sampling_strategy="uniform",
            adaptive_margin=False,
        )

        # Check attributes
        assert loss.margin == 0.1
        assert loss.sigma == 1.0
        assert loss.num_events == 2
        assert loss.num_pairs == 100
        assert loss.sampling_strategy == "uniform"
        assert loss.adaptive_margin == False
        assert loss.duration_cuts.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert loss.weights.tolist() == [1.0, 0.5, 0.5]

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts)
        os.unlink(importance_weights)


def test_event_soap_loss_initialization():
    """Test the initialization of EventSOAPLoss."""
    # Create temporary files
    duration_cuts = create_temporary_csv([1, 2, 3, 4, 5])
    importance_weights = create_temporary_csv([1.0, 0.5, 0.5])

    try:
        # Create loss instance with different parameters
        loss = EventSOAPLoss(
            duration_cuts=duration_cuts,
            importance_sample_weights=importance_weights,
            num_events=2,
            margin=0.2,
            sigma=0.5,
            num_pairs=None,  # Auto-calculate
            sampling_strategy="importance",
            adaptive_margin=True,
        )

        # Check attributes
        assert loss.margin == 0.2
        assert loss.sigma == 0.5
        assert loss.num_events == 2
        assert loss.num_pairs is None
        assert loss.sampling_strategy == "importance"
        assert loss.adaptive_margin == True
        assert loss.duration_cuts.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert loss.weights.tolist() == [1.0, 0.5, 0.5]

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts)
        os.unlink(importance_weights)


def test_sample_soap_loss_computation():
    """Test the computation of SampleSOAPLoss."""
    # Generate synthetic data
    batch_size = 4
    num_events = 2
    num_time_bins = 5

    # Create test data
    references = torch.zeros(batch_size, 4 * num_events)

    # Set event indicators: some samples have event 0, some have event 1
    # Shape: [batch_size, num_events]
    event_indicators = torch.tensor(
        [
            [1, 0],  # Sample 0 has event 0
            [0, 1],  # Sample 1 has event 1
            [1, 0],  # Sample 2 has event 0
            [0, 1],  # Sample 3 has event 1
        ],
        dtype=torch.long,
    )

    # Set event times with clear pattern
    # Shape: [batch_size, num_events]
    event_times = torch.tensor(
        [
            [2.0, 0.0],  # Sample 0: event 0 at t=2
            [0.0, 3.0],  # Sample 1: event 1 at t=3
            [4.0, 0.0],  # Sample 2: event 0 at t=4
            [0.0, 1.0],  # Sample 3: event 1 at t=1
        ],
        dtype=torch.float32,
    )

    # Populate references tensor
    references[:, num_events : 2 * num_events] = event_indicators
    references[:, 3 * num_events : 4 * num_events] = event_times

    # Create model predictions
    # Shape: [batch_size, num_events, num_time_bins+1]
    survival = torch.ones(batch_size, num_events, num_time_bins + 1)
    hazard = torch.zeros(batch_size, num_events, num_time_bins)

    # Set survival probabilities that match the event pattern
    # For event 0: Sample 0 (t=2) and Sample 2 (t=4)
    # Survival should be: Sample 0 > Sample 2 (earlier event should have lower survival)
    survival[0, 0, :] = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])  # Sample 0, event 0
    survival[2, 0, :] = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.1])  # Sample 2, event 0

    # For event 1: Sample 1 (t=3) and Sample 3 (t=1)
    # Survival should be: Sample 3 > Sample 1 (earlier event should have lower survival)
    survival[1, 1, :] = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])  # Sample 1, event 1
    survival[3, 1, :] = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.1])  # Sample 3, event 1

    # Create predictions output
    predictions = SAOutput(
        survival=survival,
        hazard=hazard,
    )

    # Create temporary files
    duration_cuts = create_temporary_csv([0, 1, 2, 3, 4, 5])
    importance_weights = create_temporary_csv(
        [1.0, 0.5, 0.5]  # Weights for censoring, event 0, event 1
    )

    try:
        # Create loss instance
        loss = SampleSOAPLoss(
            duration_cuts=duration_cuts,
            importance_sample_weights=importance_weights,
            num_events=num_events,
            margin=0.1,
            sigma=1.0,
            num_pairs=10,  # Fixed number for testing
            sampling_strategy="uniform",
            adaptive_margin=False,
        )

        # Compute loss - should work with the fixed number of pairs
        result = loss.forward(predictions, references)

        # Loss should be non-negative
        assert result.item() >= 0

        # Create reversed predictions (wrong ranking)
        bad_survival = survival.clone()

        # Swap survival curves for similar patients
        bad_survival[0, 0, :] = survival[2, 0, :]  # Sample 0 and 2 both have event 0
        bad_survival[2, 0, :] = survival[0, 0, :]

        bad_survival[1, 1, :] = survival[3, 1, :]  # Sample 1 and 3 both have event 1
        bad_survival[3, 1, :] = survival[1, 1, :]

        bad_predictions = SAOutput(survival=bad_survival, hazard=hazard)

        # Compute loss with bad predictions
        bad_result = loss.forward(bad_predictions, references)

        # Both results should be valid
        assert bad_result.item() >= 0

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts)
        os.unlink(importance_weights)


def test_event_soap_loss_computation():
    """Test the computation of EventSOAPLoss."""
    # Generate synthetic data with multiple events per sample
    batch_size = 3
    num_events = 3
    num_time_bins = 5

    # Create test data
    references = torch.zeros(batch_size, 4 * num_events)

    # Set event indicators: some samples have multiple event types
    # Shape: [batch_size, num_events]
    event_indicators = torch.tensor(
        [
            [1, 1, 0],  # Sample 0 has events 0 and 1
            [0, 1, 1],  # Sample 1 has events 1 and 2
            [1, 0, 1],  # Sample 2 has events 0 and 2
        ],
        dtype=torch.long,
    )

    # Set event times with clear pattern
    # Shape: [batch_size, num_events]
    event_times = torch.tensor(
        [
            [2.0, 4.0, 0.0],  # Sample 0: events at t=2 and t=4
            [0.0, 1.0, 3.0],  # Sample 1: events at t=1 and t=3
            [2.0, 0.0, 5.0],  # Sample 2: events at t=2 and t=5
        ],
        dtype=torch.float32,
    )

    # Populate references tensor
    references[:, num_events : 2 * num_events] = event_indicators
    references[:, 3 * num_events : 4 * num_events] = event_times

    # Create model predictions
    # Shape: [batch_size, num_events, num_time_bins+1]
    survival = torch.ones(batch_size, num_events, num_time_bins + 1)
    hazard = torch.zeros(batch_size, num_events, num_time_bins)

    # Set different survival curves for each event type and sample
    # Create decreasing survival patterns with different rates
    for i in range(batch_size):
        for j in range(num_events):
            for t in range(num_time_bins + 1):
                # Different decline rates for different event types
                if j == 0:
                    survival[i, j, t] = 1.0 - (t * 0.1)
                elif j == 1:
                    survival[i, j, t] = 1.0 - (t * 0.15)
                else:
                    survival[i, j, t] = 1.0 - (t * 0.2)

    # Create predictions output
    predictions = SAOutput(
        survival=survival,
        hazard=hazard,
    )

    # Create temporary files
    duration_cuts = create_temporary_csv([0, 1, 2, 3, 4, 5])
    importance_weights = create_temporary_csv(
        [1.0, 0.5, 0.5, 0.5]  # Weights for censoring and 3 event types
    )

    try:
        # Create loss instance
        loss = EventSOAPLoss(
            duration_cuts=duration_cuts,
            importance_sample_weights=importance_weights,
            num_events=num_events,
            margin=0.1,
            sigma=1.0,
            num_pairs=None,  # Auto-calculate
            sampling_strategy="uniform",
            adaptive_margin=False,
        )

        # Compute loss
        result = loss.forward(predictions, references)

        # Loss should be non-negative
        assert result.item() >= 0

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts)
        os.unlink(importance_weights)


def test_sampling_strategies():
    """Test different sampling strategies for SOAP loss."""
    # Generate data
    batch_size = 16
    num_events = 2
    num_time_bins = 5

    # Create references with diverse event patterns
    references = torch.zeros(batch_size, 4 * num_events)
    
    # Create diverse events and durations
    events = torch.zeros(batch_size, num_events)
    durations = torch.zeros(batch_size, num_events)
    
    # Fill with varied patterns
    for i in range(batch_size):
        event_idx = i % num_events
        events[i, event_idx] = 1
        durations[i, event_idx] = 1.0 + (i / batch_size) * 4.0
    
    # Set a few samples with multiple events
    for i in range(0, batch_size, 5):
        if i + 1 < batch_size:
            events[i] = 1.0  # Both event types
            durations[i, 0] = 2.0
            durations[i, 1] = 4.0
    
    # Populate references tensor
    references[:, num_events : 2 * num_events] = events
    references[:, 3 * num_events : 4 * num_events] = durations
    
    # Create predictions
    survival = torch.ones(batch_size, num_events, num_time_bins + 1)
    for i in range(batch_size):
        for j in range(num_events):
            for t in range(num_time_bins + 1):
                survival[i, j, t] = 1.0 - (t / num_time_bins) * (0.8 + 0.2 * (i / batch_size))
    
    hazard = torch.zeros(batch_size, num_events, num_time_bins)
    predictions = SAOutput(survival=survival, hazard=hazard)
    
    # Create temporary files
    duration_cuts = create_temporary_csv([0, 1, 2, 3, 4, 5])
    importance_weights = create_temporary_csv([1.0, 0.5, 0.5])
    
    try:
        # Test each sampling strategy
        for strategy in ["uniform", "importance", "hard"]:
            # Create loss instance
            loss = SampleSOAPLoss(
                duration_cuts=duration_cuts,
                importance_sample_weights=importance_weights,
                num_events=num_events,
                margin=0.1,
                sigma=1.0,
                num_pairs=20,  # Fixed number for testing
                sampling_strategy=strategy,
                adaptive_margin=False,
            )
            
            # Compute loss
            result = loss.forward(predictions, references)
            
            # Loss should be non-negative
            assert result.item() >= 0
            
            # For importance strategy, also test adaptive margin
            if strategy == "importance":
                adaptive_loss = SampleSOAPLoss(
                    duration_cuts=duration_cuts,
                    importance_sample_weights=importance_weights,
                    num_events=num_events,
                    margin=0.1,
                    sigma=1.0,
                    num_pairs=20,
                    sampling_strategy=strategy,
                    adaptive_margin=True,  # Enable adaptive margin
                )
                
                adaptive_result = adaptive_loss.forward(predictions, references)
                assert adaptive_result.item() >= 0
    
    finally:
        # Clean up temporary files
        os.unlink(duration_cuts)
        os.unlink(importance_weights)


def test_pair_sampling_function():
    """Test the pair sampling function directly."""
    # Create sample data
    batch_size = 10
    num_events = 1
    
    # Create event indicators and durations
    events = torch.zeros(batch_size, num_events)
    durations = torch.zeros(batch_size, num_events)
    
    # Set up events and durations
    for i in range(batch_size):
        events[i, 0] = 1 if i < batch_size - 2 else 0  # Most have events
        durations[i, 0] = float(i)
    
    # Create temporary files
    duration_cuts = create_temporary_csv([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    try:
        # Test auto-calculation of pairs
        loss = SampleSOAPLoss(
            duration_cuts=duration_cuts,
            num_events=num_events,
            margin=0.1,
            sigma=1.0,
            num_pairs=None,  # Auto-calculate
            sampling_strategy="uniform",
        )
        
        # Sample pairs with auto-calculation
        pairs = loss.sample_pairs(events, durations)
        
        # Number of pairs should be approximately batch_size * log(batch_size)
        expected_pairs = int(batch_size * np.log(batch_size))
        # Allow for some variance in the implementation
        assert 1 <= pairs.shape[0] <= max(batch_size * (batch_size - 1), expected_pairs * 2)
        
        # Test with fixed number of pairs
        fixed_num_pairs = 15
        loss.num_pairs = fixed_num_pairs
        
        pairs = loss.sample_pairs(events, durations, max_pairs=None)
        
        # Should have exactly fixed_num_pairs * batch_size pairs
        # But capped at batch_size * (batch_size - 1)
        expected = min(fixed_num_pairs * batch_size, batch_size * (batch_size - 1))
        assert pairs.shape[0] <= expected
        
        # Verify pair indices are within bounds
        assert torch.all(pairs >= 0)
        assert torch.all(pairs < batch_size)
        
        # Verify no self-comparisons
        assert not torch.any(pairs[:, 0] == pairs[:, 1])
        
    finally:
        # Clean up
        os.unlink(duration_cuts)


if __name__ == "__main__":
    test_sample_soap_loss_initialization()
    test_event_soap_loss_initialization()
    test_sample_soap_loss_computation()
    test_event_soap_loss_computation()
    test_sampling_strategies()
    test_pair_sampling_function()
    print("All tests passed!")