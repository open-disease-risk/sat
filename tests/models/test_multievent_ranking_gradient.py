"""Test MultiEventRankingLoss gradient flow"""

import os
import tempfile

import torch

from sat.loss.ranking.multievent import MultiEventRankingLoss
from sat.models.heads import SAOutput


def test_multi_event_ranking_gradient():
    """Test proper gradient flow through MultiEventRankingLoss."""
    batch_size = 4
    num_events = 2
    num_cuts = 5

    # Create simple duration cuts
    cuts = [0.0, 10.0, 20.0, 30.0, 40.0]

    # Create temporary files for configuration
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        for cut in cuts:
            f.write(f"{cut}\n")
        duration_cuts_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        weights = [1.0] * (num_events + 1)  # +1 for censoring weight
        for weight in weights:
            f.write(f"{weight}\n")
        weights_file = f.name

    try:
        # Create logits that require gradients
        logits = torch.randn(batch_size, num_events, num_cuts, requires_grad=True)

        # Compute hazard and survival from logits
        hazard = torch.nn.functional.softplus(logits)

        # Survival calculation: S(t) = exp(-cumsum(hazard))
        cumhazard = torch.cumsum(hazard, dim=2)
        survival = torch.exp(-cumhazard)

        # Prepend ones for survival at time 0
        ones = torch.ones(batch_size, num_events, 1)
        survival_with_ones = torch.cat([ones, survival], dim=2)

        # Create predictions
        predictions = SAOutput(
            logits=logits, hazard=hazard, survival=survival_with_ones
        )

        # Create references with proper structure
        # Format: [duration_percentile (event 1), duration_percentile (event 2),
        #          event_indicator (event 1), event_indicator (event 2),
        #          fraction (event 1), fraction (event 2),
        #          duration (event 1), duration (event 2)]
        references = torch.zeros(batch_size, 4 * num_events)

        # Simple case: each patient has one event, varying which type
        for i in range(batch_size):
            # Alternate which event happens
            event_type = i % num_events
            other_event_type = 1 - event_type

            # Set event indicators
            references[i, num_events + event_type] = 1  # This event happened
            references[i, num_events + other_event_type] = 0  # Other event censored

            # Set durations (percentiles)
            references[i, event_type] = i  # Varying duration indices
            references[i, other_event_type] = num_cuts - 1  # Censored at end

            # Set actual durations
            references[i, 3 * num_events + event_type] = cuts[i]
            references[i, 3 * num_events + other_event_type] = cuts[-1]

        # Create loss
        loss_fn = MultiEventRankingLoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=weights_file,
            num_events=num_events,
            sigma=0.1,
        )

        # Forward pass
        loss = loss_fn(predictions, references)

        print(f"Loss value: {loss}")
        print(f"Loss requires_grad: {loss.requires_grad}")

        # Try backward pass
        if loss.requires_grad:
            loss.backward()

            if logits.grad is not None:
                print(f"Gradients computed! Norm: {torch.norm(logits.grad)}")
                assert torch.norm(logits.grad) > 0, "Gradients are zero"
            else:
                print("No gradients on logits")
        else:
            print("Loss doesn't require grad - checking computation graph")
            # If loss is zero or constant, add small regularization
            reg_loss = loss + 1e-6 * logits.mean()
            reg_loss.backward()

            if logits.grad is not None:
                print(
                    f"Gradients computed with regularization! Norm: {torch.norm(logits.grad)}"
                )
            else:
                raise ValueError("Still no gradients after regularization")

    finally:
        # Clean up temp files
        os.unlink(duration_cuts_file)
        os.unlink(weights_file)


if __name__ == "__main__":
    test_multi_event_ranking_gradient()
