"""Tests for loss balancing strategies."""

import torch
import pytest

from sat.loss.balancing import (
    LossBalancer,
    BalancingStrategy,
    FixedWeightBalancer,
    ScaleNormalizationBalancer,
    UncertaintyWeightBalancer,
    AdaptiveWeightBalancer,
)


def test_fixed_weight_balancer():
    """Test fixed weight balancing strategy."""
    # Create balancer with custom coefficients
    coeffs = [0.5, 1.5, 2.0]
    balancer = FixedWeightBalancer(coeffs)

    # Create test losses
    losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]

    # Compute balanced loss
    total_loss = balancer(losses)

    # Expected result: 0.5*1.0 + 1.5*2.0 + 2.0*3.0 = 9.5
    expected = 0.5 + 3.0 + 6.0
    assert abs(total_loss.item() - expected) < 1e-5


def test_scale_normalization_balancer():
    """Test scale normalization balancing strategy."""
    # Create balancer
    balancer = ScaleNormalizationBalancer(2, alpha=0.5)

    # Create test losses for multiple iterations
    losses1 = [torch.tensor(10.0), torch.tensor(1.0)]
    losses2 = [torch.tensor(20.0), torch.tensor(2.0)]

    # First iteration - should apply default weights (1.0)
    total_loss1 = balancer(losses1, iteration=0)
    assert abs(total_loss1.item() - 11.0) < 1e-5

    # Second iteration - should use updated weights based on first iteration
    total_loss2 = balancer(losses2, iteration=1)

    # Get weights and check they're inversely proportional to loss magnitudes
    weights = balancer.get_weights()

    # After first update with alpha=0.5:
    # loss_scales[0] = 0.5*1.0 + 0.5*10.0 = 5.5
    # loss_scales[1] = 0.5*1.0 + 0.5*1.0 = 1.0
    # So weights should be approximately [1/5.5, 1/1.0] or [0.18, 1.0]

    assert weights[0] < weights[1]
    ratio = weights[1] / weights[0]
    # The ratio should be roughly 5.5 (ratio of the loss scales)
    assert 5.0 < ratio < 7.5


def test_uncertainty_weight_balancer():
    """Test uncertainty weighting balancing strategy."""
    # Create balancer with default parameters
    balancer = UncertaintyWeightBalancer(2)

    # Initialize with some learnable parameters
    with torch.no_grad():
        # Set one uncertainty high and one low
        balancer.log_sigma_sq[0] = torch.log(torch.tensor(4.0))
        balancer.log_sigma_sq[1] = torch.log(torch.tensor(0.25))

    # Create test losses
    losses = [torch.tensor(2.0), torch.tensor(2.0)]

    # Compute balanced loss - precision is inversely proportional to variance
    total_loss = balancer(losses)

    # Check that the weights reflect the uncertainties
    weights = balancer.get_weights()
    # Weight for first loss should be lower (higher uncertainty)
    assert weights[0] < weights[1]

    # Ratio should be roughly 0.25 / 4.0 = 0.0625
    ratio = weights[0] / weights[1]
    assert 0.05 < ratio < 0.08


def test_adaptive_weight_balancer():
    """Test adaptive weighting balancing strategy."""
    # Create balancer with custom parameters
    balancer = AdaptiveWeightBalancer(2, alpha=0.5, window_size=2)

    # Create test losses for multiple iterations
    losses1 = [torch.tensor(10.0), torch.tensor(1.0)]
    losses2 = [torch.tensor(5.0), torch.tensor(1.0)]  # First loss improving

    # Run multiple iterations to fill history buffer
    balancer(losses1, iteration=0)
    balancer(losses2, iteration=1)

    # Third iteration - should adapt weights based on improvement rates
    losses3 = [torch.tensor(4.0), torch.tensor(0.9)]
    total_loss3 = balancer(losses3, iteration=2)

    # Get weights - should give higher weight to the loss improving less
    weights = balancer.get_weights()

    # Both losses are improving, but second loss is improving less proportionally
    assert abs(weights[0] + weights[1] - 2.0) < 1e-5  # Should sum to num_losses

    # Verify correct type and shape
    assert isinstance(total_loss3, torch.Tensor)
    assert total_loss3.shape == torch.Size([])


def test_loss_balancer_factory():
    """Test LossBalancer factory method."""
    # Test creation with string strategy
    balancer1 = LossBalancer.create("fixed", 2, [1.0, 2.0])
    assert isinstance(balancer1, FixedWeightBalancer)

    # Test creation with enum strategy
    balancer2 = LossBalancer.create(BalancingStrategy.SCALE, 2)
    assert isinstance(balancer2, ScaleNormalizationBalancer)

    # Test creation with custom parameters
    balancer3 = LossBalancer.create("uncertainty", 3, init_sigma=0.5)
    assert isinstance(balancer3, UncertaintyWeightBalancer)

    # Test fallback for unknown strategy
    balancer4 = LossBalancer.create("unknown_strategy", 2)
    assert isinstance(balancer4, FixedWeightBalancer)


def test_with_gradient_computation():
    """Test balancers with gradients."""
    # Create parameters that require gradients
    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([2.0], requires_grad=True)

    # Create losses that depend on these parameters
    loss1 = a**2
    loss2 = b**2

    # Test with scale normalization balancer
    balancer = ScaleNormalizationBalancer(2)
    total_loss = balancer([loss1, loss2])

    # Should be able to backpropagate
    total_loss.backward()

    # Both parameters should have gradients
    assert a.grad is not None
    assert b.grad is not None
