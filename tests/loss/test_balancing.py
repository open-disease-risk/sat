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
        # log_var is 2*log(sigma), so we set different initial sigmas
        balancer.log_var[0] = 2.0 * torch.log(torch.tensor(2.0))  # high uncertainty
        balancer.log_var[1] = 2.0 * torch.log(torch.tensor(0.5))  # low uncertainty

    # Create test losses
    losses = [
        torch.tensor(2.0, requires_grad=True),
        torch.tensor(2.0, requires_grad=True),
    ]

    # Compute balanced loss - precision is inversely proportional to variance
    total_loss = balancer(losses)

    # Check that the weights reflect the uncertainties
    weights = balancer.get_weights()
    # Weight for first loss should be lower (higher uncertainty)
    assert weights[0] < weights[1]

    # Ratio should be roughly 0.5^2 / 2.0^2 = 0.25/4 = 0.0625
    ratio = weights[0] / weights[1]
    assert 0.05 < ratio < 0.08

    # Test gradient flow - the loss should be able to backpropagate
    total_loss.backward()

    # Verify that log_var parameters received gradients
    assert balancer.log_var.grad is not None
    assert balancer.log_var.grad.shape == torch.Size([2])


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


def test_safe_uncertainty_function_optimization():
    """Test that SafeUncertaintyFunction allows proper gradient updates."""
    from sat.loss.balancing import SafeUncertaintyFunction

    # Create a simple optimization scenario
    # We'll create two dummy losses with different scales
    loss1 = torch.tensor(10.0, requires_grad=True)  # Large loss
    loss2 = torch.tensor(1.0, requires_grad=True)  # Small loss

    # Initialize log variances as learnable parameters
    log_var1 = torch.tensor(0.0, requires_grad=True)  # Start with equal weights
    log_var2 = torch.tensor(0.0, requires_grad=True)  # Start with equal weights

    # Create optimizer for log variances
    optimizer = torch.optim.SGD([log_var1, log_var2], lr=0.1)

    # Run several optimization steps
    initial_log_var1 = log_var1.item()
    initial_log_var2 = log_var2.item()

    for _ in range(5):
        # Clear gradients
        optimizer.zero_grad()

        # Apply uncertainty weighting to each loss
        weighted_loss1 = SafeUncertaintyFunction.apply(loss1, log_var1)
        weighted_loss2 = SafeUncertaintyFunction.apply(loss2, log_var2)

        # Total loss
        total_loss = weighted_loss1 + weighted_loss2

        # Backward pass
        total_loss.backward()

        # Update log variances
        optimizer.step()

    # Check that log variances have been updated
    assert log_var1.item() != initial_log_var1
    assert log_var2.item() != initial_log_var2

    # The larger loss should get a larger log variance (more uncertainty, less weight)
    # because otherwise it would dominate the total loss
    assert log_var1.item() > log_var2.item()

    # Convert to precisions (weights)
    precision1 = torch.exp(-log_var1).item()
    precision2 = torch.exp(-log_var2).item()

    # The smaller loss should get higher weight
    assert precision1 < precision2


def test_uncertainty_balancer_optimization():
    """Test that UncertaintyWeightBalancer learns appropriate weights through optimization."""
    # Create model parameters
    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([2.0], requires_grad=True)

    # Create balancer
    balancer = UncertaintyWeightBalancer(2)

    # Create optimizers
    model_optimizer = torch.optim.SGD([a, b], lr=0.1)
    balancer_optimizer = torch.optim.SGD(balancer.parameters(), lr=0.1)

    # Save initial log variances
    initial_log_var = balancer.log_var.detach().clone()

    # Run several optimization steps
    for _ in range(10):
        # Clear gradients
        model_optimizer.zero_grad()
        balancer_optimizer.zero_grad()

        # Create losses with different scales
        loss1 = 10.0 * (a**2)  # Larger scale loss
        loss2 = 1.0 * (b**2)  # Smaller scale loss

        # Apply balancer
        total_loss = balancer([loss1, loss2])

        # Backward pass
        total_loss.backward()

        # Update parameters
        model_optimizer.step()
        balancer_optimizer.step()

    # Check that log variances have been updated
    assert not torch.allclose(balancer.log_var, initial_log_var)

    # Get final weights
    weights = balancer.get_weights()

    # The larger loss (loss1) should get a smaller weight to balance optimization
    assert weights[0] < weights[1]
