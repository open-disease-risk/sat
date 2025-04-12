"""
Tests for the specialized survival distributions.
"""

import torch

from sat.distributions import (
    LogNormalDistribution,
    LogNormalMixtureDistribution,
    MixtureDistribution,
    WeibullDistribution,
    WeibullMixtureDistribution,
)
from sat.distributions.utils import (
    create_conditional_dsm_distribution,
    create_dsm_distribution,
)


def test_weibull_distribution():
    """Test basic functions of the Weibull distribution."""
    batch_size = 4

    # Create random parameters
    shape = torch.rand(batch_size) * 2.0 + 0.5  # 0.5 to 2.5
    scale = torch.rand(batch_size) * 9.0 + 1.0  # 1.0 to 10.0

    # Create distribution
    dist = WeibullDistribution(shape, scale)

    # Create time points for evaluation
    time = torch.linspace(0.1, 10.0, 20).unsqueeze(0).expand(batch_size, -1)

    # Test survival function
    survival = dist.survival_function(time)
    assert survival.shape == (batch_size, 20)
    assert torch.all(survival >= 0.0) and torch.all(survival <= 1.0)
    assert torch.all(
        survival[:, 0] > survival[:, -1]
    )  # Survival should decrease with time

    # Test hazard function
    hazard = dist.hazard_function(time)
    assert hazard.shape == (batch_size, 20)
    assert torch.all(hazard >= 0.0)

    # Test log likelihood
    event_time = torch.rand(batch_size) * 10.0
    log_likelihood = dist.log_likelihood(event_time)
    assert log_likelihood.shape == (batch_size,)
    assert torch.all(torch.isfinite(log_likelihood))

    # Test mean
    mean = dist.mean()
    assert mean.shape == (batch_size,)
    assert torch.all(mean > 0.0)


def test_weibull_mixture_distribution():
    """Test basic functions of the Weibull mixture distribution."""
    batch_size = 4
    num_mixtures = 3

    # Create random parameters
    shape = torch.rand(batch_size, num_mixtures) * 2.0 + 0.5  # 0.5 to 2.5
    scale = torch.rand(batch_size, num_mixtures) * 9.0 + 1.0  # 1.0 to 10.0
    logits = torch.randn(batch_size, num_mixtures)

    # Create distribution
    dist = WeibullMixtureDistribution(shape, scale, logits)

    # Create time points for evaluation
    time = torch.linspace(0.1, 10.0, 20).unsqueeze(0).expand(batch_size, -1)

    # Test survival function
    survival = dist.survival_function(time)
    assert survival.shape == (batch_size, 20)
    assert torch.all(survival >= 0.0) and torch.all(survival <= 1.0)
    assert torch.all(
        survival[:, 0] > survival[:, -1]
    )  # Survival should decrease with time

    # Test hazard function
    hazard = dist.hazard_function(time)
    assert hazard.shape == (batch_size, 20)
    assert torch.all(hazard >= 0.0)

    # Test log likelihood
    event_time = torch.rand(batch_size) * 10.0
    log_likelihood = dist.log_likelihood(event_time)
    assert log_likelihood.shape == (batch_size,)
    assert torch.all(torch.isfinite(log_likelihood))

    # Test mean
    mean = dist.mean()
    assert mean.shape == (batch_size,)
    assert torch.all(mean > 0.0)


def test_lognormal_distribution():
    """Test basic functions of the LogNormal distribution."""
    batch_size = 4

    # Create random parameters
    loc = torch.randn(batch_size)  # Mean of log(X)
    scale = torch.rand(batch_size) * 0.9 + 0.1  # 0.1 to 1.0

    # Create distribution
    dist = LogNormalDistribution(loc, scale)

    # Create time points for evaluation
    time = torch.linspace(0.1, 10.0, 20).unsqueeze(0).expand(batch_size, -1)

    # Test survival function
    survival = dist.survival_function(time)
    assert survival.shape == (batch_size, 20)
    assert torch.all(survival >= 0.0) and torch.all(survival <= 1.0)
    assert torch.all(
        survival[:, 0] > survival[:, -1]
    )  # Survival should decrease with time

    # Test hazard function
    hazard = dist.hazard_function(time)
    assert hazard.shape == (batch_size, 20)
    assert torch.all(hazard >= 0.0)

    # Test log likelihood
    event_time = torch.rand(batch_size) * 10.0
    log_likelihood = dist.log_likelihood(event_time)
    assert log_likelihood.shape == (batch_size,)
    assert torch.all(torch.isfinite(log_likelihood))

    # Test mean
    mean = dist.mean()
    assert mean.shape == (batch_size,)
    assert torch.all(mean > 0.0)


def test_lognormal_mixture_distribution():
    """Test basic functions of the LogNormal mixture distribution."""
    batch_size = 4
    num_mixtures = 3

    # Create random parameters
    loc = torch.randn(batch_size, num_mixtures)  # Mean of log(X)
    scale = torch.rand(batch_size, num_mixtures) * 0.9 + 0.1  # 0.1 to 1.0
    logits = torch.randn(batch_size, num_mixtures)

    # Create distribution
    dist = LogNormalMixtureDistribution(loc, scale, logits)

    # Create time points for evaluation
    time = torch.linspace(0.1, 10.0, 20).unsqueeze(0).expand(batch_size, -1)

    # Test survival function
    survival = dist.survival_function(time)
    assert survival.shape == (batch_size, 20)
    assert torch.all(survival >= 0.0) and torch.all(survival <= 1.0)
    assert torch.all(
        survival[:, 0] > survival[:, -1]
    )  # Survival should decrease with time

    # Test hazard function
    hazard = dist.hazard_function(time)
    assert hazard.shape == (batch_size, 20)
    assert torch.all(hazard >= 0.0)

    # Test log likelihood
    event_time = torch.rand(batch_size) * 10.0
    log_likelihood = dist.log_likelihood(event_time)
    assert log_likelihood.shape == (batch_size,)
    assert torch.all(torch.isfinite(log_likelihood))

    # Test mean
    mean = dist.mean()
    assert mean.shape == (batch_size,)
    assert torch.all(mean > 0.0)


def test_factory_functions():
    """Test the distribution factory functions."""
    batch_size = 4
    num_events = 2
    num_mixtures = 3

    # Create parameters for DSM model
    shape = torch.rand(batch_size, num_events, num_mixtures) * 2.0 + 0.5  # 0.5 to 2.5
    scale = torch.rand(batch_size, num_events, num_mixtures) * 9.0 + 1.0  # 1.0 to 10.0
    logits_g = torch.randn(batch_size, num_events, num_mixtures)

    # Create event dependency matrix for MENSA
    dependency_matrix = torch.zeros(num_events, num_events)
    dependency_matrix[1, 0] = 0.5  # Event 1 depends on event 0

    # Create observed events and times for conditional distribution
    observed_events = torch.zeros(batch_size, num_events)
    observed_events[:2, 0] = 1.0  # First two samples have event 0

    observed_times = torch.zeros(batch_size, num_events)
    observed_times[:2, 0] = torch.tensor([2.0, 4.0])  # Times for observed events

    # Test simple distribution creation
    for dist_type in ["weibull", "lognormal"]:
        # DSM distribution
        dsm_dist = create_dsm_distribution(
            shape, scale, logits_g, distribution_type=dist_type, event_idx=0
        )

        # Verify it's a mixture distribution
        assert isinstance(dsm_dist, MixtureDistribution)

        # Conditional MENSA distribution
        mensa_dist = create_conditional_dsm_distribution(
            shape,
            scale,
            logits_g,
            dependency_matrix,
            event_idx=1,
            observed_events=observed_events,
            observed_times=observed_times,
            distribution_type=dist_type,
        )

        # Verify it's a mixture distribution
        assert isinstance(mensa_dist, MixtureDistribution)

        # Create time points for evaluation
        time = torch.linspace(0.1, 10.0, 20).unsqueeze(0).expand(batch_size, -1)

        # Test that both distributions produce valid survival functions
        dsm_survival = dsm_dist.survival_function(time)
        mensa_survival = mensa_dist.survival_function(time)

        assert torch.all(torch.isfinite(dsm_survival))
        assert torch.all(torch.isfinite(mensa_survival))
        assert torch.all(dsm_survival >= 0.0) and torch.all(dsm_survival <= 1.0)
        assert torch.all(mensa_survival >= 0.0) and torch.all(mensa_survival <= 1.0)


def test_numerical_stability():
    """Test distributions with extreme parameter values."""
    batch_size = 4

    # Create extreme parameter values
    small_shape = torch.tensor([0.01, 0.1, 1.0, 10.0])
    large_shape = torch.tensor([0.5, 1.0, 10.0, 100.0])

    small_scale = torch.tensor([0.01, 0.1, 1.0, 10.0])
    large_scale = torch.tensor([0.1, 1.0, 10.0, 1000.0])

    extreme_loc = torch.tensor([-100.0, -10.0, 10.0, 100.0])
    extreme_scale = torch.tensor([0.01, 0.1, 1.0, 10.0])

    # Test Weibull with extreme parameters
    dist_small = WeibullDistribution(small_shape, small_scale)
    dist_large = WeibullDistribution(large_shape, large_scale)

    # Create time points including extreme values
    time = torch.tensor([[1e-6, 1e-3, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e6]]).expand(
        batch_size, -1
    )

    # Test survival functions
    survival_small = dist_small.survival_function(time)
    survival_large = dist_large.survival_function(time)

    # Ensure results are finite and within valid range
    assert torch.all(torch.isfinite(survival_small))
    assert torch.all(torch.isfinite(survival_large))
    assert torch.all(survival_small >= 0.0) and torch.all(survival_small <= 1.0)
    assert torch.all(survival_large >= 0.0) and torch.all(survival_large <= 1.0)

    # Test LogNormal with extreme parameters
    dist_extreme = LogNormalDistribution(extreme_loc, extreme_scale)

    # Test survival function
    survival_extreme = dist_extreme.survival_function(time)

    # Ensure results are finite and within valid range
    assert torch.all(torch.isfinite(survival_extreme))
    assert torch.all(survival_extreme >= 0.0) and torch.all(survival_extreme <= 1.0)


def test_backward():
    """Test that distributions support backpropagation."""
    batch_size = 4
    num_mixtures = 3

    # Create trainable parameters - use raw tensors
    shape_raw = torch.rand(batch_size, num_mixtures) * 2.0 + 0.5
    scale_raw = torch.rand(batch_size, num_mixtures) * 9.0 + 1.0
    logits_raw = torch.randn(batch_size, num_mixtures)

    # Make them require gradients
    shape = shape_raw.clone().detach().requires_grad_(True)
    scale = scale_raw.clone().detach().requires_grad_(True)
    logits = logits_raw.clone().detach().requires_grad_(True)

    # Create distribution
    dist = WeibullMixtureDistribution(shape, scale, logits)

    # Create time points and target survival values
    time = torch.linspace(0.1, 10.0, 20).unsqueeze(0).expand(batch_size, -1)

    # Compute survival
    survival = dist.survival_function(time)

    # Create a simple loss (mean of survival values)
    loss = survival.mean()

    # Backward pass
    loss.backward()

    # Check if gradients are computed and are finite
    assert shape.grad is not None
    assert scale.grad is not None
    assert logits.grad is not None

    assert torch.all(torch.isfinite(shape.grad))
    assert torch.all(torch.isfinite(scale.grad))
    assert torch.all(torch.isfinite(logits.grad))

    # Repeat for LogNormal
    loc_raw = torch.randn(batch_size, num_mixtures)
    scale_raw = torch.rand(batch_size, num_mixtures) * 0.9 + 0.1
    logits_raw = torch.randn(batch_size, num_mixtures)

    # Make them require gradients
    loc = loc_raw.clone().detach().requires_grad_(True)
    scale = scale_raw.clone().detach().requires_grad_(True)
    logits = logits_raw.clone().detach().requires_grad_(True)

    # Create distribution
    dist = LogNormalMixtureDistribution(loc, scale, logits)

    # Compute survival
    survival = dist.survival_function(time)

    # Create a simple loss
    loss = survival.mean()

    # Backward pass
    loss.backward()

    # Check gradients
    assert loc.grad is not None
    assert scale.grad is not None
    assert logits.grad is not None

    assert torch.all(torch.isfinite(loc.grad))
    assert torch.all(torch.isfinite(scale.grad))
    assert torch.all(torch.isfinite(logits.grad))
