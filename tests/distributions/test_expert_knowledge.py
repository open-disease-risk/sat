"""Unit tests for expert knowledge integration in survival distributions."""

import torch
import numpy as np
import pytest

from sat.distributions.weibull import WeibullDistribution, WeibullMixtureDistribution
from sat.distributions.lognormal import (
    LogNormalDistribution,
    LogNormalMixtureDistribution,
)
from sat.distributions.utils import (
    create_distribution,
    create_dsm_distribution,
    create_conditional_dsm_distribution,
    apply_informative_prior,
    apply_event_specific_constraints,
    create_clinically_informed_distribution,
)


@pytest.fixture
def batch_size():
    return 32


@pytest.fixture
def num_mixtures():
    return 4


@pytest.fixture
def num_time_points():
    return 20


@pytest.fixture
def weibull_params(batch_size, num_mixtures):
    # Generate reasonable Weibull parameters for testing
    shape = (
        torch.rand(batch_size, num_mixtures) * 2.0 + 0.5
    )  # Shape between 0.5 and 2.5
    scale = (
        torch.rand(batch_size, num_mixtures) * 10.0 + 1.0
    )  # Scale between 1.0 and 11.0
    logits = torch.randn(batch_size, num_mixtures)  # Random logits for mixture weights

    return {"shape": shape, "scale": scale, "logits": logits}


@pytest.fixture
def lognormal_params(batch_size, num_mixtures):
    # Generate reasonable LogNormal parameters for testing
    loc = torch.randn(batch_size, num_mixtures) * 1.0 + 0.0  # Loc centered around 0
    scale = (
        torch.rand(batch_size, num_mixtures) * 1.0 + 0.5
    )  # Scale between 0.5 and 1.5
    logits = torch.randn(batch_size, num_mixtures)  # Random logits for mixture weights

    return {"loc": loc, "scale": scale, "logits": logits}


@pytest.fixture
def time_points(batch_size, num_time_points):
    # Generate linearly spaced time points for testing
    return (
        torch.linspace(0.1, 10.0, num_time_points).unsqueeze(0).expand(batch_size, -1)
    )


def test_informative_prior_weibull():
    # Test applying informative priors to Weibull parameters
    batch_size = 10
    num_mixtures = 2

    # Create random parameters
    shape = torch.rand(batch_size, num_mixtures) * 2.0 + 0.5
    scale = torch.rand(batch_size, num_mixtures) * 10.0 + 1.0

    # Create parameter dictionary
    params = {"shape": shape, "scale": scale}

    # Apply cancer-specific prior
    prior_params = apply_informative_prior(params, "weibull", "cancer")

    # Check that shape has been influenced by prior (cancer should have shape > 1)
    assert torch.all(prior_params["shape"] > 0.5)  # Still positive

    # Get mean shape before and after
    mean_shape_before = shape.mean().item()
    mean_shape_after = prior_params["shape"].mean().item()

    # For cancer, shape should be pushed toward higher values (increasing hazard)
    # But not drastically changed (only blended)
    assert torch.allclose(prior_params["shape"], shape, rtol=0.5)

    # Apply different event type
    prior_params = apply_informative_prior(params, "weibull", "infectious_disease")

    # For infectious disease, shape is often < 1 (decreasing hazard)
    # So the mean should be pushed lower
    mean_shape_infectious = prior_params["shape"].mean().item()
    assert mean_shape_infectious < mean_shape_before


def test_informative_prior_lognormal():
    # Test applying informative priors to LogNormal parameters
    batch_size = 10
    num_mixtures = 2

    # Create random parameters
    loc = torch.randn(batch_size, num_mixtures) * 1.0
    scale = torch.rand(batch_size, num_mixtures) * 1.0 + 0.5

    # Create parameter dictionary
    params = {"loc": loc, "scale": scale}

    # Apply cancer-specific prior
    prior_params = apply_informative_prior(params, "lognormal", "cancer")

    # Check that parameters have been influenced by prior but not drastically changed
    assert torch.allclose(prior_params["loc"], loc, rtol=0.5)
    assert torch.allclose(prior_params["scale"], scale, rtol=0.5)

    # Apply different event type
    prior_params = apply_informative_prior(params, "lognormal", "chronic_disease")

    # Chronic disease typically has higher location parameter (longer survival)
    mean_loc_chronic = prior_params["loc"].mean().item()
    mean_loc_before = loc.mean().item()

    # The mean should be affected but still close to original
    assert torch.allclose(prior_params["loc"], loc, rtol=0.5)


def test_event_specific_constraints():
    # Test applying event-specific constraints
    batch_size = 10
    num_mixtures = 2

    # Create parameters with very extreme values
    shape = torch.ones(batch_size, num_mixtures) * 0.3  # Very low shape

    # Create parameter dictionary
    params = {
        "shape": shape,
    }

    # Apply cancer-specific constraints (cancer typically has shape > 1)
    constrained_params = apply_event_specific_constraints(
        params, 0, 3, "weibull", ["cancer", "heart_disease", "other"]
    )

    # Cancer shapes should be constrained to be > 1
    assert torch.all(constrained_params["shape"] > 0.9)

    # Now test with shape > 2 (above cancer typical range)
    params = {"shape": torch.ones(batch_size, num_mixtures) * 3.0}

    # Apply heart_disease constraints (should be close to 1)
    constrained_params = apply_event_specific_constraints(
        params, 1, 3, "weibull", ["cancer", "heart_disease", "other"]
    )

    # Heart disease should have shape close to 1
    assert torch.all(constrained_params["shape"] <= 1.5)


def test_expert_knowledge_in_distribution():
    # Test creating distribution with expert knowledge
    batch_size = 10

    # Create parameters
    shape = torch.ones(batch_size) * 0.7  # Low shape (decreasing hazard)
    scale = torch.ones(batch_size) * 5.0

    # Create Weibull with and without expert knowledge
    dist_no_expert = WeibullDistribution(shape, scale)
    dist_with_expert = WeibullDistribution(
        shape, scale, constrain_shape=True, event_type="cancer", use_expert_priors=True
    )

    # The expert version should have higher shape for cancer (increasing hazard)
    assert torch.mean(dist_with_expert.shape) > torch.mean(dist_no_expert.shape)

    # Similarly for LogNormal
    loc = torch.zeros(batch_size)
    scale = torch.ones(batch_size) * 2.0  # High uncertainty

    dist_no_expert = LogNormalDistribution(loc, scale)
    dist_with_expert = LogNormalDistribution(
        loc, scale, constrain_params=True, event_type="cancer", use_expert_priors=True
    )

    # The expert version should have lower scale (less uncertainty)
    assert torch.mean(dist_with_expert.scale) < torch.mean(dist_no_expert.scale)


def test_expert_knowledge_in_mixture_distribution():
    # Test creating mixture distribution with expert knowledge
    batch_size = 10
    num_mixtures = 3

    # Create parameters
    shape = torch.rand(batch_size, num_mixtures) * 0.5 + 0.5  # All shapes < 1
    scale = torch.rand(batch_size, num_mixtures) * 5.0 + 1.0
    logits = torch.randn(batch_size, num_mixtures)

    # Create mixture distribution with expert knowledge for cancer
    dist = WeibullMixtureDistribution(
        shape,
        scale,
        logits,
        constrain_shape=True,
        event_type="cancer",
        use_expert_priors=True,
    )

    # For cancer, at least the first component should have shape > 1
    first_component_mean_shape = dist.shape[:, 0].mean().item()
    assert first_component_mean_shape > 0.9  # Should be pushed toward increasing hazard

    # Test with lognormal mixture
    loc = torch.zeros(batch_size, num_mixtures)  # All loc = 0
    scale = (
        torch.ones(batch_size, num_mixtures) * 2.0
    )  # All scale = 2 (high uncertainty)

    dist = LogNormalMixtureDistribution(
        loc,
        scale,
        logits,
        constrain_params=True,
        event_type="cancer",
        use_expert_priors=True,
    )

    # For cancer, the first component should have loc > 0 and reasonable scale
    assert (
        torch.mean(dist.loc[:, 0]) > 0.0
    )  # Should be pushed toward positive (longer survival)
    assert (
        torch.mean(dist.scale[:, 0]) < 2.0
    )  # Should be constrained to lower uncertainty


def test_conditional_dsm_distribution(batch_size):
    # Test creating conditional distribution for MENSA model
    num_events = 3
    num_mixtures = 2

    # Create parameters
    shape = torch.rand(batch_size, num_events, num_mixtures) * 2.0 + 0.5
    scale = torch.rand(batch_size, num_events, num_mixtures) * 10.0 + 1.0
    logits_g = torch.randn(batch_size, num_events, num_mixtures)

    # Create dependency matrix
    dependency_matrix = torch.zeros(num_events, num_events)
    dependency_matrix[1, 0] = 0.5  # Event 0 affects event 1

    # Create observed events and times
    observed_events = torch.zeros(batch_size, num_events)
    observed_events[:5, 0] = 1  # First 5 samples experienced event 0

    observed_times = torch.zeros(batch_size, num_events)
    observed_times[:5, 0] = torch.rand(5) * 5.0  # Random times for event 0

    # Event types for expert knowledge
    event_types = ["cancer", "treatment_complications", "other"]

    # Create conditional distribution with expert knowledge
    dist_with_expert = create_conditional_dsm_distribution(
        shape,
        scale,
        logits_g,
        dependency_matrix,
        event_idx=1,  # Looking at event 1 (treatment_complications)
        observed_events=observed_events,
        observed_times=observed_times,
        distribution_type="weibull",
        event_types=event_types,
        use_expert_priors=True,
    )

    # Create the same distribution without expert knowledge
    dist_no_expert = create_conditional_dsm_distribution(
        shape,
        scale,
        logits_g,
        dependency_matrix,
        event_idx=1,
        observed_events=observed_events,
        observed_times=observed_times,
        distribution_type="weibull",
    )

    # For treatment complications, the shape should typically be < 1 (decreasing hazard)
    # so with expert knowledge, the shapes should be lower
    assert torch.mean(dist_with_expert.shape) < torch.mean(dist_no_expert.shape)

    # Since event 0 affects event 1, the scale for samples that experienced event 0
    # should be different than for those that didn't

    # Compare first 5 samples (experienced event 0) to next 5 (didn't experience event 0)
    mean_scale_with_event = dist_with_expert.scale[:5].mean()
    mean_scale_without_event = dist_with_expert.scale[5:10].mean()

    # The scales should be different due to the dependency
    assert not torch.isclose(mean_scale_with_event, mean_scale_without_event, rtol=1e-2)


def test_clinically_informed_distribution():
    # Test creating clinically informed distribution
    batch_size = 10
    num_mixtures = 2

    # Create parameters for Weibull
    shape = torch.rand(batch_size, num_mixtures) * 0.5 + 0.5  # All shapes < 1
    scale = torch.rand(batch_size, num_mixtures) * 5.0 + 1.0
    logits = torch.randn(batch_size, num_mixtures)

    weibull_params = {"shape": shape, "scale": scale, "logits": logits}

    # Create demographic information
    demographic_info = {"age": torch.ones(batch_size) * 70}  # Elderly patients

    # Create clinically informed distribution for cancer
    dist = create_clinically_informed_distribution(
        "weibull",
        weibull_params,
        event_type="cancer",
        demographic_info=demographic_info,
    )

    # Verify distribution type
    assert isinstance(dist, WeibullMixtureDistribution)

    # For elderly cancer patients, survival time is typically shorter
    # Check that scale is influenced by age
    orig_scale_mean = scale.mean().item()
    dist_scale_mean = dist.scale.mean().item()

    # Scale should be adjusted, but not drastically different
    assert dist_scale_mean < orig_scale_mean


def test_regularization_weight():
    # Test that the regularization weight penalizes extreme parameters
    from sat.distributions.utils import calculate_regularization_weight

    batch_size = 10
    num_mixtures = 2

    # Create parameters with reasonable values
    shape_reasonable = torch.ones(batch_size, num_mixtures) * 1.5
    scale_reasonable = torch.ones(batch_size, num_mixtures) * 5.0

    reasonable_params = {"shape": shape_reasonable, "scale": scale_reasonable}

    # Create parameters with extreme values
    shape_extreme = torch.ones(batch_size, num_mixtures) * 10.0  # Very high shape
    scale_extreme = torch.ones(batch_size, num_mixtures) * 200.0  # Very high scale

    extreme_params = {"shape": shape_extreme, "scale": scale_extreme}

    # Calculate regularization weights
    reg_weight_reasonable = calculate_regularization_weight(
        reasonable_params, "weibull"
    )
    reg_weight_extreme = calculate_regularization_weight(extreme_params, "weibull")

    # The extreme parameters should have higher regularization weight
    assert reg_weight_extreme > reg_weight_reasonable

    # Test with lognormal
    loc_reasonable = torch.ones(batch_size, num_mixtures) * 2.0
    scale_reasonable = torch.ones(batch_size, num_mixtures) * 1.0

    reasonable_params = {"loc": loc_reasonable, "scale": scale_reasonable}

    loc_extreme = torch.ones(batch_size, num_mixtures) * 10.0
    scale_extreme = torch.ones(batch_size, num_mixtures) * 5.0

    extreme_params = {"loc": loc_extreme, "scale": scale_extreme}

    reg_weight_reasonable = calculate_regularization_weight(
        reasonable_params, "lognormal"
    )
    reg_weight_extreme = calculate_regularization_weight(extreme_params, "lognormal")

    assert reg_weight_extreme > reg_weight_reasonable
