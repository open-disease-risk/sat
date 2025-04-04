# Specialized Survival Distributions Module

This module provides a collection of specialized probability distributions for survival analysis, with a focus on numerical stability and performance. The distributions are designed to be used with both Deep Survival Machines (DSM) and Multi-Event Neural Survival Analysis (MENSA) models.

## Features

- Numerically stable implementations of common survival distributions (Weibull, LogNormal)
- Support for mixture distributions with flexible component weighting
- Consistent API for all distributions with PyTorch-style implementation
- Careful handling of extreme values to prevent NaN/Inf errors
- Full support for automatic differentiation (backpropagation)
- Utility functions for creating distributions for DSM and MENSA models

## Available Distributions

1. **Weibull Distribution**
   - Standard and mixture variants
   - Parameterized by shape (k) and scale (λ)
   - Commonly used for accelerated and decelerated failure rates

2. **LogNormal Distribution**
   - Standard and mixture variants
   - Parameterized by location (μ) and scale (σ)
   - Good for modeling survival times with a skewed distribution

## Basic Usage

```python
import torch
from sat.distributions import WeibullDistribution, WeibullMixtureDistribution

# Create a Weibull distribution
batch_size = 4
shape = torch.tensor([1.5, 0.8, 2.0, 1.2])  # Shape parameter (k)
scale = torch.tensor([10.0, 8.0, 5.0, 12.0])  # Scale parameter (λ)

dist = WeibullDistribution(shape, scale)

# Evaluate survival function at different time points
time = torch.linspace(0.1, 20.0, steps=50).unsqueeze(0).expand(batch_size, -1)
survival = dist.survival_function(time)

# Compute hazard function
hazard = dist.hazard_function(time)

# For training models, compute log likelihood of observed times
event_times = torch.tensor([5.0, 3.0, 8.0, 2.0])
log_likelihood = dist.log_likelihood(event_times)
```

## Using with DSM and MENSA Models

The module provides utility functions to easily create distributions for DSM and MENSA models:

```python
from sat.distributions.utils import create_dsm_distribution, create_conditional_dsm_distribution

# For DSM model
batch_size, num_events, num_mixtures = 16, 2, 4
shape = model_output.shape  # From model output [batch_size, num_events, num_mixtures]
scale = model_output.scale  # From model output [batch_size, num_events, num_mixtures]
logits_g = model_output.logits_g  # From model output [batch_size, num_events, num_mixtures]

# Create distribution for a specific event
event_idx = 0
dsm_dist = create_dsm_distribution(
    shape, scale, logits_g, 
    distribution_type='weibull',
    event_idx=event_idx
)

# Evaluate survival function
time_points = torch.linspace(0.1, 20.0, steps=50).unsqueeze(0).expand(batch_size, -1)
survival = dsm_dist.survival_function(time_points)

# For MENSA model with event dependencies
dependency_matrix = model_output.event_dependency_matrix
observed_events = torch.zeros(batch_size, num_events)
observed_events[:, 0] = 1  # Assuming event 0 has occurred
observed_times = torch.zeros(batch_size, num_events)
observed_times[:, 0] = 5.0  # Time at which event 0 occurred

# Create conditional distribution for event 1
event_idx = 1
mensa_dist = create_conditional_dsm_distribution(
    shape, scale, logits_g,
    dependency_matrix=dependency_matrix,
    event_idx=event_idx,
    observed_events=observed_events,
    observed_times=observed_times,
    distribution_type='weibull'
)

# Evaluate conditional survival
conditional_survival = mensa_dist.survival_function(time_points)
```

## Numerical Stability

All distributions in this module employ techniques to ensure numerical stability:

1. **Log-domain operations**: Critical calculations use log-domain to prevent overflow/underflow
2. **Appropriate clamping**: Values are clamped to reasonable ranges at key points 
3. **Handling of extreme values**: Special handling for very small or large values
4. **Edge case detection**: Special handling for edge cases (e.g., shape < 1 in Weibull)

This makes them suitable for use with real-world data containing extreme values and for
integration with deep learning models during training.

## Integration with Model Heads

This module can be used to refactor the DSM and MENSA model heads for improved numerical stability.
Instead of implementing distribution calculations directly, the model heads can use these
specialized distributions to compute survival, hazard, and likelihood values.

Example integration with DSM model head:

```python
from sat.distributions.utils import create_dsm_distribution

# Inside DSMTaskHead._compute_survival_function
def _compute_survival_function(self, time_points, shape, scale, logits_g):
    """Compute survival function using specialized distributions."""
    batch_size, num_time_points = time_points.shape
    device = time_points.device
    
    # Initialize survival tensor
    survival = torch.zeros(batch_size, self.num_events, num_time_points, device=device)
    
    # Process each event separately
    for event_idx in range(self.num_events):
        # Create distribution for this event
        dist = create_dsm_distribution(
            shape, scale, logits_g, 
            distribution_type=self.distribution,
            event_idx=event_idx
        )
        
        # Compute survival function
        survival[:, event_idx, :] = dist.survival_function(time_points)
    
    return survival
```

Similar refactoring can be applied to hazard function calculation and loss computation.