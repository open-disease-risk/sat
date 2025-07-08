# Survival Analysis Toolkit (SAT) Loss Functions

This directory contains various loss functions used in the Survival Analysis Toolkit.

## SurvivalFocalLoss

The `SurvivalFocalLoss` class implements focal loss for survival analysis tasks, particularly useful for imbalanced datasets and competing risks scenarios. It applies focal loss to the survival predictions directly, focusing the model's attention on harder-to-predict survival outcomes.

The focal loss formula is:
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

where:
- `p_t` is the model's estimated survival probability for the event
- `γ` (gamma) is the focusing parameter that down-weights easy examples
- `α_t` is a weighting factor for class imbalance, implemented using importance weights

This loss is specifically designed to work with the `SAOutput` from survival models, using the survival probabilities directly. It works very well for competing risks scenarios where you want to focus learning on specific event types that may be harder to predict.

### Multi-Focal Parameters

The implementation supports both a single global gamma value or different gamma values for each event type (multi-focal parameters).

When dealing with competing risks or multiple event types, each event might have different characteristics:
- Some events might be rarer and harder to predict
- Some events might have more imbalanced positive/negative examples
- Some events might benefit from different focusing strengths

Using multi-focal parameters allows you to customize the focusing strength for each event type individually, potentially improving model performance.

### Importance Weights

Instead of directly specifying alpha values in the code, the FocalLoss implementation uses importance weights loaded from a CSV file. This ensures consistency with other loss functions in the framework that also use importance weights.

The importance weights file should contain one weight per line, with:
- First line: Weight for the background/no-event class
- Subsequent lines: Weights for each event type

These weights are used to give more or less importance to different classes, helping to address class imbalance.

### Usage Examples

#### Basic Usage (Single Gamma)
```python
from sat.loss import FocalLoss

# Create a focal loss with a single gamma value for all events
loss_fn = FocalLoss(
    gamma=2.0,                             # Single gamma for all events
    importance_sample_weights="path/to/weights.csv",  # Class weighting
    num_events=2,                          # Number of competing events
    reduction="mean"                       # Reduction method (mean, sum, none)
)
```

#### Multi-Focal Usage (Multiple Gamma Values)
```python
from sat.loss import FocalLoss

# Create a focal loss with different gamma values for each event type
loss_fn = FocalLoss(
    gamma=[1.0, 3.0],                      # Lower gamma for event 0, higher for event 1
    importance_sample_weights="path/to/weights.csv",  # Class weighting
    num_events=2,
    reduction="mean"
)
```

#### Configuration Examples

Single gamma (conf/tasks/losses/survival_focal.yaml):
```yaml
# @package _group_.loss
_target_: sat.loss.SurvivalFocalLoss
gamma: 2.0           # Single focusing parameter
importance_sample_weights: ${paths.importance_sample_weights_path}
reduction: mean
num_events: ${data.num_events}
```

Multi-focal parameters (conf/tasks/losses/multi_survival_focal.yaml):
```yaml
# @package _group_.loss
_target_: sat.loss.SurvivalFocalLoss
gamma: [2.0, 3.0]    # Multiple focusing parameters, one per event type
importance_sample_weights: ${paths.importance_sample_weights_path}
reduction: mean
num_events: ${data.num_events}
```

## MetaLoss

The `MetaLoss` class provides a framework for combining multiple loss components with flexible weighting strategies.

It can be used to combine FocalLoss with other loss functions or to combine multiple FocalLoss instances with different configurations.

## Other Loss Functions

The SAT toolkit includes various other loss functions for different tasks:
- Regression losses (L1, MSE, Quantile)
- Classification losses (CrossEntropy)
- Ranking losses (SampleRanking, MultiEventRanking)
- Survival-specific losses (NLLPCHazard, Mismatch, DeepHit components, DSM loss, MENSA loss)

### MENSA Loss

The `MENSALoss` class implements a specialized loss function for the Multi-Event Neural Survival Analysis (MENSA) model. It combines:

1. Negative log-likelihood for uncensored data
2. Negative log-survival for censored data
3. Regularization for the event dependency matrix

This loss is particularly useful for multi-event scenarios where events may have dependencies on each other. For detailed information about MENSA, see the [MENSA Documentation](mensa.md).

#### Usage Example

```yaml
# conf/tasks/losses/mensa.yaml
_target_: sat.loss.survival.MENSALoss
duration_cuts: ${paths.duration_cuts}
importance_sample_weights: ${paths.importance_sample_weights}
num_events: ${..num_events}
distribution: "weibull"
discount: 1.0
elbo: false
dependency_regularization: 0.01  # Controls the strength of dependency regularization
```

The `dependency_regularization` parameter controls how strongly the model is encouraged to use sparse dependencies between events. Higher values promote more independence, while lower values allow more complex dependency structures.
