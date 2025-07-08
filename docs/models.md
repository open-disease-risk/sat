# Survival Analysis Models

This document provides an overview of the different survival analysis models implemented in the Survival Analysis Toolkit (SAT).

## Model Overview

SAT implements several state-of-the-art survival analysis models that can be used for different survival prediction tasks:

| Model | Description | Best For | Documentation |
|-------|-------------|----------|--------------|
| Survival Task Head | Standard survival model with discrete time bins | General survival analysis | [Basic Documentation](#survival-task-head) |
| DeepHit | Discrete survival model with ranking-based loss | Competing risks, discrete time | [Full Documentation](deephit.md) |
| DSM | Deep Survival Machines with mixture of parametric distributions | Flexible distribution shapes | [Full Documentation](dsm.md) |
| MENSA | Multi-Event Neural Survival Analysis with event dependencies | Multi-event scenarios with dependencies | [Full Documentation](mensa.md) |

## Choosing the Right Model

### Single-Event Scenarios

- **Survival Task Head**: Good baseline model for standard survival analysis
- **DSM**: Better for capturing complex distribution shapes and uncertainty estimation

### Multi-Event (Competing Risks) Scenarios

- **DeepHit**: Strong performance for competing risks with ranking-based training
- **MENSA**: Best when events have dependencies and continuous time modeling is needed

### Considerations

- **Data Size**: For smaller datasets, simpler models like Survival Task Head may be more robust
- **Time Representation**: DSM and MENSA offer continuous time modeling, while others use discrete bins
- **Interpretation**: MENSA offers interpretable event dependencies
- **Performance**: DeepHit tends to perform well in benchmarks for competing risks
- **Uncertainty**: DSM and MENSA provide distribution parameters that can be used for uncertainty quantification

## Performance Comparison

Based on our benchmarks, here's how the different models compare in terms of performance:

| Model | Training Speed | Inference Speed | Scaling with Events | Memory Usage |
|-------|---------------|-----------------|---------------------|--------------|
| Survival Task Head | ★★★★☆ | ★★★★★ | ★★★★☆ | ★★★★★ |
| DeepHit | ★★★☆☆ | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| DSM | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ | ★★★☆☆ |
| MENSA (no deps) | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★★☆☆ |
| MENSA (with deps) | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ | ★★☆☆☆ |

### Detailed Performance Insights

1. **Training Time** (2 events, batch size 32):
   - MENSA (no dependencies): 0.69s/epoch
   - MENSA (with dependencies): 0.79s/epoch
   - DSM: 1.69s/epoch

2. **Inference Time** (2 events, batch size 32):
   - MENSA (no dependencies): 0.022s
   - MENSA (with dependencies): 0.027s
   - DSM: 0.064s

3. **Scaling with Event Count**:
   - As the number of events increases from 2 to 8:
     - DSM: 17.8% increase in training time
     - MENSA (no dependencies): 56.5% increase
     - MENSA (with dependencies): 75.9% increase

4. **Memory Usage**:
   - MENSA with dependencies requires additional memory for the dependency matrix (O(E²) where E is the number of events)
   - This becomes more significant as the number of events increases

For complete benchmark results, see the [Performance Comparison](performance_comparison.md) documentation.

## Survival Task Head

The Survival Task Head is a basic neural network model for survival analysis. It predicts discrete hazard rates at predefined time points.

### Key Features

- Discrete time bins for survival prediction
- Compatible with transformer-based feature extractors
- Configurable for both single-event and competing risks scenarios

### Usage Example

```yaml
# conf/tasks/survival.yaml
_target_: sat.models.heads.SurvivalTaskHead
num_features: ${..num_features}
num_events: ${..num_events}
num_hidden_layers: 2
intermediate_size: 64
batch_norm: true
bias: true
hidden_dropout_prob: 0.1
loss_weight: 1.0
```

## Model Integration

All survival models in SAT are designed to work with the transformer architecture:

1. Input data → Transformer model → Feature embeddings
2. Feature embeddings → Survival model → Survival predictions

This consistent interface makes it easy to swap different survival models while keeping the same transformer backbone.

## Multi-Task Learning Integration

All survival models can be used as component heads in multi-task learning setups. For example:

```yaml
# Using MENSA in a multi-task learning configuration
task_heads:
  - _target_: sat.models.heads.MENSATaskHead
    num_features: ${..num_features}
    num_events: ${..num_events}
    # ... other MENSA parameters

  - _target_: sat.models.heads.EventClassificationTaskHead
    # ... classification parameters
```

## Metrics and Evaluation

Different models may excel on different evaluation metrics:

- **C-index**: All models optimize for discrimination (ranking)
- **Calibration**: DSM and MENSA may offer better calibration due to distribution modeling
- **Brier Score**: Measures both discrimination and calibration

## Further Reading

- [Loss Functions Documentation](loss.md)
- [Performance Comparisons](performance_comparison.md)
- Specific model documentation:
  - [DeepHit Documentation](deephit.md)
  - [DSM Documentation](dsm.md)
  - [MENSA Documentation](mensa.md)
