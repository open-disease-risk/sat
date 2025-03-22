# Multi-level Loss Optimization in Survival Analysis

The Survival Analysis Toolkit (SAT) implements a comprehensive multi-level approach to loss optimization, addressing different aspects of the training challenge in survival analysis. This document explains the three-tiered strategy that allows for fine-grained control over the learning process.

## Overview of the Multi-layered Approach

SAT's loss optimization framework operates at three distinct levels:

1. **Class/Sample Level**: Addressing class imbalance with importance weighting
2. **Difficulty Level**: Adapting to sample difficulty through focal parameters 
3. **Component Level**: Balancing multiple loss components dynamically

These levels work together to create a flexible and powerful training framework that can be tuned to the specific characteristics of different survival analysis datasets and tasks.

## 1. Class/Sample Level: Importance Weighting

At the most basic level, SAT addresses class imbalance through importance weighting mechanisms.

### Problem
In survival analysis, particularly with competing risks, there's often significant imbalance between:
- Event occurrences vs. censored observations
- Different event types (some events may be much rarer than others)
- Temporal intervals (later time bins typically have fewer events)

### Solution: Importance Weighting
- **Implementation**: All SAT loss functions support loading importance weights from a CSV file via the `importance_sample_weights` parameter
- **Weighting Scheme**: 
  - First weight in the file corresponds to background/censored class
  - Subsequent weights correspond to each event type
- **Effect**: Rare events or underrepresented classes are given higher weight during training

### Example Configuration
```yaml
_target_: sat.loss.SATNLLPCHazardLoss
importance_sample_weights: ${paths.importance_sample_weights_path}
num_events: ${data.num_events}
```

## 2. Difficulty Level: Focal Parameters

Beyond addressing class imbalance, SAT can adjust learning based on sample difficulty using focal parameters.

### Problem
Even within a class, some samples are harder to classify correctly than others:
- Ambiguous cases near decision boundaries
- Outliers or unusual presentations
- Samples with complex feature interactions

### Solution: Survival Focal Loss with Multi-focal Parameters
- **Implementation**: `SurvivalFocalLoss` class with configurable gamma parameters
- **Focusing Mechanism**: Down-weights easily predicted survival outcomes to focus learning on more difficult predictions
- **Multi-focal Extension**: Different focusing parameters (gamma) for each event type
- **Effect**: 
  - Higher gamma values increase focus on harder samples
  - Allows different focusing strengths for different event types

### Example Configuration
```yaml
_target_: sat.loss.SurvivalFocalLoss
gamma: [2.0, 3.0]  # Different focusing parameters per event type
importance_sample_weights: ${paths.importance_sample_weights_path}
num_events: ${data.num_events}
```

## 3. Component Level: Loss Balancing Framework

At the highest level, SAT provides sophisticated mechanisms to balance multiple loss components.

### Problem
Survival models often combine multiple loss objectives:
- Likelihood losses (measure fit to observed data)
- Ranking losses (ensure correct ordering of risk predictions)
- Calibration losses (ensure predicted probabilities match empirical frequencies)
- Additional task-specific losses

These components can have different:
- Scales (orders of magnitude)
- Convergence rates
- Optimization landscapes
- Relative importance to final performance

### Solution: Dynamic Loss Balancing
- **Implementation**: `MetaLoss` with multiple balancing strategies
- **Strategy Options**:
  1. `fixed`: Standard fixed coefficients
  2. `scale`: Normalizes by loss magnitude using exponential moving average
  3. `grad`: Balances by gradient magnitude
  4. `uncertainty`: Homoscedastic uncertainty weighting (learned)
  5. `adaptive`: Adjusts based on loss improvement trends

- **Effect**: Prevents one loss component from dominating others, ensures balanced optimization

### Example Configuration
```yaml
_target_: sat.loss.MetaLoss
losses:
  - _target_: sat.loss.FocalLoss
    gamma: [2.0, 3.0]
    importance_sample_weights: ${paths.importance_sample_weights_path}
    num_events: 2
  - _target_: sat.loss.SampleRankingLoss
    duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
    num_events: 2
balance_strategy: "grad"  # Balance by gradient magnitudes
balance_params:
  alpha: 0.9  # Smoothing factor for running averages
```

## Combining All Levels

SAT's power comes from the ability to combine these different levels of loss optimization:

```yaml
_target_: sat.loss.MetaLoss  # Component level balancing
losses:
  - _target_: sat.loss.SurvivalFocalLoss  # Difficulty level focusing
    gamma: [2.0, 3.0]  
    importance_sample_weights: ${paths.importance_sample_weights_path}  # Class level weighting
    num_events: ${data.num_events}
  - _target_: sat.loss.SampleRankingLoss
    duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
    importance_sample_weights: ${paths.importance_sample_weights_path}  # Class level weighting
    num_events: ${data.num_events}
balance_strategy: "adaptive"  # Component level strategy
balance_params:
  alpha: 0.9
  window_size: 100
  adaptation_rate: 0.01
```

## Benefits of the Multi-layered Approach

This comprehensive approach offers several advantages:

1. **Precision**: Fine-grained control over the learning process
2. **Adaptability**: Can be tuned to the specific characteristics of each dataset
3. **Robustness**: Handles imbalanced data, difficult samples, and competing objectives
4. **Modularity**: Each level can be configured independently or disabled if not needed
5. **Transparency**: Loss components and weights can be monitored during training

## Monitoring and Tuning

To effectively use this multi-level optimization:

1. **Monitor Loss Components**: Use the loss weight logging feature to track component weights and contributions
   ```bash
   tensorboard --logdir logs/your-experiment-dir
   ```

2. **Tuning Strategy**:
   - Start with simple configurations and add complexity as needed
   - Begin with appropriate importance weights for class imbalance
   - Add focal parameters if some classes show poor performance
   - Finally, tune component-level balancing if using multiple loss types

3. **Hyperparameter Recommendations**:
   - Focal gamma: Start with 2.0, increase for more focus on hard examples
   - Balance strategy: Start with `scale` for stability, move to `adaptive` or `uncertainty` for advanced tuning
   - Component coefficients: Start with equal weights (1.0) when using dynamic balancing

## Implementation Examples

For complete implementation examples, see:
- [loss.md](loss.md) - Detailed documentation of available loss functions
- [loss_weight_logging.md](loss_weight_logging.md) - Guide to monitoring loss weights

## References

The multi-level approach draws inspiration from several key papers:

1. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (2017)
2. **Multi-Task Uncertainty Weighting**: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (2018)
3. **Gradient Normalization**: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks" (2018)
4. **Dynamic Task Prioritization**: Guo et al., "Dynamic Task Prioritization for Multitask Learning" (2018)