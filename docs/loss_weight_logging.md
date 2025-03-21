# Loss Weight Logging

This document explains how to use and interpret the loss weight logging feature for multi-objective training.

## Overview

When training models with multiple loss components, it's important to understand how the weighting between these components evolves over time. The loss weight logging feature visualizes these weights in TensorBoard, allowing you to monitor and debug the balancing behavior.

## Enabling Loss Weight Logging

To enable loss weight logging, include the `loss_weight_logger.yaml` configuration in your callback list:

```yaml
# In your experiment configuration
callbacks:
  - _target_: transformers.EarlyStoppingCallback
    early_stopping_patience: 10
  - _target_: sat.transformers.callbacks.LossWeightLoggerCallback
    log_freq: 1
    prefix: "loss_weights"
    log_eval: true
    log_train: true
```

Alternatively, you can simply include the `loss_weight_logger` config group:

```yaml
# In your experiment configuration
defaults:
  - callbacks@callbacks: default
  - callbacks@callbacks: loss_weight_logger
```

## Configuration Options

The `LossWeightLoggerCallback` supports the following options:

- `log_freq`: Logging frequency (every N evaluation steps)
- `prefix`: Prefix for the logged metrics in TensorBoard
- `log_eval`: Whether to log during evaluation (default: true)
- `log_train`: Whether to log during training (default: true)

## Visualizing Loss Weights

After training, you can visualize the loss weights in TensorBoard:

1. Launch TensorBoard with your logging directory:
   ```
   tensorboard --logdir logs/your-experiment-dir
   ```

2. Navigate to the "Scalars" tab

3. Look for metrics with the prefix `loss_weights/`:
   - `loss_weights/train/weight_0`, `loss_weights/train/weight_1`, etc. for training
   - `loss_weights/eval/weight_0`, `loss_weights/eval/weight_1`, etc. for evaluation

## Interpreting Loss Weights

The interpretation of loss weights depends on the balancing strategy:

### Fixed Weighting
Fixed weights should remain constant throughout training.

### Scale Normalization
Weights will adjust to compensate for differences in loss scales. Larger losses will get smaller weights to ensure balanced contributions.

### Gradient Normalization
Weights will adjust based on gradient magnitudes. Components with larger gradients will get smaller weights.

### Uncertainty Weighting
Weights reflect the learned precision (1/variance) for each task. A higher weight indicates the model is more confident about that task.

### Adaptive Weighting
Weights will adjust dynamically based on loss improvement rates. Tasks that are improving slowly will receive higher weights.

## Debugging Balancing Issues

If you notice issues with your multi-objective training, check the loss weight logs for:

1. **Extreme Values**: Weights approaching zero or becoming extremely large may indicate numerical issues
2. **Rapid Oscillations**: Unstable weights that oscillate rapidly may indicate conflicts between objectives
3. **Domination**: One weight becoming much larger than others may indicate one task dominating training

## Implementation Details

The callback works by:
1. Checking if the model's loss function implements `get_loss_weights()`
2. Retrieving the current weights during training/evaluation steps
3. Logging these weights to TensorBoard with appropriate prefixes

For MetaLoss and other loss functions implementing the balancing framework, the weights are automatically available.