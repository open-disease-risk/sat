# Loss Weight Logging

This document explains how to interpret the loss weight logging feature for multi-objective training.

## Overview

When training models with multiple loss components, it's important to understand how the weighting between these components evolves over time. The loss weight logging feature visualizes these weights in TensorBoard, allowing you to monitor and debug the balancing behavior.

## Automatic Logging

Loss weights are automatically logged to TensorBoard during training. The `LossWeightLoggerCallback` is added by default in the training pipeline and requires no manual configuration.

## Visualizing Loss Weights

After training, you can visualize the loss weights in TensorBoard:

1. Launch TensorBoard with your logging directory:
   ```
   tensorboard --logdir logs/your-experiment-dir
   ```

2. Navigate to the "Scalars" tab

3. Look for metrics with the following patterns:
   - `train/weight_0`, `train/weight_1`, etc. for training
   - `eval/weight_0`, `eval/weight_1`, etc. for evaluation
   - For multi-task learning models: `train/head_0_weight_0`, `train/head_1_weight_0`, etc.

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
1. Automatically detecting if the model's loss function implements `get_loss_weights()`
2. Retrieving the current weights during training/evaluation steps
3. Logging these weights to TensorBoard with appropriate prefixes

For MetaLoss and other loss functions implementing the balancing framework, the weights are automatically available.
