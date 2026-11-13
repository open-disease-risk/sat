# Numerical Stability in SAT

This document explains the numerical stability issues encountered in the SAT (Survival Analysis using Transformers) project and the solutions implemented.

## Problem Description

We encountered two main issues during training:

1. **The model wasn't learning effectively**: Loss plateaued early in training
2. **Numerical instabilities**: NaN and Inf values appeared during gradient calculations

These issues were most prominent when training on the metabric dataset using the survival configuration. The primary sources of instability were:

- **NLLPC Hazard Loss function**: The negative log-likelihood for PC-Hazard calculation involves operations prone to numerical instability (log, softplus, cumsum)
- **Small values in intermediate calculations**: Values close to zero created underflow issues
- **Extreme gradient values**: Occasional gradient explosions propagated NaN values

## Solutions Implemented

We implemented several solutions across multiple levels of the codebase:

### 1. Core Utility Functions

- **pad_col**: Increased default value from 1e-8 to 1e-6 for better numerical stability
- **log_softplus**: Changed threshold from -15.0 to -10.0 for improved approximation
- Added extensive error handling and logging to both functions
- Added input validation and NaN/Inf detection with automatic correction

### 2. Loss Function Enhancements

- Added NaN/Inf detection and cleanup in loss calculations
- Implemented fallback mechanisms to return valid loss values when issues are detected
- Added error boundaries to prevent propagation of extreme values
- Improved type handling (e.g., explicit bool to float conversion)

### 3. Model Architecture Improvements

- Created `StableLinear` layer with controlled weight initialization and gradient scaling
- Implemented `StableMLPSequence` with proper normalization and extensive stability measures
- Added residual connections to improve gradient flow
- Added output value clamping to prevent extreme activations

### 4. Configuration System

Created multiple stability configurations with progressively stronger measures:

- **metabric_mild.yaml**: Slight parameter constraining
- **metabric_medium.yaml**: Moderate constraints that work for most cases
- **metabric_strong.yaml**: Aggressive constraints for difficult cases
- **metabric_extreme.yaml**: Maximum stability at the cost of learning capacity

### 5. Debugging and Monitoring

- Implemented `StabilityTracker` for monitoring model parameters and activations
- Created `debug_loss.py` with detailed diagnostics for loss function troubleshooting
- Added extensive logging for tracking numerical issues during training
- Enhanced `Trainer` with stability monitoring to catch issues early

## Results and Recommendations

Our experiments show:

1. The **medium stability configuration** provides the best balance between stability and learning capacity. It successfully completes training without NaN errors in most cases, while still allowing effective learning.

2. The strong and extreme configurations apply constraints that are too aggressive, preventing the model from learning effectively or causing numerical issues in the backward pass.

3. The loss function modifications and utility function improvements significantly reduce NaN/Inf occurrences.

**Recommended Configuration for metabric survival training**:

```yaml
# Medium stability parameters for metabric dataset
stability:
  grad_scale: 0.1
  eps: 1e-05
  max_norm: 0.2
  init_gain: 0.01
  weight_clamp_value: 0.02
  noise_scale: 5e-05
  nan_value: 0.0
  posinf_value: 0.5
  neginf_value: -0.5
  output_min: -8.0
  output_max: 8.0
  weight_init_std: 0.01
  softplus_beta: 0.5
  leaky_relu_negative_slope: 0.005
  use_residual: True
  layer_norm_eps: 1e-06
```

Recommended command:
```bash
HYDRA_FULL_ERROR=1 poetry run python -m sat.finetune experiments=metabric/survival stability=metabric_medium
```

## Further Improvements

For continued enhancement of numerical stability:

1. Consider implementing custom backward functions for critical operations
2. Explore alternative loss formulations that are inherently more stable
3. Further optimize hyperparameters, especially learning rate and batch size
4. Add gradient clipping directly in the loss computation
5. Investigate alternative activation functions with better stability properties

## Conclusion

The numerical stability issues in the SAT project have been significantly mitigated through a multi-level approach targeting utility functions, loss calculations, and model architecture. The medium stability configuration provides a good balance between stability and learning effectiveness for the metabric survival task.