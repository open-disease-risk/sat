# TorchSurv Integration

This document describes the integration of TorchSurv into the SAT project for more robust survival analysis metrics.

## Background

The project previously used scikit-survival's implementations of the Brier score and concordance index (c-index) for evaluating survival models. However, scikit-survival's implementation of the Brier score can be problematic in certain situations, particularly when evaluation time points are close to the maximum observed time in the training data.

To address these issues, we've integrated TorchSurv's implementations of these metrics, which offer several advantages:

- Native PyTorch implementation for improved numerical stability
- Better handling of edge cases
- Built-in statistical testing capabilities
- More flexible weighting options
- Active development and maintenance

## Implementation

The integration consists of:

1. **Wrapper Functions**: We've created wrapper functions in `sat/evaluate/torchsurv_wrappers.py` that provide the same API as scikit-survival but use TorchSurv's implementations.

2. **Fallback Mechanism**: The system automatically falls back to scikit-survival's implementations if TorchSurv is not available, ensuring backward compatibility.

3. **Enhanced Diagnostics**: The wrappers include detailed diagnostics for troubleshooting metric calculations.

## Installation

To use TorchSurv, you'll need to install it:

```bash
pip install torchsurv
```

TorchSurv requires PyTorch, which should be installed first if not already present:

```bash
pip install torch
```

## Usage

No changes are needed to your existing code. The system will automatically use TorchSurv's implementations if available, with a fallback to scikit-survival if not.

If you want to explicitly use TorchSurv's implementations in your own code:

```python
from sat.evaluate.torchsurv_wrappers import (
    torchsurv_brier_score,
    torchsurv_integrated_brier_score,
    torchsurv_concordance_index_censored
)

# Then use these functions as drop-in replacements for their scikit-survival equivalents
```

## Metrics Implemented

### Brier Score

The Brier score measures the accuracy of survival probability predictions at specific time points. TorchSurv's implementation handles inverse probability of censoring weighting (IPCW) to account for censored observations.

A key difference from scikit-survival is that TorchSurv requires IPCW weights not only for subject times but also for evaluation time points (through the `weight_new_time` parameter). Our wrappers automatically calculate and provide these weights using `get_ipcw()`.

#### Implementation details:

```python
# Calculate IPCW at subject times
subject_ipcw = get_ipcw(train_events, train_times)

# Calculate IPCW at evaluation time points
# Note: In TorchSurv 0.1.4, use 'new_time' parameter instead of 'times'
eval_ipcw = get_ipcw(train_events, train_times, new_time=eval_times)

# Both sets of weights must be provided
brier_score(predictions, test_events, test_times, 
           weight=subject_ipcw, new_time=eval_times, weight_new_time=eval_ipcw)
```

### Integrated Brier Score

The integrated Brier score provides an overall measure of prediction accuracy across all time points. TorchSurv's implementation includes more robust numerical integration and statistical testing.

Our implementation includes several fallback mechanisms for maximum robustness:

1. **Parameter Name Adaptability**: The code tries multiple parameter formats for `get_ipcw()` to accommodate different TorchSurv versions.

2. **Per-time-point Computation**: If the integrated calculation fails, we compute the Brier score at each time point individually and then integrate them.

3. **Shape Mismatch Handling**: When tensor shape mismatches occur (common when the number of subjects and time points differ), we gracefully fall back to simplified approaches.

This multi-layered fallback strategy ensures maximum robustness for different data scenarios and TorchSurv versions.

### Concordance Index (c-index)

The concordance index measures the ability of a model to correctly rank survival times. TorchSurv's implementation offers statistical testing, confidence intervals, and flexible handling of ties.

## Benefits over scikit-survival

1. **Robustness**: Better handling of edge cases, especially when evaluating time points near the maximum observed time.

2. **Performance**: PyTorch-based implementation with potential for GPU acceleration for large datasets.

3. **Statistical Rigor**: Built-in confidence intervals and hypothesis testing.

4. **Active Development**: TorchSurv is actively maintained by Novartis and the community.

5. **Flexibility**: More options for handling censored data and ties.

## Debugging

If you encounter issues with the metrics, both implementations provide detailed logging to help diagnose problems:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

The TorchSurv wrappers also include a `return_diagnostics` parameter that provides detailed information about the calculations.

## Citations

If you use TorchSurv in your research, please cite the original package:

```
@article{coroller2023torchsurv,
  title={TorchSurv: A PyTorch Package for Deep Learning in Survival Analysis},
  author={Coroller, Thibaud and Monod, M{\'e}lodie},
  journal={Journal of Open Source Software},
  year={2023}
}
```