# Loss Functions

This directory contains various loss functions utilized in the Survival Analysis Toolkit (SAT). 

## Loss Types

### Classification Losses
- `CrossEntropyLoss`: Standard cross-entropy loss for classification tasks
- `FocalLoss`: A variant of cross-entropy loss that focuses on hard-to-classify examples

### Regression Losses
- `L1Loss`: Mean absolute error loss (L1 norm)
- `MSELoss`: Mean squared error loss (L2 norm)

### Quantile Losses
- `QuantileLoss`: Loss function for quantile regression
- `SmoothQuantileLoss`: A smooth variant of quantile loss

## Handling Censored Data

All classification and regression losses are adapted to handle censored data, offering three configuration options:

### 1. Uncensored Mode
This mode falls back to the original calculation of the loss, using only uncensored data points (where the event was observed). Predictions for censored data points do not contribute to the loss calculation.

```python
loss = L1Loss(training_set=None, l1_type="uncensored")
```

### 2. Hinge Mode
In hinge mode, predictions for censored events are constrained with a lower bound of zero. This enforces that the model doesn't predict that events will occur before the censoring time.

For a censored data point where the subject was last observed at time `t` without experiencing the event:
- If the model predicts a time `p` where `p > t` (event after censoring), no penalty is applied
- If the model predicts a time `p` where `p < t` (event before censoring), a penalty proportional to `t - p` is applied

```python
loss = L1Loss(training_set=None, l1_type="hinge")
```

### 3. Margin Mode
Margin mode uses Kaplan-Meier estimates to generate a "best guess" for each censored data point, based on the population statistics. 

```python
loss = L1Loss(training_set="path/to/training_data.csv", l1_type="margin")
```

#### How Margin Computation Works

The margin approach handles censored data through these steps:

1. **Kaplan-Meier Estimation**: During initialization, the Kaplan-Meier survival curve is fitted using the training data for each event type. This provides an estimate of the survival function S(t).

2. **Weight Calculation**: For each censored data point with censoring time `t`, a weight `w = 1 - S(t)` is computed. This weight represents the probability that the event would have occurred before the censoring time.

3. **Best Guess Calculation**: For each censored time, a "best guess" of when the event would occur is calculated using the conditional expectation of event time given that the event occurs after the censoring time.

4. **Weighted Loss**: The loss for censored data points is computed as:
   ```
   loss = w * (best_guess - prediction)^2   # For MSELoss
   loss = w * |best_guess - prediction|     # For L1Loss
   ```
   
   Where:
   - `w` is the weight described above
   - `best_guess` is the expected event time conditional on surviving past censoring time
   - `prediction` is the model's prediction

This approach leverages population-level statistics to inform the loss calculation for censored data points, effectively imputing probable outcomes for censored observations based on observed patterns in the data. This provides more information to the model than simply ignoring censored data or using a hinge constraint.

## Loss Balancing API

All loss functions support a loss balancing API that allows for different strategies to balance multiple loss components. The balancing strategies include:

- `NONE`: No balancing is applied
- `SCALE_NORMALIZATION`: Normalize loss components by their scale
- `GRAD_NORMALIZATION`: Normalize loss components by their gradient magnitude
- `HOMOSCEDASTIC`: Learn optimal weights for each loss component
- `ADAPTIVE`: Dynamically adjust weights based on loss history

See each loss function's documentation for details on their specific implementation and configuration options.
