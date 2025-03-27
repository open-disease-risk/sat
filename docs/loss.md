# SAT Loss Framework

The Survival Analysis Toolkit (SAT) features a comprehensive loss framework for survival analysis and other related tasks. This document provides an overview of the available loss functions, their mathematical foundations, and guidance on how to combine them effectively.

## Loss Framework Architecture

The SAT loss framework is built on four key components:

1. **Base Loss Functions**: Individual loss functions for specific objectives (survival analysis, ranking, classification, etc.)
2. **MetaLoss**: A composable container that combines multiple loss functions with configurable weighting
3. **Loss Balancing Strategies**: Methods to dynamically adjust weights between loss components
4. **Task-Specific Compositions**: Pre-configured combinations for common objectives like survival analysis

## 1. Core Survival Loss Functions

### NLLPCHazard Loss

**Purpose**: Negative log-likelihood loss based on piece-wise constant hazard rates, the fundamental survival analysis loss.

**Equation**:
```
L = -∑[δᵢ * log(λ(tᵢ)) - ∫₀^tᵢ λ(u)du]
```
where:
- δᵢ is the event indicator (1 if event occurred, 0 if censored)
- λ(t) is the hazard function at time t
- tᵢ is the event/censoring time for subject i

**Pros**:
- Statistically principled maximum likelihood estimation
- Handles censored data natively
- Allows for direct hazard modeling

**Cons**:
- May struggle with heavily censored data
- Does not directly optimize ranking metrics like concordance index

**Configuration**:
```yaml
_target_: sat.loss.survival.NLLPCHazard
duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
num_events: ${data.num_events}
importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
reduction: mean
```

### DeepHit Loss

**Purpose**: Specialized survival loss that combines NLL with ranking terms to optimize both likelihood and ranking performance.

**Equation**:
```
L = α * L_nll + (1-α) * L_rank
```
where:
- L_nll is the negative log-likelihood loss
- L_rank is a ranking-based loss term
- α is a balancing weight between the two

**Pros**:
- Combines the strengths of likelihood and ranking approaches
- Often achieves better concordance index than NLL alone
- Built-in handling of competing risks

**Cons**:
- More hyperparameters to tune (α, ranking term parameters)
- More computationally intensive

**Configuration**:
```yaml
_target_: sat.loss.survival.DeepHitLoss
duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
num_events: ${data.num_events}
alpha: 0.5  # Weight between likelihood and ranking terms
sigma: 0.1  # Scaling factor for ranking loss
importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
reduction: mean
```

### Focal Loss for Survival

**Purpose**: Adapts focal loss concept from object detection to survival analysis, focusing learning on harder examples.

**Equation**:
```
FL(p) = -α * (1-p)^γ * log(p)
```
where:
- p is the predicted survival probability
- γ (gamma) is the focusing parameter
- α is a class balancing factor

**Pros**:
- Helps with imbalanced survival data
- Focuses learning on harder examples
- Particularly useful for rare events in competing risks

**Cons**:
- Requires tuning of γ parameter
- May converge slower than standard NLL

**Configuration**:
```yaml
_target_: sat.loss.survival.SurvivalFocalLoss
gamma: 2.0  # Single focusing parameter for all events
# Or for multi-focal: gamma: [2.0, 3.0]  # Different focus per event
importance_sample_weights: ${paths.importance_sample_weights_path}
num_events: ${data.num_events}
reduction: mean
```

## 2. Ranking Loss Functions

Ranking losses focus on the order of predictions rather than their absolute values, often important for survival analysis where proper ordering of risk is critical.

### Sample Ranking Loss

**Purpose**: Ensures subjects with earlier events have higher predicted risk than those with later events.

**Equation**:
```
L = -log(sigmoid(σ * (r_i - r_j)))
```
where:
- r_i, r_j are risk scores for subjects i and j
- σ is a scaling factor
- Subject i should have higher risk than j

**Pros**:
- Directly optimizes concordance-like metrics
- Often improves discrimination performance
- Efficient implementation with tensor operations

**Cons**:
- Does not account for calibration
- Scales quadratically with batch size

**Configuration**:
```yaml
_target_: sat.loss.ranking.SampleRankingLoss
duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
sigma: 0.1  # Scaling factor
margin: 0.0  # Minimum required difference between scores
num_events: ${data.num_events}
```

### Multi-Event Ranking Loss

**Purpose**: For competing risks, ensures correct ranking of different event types within a subject.

**Equation**: Similar to Sample Ranking Loss but compares different event types for the same subject.

**Pros**:
- Essential for competing risks modeling
- Complements within-event ranking
- Scales linearly with number of events

**Cons**:
- Only applicable to competing risks problems
- May require careful balancing with other losses

**Configuration**:
```yaml
_target_: sat.loss.ranking.MultiEventRankingLoss
duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
sigma: 0.1
num_events: ${data.num_events}
```

### ListMLE Losses

**Purpose**: List-based ranking losses that optimize the likelihood of the correct permutation of samples.

#### SampleListMLELoss

Optimizes ranking between different samples.

**Equation**:
```
L = -log(∏ᵢ P(sᵢ|Sᵢ))
```
where P(sᵢ|Sᵢ) is the probability that element sᵢ is selected first from the remaining set Sᵢ.

**Pros**:
- Better scaling than pairwise methods (O(n log n) vs O(n²))
- Often more stable than pairwise approaches
- Considers global context rather than just pairs

**Configuration**:
```yaml
_target_: sat.loss.ranking.SampleListMLELoss
duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
temperature: 1.0  # Controls sharpness of probability distribution
num_events: ${data.num_events}
```

#### EventListMLELoss

Similar to SampleListMLELoss but for ranking between event types.

## 3. Auxiliary Losses

### Brier Score Loss

**Purpose**: Measures calibration of probability predictions.

**Equation**:
```
BS(t) = 1/N ∑ᵢ (S(t|xᵢ) - I(tᵢ > t))²
```
where:
- S(t|xᵢ) is the predicted survival at time t
- I(tᵢ > t) is 1 if subject i survived past time t, 0 otherwise

**Pros**:
- Measures both discrimination and calibration
- Provides interpretable probabilities
- Penalizes overconfident incorrect predictions

**Cons**:
- Less focus on ranking than concordance-based metrics
- Requires proper handling of censoring

### Quantile Regression Loss

**Purpose**: Optimize specific quantiles of the survival distribution.

**Equation**:
```
L = ∑ᵢ (τ - I(yᵢ < ŷᵢ)) * (yᵢ - ŷᵢ)
```
where:
- τ is the target quantile (e.g., 0.5 for median)
- yᵢ is the true time
- ŷᵢ is the predicted time

**Pros**:
- Provides prediction intervals, not just point estimates
- More robust to outliers than mean-based losses
- Allows for asymmetric error penalties

**Cons**:
- More complex to optimize
- Requires special handling for censoring

## 4. Loss Balancing Strategies

The `MetaLoss` class supports multiple strategies for balancing different loss components:

### Fixed Weighting

**Purpose**: Static predetermined weights for each loss component.

**Equation**: 
```
L = ∑ᵢ wᵢ * Lᵢ
```

**Pros**:
- Simple to implement and understand
- Predictable behavior
- No computational overhead

**Cons**:
- Requires manual tuning
- Cannot adapt to changing loss scales during training

**Configuration**:
```yaml
_target_: sat.loss.MetaLoss
losses:
  - _target_: sat.loss.survival.NLLPCHazard
    # ...
  - _target_: sat.loss.ranking.SampleRankingLoss
    # ...
coeffs: [1.0, 0.1]  # Fixed weights
balance_strategy: "fixed"
```

### Scale Normalization

**Purpose**: Dynamically adjusts weights to normalize loss scales.

**Equation**: 
```
wᵢ = 1 / (sᵢ + ε)
```
where sᵢ is the exponential moving average of loss i's scale.

**Pros**:
- Prevents one loss from dominating due to scale differences
- Adapts to changing loss scales during training
- No additional hyperparameters to tune

**Cons**:
- May not reflect true importance of losses
- Could fluctuate during training

**Configuration**:
```yaml
_target_: sat.loss.MetaLoss
losses:
  # Loss components...
balance_strategy: "scale"
balance_params:
  alpha: 0.9  # EMA decay rate
  eps: 1e-8   # Numerical stability constant
```

### Gradient Normalization

**Purpose**: Balances losses based on gradient magnitudes.

**Equation**: Similar to scale normalization but using gradient norms instead of loss values.

**Pros**:
- Focuses on actual impact on model parameters
- Better theoretical justification than scale normalization
- Adapts to loss "difficulty"

**Cons**:
- Computationally more expensive
- May oscillate during training

**Configuration**:
```yaml
_target_: sat.loss.MetaLoss
losses:
  # Loss components...
balance_strategy: "grad"
balance_params:
  alpha: 0.9
```

### Uncertainty Weighting

**Purpose**: Learns optimal weights through homoscedastic uncertainty.

**Equation**:
```
L = ∑ᵢ 1/(2σᵢ²) * Lᵢ + log(σᵢ)
```
where σᵢ² is the learned variance for task i.

**Pros**:
- Theoretically principled approach
- Automatically learns task weighting
- Adapts to task difficulty

**Cons**:
- More parameters to optimize
- Requires sufficient training data

**Configuration**:
```yaml
_target_: sat.loss.MetaLoss
losses:
  # Loss components...
balance_strategy: "uncertainty"
balance_params:
  init_sigma: 1.0     # Initial uncertainty
  log_interval: 10    # TensorBoard logging frequency
```

### Adaptive Weighting

**Purpose**: Dynamically adjusts weights based on loss improvement rates.

**Equation**: Increases weights for losses showing slower improvement.

**Pros**:
- Balances progress across all objectives
- Helps prevent plateauing on individual losses
- Works well for objectives with different convergence rates

**Cons**:
- Sensitive to learning dynamics
- May require tuning of adaptation parameters

**Configuration**:
```yaml
_target_: sat.loss.MetaLoss
losses:
  # Loss components...
balance_strategy: "adaptive"
balance_params:
  alpha: 0.9
  window_size: 10         # Window for improvement calculation
  adaptation_rate: 0.01   # Rate of weight adjustment
```

## 5. Loss Recipe Examples

### Balanced Survival Analysis

**Recipe**: Combine likelihood, calibration, and ranking objectives.

```yaml
_target_: sat.loss.MetaLoss
losses:
  - _target_: sat.loss.survival.NLLPCHazard
    duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
    num_events: ${data.num_events}
  
  - _target_: sat.loss.ranking.SampleRankingLoss
    duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
    sigma: 0.1
    num_events: ${data.num_events}
  
  - _target_: sat.evaluate.BrierScore
    duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
    num_events: ${data.num_events}
    as_loss: true

balance_strategy: "uncertainty"  # Learn optimal weights
```

### Competing Risks with Focal Enhancement

**Recipe**: For competing risks with imbalanced event types.

```yaml
_target_: sat.loss.MetaLoss
losses:
  - _target_: sat.loss.survival.SurvivalFocalLoss
    gamma: [2.0, 3.0]  # Different focus per event type
    num_events: ${data.num_events}
    duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
  
  - _target_: sat.loss.ranking.SampleRankingLoss
    duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
    sigma: 0.1
    num_events: ${data.num_events}
  
  - _target_: sat.loss.ranking.MultiEventRankingLoss
    duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
    sigma: 0.1
    num_events: ${data.num_events}

balance_strategy: "scale"  # Normalize loss scales
```

### Multi-Task Learning

**Recipe**: Combine survival, classification and regression tasks.

```yaml
# First, configure each task head with appropriate loss
# In the MTLForSurvival class configuration:
survival_loss:
  _target_: sat.loss.survival.NLLPCHazard
  # ...

classification_loss:
  _target_: sat.loss.classification.CrossEntropyLoss
  # ...

regression_loss:
  _target_: sat.loss.regression.MSELoss
  # ...

# Then configure MTL balancing
mtl_balance_strategy: "uncertainty"
mtl_balance_params:
  init_sigma: 1.0
```

## 6. Integration with Multi-Task Learning

The SAT framework integrates loss balancing at two levels:

1. **Within Task Heads**: Each task head (survival, classification, regression) can use `MetaLoss` to combine multiple objectives.

2. **Between Task Heads**: The `MTLForSurvival` class balances losses between different task heads.

This creates a hierarchical loss balancing structure:

```
MTLForSurvival
├── Survival Head
│   └── MetaLoss
│       ├── NLLPCHazard
│       ├── SampleRankingLoss
│       └── BrierScore
├── Classification Head
│   └── CrossEntropyLoss
└── Regression Head
    └── MSELoss
```

Each level can use different balancing strategies, enabling fine-grained control over multi-objective optimization.

## 7. Monitoring Loss Balancing

To monitor loss balancing during training, use the `LossWeightLoggerCallback`:

```yaml
callbacks:
  - _target_: sat.transformers.callbacks.LossWeightLoggerCallback
    log_freq: 1
    prefix: "loss_weights"
    log_eval: true
    log_train: true
```

This logs the evolving loss weights to TensorBoard, allowing you to:
- Understand how balancing strategies behave
- Identify potential issues with loss dominance
- Fine-tune balance parameters

## 8. Best Practices

1. **Start with Fixed Weights**: Begin with fixed weights to establish baselines.

2. **Scale Matters**: Ranking losses often need smaller weights (0.01-0.1) compared to likelihood losses.

3. **Consider Task Relationships**:
   - Use `SampleRankingLoss` for within-event ranking
   - Use `MultiEventRankingLoss` for between-event ranking in competing risks

4. **Adaptive Strategies for Complex Tasks**: For multi-objective tasks, consider uncertainty or adaptive weighting instead of fixed weights.

5. **Monitor TensorBoard**: Watch loss weight evolution to ensure balanced optimization.

6. **Layer Your Losses**: For complex tasks, consider hierarchical loss composition rather than a flat structure.

7. **Use Specialized Loss Functions**:
   - Prefer `ListMLE` over pairwise ranking for larger datasets
   - Use `FocalLoss` for imbalanced event types

8. **Combine Compatible Losses**: Ensure losses are complementary rather than conflicting.