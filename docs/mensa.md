# MENSA: Multi-Event Neural Survival Analysis

## Overview

MENSA (Multi-Event Neural Survival Analysis) is a specialized neural network architecture designed for survival analysis of multiple, potentially dependent events. It models survival distributions as mixtures of Weibull distributions with explicit dependencies between different event types, which is particularly valuable for complex medical scenarios where multiple risks or outcomes need to be predicted simultaneously.

## Technical Approach

### Core Components

1. **Mixture of Weibull Distributions**: 
   - Each event's survival function is modeled as a mixture of K Weibull distributions.
   - Parameters (shape, scale) for each mixture component are learned from data.
   - Mixture weights determine the contribution of each component to the overall distribution.

2. **Event Dependency Modeling**:
   - A learnable dependency matrix captures relationships between events.
   - The dependency matrix is normalized using softmax to create a weighted influence.
   - Distribution parameters for each event are adjusted based on these dependencies.

3. **SELU Activation Functions**:
   - Self-normalizing neural networks with SELU activations provide improved stability.
   - SELU helps maintain normalized activations through deep networks.

4. **Shared Representation Learning**:
   - Lower layers capture shared information across all events.
   - Event-specific layers model unique characteristics of each event.

### Mathematical Formulation

1. **Survival Function**:
   For an event j, the survival function is:
   ```
   S_j(t|x) = ∑_{k=1}^K π_jk(x) * exp(-(t/λ_jk(x))^α_jk(x))
   ```
   where:
   - π_jk(x) are mixture weights
   - λ_jk(x) are scale parameters
   - α_jk(x) are shape parameters
   - All parameters are functions of input covariates x

2. **Event Dependency**:
   Parameters for event i are adjusted as:
   ```
   α_i'(x) = ∑_{j=1}^J w_{ij} * α_j(x)
   λ_i'(x) = ∑_{j=1}^J w_{ij} * λ_j(x)
   ```
   where w_{ij} is the (i,j) element of the dependency weight matrix.

3. **Loss Function**:
   The loss combines negative log-likelihood for uncensored data, negative log-survival for censored data, and regularization for the dependency matrix:
   ```
   L = L_uncensored + discount * L_censored + reg_strength * R_dependency
   ```

## Differentiation from Other Approaches

### Compared to DeepHit

1. **Parametric vs. Non-parametric**:
   - DeepHit: Discretizes time and uses a non-parametric approach.
   - MENSA: Uses parametric Weibull mixtures for continuous time modeling.

2. **Event Dependencies**:
   - DeepHit: Implicitly models dependencies through shared networks.
   - MENSA: Explicitly models dependencies with a learnable matrix.

3. **Distribution Modeling**:
   - DeepHit: Produces discrete probability mass functions for each time bin.
   - MENSA: Creates continuous parametric survival functions.

### Compared to DSM (Deep Survival Machines)

1. **Event Handling**:
   - DSM: Originally designed for single-event scenarios (extensions exist).
   - MENSA: Built specifically for multiple, dependent events.

2. **Dependencies**:
   - DSM: Doesn't explicitly model dependencies between events.
   - MENSA: Contains a specific mechanism for event dependencies.

3. **Activation Functions**:
   - DSM: Typically uses ReLU or LeakyReLU.
   - MENSA: Uses SELU for self-normalization properties.

### Compared to Cox Proportional Hazards

1. **Assumptions**:
   - Cox: Assumes proportional hazards and linear relationships.
   - MENSA: Makes no proportionality assumptions and can model complex non-linear relationships.

2. **Multiple Events**:
   - Cox: Requires separate models for each event type.
   - MENSA: Handles multiple events simultaneously with shared information.

## Advantages

1. **Continuous Time Modeling**: Produces smooth survival curves rather than discretized predictions.

2. **Explicit Event Dependencies**: Captures and quantifies relationships between different event types.

3. **Interpretable Dependencies**: The learned dependency matrix provides insights into event relationships.

4. **Flexible Distribution Modeling**: Mixture of Weibulls can approximate many different distribution shapes.

5. **Transformer Compatibility**: Works seamlessly with transformer-based feature extraction.

6. **Uncertainty Quantification**: Parameters of the distribution enable uncertainty estimates.

7. **Efficiency**: Can model multiple events simultaneously, leveraging shared information.

## Limitations and Disadvantages

1. **Computational Complexity**: More parameters to learn compared to simpler models, especially with many events.

2. **Distributional Assumptions**: Relies on Weibull mixtures, which may not fit all datasets perfectly.

3. **Training Stability**: Mixture models can be sensitive to initialization and may converge to local optima.

4. **Hyperparameter Sensitivity**: Performance depends on proper tuning of mixture components, regularization strength, etc.

5. **Data Requirements**: May require more data to learn reliable dependencies between events.

6. **Interpretability Trade-offs**: While the dependency matrix is interpretable, the overall model remains a black box.

## Practical Considerations

### When to Use MENSA

- Multi-event scenarios where events may influence each other
- When continuous survival functions are needed
- When using transformer-based feature extraction
- When interpretable event dependencies are valuable

### When to Consider Alternatives

- Simple single-event scenarios (consider DSM instead)
- Very limited data (consider simpler models)
- When computational efficiency is the primary concern
- When assumptions of parametric distributions are violated

## Implementation Details

In our implementation, MENSA is integrated with transformer backends, allowing it to leverage powerful representation learning. The model consists of:

1. **Transformer Encoder**: Processes input features to create embeddings.
2. **MENSA Parameter Network**: Generates Weibull distribution parameters.
3. **Event Dependency Mechanism**: Adjusts parameters based on learned dependencies.
4. **Loss Function**: Combines likelihood terms with dependency regularization.

## Performance Characteristics

We've conducted extensive benchmarks comparing MENSA against other survival models across different configurations. Here are the key findings:

### Training and Inference Times

| Model | 2 Events (Train) | 2 Events (Infer) | 4 Events (Train) | 4 Events (Infer) | 8 Events (Train) | 8 Events (Infer) |
|-------|------------------|------------------|------------------|------------------|------------------|------------------|
| DSM | 1.69s | 0.064s | 1.93s | 0.052s | 1.99s | 0.082s |
| MENSA (no dependencies) | 0.69s | 0.022s | 0.92s | 0.031s | 1.08s | 0.026s |
| MENSA (with dependencies) | 0.79s | 0.027s | 0.84s | 0.025s | 1.39s | 0.041s |

### Scaling Characteristics

1. **MENSA without Dependencies**:
   - Training time increases ~56% when scaling from 2 to 8 events
   - More efficient than DSM across all configurations tested
   - Provides a good balance of performance and flexible distribution modeling

2. **MENSA with Dependencies**:
   - Training time increases ~76% when scaling from 2 to 8 events
   - The dependency matrix calculations add computational overhead as event count increases
   - Most valuable when explicit modeling of event relationships is important

### Learned Dependency Matrices

The dependency matrices show interesting patterns across different numbers of events:

**2 Events**:
```
[[0.74 0.26]
 [0.26 0.74]]
```
Clear diagonal dominance indicating primary self-dependency with moderate cross-event influence.

**4 Events**:
```
[[0.48 0.18 0.17 0.17]
 [0.17 0.48 0.17 0.17]
 [0.17 0.18 0.48 0.17]
 [0.18 0.17 0.17 0.48]]
```
Strong diagonal dominance with more uniform off-diagonal dependencies.

**8 Events**:
```
[[0.28 0.10 0.10 ...]
 [0.10 0.28 0.10 ...]
 ...
```
More uniform dependencies with weaker diagonal dominance.

These patterns suggest that as the number of events increases, the relative influence of each event becomes more distributed, though self-dependency (diagonal elements) remains strongest.

### Configuration

MENSA can be configured in YAML files:

```yaml
# Task configuration
_target_: sat.models.heads.MENSATaskHead
num_features: ${..num_features}
num_events: ${..num_events}
num_hidden_layers: 2
intermediate_size: 64
indiv_intermediate_size: 32
indiv_num_hidden_layers: 1
batch_norm: true
bias: true
hidden_dropout_prob: 0.1
num_mixtures: 4
distribution: "weibull"
event_dependency: true  # Set to false to disable dependency modeling
temp: 1000.0
discount: 1.0
loss_weight: 1.0

# Loss configuration
_target_: sat.loss.survival.MENSALoss
duration_cuts: ${paths.duration_cuts}
importance_sample_weights: ${paths.importance_sample_weights}
num_events: ${..num_events}
distribution: "weibull"
discount: 1.0
elbo: false
dependency_regularization: 0.01  # Controls the strength of dependency regularization
```

### Analyzing Event Dependencies

The learned dependency matrix provides valuable insights into relationships between different events:

```python
# Extract the learned dependency matrix
if hasattr(model.nets, "event_dependency_matrix"):
    # Get the normalized dependency weights
    dependency_matrix = F.softmax(model.nets.event_dependency_matrix, dim=1)
    print(f"Learned dependency matrix:\n{dependency_matrix.detach().cpu().numpy()}")
    
    # Visualize as a heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(dependency_matrix.detach().cpu().numpy(), 
                annot=True, cmap="viridis", vmin=0, vmax=1)
    plt.title("Event Dependency Matrix")
    plt.xlabel("Source Event")
    plt.ylabel("Target Event")
    plt.savefig("event_dependencies.png")
```

High values in the (i,j) position indicate that event j strongly influences event i.

## References

- "MENSA: Multi-Event Neural Survival Analysis" (2024) [arXiv:2409.06525](https://arxiv.org/html/2409.06525v1)