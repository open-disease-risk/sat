# Momentum Contrast (MoCo) for Survival Analysis

## Introduction

Momentum Contrast (MoCo) is a technique adapted from computer vision and self-supervised learning to enhance survival analysis models. It addresses a fundamental challenge in survival data - the sparsity of events and highly imbalanced censoring patterns - by maintaining a memory buffer of past embeddings. This approach effectively increases the "event density" during training, leading to more stable and accurate models, especially when facing high censoring rates.

## Core Concepts

### Problem: Event Sparsity in Survival Analysis

In survival analysis, particularly for medical datasets:
- Many samples are censored (event never observed)
- Events may be rare, with censoring rates often exceeding 70-80%
- A typical mini-batch might contain very few events
- This leads to unstable gradients and poor convergence

### Solution: MoCo's Memory Buffer

MoCo addresses this by:
1. Maintaining a queue of past sample embeddings and their event/time information
2. Using these stored embeddings alongside the current batch during loss computation
3. Creating an "effective batch" with significantly more events
4. Gradually growing the buffer size as training progresses
5. Dynamically adjusting buffer usage based on training dynamics

## MoCo Implementation Modes

SAT offers three MoCo implementations, each with specific use cases:

### 1. Standard MoCoSurvivalLoss

The base implementation for datasets with moderate censoring rates (40-70%).

**Key parameters:**
- `moco_buffer_size`: Maximum size of the memory queue (e.g., 1024)
- `moco_use_buffer`: Whether to use the buffer (default: True)
- `moco_current_batch_weight`: Weight of loss on current batch (default: 1.0)
- `moco_buffer_weight`: Weight of loss on combined batch+buffer (default: 1.0)
- `moco_dynamic_buffer`: Whether to grow buffer size during training (default: True)
- `moco_initial_buffer_size`: Initial buffer size when using dynamic growth

**When to use:**
- Standard survival analysis tasks
- Moderate censoring rates
- Balanced datasets
- Simple survival problems with sufficient events

### 2. DynamicWeightMoCoLoss

An enhanced implementation that gradually transitions weight from the current batch to the buffer during training. Ideal for high censoring rates (70-85%).

**Additional parameters:**
- `moco_initial_batch_weight`: Starting weight for current batch loss (1.0)
- `moco_final_batch_weight`: Final weight for current batch loss (0.5)
- `moco_initial_buffer_weight`: Starting weight for buffer loss (0.0)
- `moco_final_buffer_weight`: Final weight for buffer loss (1.0)
- `moco_warmup_steps`: Number of steps to transition between initial and final weights

**When to use:**
- High censoring rates (70-85%)
- Multi-event datasets
- When standard MoCo shows instability
- When loss oscillates during training

### 3. AdaptiveMoCoLoss

The most advanced implementation that automatically adjusts buffer usage based on loss variance monitoring. Best for very high censoring (>85%) or competing risks tasks.

**Additional parameters:**
- `moco_adaptive_buffer`: Enable adaptive buffer size adjustments
- `moco_track_variance`: Track loss variance over time
- `moco_variance_window`: Window size for variance calculation (e.g., 10)
- `moco_variance_threshold`: Threshold for significant variance change (e.g., 0.15)
- `moco_min_buffer_ratio`: Minimum buffer size as fraction of max (e.g., 0.25)
- `moco_max_buffer_ratio`: Maximum buffer size as fraction of max (e.g., 1.0)

**When to use:**
- Very high censoring rates (>85%)
- Competing risks with multiple event types
- Complex datasets with varying event densities
- Research projects requiring maximum performance
- When training shows high instability

## Using the MoCo Recommender Tool

SAT includes a dedicated `moco_recommend` tool that analyzes your dataset and recommends the optimal MoCo configuration.

### Basic usage:

```
poetry run python -m sat.moco_recommend -cn experiments/metabric/survival
```

### What the tool does:

1. **Analyzes dataset characteristics:**
   - Sample count
   - Censoring rate
   - Event counts and types
   - Time distributions

2. **Evaluates training configuration:**
   - Batch size
   - Hardware (CPU/GPU)
   - Computational constraints

3. **Recommends the optimal setup:**
   - Which MoCo implementation to use
   - Buffer size parameters
   - Dynamic growth settings
   - Weight parameters

4. **Provides batch size analysis:**
   - Generates plots showing the relationship between batch size and buffer requirements
   - Helps you optimize memory usage and training efficiency

### Output example:

```
MoCo Buffer Size Recommendations
===============================

Dataset Statistics:
  Total samples: 1142
  Censoring rate: 42.8%
  Event types: 1

Training Configuration:
  Batch size: 256
  Min events per batch: 10

Buffer Recommendations:
  Recommended buffer size: 256
  Initial buffer size: 128
  Expected events in batch: 146.5
  Expected events with buffer: 293.0

Implementation Recommendation:
Use standard MoCoSurvivalLoss for this dataset.

Configuration:
  moco_buffer_size: 256
  moco_initial_buffer_size: 128
  moco_use_buffer: True
  moco_dynamic_buffer: True
  moco_batch_weight: 1.0
  moco_buffer_weight: 1.0
```

## Parameter Tuning Guidelines

### Buffer Size Selection

- **Rule of thumb**: Buffer size should scale with censoring rate
- **Low censoring (<30%)**: 1-2x batch size
- **Moderate censoring (30-70%)**: 2-5x batch size
- **High censoring (70-85%)**: 5-10x batch size
- **Very high censoring (>85%)**: 10-20x batch size

### Batch Size Considerations

- **For GPUs**: Maximize batch size within memory constraints
- **For CPUs**: Smaller batches with larger buffers often work better
- **Min events guideline**: Aim for at least 10-20 events per effective batch
- **Calculation**: events_per_batch = batch_size * (1 - censoring_rate)

### Weight Parameters

- **Standard cases**: Equal weights (1.0) for batch and buffer
- **High censoring**: Gradually increase buffer weight using DynamicWeightMoCoLoss
- **Unstable training**: Use AdaptiveMoCoLoss to automatically adjust weights

## Implementation Details

MoCo maintains two core components:
1. A queue of past embeddings from the model
2. A corresponding queue of event/time information

During forward pass:
1. The current batch is processed normally
2. Past embeddings from the buffer are combined with current batch
3. Loss is computed on both current batch and combined data
4. The buffer is updated with current batch embeddings
5. Weights and buffer size are adjusted based on training progress

## Best Practices

1. **Always run the recommender** before deciding on MoCo configuration
2. **Start with standard MoCo** and move to more advanced versions if needed
3. **Monitor buffer statistics** during training
4. **For highly censored data**, use DynamicWeightMoCoLoss or AdaptiveMoCoLoss
5. **For competing risks**, AdaptiveMoCoLoss usually performs best
6. **Balance memory usage** with performance requirements
7. **Create checkpoint directories** before training

## Advanced Topics

### Buffer Adjustment Logic

The adaptive buffer monitors loss variance and increases buffer size when:
- Loss variance increases significantly (>50% change)
- Training becomes unstable
- Gradient norms show high fluctuation

This ensures that the model automatically adapts to the difficulty of the dataset.

### Combining with Other Loss Functions

MoCo works particularly well when combined with:
- Ranking losses (RankNet, ListMLE, SOAP)
- Focal loss adaptations
- Multi-task learning objectives

Simply wrap your base loss with the appropriate MoCo implementation.

### Memory Considerations

For large datasets, consider:
- Starting with smaller buffer sizes and enabling dynamic growth
- Using mixed precision training to reduce memory footprint
- Monitoring memory usage during training
- Adjusting batch size and buffer size to optimize GPU utilization

## Conclusion

MoCo significantly improves survival analysis models, especially for highly censored datasets. The three implementation modes (Standard, Dynamic, Adaptive) provide flexible options for different scenarios, while the recommender tool simplifies configuration. For best results, use the recommender to analyze your specific dataset characteristics and follow the suggested configuration.