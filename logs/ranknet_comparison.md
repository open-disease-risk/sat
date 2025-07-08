# RankNet Performance Comparison

This document summarizes the performance of RankNet compared to other ranking losses.

## Speedup by Batch Size (vs. SampleRankingLoss)

| Batch Size | multi_event | sample_list_mle | sample_soap | sample_ranknet | event_ranknet | Best Method |
|-----------|---------------|-------------------|---------------|------------------|-----------------|---------------|
| 16 | 7.03x | 9.23x | 3.21x | 5.32x | 4.13x | sample_list_mle (9.23x) |
| 32 | 1.38x | 1.81x | 0.71x | 1.12x | 0.35x | sample_list_mle (1.81x) |
| 64 | 3.26x | 4.99x | 1.75x | 2.92x | 0.56x | sample_list_mle (4.99x) |
| 128 | 7.25x | 10.30x | 3.05x | 4.42x | 0.60x | sample_list_mle (10.30x) |
| 256 | 8.17x | 13.42x | 4.25x | 0.08x | 0.37x | sample_list_mle (13.42x) |

## Key Findings

- **sample_list_mle**: Average 7.95x speedup, Max 13.42x
- **multi_event**: Average 5.42x speedup, Max 8.17x
- **sample_ranknet**: Average 2.77x speedup, Max 5.32x
- **sample_soap**: Average 2.59x speedup, Max 4.25x
- **event_ranknet**: Average 1.20x speedup, Max 4.13x

**Overall Best Performer**: sample_list_mle (Average: 7.95x)

## Method Characteristics

### SampleRankingLoss (Baseline)
- Traditional pairwise ranking approach using margin-based loss
- O(n²) complexity with batch size
- Complete pairwise comparison of all samples

### MultiEventRankingLoss
- Compares different event types within the same observation
- O(e²) complexity with number of events
- Margin-based pairwise comparisons

### SampleListMLELoss
- Listwise ranking approach using Plackett-Luce model
- O(n log n) complexity
- Directly optimizes probability of correct ordering

### SampleSOAPLoss
- Accelerated pairwise approach with strategic pair sampling
- Reduces complexity to approximately O(n log n)
- Uses margin-based comparisons with optimized implementation

### SampleRankNetLoss
- Probabilistic pairwise ranking using logistic function
- Smooth differentiable loss with cross-entropy
- Adaptive sampling of pairs to reduce computation

### EventRankNetLoss
- RankNet approach for comparing different event types
- Probabilistic ranking between competing risks
- Particularly effective for multi-event scenarios

