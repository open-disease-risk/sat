# Ranking Loss Benchmark Summary

This document summarizes the performance comparison between different ranking loss implementations.

## Batch Size Scaling

The table below shows the speedup factor compared to SampleRankingLoss (higher is better).

| Batch Size | MultiEventRankingLoss | SampleListMLELoss | EventListMLELoss |
|------------|----------------------|-------------------|------------------|
| 16 | 0.18x | 0.52x | 0.23x |
| 32 | 1.43x | 1.78x | 0.17x |

## Event Count Scaling

The table below shows the speedup factor compared to SampleRankingLoss (higher is better).

| Event Count | MultiEventRankingLoss | SampleListMLELoss | EventListMLELoss |
|------------|----------------------|-------------------|------------------|
| 1 | 1.69x | 2.52x | 21.38x |
| 2 | 1.25x | 1.64x | 0.14x |

## Key Findings

- **MultiEventRankingLoss**: Average 1.14x speedup compared to SampleRankingLoss
- **SampleListMLELoss**: Average 1.61x speedup compared to SampleRankingLoss
- **EventListMLELoss**: Average 5.48x speedup compared to SampleRankingLoss

**Overall Best Performer**: EventListMLELoss

### Scaling Observations

- MultiEventRankingLoss scales better with larger batch sizes
- SampleListMLELoss scales better with larger batch sizes
- EventListMLELoss scales worse with larger batch sizes
