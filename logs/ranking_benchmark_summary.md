# Ranking Loss Benchmark Results

## Forward Pass Time (seconds)

|                                           |   ('forward_time', 32) |
|:------------------------------------------|-----------------------:|
| ('MultiEventRankingLoss (Original)', 1)   |            0.00207472  |
| ('MultiEventRankingLoss (Vectorized)', 1) |            0.000209093 |
| ('SampleRankingLoss (Original)', 1)       |            0.046617    |
| ('SampleRankingLoss (Vectorized)', 1)     |            0.000264883 |

## Backward Pass Time (seconds)

|                                           |   ('backward_time', 32) |
|:------------------------------------------|------------------------:|
| ('MultiEventRankingLoss (Original)', 1)   |             0.0029943   |
| ('MultiEventRankingLoss (Vectorized)', 1) |             0.000254154 |
| ('SampleRankingLoss (Original)', 1)       |             0.110095    |
| ('SampleRankingLoss (Vectorized)', 1)     |             0.000554085 |

## Total Time (seconds)

|                                           |   ('total_time', 32) |
|:------------------------------------------|---------------------:|
| ('MultiEventRankingLoss (Original)', 1)   |          0.00506902  |
| ('MultiEventRankingLoss (Vectorized)', 1) |          0.000463247 |
| ('SampleRankingLoss (Original)', 1)       |          0.156712    |
| ('SampleRankingLoss (Vectorized)', 1)     |          0.000818968 |

## SampleRankingLoss Speedup Factors

|    |   batch_size |   num_events |   forward_speedup |   backward_speedup |   total_speedup |
|---:|-------------:|-------------:|------------------:|-------------------:|----------------:|
|  0 |           32 |            1 |           175.991 |            198.697 |         191.353 |

## MultiEventRankingLoss Speedup Factors

|    |   batch_size |   num_events |   forward_speedup |   backward_speedup |   total_speedup |
|---:|-------------:|-------------:|------------------:|-------------------:|----------------:|
|  0 |           32 |            1 |           9.92246 |            11.7814 |         10.9424 |

### Average Speedup

- SampleRankingLoss: 191.35x
- MultiEventRankingLoss: 10.94x

## Visualization

See the generated PNG files in the same directory for visualizations of these results.
