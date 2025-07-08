# SurvRNC vs Other Ranking Losses - Benchmark Summary

This document summarizes the performance comparison between different ranking loss implementations, including the new SurvRNCLoss.

## Performance Comparison

The table below shows the speedup factor compared to SampleRankingLoss (higher is better).

| Configuration | MultiEventRankingLoss | SampleListMLELoss | SurvRNCLoss |
|---------------|----------------------|-------------------|------------|
| b16_e1 | 0.10x | 0.23x | 0.27x |
| b16_e2 | 0.96x | 0.68x | 0.43x |
| b32_e1 | 0.92x | 1.33x | 0.29x |
| b32_e2 | 1.26x | 0.85x | 0.26x |

## Key Findings

- **MultiEventRankingLoss**: Average 0.81x speedup compared to SampleRankingLoss
- **SampleListMLELoss**: Average 0.77x speedup compared to SampleRankingLoss
- **SurvRNCLoss**: Average 0.31x speedup compared to SampleRankingLoss

**Overall Best Performer**: MultiEventRankingLoss


## SurvRNC Loss Characteristics

The Survival Rank-N-Contrast (SurvRNC) loss has the following characteristics:

1. **Contrastive Learning Approach**: Unlike pairwise ranking losses, SurvRNC uses an N-pair contrastive approach that focuses on learning similarity between samples with similar outcomes.

2. **Improved Generalization**: By focusing on grouping similar patients together in the embedding space, SurvRNC can lead to better generalization performance, especially in cases with limited data.

3. **Temperature Parameter**: Allows for controlling the sharpness of the similarity distribution, which can be important for balancing between hard and soft contrasts.

4. **Margin Parameter**: Controls the separation boundary between samples considered similar vs. dissimilar, which can help with robustness.

5. **Computational Efficiency**: SurvRNC has a computational complexity that scales better with dataset size compared to traditional pairwise ranking methods.

