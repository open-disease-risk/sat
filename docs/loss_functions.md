# Survival Analysis Loss Functions

This document provides a comprehensive overview of all loss functions implemented in the Survival Analysis Toolkit (SAT), their mathematical formulations, appropriate use cases, and relationships to the research literature.

## Table of Contents

1. [Introduction](#introduction)
2. [Classification and Regression Losses with Censoring](#classification-and-regression-losses-with-censoring)
   - [L1 Loss with Censoring](#l1-loss-with-censoring)
   - [Quantile Loss with Censoring](#quantile-loss-with-censoring)
3. [Parametric Survival Losses](#parametric-survival-losses)
   - [Negative Log-Likelihood Piecewise Constant Hazard (NLLPCHazard)](#negative-log-likelihood-piecewise-constant-hazard-nllpchazard)
   - [Deep Survival Machines (DSM)](#deep-survival-machines-dsm)
   - [Multi-Event Neural Survival Analysis (MENSA)](#multi-event-neural-survival-analysis-mensa)
   - [DeepHit Likelihood](#deephit-likelihood)
   - [DeepHit Calibration](#deephit-calibration)
4. [Ranking-Based Losses](#ranking-based-losses)
   - [Pairwise Ranking Losses](#pairwise-ranking-losses)
     - [SOAP (Statistically Optimal Accelerated Pairwise)](#soap-statistically-optimal-accelerated-pairwise)
     - [RankNet](#ranknet)
     - [SurvRNC (Survival Rank-N-Contrast)](#survrnc-survival-rank-n-contrast)
   - [Listwise Ranking Losses](#listwise-ranking-losses)
     - [ListMLE (List Maximum Likelihood Estimation)](#listmle-list-maximum-likelihood-estimation)
   - [Event-Specific and Sample-Based Ranking](#event-specific-and-sample-based-ranking)
     - [Sample Ranking](#sample-ranking)
     - [Multi-Event Ranking](#multi-event-ranking)
5. [Meta-Loss Approaches](#meta-loss-approaches)
   - [Balancing Strategies](#balancing-strategies)
   - [Focal Loss Adaptation](#focal-loss-adaptation)
6. [Future Directions](#future-directions)

## Introduction

Survival analysis poses unique challenges due to censoring, where the event of interest may not be observed for all samples. Traditional loss functions from classification and regression must be adapted to account for this partial information.

The SAT library implements a diverse set of loss functions that address different aspects of survival prediction:

1. **Classification and regression losses** adapted to handle censoring through margin-based or hinge formulations
2. **Parametric survival losses** that explicitly model the survival and hazard functions
3. **Ranking-based losses** that focus on correctly ordering patients by risk
4. **Meta-loss approaches** that combine multiple loss components

Each loss function is suitable for different scenarios and modeling goals.

## Classification and Regression Losses with Censoring

These losses adapt standard classification and regression objectives to handle censored data through special formulations.

### L1 Loss with Censoring

The L1 (absolute error) loss is modified to handle censoring with three different approaches:

#### Uncensored L1

```
L = 1/N * Σ |t_i - ŷ_i| * I(δ_i = 1)
```

where:
- t_i is the observed time
- ŷ_i is the predicted time
- δ_i is the event indicator (1 = event, 0 = censored)
- I() is the indicator function

This only computes the loss for uncensored examples, ignoring censored data.

#### Hinge L1

```
L = 1/N * Σ |max(0, t_i - ŷ_i)| * I(δ_i = 0) + |t_i - ŷ_i| * I(δ_i = 1)
```

This applies a hinge constraint to censored samples, penalizing only when predictions are shorter than observed censoring times.

#### Margin L1

```
L = 1/N * (Σ |t_i - ŷ_i| * I(δ_i = 1) + Σ w_i|t̃_i - ŷ_i| * I(δ_i = 0))
```

where:
- w_i is a weight based on Kaplan-Meier estimates
- t̃_i is the expected time of event given censoring at time t_i

This uses the Kaplan-Meier estimator to inform loss for censored samples.

**References:**
- Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018). DeepHit: A deep learning approach to survival analysis with competing risks. AAAI.
- Kvamme, H., & Borgan, Ø. (2019). Continuous and discrete-time survival prediction with neural networks. arXiv preprint arXiv:1910.06724.

**When to use:**
- When direct time prediction is required
- For models that need to output a single time point
- When interpretability of predictions is important

### Quantile Loss with Censoring

Extends quantile regression to account for censoring:

```
L_q(y, ŷ) = q * max(0, y - ŷ) + (1-q) * max(0, ŷ - y)
```

where q is the quantile of interest (e.g., 0.5 for median).

For censored data, we use either the uncensored approach (ignoring censored samples) or the margin approach (using Kaplan-Meier estimates).

**References:**
- Koenker, R., & Bassett Jr, G. (1978). Regression quantiles. Econometrica, 33-50.
- Lee, C., & Van Der Schaar, M. (2019). Temporal quantile regression with neural networks: An application to remaining lifetime prediction. Advances in neural information processing systems.

**When to use:**
- When prediction intervals are more important than point estimates
- For uncertainty estimation in survival times
- When interested in specific quantiles of the survival distribution (e.g., median survival)

## Parametric Survival Losses

These losses explicitly model the survival and hazard functions using different distributional assumptions.

### Negative Log-Likelihood Piecewise Constant Hazard (NLLPCHazard)

Models the hazard function as piecewise constant, maximizing the likelihood of observed data:

```
L = -Σ [ δ_i * log(h(t_i)) + log(S(t_i)) ]
```

where:
- h(t) is the hazard function at time t
- S(t) is the survival function at time t
- δ_i is the event indicator
- t_i is the observed time

**References:**
- Kvamme, H., Borgan, Ø., & Scheel, I. (2019). Time-to-event prediction with neural networks and Cox regression. Journal of Machine Learning Research, 20(129), 1-30.

**When to use:**
- When a non-parametric approach is desired
- For flexible modeling of the hazard function
- When survival curves need to be estimated at many time points

### Deep Survival Machines (DSM)

Models the survival distribution as a mixture of parametric distributions (Weibull or log-normal):

```
L = -Σ [ δ_i * log(f(t_i)) + (1-δ_i) * log(S(t_i)) ]
```

where:
- f(t) is the probability density function
- S(t) is the survival function
- Both are modeled as mixtures: S(t) = Σ_k w_k * S_k(t)

**References:**
- Nagpal, C., Li, X., & Dubrawski, A. (2021). Deep survival machines: Fully parametric survival regression and representation learning for censored data with competing risks. IEEE Journal of Biomedical and Health Informatics.

**When to use:**
- When a flexible parametric approach is desired
- For scenarios where understanding the entire survival distribution is important
- When dealing with competing risks
- For more stable training compared to non-parametric approaches

### Multi-Event Neural Survival Analysis (MENSA)

An extension of DSM that explicitly models dependencies between different event types:

```
L = -Σ [δ_i,j * log(f_j(t_i)) + (1-δ_i,j) * log(S_j(t_i))] + R(D)
```

where:
- f_j(t) is the density function for event type j
- S_j(t) is the survival function for event type j
- R(D) is a regularization term on the dependency matrix D

**References:**
- For the detailed implementation of MENSA, see [MENSA Documentation](mensa.md).

**When to use:**
- For competing risks scenarios with potential dependencies between events
- When more accurate modeling of inter-event relationships is required
- For complex multi-event survival analysis

### DeepHit Likelihood

Models the discrete-time survival function directly for competing risks:

```
L = -Σ [ δ_i,j * log(h_j(t_i) * S(t_i-)) + (1-Σ_j δ_i,j) * log(S(t_i)) ]
```

where:
- h_j(t) is the cause-specific hazard for event j at time t
- S(t) is the overall survival function
- S(t-) is the survival function just before time t

**References:**
- Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018). DeepHit: A deep learning approach to survival analysis with competing risks. AAAI.

**When to use:**
- For competing risks scenarios
- When a discrete-time approach is preferred
- For datasets with many tied event times

### DeepHit Calibration

Ensures the model's predictions match observed event frequencies:

```
L = 1/N * Σ_t [ (1-S_j(t) - F_j(t))^2 ]
```

where:
- S_j(t) is the predicted survival for event j at time t
- F_j(t) is the observed frequency of event j by time t

**References:**
- Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018). DeepHit: A deep learning approach to survival analysis with competing risks. AAAI.

**When to use:**
- In combination with other losses (especially DeepHit likelihood)
- When calibration of predictions is important
- For improved reliability of survival predictions

## Ranking-Based Losses

These losses focus on correctly ordering patients by risk, rather than predicting exact survival times.

### Pairwise Ranking Losses

#### SOAP (Statistically Optimal Accelerated Pairwise)

Optimizes the ordering of pairs of samples with an efficient sampling strategy:

```
L = 1/N * Σ_i,j [ max(0, m - sign(t_i - t_j)*(r_i - r_j)) ]
```

where:
- r_i, r_j are risk scores
- t_i, t_j are observed times
- m is a margin hyperparameter

SOAP uses strategic sampling to reduce the O(n²) complexity of pairwise comparisons.

**References:**
- Kvamme, H., & Borgan, Ø. (2019). Efficient comparison of concordance indices for survival analyses. Statistics in Medicine.

**When to use:**
- For large datasets where full pairwise comparison is prohibitive
- When ranking performance (concordance) is the primary objective
- For models that need to be efficiently trained on ranking tasks

#### RankNet

Uses a probabilistic framework for pairwise ranking:

```
L = -Σ_i,j [ y_ij * log(p_ij) + (1-y_ij) * log(1-p_ij) ]
```

where:
- y_ij = 1 if sample i should rank higher than j
- p_ij = sigmoid(σ * (s_i - s_j)) is the predicted probability that i ranks higher than j
- σ is a temperature parameter

**References:**
- Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., & Hullender, G. (2005). Learning to rank using gradient descent. ICML.

**When to use:**
- For a probabilistic interpretation of ranking performance
- When a smooth differentiable ranking loss is desired
- For models that need to output well-calibrated risk scores

#### SurvRNC (Survival Rank-N-Contrast)

Combines ranking and contrastive learning for survival analysis:

```
L = -1/N * Σ_i [ log(exp(sim(a_i, p_i)/τ) / Σ_j exp(sim(a_i, j)/τ)) ]
```

where:
- sim(a,b) is the similarity between samples a and b
- p_i is a positive sample (similar outcome to anchor a_i)
- τ is a temperature parameter

SurvRNC reduces complexity from O(n²) to approximately O(n).

**References:**
- Kvamme, H., & Borgan, Ø. (2021). Learning representations for survival analysis with contrastive learning. arXiv preprint arXiv:2107.07334.

**When to use:**
- For large datasets where efficiency is important
- When better representation learning is a goal
- For improved generalization compared to traditional pairwise losses

### Listwise Ranking Losses

#### ListMLE (List Maximum Likelihood Estimation)

Optimizes the likelihood of the correct ordering of a list:

```
L = -Σ_i log P(π_i|s_i)
```

where:
- π_i is the correct ordering of samples
- s_i are the predicted scores
- P(π|s) is modeled using the Plackett-Luce distribution

**References:**
- Xia, F., Liu, T. Y., Wang, J., Zhang, W., & Li, H. (2008). Listwise approach to learning to rank: theory and algorithm. ICML.

**When to use:**
- When optimizing the entire ordering, not just pairwise comparisons
- For improved ranking performance with reduced complexity
- When the listwise context is important for predictions

### Event-Specific and Sample-Based Ranking

#### Sample Ranking

Efficiently implements ranking for samples within a mini-batch:

```
L = 1/N * Σ_i,j [ max(0, m - sign(t_i - t_j)*(r_i - r_j)) * I(δ_i = 1) * I(δ_j = 1 or t_j > t_i) ]
```

Optimized implementation for better computational efficiency than standard pairwise ranking.

**References:**
- Antolini, L., Boracchi, P., & Biganzoli, E. (2005). A time-dependent discrimination index for survival data. Statistics in medicine, 24(24), 3927-3944.

**When to use:**
- For efficient mini-batch training
- When standard pairwise ranking is too slow
- For models where ranking is more important than exact survival times

#### Multi-Event Ranking

Extends ranking approaches to handle multiple competing events:

```
L = 1/N * Σ_k [ Σ_i,j [ max(0, m - sign(t_i,k - t_j,k)*(r_i,k - r_j,k)) * I(δ_i,k = 1) * I(δ_j,k = 1 or t_j,k > t_i,k) ] ]
```

where k indexes the event type.

**References:**
- Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018). DeepHit: A deep learning approach to survival analysis with competing risks. AAAI.

**When to use:**
- For competing risks scenarios where ranking is important
- When event-specific ordering is more relevant than absolute times
- For large datasets with multiple event types

## Meta-Loss Approaches

These approaches combine multiple loss components or modify existing losses for specific scenarios.

### Balancing Strategies

Combines multiple loss components with dynamic weighting:

```
L = Σ_i w_i * L_i
```

where w_i can be determined by fixed weights, adaptive strategies, or learned during training.

**References:**
- Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. CVPR.

**When to use:**
- When combining multiple loss components
- For multi-task learning scenarios
- When different loss components have different scales or importance

### Focal Loss Adaptation

Adjusts the focus of the loss on hard-to-predict samples:

```
L = -Σ [ (1-p_t)^γ * log(p_t) ]
```

where:
- p_t is the predicted probability for the true class
- γ is a focusing parameter that down-weights easy examples

**References:**
- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. ICCV.

**When to use:**
- For imbalanced datasets
- When some events are harder to predict than others
- For competing risks scenarios with varying event frequencies

## Future Directions

Several promising directions for future loss function development in survival analysis include:

1. **Deep representation learning for survival**
   - Self-supervised approaches specific to time-to-event data
   - Contrastive learning frameworks that leverage censoring information

2. **Causal survival analysis**
   - Loss functions that explicitly account for treatment effects
   - Methods to reduce selection bias in observational survival data

3. **Uncertainty quantification**
   - Bayesian approaches with proper handling of censoring
   - Conformal prediction for survival analysis

4. **Multi-modal survival loss functions**
   - Specialized losses for combining imaging, genomic, and clinical data
   - Attention mechanisms for interpretable feature importance

5. **Dynamic/longitudinal survival prediction**
   - Loss functions that handle time-varying covariates
   - Recurrent architectures with specialized survival objectives

6. **Fairness-aware survival losses**
   - Loss functions that enforce fairness constraints across protected groups
   - Balancing model performance with group parity in survival predictions

7. **Transfer learning for survival**
   - Pre-training objectives specific to survival data
   - Domain adaptation losses for survival across different populations

Future research will likely focus on combining the strengths of different approaches, such as the ranking benefits of SurvRNC with the interpretability of parametric models like MENSA, or developing hybrid losses that simultaneously optimize for different clinical objectives.

The optimal loss function depends on the specific application, available data, and modeling goals. The SAT library provides a comprehensive toolbox for experimenting with different loss functions to find the best approach for each scenario.
