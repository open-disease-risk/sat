# Numerical Stability in Survival Analysis Models

This document details the numerical stability measures implemented in the distribution models used for survival analysis. These techniques ensure robust training and inference while maintaining theoretical consistency.

## Theoretical Properties and Constraints

### Survival Function (S(t))

- **Mathematical Definition**: S(t) = P(T > t), the probability of surviving beyond time t
- **Theoretical Constraints**:
  - Range: S(t) ∈ [0, 1]
  - S(0) = 1 (probability of surviving beyond time 0 is 100%)
  - S(∞) → 0 (probability of surviving indefinitely approaches 0)
  - Monotonically decreasing: For t₁ < t₂, S(t₁) ≥ S(t₂)
- **Relationships**:
  - S(t) = exp(-H(t)) where H(t) is the cumulative hazard
  - log(S(t)) = -H(t)

### Hazard Function (h(t))

- **Mathematical Definition**: h(t) = -d/dt[log(S(t))], the instantaneous rate of failure at time t
- **Theoretical Constraints**:
  - Non-negative: h(t) ≥ 0
  - No upper bound: h(t) can be arbitrarily large
  - Different shapes depending on the distribution:
    - Constant (exponential distribution): h(t) = λ
    - Increasing (Weibull with k > 1): risk increases over time
    - Decreasing (Weibull with k < 0): risk decreases over time
    - Non-monotonic (LogNormal): risk increases then decreases
- **Relationships**:
  - h(t) = f(t)/S(t) where f(t) is the probability density function
  - h(t) = d/dt[H(t)]

### Cumulative Hazard Function (H(t))

- **Mathematical Definition**: H(t) = ∫₀ᵗ h(u)du, the integral of the hazard function
- **Theoretical Constraints**:
  - Non-negative: H(t) ≥ 0
  - H(0) = 0 (no accumulated risk at time 0)
  - Monotonically increasing: For t₁ < t₂, H(t₁) ≤ H(t₂)
  - No upper bound
- **Relationships**:
  - H(t) = -log(S(t))
  - S(t) = exp(-H(t))

## Numerical Stability Strategies

### 1. Handling Time Bounds and Edge Cases

```python
# Ensure time is positive and within reasonable bounds
time_safe = torch.clamp(time, min=self.eps, max=1e10)

# Special handling for t=0 (hazard is undefined at t=0)
if time_points.shape[1] > 0 and time_points[:, 0].min() < 1e-5:
    safe_time_points = time_points.clone()
    safe_time_points[:, 0] = torch.max(safe_time_points[:, 0], torch.tensor(1e-5, device=device))
```

**Rationale**: 
- Time values must be positive for survival distributions. The upper bound of 1e10 prevents overflow while allowing for essentially any realistic time value.
- Special attention is needed for t=0, where hazard functions are theoretically undefined (division by zero in density/survival).
- For hazard calculations, we may skip the first time point if it's very close to zero, or replace it with a small positive value.
- We ensure minimum spacing between time points to prevent numerical issues in discrete hazard approximations.

### 2. Survival Function Stability

```python
# Ensure survival probabilities are valid
survival = torch.clamp(survival, min=0.0, max=1.0)
```

**Rationale**: Directly enforces the theoretical constraint that S(t) ∈ [0, 1]. This prevents invalid probabilities that could arise from numerical approximations.

### 3. Hazard Function Stability

```python
# Ensure non-negative and bounded hazard
mixture_hazard = torch.clamp(mixture_hazard, min=0.0, max=1e3)
```

**Rationale**: Enforces the theoretical constraint that h(t) ≥ 0. The upper bound of 1e3 is a practical limit to prevent numerical overflow during training while still allowing for high hazard rates that might occur in realistic scenarios.

### 4. Log-Domain Calculations

```python
# Compute in log domain for numerical stability
log_survival = torch.log(torch.clamp(survival, min=self.eps))
log_survival_safe = torch.clamp(log_survival, min=-100.0, max=0.0)
```

**Rationale**: Log-domain calculations avoid underflow for very small probabilities. The lower bound of -100.0 corresponds to a survival probability of approximately 3.7e-44, which is sufficiently small for any practical application while preventing -∞ values that would cause gradient issues.

### 5. Mixture Weight Normalization

```python
# Ensure weights are normalized and valid
weights_safe = torch.clamp(self.weights, min=self.eps, max=1.0 - self.eps)
weights_sum = weights_safe.sum(dim=1, keepdim=True)
weights_normalized = weights_safe / weights_sum
```

**Rationale**: Ensures mixture weights are valid probabilities (between 0 and 1) and sum to 1, maintaining the proper interpretation of mixture distributions.

### 6. Safe LogSumExp Implementation

```python
# Manual implementation of logsumexp for better control over numerical stability
max_values, _ = torch.max(combined_values_safe, dim=1, keepdim=True)
exp_values = torch.exp(combined_values_safe - max_values)
sum_exp = torch.sum(exp_values, dim=1)
log_likelihood = max_values.squeeze(1) + torch.log(sum_exp)
```

**Rationale**: Implementing logsumexp manually allows for better control over numerical stability compared to using torch.logsumexp directly. This approach prevents both underflow and overflow by shifting values by their maximum before exponentiation.

### 7. NaN and Infinity Handling

```python
# Handle any remaining NaN or Inf values
log_likelihood = torch.nan_to_num(log_likelihood, nan=-10.0, posinf=10.0, neginf=-10.0)
```

**Rationale**: Replaces any NaN or infinite values with finite approximations that allow training to continue. This prevents gradient explosions that would otherwise halt training.

## Distribution-Specific Considerations

### Weibull Distribution

- **Shape parameter (k)**: Clamped to be positive (min=eps) with a reasonable upper bound (max=100.0)
  - k < 1: Decreasing hazard (e.g., early mortality phase)
  - k = 1: Constant hazard (exponential distribution)
  - k > 1: Increasing hazard (e.g., wear-out phase, aging)

- **Scale parameter (λ)**: Clamped to be positive (min=eps) with a reasonable upper bound (max=1000.0)
  - Larger λ corresponds to longer survival times

### LogNormal Distribution

- **Location parameter (μ)**: Clamped within a wide range (min=-100.0, max=100.0)
  - Median survival time is exp(μ)
  
- **Scale parameter (σ)**: Clamped to be positive (min=eps) with a reasonable upper bound (max=100.0)
  - Larger σ corresponds to greater variance in survival times

## Expert Knowledge Integration

When expert knowledge is applied, additional constraints ensure the distribution parameters align with domain expertise:

### Cancer Event Type

```python
# Weibull shape for cancer typically shows increasing hazard
if event_type == "cancer":
    shape = torch.clamp(shape, min=1.2, max=2.5)
```

**Rationale**: Medical literature suggests cancer hazard rates typically increase over time, corresponding to Weibull shape parameters greater than 1.

### Heart Disease Event Type

```python
# Heart disease often has nearly constant hazard
if event_type == "heart_disease":
    shape = torch.clamp(shape, min=0.95, max=1.5)
```

**Rationale**: Heart disease often shows a pattern of slightly increasing hazard over time, with Weibull shape parameters close to 1.

### Infection Event Type

```python
# Infections often show decreasing hazard
if event_type == "infection":
    shape = torch.clamp(shape, min=0.5, max=0.9)
```

**Rationale**: Infectious disease mortality often shows a pattern of decreasing hazard over time, with Weibull shape parameters less than 1.

## Conclusion

These numerical stability measures ensure robust training and inference while maintaining the theoretical properties required for valid survival analysis. The constraints are carefully chosen to be conservative enough for stability while still allowing the model to represent virtually any realistic survival scenario.

## References

- Kalbfleisch, J. D., & Prentice, R. L. (2011). The statistical analysis of failure time data (Vol. 360). John Wiley & Sons.
- Ibrahim, J. G., Chen, M. H., & Sinha, D. (2001). Bayesian survival analysis. Springer Science & Business Media.
- Kleinbaum, D. G., & Klein, M. (2012). Survival analysis (Vol. 3). Springer.
- Royston, P., & Parmar, M. K. (2002). Flexible parametric proportional-hazards and proportional-odds models for censored survival data. Statistics in medicine, 21(15), 2175-2197.