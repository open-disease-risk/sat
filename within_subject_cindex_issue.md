# Implementation Plan for Within-Subject C-Index

## Problem Statement
Currently, the concordance index (c-index) is calculated across patients to evaluate how well the model ranks risk between different patients. We need to extend this functionality to calculate c-index within a single patient for competing risks scenarios.

## Implementation Plan

### 1. Create New Within-Subject C-Index Module
- Create a new module `within_subject_concordance/within_subject_concordance.py`
- Leverage existing code structure from `concordance_index_ipcw.py`
- Adapt to focus on competing risks within the same patient

### 2. Core Implementation Details
- Modify the input structure to accept multi-event data for each patient
- Group events by patient ID
- For each patient with multiple events, calculate pairwise concordance
- Aggregate results across all patients

### 3. Technical Considerations
- Reuse the vectorized ranking approach from `MultiEventRankingLoss`
- Handle censored vs. uncensored event comparisons appropriately
- Ensure numerical stability with proper handling of edge cases

### 4. Integration Points
- Add the new metric to `sat/evaluate/eval_modules.py`
- Register in the metrics configuration
- Ensure compatibility with existing evaluation pipeline

### 5. Validation Strategy
- Add comprehensive unit tests with synthetic multi-event data
- Validate on existing datasets like HSA synthetic with known competing risks

## Expected Benefits
- Better evaluation of competing risk models
- More accurate assessment of within-subject event ranking capability
- Enhanced model selection for competing risks scenarios

## Resources Required
- Familiarity with concordance index calculation
- Understanding of competing risks in survival analysis
- Knowledge of PyTorch tensor operations for efficient implementation

## Technical Implementation Details

The current c-index is calculated over ranking of events between patients. The same implementation approach can be adapted for ranking of events within a patient by:

1. Modifying the input structure to work with patient IDs and multi-event data
2. Grouping events by patient ID instead of treating each sample independently
3. Computing concordance pairs only within each patient's events
4. Aggregating the within-patient concordance metrics

### Code Structure

```python
def within_subject_concordance(
    event_times: torch.Tensor,  # [batch, num_events]
    event_indicators: torch.Tensor,  # [batch, num_events] 
    predicted_risks: torch.Tensor,  # [batch, num_events]
    patient_ids: torch.Tensor,  # [batch]
):
    """
    Calculate concordance index for events within each patient.
    
    Args:
        event_times: Time-to-event values for each patient and event type
        event_indicators: Binary indicators of whether events occurred (1) or were censored (0)
        predicted_risks: Model's predicted risk scores for each patient and event type
        patient_ids: Unique identifier for each patient to group events
        
    Returns:
        float: Within-subject concordance index
    """
    # Implementation will group by patient_ids and compute concordance within each group
    # ...
```

This functionality will be particularly valuable for evaluating models on datasets with competing risks like the HSA synthetic dataset, enabling better assessment of a model's ability to correctly rank different event types for the same patient.