# Implement Streaming Support for Large-Scale Dataset Processing

## Summary
Enable true streaming support throughout the SAT codebase to handle arbitrarily large datasets without memory constraints. This requires refactoring components that currently require full dataset access to use online statistics, batch-wise approximations, and reservoir sampling techniques.

## Motivation
Currently, several key components require loading entire datasets into memory:
- MoCo buffer optimization
- Distribution fitting and EDA
- Label transformations (discretization)
- Ranking loss functions
- IPCW-based evaluation metrics
- Patient-level grouping operations

This limits the ability to work with very large medical datasets that don't fit in memory.

## Proposed Solution

### 1. Online Statistics for Global Computations

#### A. MoCo Buffer Optimization
Replace full dataset scans with online statistics:
```python
class OnlineMoCoEstimator:
    def __init__(self):
        self.event_counts = defaultdict(int)
        self.censoring_rate = OnlineMean()
        self.time_stats = OnlineStats()
        
    def update_batch(self, batch):
        # Update statistics incrementally
        for sample in batch:
            self.event_counts[sample['event']] += 1
            self.censoring_rate.update(sample['censored'])
            self.time_stats.update(sample['time'])
    
    def get_buffer_recommendation(self):
        # Calculate recommendations from running statistics
        return calculate_buffer_sizes(
            self.event_counts,
            self.censoring_rate.mean,
            self.time_stats
        )
```

#### B. Distribution Fitting
Implement online distribution parameter estimation:
```python
class OnlineDistributionFitter:
    def __init__(self, distribution_type):
        self.distribution_type = distribution_type
        self.sufficient_stats = OnlineSufficientStatistics()
        
    def update_batch(self, times, events):
        # Update sufficient statistics for MLE
        self.sufficient_stats.update(times, events)
    
    def get_parameters(self):
        # Calculate MLE from sufficient statistics
        return self.distribution_type.fit_from_stats(
            self.sufficient_stats
        )
```

### 2. Batch-wise Approximations

#### A. Ranking Loss Functions
Implement batch-wise ranking approximations:
```python
class BatchWiseRankingLoss:
    def forward(self, predictions, targets, batch_info):
        # Use batch-local pairs/triplets
        pairs = generate_batch_pairs(predictions, targets)
        
        # Maintain running statistics for global context
        self.global_stats.update(predictions, targets)
        
        # Adjust loss with global statistics
        loss = compute_ranking_loss(pairs)
        adjusted_loss = self.adjust_with_global_context(
            loss, self.global_stats
        )
        return adjusted_loss
```

#### B. Label Transformation
Implement online quantile estimation:
```python
class OnlineQuantileDiscretizer:
    def __init__(self, n_bins):
        self.quantile_estimator = OnlineQuantileEstimator(
            quantiles=np.linspace(0, 1, n_bins+1)
        )
    
    def fit_batch(self, batch_times):
        self.quantile_estimator.update(batch_times)
    
    def transform(self, times):
        # Use current quantile estimates
        return np.digitize(times, self.quantile_estimator.quantiles)
```

### 3. Reservoir Sampling

#### A. Momentum Buffer Management
Replace FIFO queues with reservoir sampling:
```python
class ReservoirMomentumBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
        self.samples_seen = 0
    
    def update(self, new_samples):
        for sample in new_samples:
            self.samples_seen += 1
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(sample)
            else:
                # Reservoir sampling algorithm
                idx = random.randint(0, self.samples_seen - 1)
                if idx < self.buffer_size:
                    self.buffer[idx] = sample
    
    def get_negative_samples(self, n_samples):
        # Sample from reservoir
        return random.sample(self.buffer, min(n_samples, len(self.buffer)))
```

#### B. Reference Set for IPCW
Maintain a reservoir sample for censoring distribution:
```python
class StreamingIPCW:
    def __init__(self, reservoir_size=10000):
        self.reservoir = ReservoirSampler(reservoir_size)
        self.km_estimator = None
    
    def update_batch(self, batch_times, batch_events):
        # Update reservoir with uncensored samples
        uncensored = [(t, e) for t, e in zip(batch_times, batch_events) if e]
        self.reservoir.update(uncensored)
    
    def compute_weights(self, eval_times):
        # Fit KM on reservoir sample
        if self.km_estimator is None:
            reservoir_data = self.reservoir.get_sample()
            self.km_estimator = KaplanMeierFitter()
            self.km_estimator.fit(reservoir_data)
        
        return self.km_estimator.survival_function_at_times(eval_times)
```

### 4. Streaming-Compatible Data Processing

#### A. Patient Event Grouping
Implement streaming patient grouping with bounded memory:
```python
class StreamingPatientGrouper:
    def __init__(self, max_patients_in_memory=10000):
        self.patient_buffer = LRUCache(max_patients_in_memory)
        self.completed_patients = []
    
    def process_event(self, event):
        patient_id = event['patient_id']
        
        if patient_id not in self.patient_buffer:
            self.patient_buffer[patient_id] = []
        
        self.patient_buffer[patient_id].append(event)
        
        # Check if patient is complete (e.g., death event or max follow-up)
        if self.is_patient_complete(patient_id):
            patient_data = self.patient_buffer.pop(patient_id)
            self.completed_patients.append(patient_data)
    
    def get_completed_patients(self):
        # Return and clear completed patients
        result = self.completed_patients
        self.completed_patients = []
        return result
```

### 5. Implementation Plan

1. **Phase 1**: Implement online statistics utilities
   - `OnlineMean`, `OnlineVariance`, `OnlineQuantile` classes
   - `OnlineSufficientStatistics` for common distributions

2. **Phase 2**: Refactor MoCo and distribution fitting
   - Convert `MoCoBufferEstimator` to use online statistics
   - Implement streaming distribution fitting

3. **Phase 3**: Update loss functions
   - Batch-wise ranking losses with global context
   - Reservoir sampling for momentum buffers

4. **Phase 4**: Streaming evaluation metrics
   - Implement streaming IPCW
   - Batch-wise concordance index approximations

5. **Phase 5**: Data processing pipeline
   - Streaming patient grouping
   - Memory-bounded event aggregation

### 6. Configuration Changes

Add streaming configuration options:
```yaml
streaming:
  enabled: true
  batch_size: 1024
  reservoir_size: 10000
  online_stats:
    window_size: 100000
    update_frequency: 1000
  patient_grouping:
    max_patients_in_memory: 10000
    timeout_seconds: 3600
```

### 7. Testing Strategy

1. Implement comparison tests between batch and streaming versions
2. Verify statistical properties of online estimators
3. Test memory usage remains bounded
4. Benchmark performance on large synthetic datasets

## Benefits

1. **Memory Efficiency**: Process datasets of any size with bounded memory
2. **Scalability**: Handle growing medical datasets without infrastructure changes
3. **Flexibility**: Support both batch and streaming modes
4. **Performance**: Enable processing of datasets that don't fit in memory

## Backward Compatibility

- Maintain existing batch processing as default
- Add `streaming=True` parameter to enable streaming mode
- Ensure results are statistically equivalent between modes

## References

- [Online Algorithms for Statistics](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm)
- [Reservoir Sampling](https://en.wikipedia.org/wiki/Reservoir_sampling)
- [Online Quantile Estimation](https://arxiv.org/abs/1801.01143)