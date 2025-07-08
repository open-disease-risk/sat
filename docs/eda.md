# Exploratory Data Analysis (EDA) Framework

The SAT Exploratory Data Analysis (EDA) framework provides comprehensive tools for analyzing survival analysis datasets. This document explains the framework's capabilities, configuration options, and interpretation of results.

## Performance Optimizations

The EDA framework includes two key performance enhancements:

1. **Polars Integration**: For faster data processing, the framework uses the [Polars](https://www.pola.rs/) dataframe library, a high-performance alternative to pandas written in Rust.

2. **CSV Export**: All analysis results can be exported to CSV files for import into external visualization tools like LaTeX/PGFPlots.

## Overview

Survival analysis datasets have unique characteristics that require specialized analysis techniques. The SAT EDA framework addresses these needs with three main analysis components:

1. **Distribution Analysis**: Identifies the best parametric distribution(s) for time-to-event data
2. **Censoring Analysis**: Examines censoring patterns and potential biases
3. **Covariate Analysis**: Explores feature relationships with survival outcomes

## Running EDA

Basic usage:

```bash
python -m sat.eda dataset=metabric
```

Customizing the analysis:

```bash
python -m sat.eda dataset=seer analysis.run_distribution_analysis=true analysis.distributions=[weibull,lognormal]
```

## Distribution Analysis

The distribution analysis module fits various parametric distributions to event times and evaluates their goodness of fit.

### Key Capabilities

- Fits Weibull, LogNormal, and LogLogistic distributions to event times
- Computes AIC and BIC metrics to compare distribution fit quality
- Creates distribution fit visualizations with empirical data
- Generates recommendations for optimal distribution choice
- Automatically creates DSM model configurations based on analysis

### Configuration

```yaml
analysis:
  run_distribution_analysis: true
  distributions:
    - weibull
    - lognormal
    - loglogistic
  top_recommendations: 2
  prefer_metric: bic  # 'aic' or 'bic'
  create_config: true
```

### Interpreting Results

The distribution analysis generates:

1. **Fit statistics**: AIC and BIC values for each distribution
2. **Visualization plots**: Distribution PDF/CDF against empirical data
3. **Recommendations**: Ranked list of distributions by fit quality
4. **DSM configurations**: Auto-generated configuration for Deep Survival Machines

Lower AIC/BIC values indicate better fit. The framework recommends distributions based on the specified metric.

## Censoring Analysis

The censoring analysis module examines censoring patterns and tests for potential biases that could affect model training.

### Key Capabilities

- Calculates censoring rates and patterns over time
- Visualizes censoring distributions with Kaplan-Meier curves
- Tests for informative censoring using statistical tests
- Analyzes relationships between covariates and censoring
- Examines competing risks interactions when multiple event types exist

### Configuration

```yaml
analysis:
  run_censoring_analysis: true
  use_covariates: true
  censoring:
    alpha: 0.05
    plot_survival_curves: true
    analyze_informative_censoring: true
```

### Interpreting Results

The censoring analysis generates:

1. **Censoring statistics**: Rates and patterns of censoring
2. **Informative censoring tests**: p-values for covariate association with censoring
3. **Visualization plots**: Kaplan-Meier curves and censoring distributions
4. **Competing risks analysis**: Interaction between different event types

Pay particular attention to:
- High censoring rates (>70%) which may indicate insufficient follow-up
- Statistically significant informative censoring (p < alpha), which suggests censoring bias
- Changes in censoring patterns over time, which may require specialized modeling

## Covariate Analysis

The covariate analysis module explores the relationships between features and survival outcomes.

### Key Capabilities

- Analyzes the distribution of each covariate
- Identifies potentially important risk factors using multiple methods
- Ranks features by predictive importance for survival
- Creates stratified survival curves for key covariates
- Provides statistical measures of association with survival

### Configuration

```yaml
analysis:
  run_covariate_analysis: true
  covariates:
    # Optional: explicitly define columns to use as covariates
    # columns: []
    alpha: 0.05
    top_n_features: 10
    importance_methods:
      - cox_ph
      - mutual_information
    risk_factor_methods:
      - univariate_cox
      - correlation
    visualizations:
      distribution_plots: true
      stratified_survival_curves: true
```

### Interpreting Results

The covariate analysis generates:

1. **Feature distributions**: Statistical summaries and visualizations
2. **Feature importance**: Ranked lists of features by predictive power
3. **Risk factor analysis**: Statistical associations with survival
4. **Stratified survival curves**: Survival patterns across feature values

Key insights include:
- Identifying the most predictive features for survival outcomes
- Understanding how different feature values affect survival probability
- Discovering potential non-linear relationships with survival
- Identifying features that may require special handling (transformations, encoding, etc.)

## Output Structure

All EDA results are saved in a structured directory format:

```
outputs/eda/{dataset}/
│
├── eda_summary.json                  # Overall summary of all analyses
│
├── distribution_analysis/
│   ├── distribution_fits.png         # Visualization of distribution fits
│   ├── recommended_distributions.json # Distribution recommendations
│   └── dsm_config.yaml               # Auto-generated DSM configuration
│
├── censoring_analysis/
│   ├── {event_type}/
│   │   ├── censoring_pattern.png     # Censoring pattern visualization
│   │   ├── km_curve.png              # Kaplan-Meier curve
│   │   └── censoring_statistics.json # Censoring statistics
│   └── censoring_summary.json        # Overall censoring analysis
│
└── covariate_analysis/
    ├── distributions/                 # Covariate distribution plots
    └── {event_type}/
        ├── importance/                # Feature importance results
        ├── risk_factors/              # Risk factor analysis
        └── covariate_summary.json     # Covariate analysis summary
```

## Integration with Model Training

The EDA framework is designed to integrate seamlessly with the SAT model training pipeline:

1. Run EDA on your dataset to understand its characteristics
2. Use the auto-generated configurations for optimal model setup
3. Apply insights from censoring analysis to handle potential biases
4. Focus on the most important covariates identified in feature analysis

Example workflow:

```bash
# Run EDA
python -m sat.eda dataset=metabric

# Use EDA-generated DSM configuration
python -m sat.finetune experiments=metabric/dsm
```

## Common Use Cases

1. **New Dataset Exploration**:
   ```bash
   python -m sat.eda dataset=new_dataset
   ```

2. **Distribution Analysis Only**:
   ```bash
   python -m sat.eda dataset=metabric analysis.run_censoring_analysis=false analysis.run_covariate_analysis=false
   ```

3. **Focus on Specific Distributions**:
   ```bash
   python -m sat.eda dataset=seer analysis.distributions=[weibull,lognormal]
   ```

4. **Custom Output Directory**:
   ```bash
   python -m sat.eda dataset=metabric outputs.dir=my_custom_eda_results
   ```

5. **Using Polars for Faster Processing**:
   ```bash
   python -m sat.eda_polars dataset=metabric
   ```

6. **Exporting CSV Files for LaTeX/PGFPlots**:
   ```bash
   python -m sat.eda dataset=metabric performance.export_csv=true
   ```

## Performance Considerations

### Polars for Improved Speed

The EDA framework uses Polars, a high-performance dataframe library written in Rust, to significantly improve performance, especially for large datasets.

Polars provides several performance benefits compared to pandas:
- Faster data loading and filtering operations (often 5-10x faster)
- More efficient memory usage (up to 60% less memory)
- Improved parallel processing through vectorized operations
- Reduced compute time for large datasets
- Better handling of string operations

The framework uses Polars for initial data loading and extraction, with a fallback to pandas only when necessary for compatibility with specific operations or libraries like lifelines and scikit-learn. This hybrid approach gives us the performance benefits of Polars while maintaining compatibility with the statistical and modeling libraries built for pandas.

For the largest datasets, the performance gains are substantial:
- Data loading and preprocessing: Up to 10x faster
- Feature extraction: 3-5x faster
- Overall analysis time: 2-4x faster

The framework currently retains some pandas operations in specific analysis modules to maintain compatibility with established statistical libraries. In future versions, we plan to migrate more operations to pure Polars as the ecosystem matures.

### CSV Exports for External Visualization

All analysis results can be exported as CSV files, making it easy to use external visualization tools:

- **Distribution analysis**: Exports distribution parameters, AIC/BIC scores, and survival function data points
- **Censoring analysis**: Exports censoring statistics, KM curve data, and bias test results
- **Covariate analysis**: Exports feature importance rankings and risk factor statistics

Key CSV files generated:
- `{event_type}_distribution_parameters.csv`: Parameters for each fitted distribution
- `{event_type}_survival_functions.csv`: Survival function values for plotting
- `{event_type}_full_metrics.csv`: AIC/BIC and other goodness-of-fit metrics
- `censoring_statistics.csv`: Censoring rates and followup statistics
- `censoring_km_data.csv`: KM estimator data with confidence intervals
- `mutual_info_importance.csv`: Feature importance scores
- `cox_feature_importance.csv`: Hazard ratios and significance values

These CSV files allow you to:
1. Create publication-quality figures using LaTeX/PGFPlots
2. Use custom visualization frameworks
3. Perform additional analyses in external tools
4. Share results with collaborators who don't use Python
