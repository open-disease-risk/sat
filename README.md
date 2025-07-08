Folder Structure
============================

### Top-level directory layout

    .
    ├── conf                    # Hydra configuration files
    ├── data                    # Data files
    ├── docs                    # Documentation
    ├── envs                    # Conda environment files for Azure execution
    ├── sat                     # Python source files
    ├── scripts                 # Utility scripts
    ├── pyproject.toml          # Project definitions/dependencies
    ├── .pre-commit-config.yaml # Pre-commit hooks configuration
    └── README.md

### Development Setup

We use pre-commit hooks to enforce code quality standards. To set up the development environment:

```bash
# Using the setup script
./scripts/setup_dev_env.sh

# Or manually
pip install poetry pre-commit
poetry install
pre-commit install
```

For more information on development practices, see [Development Guide](docs/development_guide.md).

### Configuration structure

> Configuration of the tasks is done using Hydra.

    .
    └── conf
          ├── aml/                      # Azure ML compute configuration
          ├── callbacks/                # HF Training callbacks
          ├── data/                     # Data load/transform/parsing configuration
          ├── debug/                    # Configuration when debugging an execution
          ├── experiments/              # Experiment settings
          ├── hydra/                    # Hydra configuration
          ├── inputs/                   # Input configuration for job execution on Azure
          ├── local/                    # Definition of the compute targets on Azure (private)
          ├── outputs/                  # Output configuration for job execution on Azure
          ├── paths/                    # Path configurations
          ├── sweep/                    # Hyper-parameter configuration for Azure
          ├── tasks/                    # Survival task head configurations
          ├── tokenizers/               # Different HF tokenization schemes
          ├── trainer/                  # HF training arguments
          ├── transformers/             # Transformer configurations
          ├── defaults.yaml             # Default configuration all scripts share
          ├── eval.yaml                 # evaluate a trained transformer
          ├── finetune.yaml             # fine-tune a transformer
          ├── infer.yaml                # run inference on a dataset
          ├── prepare_data.yaml         # parse a dataset into the format required for sat
          ├── preprocess_data.yaml      # preprocess especially large datasets and serialize
          ├── pretrain.yaml             # pretrain transformer
          ├── train_labeltransform.yaml # train the label transformation for event/duration
          └── train_tokenizer.yaml      # train a tokenizer


Training Survival Models
============================

### Using the scripts
1. Use e.g., `pyenv` to install python 3.9 or 3.10
2. In the `sat` root project directory set the python version: `pyenv local 3.9.xx`
3. Install the project dependencies: `poetry install`
4. Activate the installed environment: `poetry shell`
5. Execute any of the scripts: `python -m sat.<script_name> experiments=<data>/<task>`
 - where `<script_name>` is any of the scripts in the `sat` package namespace
 - `<data>` is the dataset to run experiments on, i.e., see the
   `conf/experiments` folder
 - `<task>` is the survival task to execute, i.e., see `conf/experiments/<data>` folder

*Note 1*: If you are finetuning on an Apple Mac M1/M2, you would need to provide
an MPS fallback option to pytorch for the welling tasks via:
`PYTORCH_ENABLE_MPS_FALLBACK=1 python -m sat.finetune experiments=seer/welling`.

*Note 2*: You can overwrite any configuration variable on the command-line.
E.g., if you wish to use a specific learning rate, you can execute the
finetuning script like so `python -m sat.finetune experiments=seer/survival
learning_rate=0.001`. You can also append any other configuration variable shall
you wish to change more than one.

*Note 3*: Use `TOKENIZERS_PARALLELISM=false poetry run ...` to execute
finetuning to avoid potential performance (on small datasets) and deadlock
issues. The finetuning script may need to address the tokenization scheme such
that parallelization can be utilized within the training procedure.

### Run the scripts in Azure
1. Go to `ml.azure.com` and select the workspace to use for the experiments
2. Download the `config.json` of the workspace configuration and put it into the
   project root directory
3. Specify the compute target for CPU and GPU in `conf/local`, e.g.,

> Example GPU configuration using NC6s VMs

    # @package _global_

    hydra:
      launcher:
        compute:
          target: "dd-nc6s-v3"

4. Execute any script remotely: `python -m sat.<script_name> experiments=<data>/<task> -m`
5. Authenticate with `ml.azure.com` when prompted
6. The job opens automatically in a browser window

*Note*: Some scripts require a CPU target and some a GPU target. Check the
script configuration for the environment that is being referenced. All scripts
that use `envs/satcuda118py39.yml` execute on a GPU. It is necessary to specify
therefore at least two files in `conf/local`: one for CPU targets
(`conf/local/cpu.yaml`) and one for GPU targets (`conf/local/gpu.yaml`) in your
workspace. If these targets do not exist, they will be created according to the
configuration in `conf/aml/`.


### Order of script execution
Taking the SEER data as an example, these are the concrete steps to execute all
relevant scripts to perform finetuning.

1. `python -m sat.prepare_data experiments=seer/survival`
2. `python -m sat.train_tokenizer experiments=seer/survival`
3. `python -m sat.train_labeltransform experiments=seer/survival`
4. `python -m sat.finetune experiments=seer/survival`

*Note 1*: Different tasks share the data and pre-processing steps. E.g., in the
above example, you could execute any other finetuning task without needing to
execute steps 1.-3..

Composable Loss Framework
============================

SAT features a comprehensive, composable loss framework that allows flexible combination of multiple loss functions with advanced balancing strategies.

### Core Components

1. **Base Loss Functions**: Individual loss functions for survival analysis, ranking, classification, and more
2. **MetaLoss**: A flexible container that combines multiple loss functions with dynamic weighting
3. **Loss Balancing Strategies**: Methods to adjust weights between loss components (fixed, scale, gradient, uncertainty, adaptive)
4. **Multi-level Balancing**: Support for balancing both within task heads and between task heads in multi-task learning
5. **Momentum Contrast (MoCo)**: Buffer-based approach to enhance training stability, especially for highly censored datasets

### Key Loss Functions

#### Survival Loss Functions
- **NLLPCHazard**: Negative log-likelihood loss for piece-wise constant hazard models
- **DeepHit Loss**: Combines likelihood, ranking, and calibration components for competing risks
- **Survival Focal Loss**: Down-weights easily predicted examples to focus on harder cases
- **MoCo-Enhanced Loss**: Wraps any survival loss with a memory buffer to improve training stability for highly censored data

#### Ranking Loss Functions
- **SampleRankingLoss**: Ensures proper ordering of different samples within the same event type
- **MultiEventRankingLoss**: Ranks different event types for the same sample in competing risks
- **ListMLE Losses**: Efficient list-based ranking losses that scale better than pairwise approaches

#### Auxiliary Loss Functions
- **Quantile Regression Loss**: Optimizes specific quantiles of the survival distribution

### Example Loss Recipe

For competing risks with imbalanced event types:

```yaml
_target_: sat.loss.MetaLoss
losses:
  - _target_: sat.loss.survival.SurvivalFocalLoss
    gamma: [2.0, 3.0]  # Different focus per event type
    num_events: ${data.num_events}

  - _target_: sat.loss.ranking.SampleRankingLoss
    sigma: 0.1
    num_events: ${data.num_events}

  - _target_: sat.loss.ranking.MultiEventRankingLoss
    sigma: 0.1
    num_events: ${data.num_events}

balance_strategy: "uncertainty"  # Learn optimal weights
```

For detailed information about all available loss functions, mathematical formulations, pros and cons, and recipes for combining them, see the comprehensive documentation in [docs/loss.md](docs/loss.md).

See [docs/loss_weight_logging.md](docs/loss_weight_logging.md) for information on monitoring loss weights during training, and [docs/loss_optimization.md](docs/loss_optimization.md) for details on our optimized loss implementations.

Multi-Task Learning Integration
============================

The SAT framework provides comprehensive support for multi-task learning, allowing models to simultaneously optimize for multiple objectives. This is particularly valuable in survival analysis where different aspects like likelihood, ranking, and calibration all contribute to model performance.

### Multi-level Loss Balancing

The framework integrates loss balancing at two distinct levels:

1. **Within Task Heads**: Each individual task head (survival, classification, regression) can use `MetaLoss` to combine multiple loss functions with configurable weighting.

2. **Between Task Heads**: The `MTLForSurvival` class balances losses from different task heads, allowing competition and cooperation between tasks.

This creates a hierarchical loss structure:

```
MTLForSurvival
├── Survival Head
│   └── MetaLoss
│       ├── NLLPCHazard
│       └── SampleRankingLoss
├── Classification Head
│   └── CrossEntropyLoss
└── Regression Head
    └── MSELoss
```

### Available Balancing Strategies

Five different balancing strategies are supported at both levels:

1. **Fixed Weighting** (`fixed`): Standard approach using predefined coefficients
   - Simple and predictable
   - Requires manual tuning of weights

2. **Scale Normalization** (`scale`): Dynamically normalizes losses based on their magnitudes
   - Prevents losses with larger scales from dominating
   - Uses exponential moving average (EMA) tracking

3. **Gradient Normalization** (`grad`): Balances losses based on gradient magnitudes
   - Focuses on model optimization dynamics
   - Adapts to different loss optimization characteristics

4. **Homoscedastic Uncertainty Weighting** (`uncertainty`): Learns optimal weights
   - Automatically learns importances through principled approach
   - Adapts during training via gradient descent

5. **Adaptive Weighting** (`adaptive`): Adjusts weights based on improvement rates
   - Gives more weight to tasks that are improving slowly
   - Encourages balanced optimization across objectives

### Configuration Example

```yaml
# MTL survival model with automatic balancing between heads
model:
  _target_: sat.models.heads.MTLForSurvival

  # Configure the survival head
  survival_head:
    loss:
      _target_: sat.loss.MetaLoss
      losses:
        - _target_: sat.loss.survival.NLLPCHazard
          # ...
        - _target_: sat.loss.ranking.SampleRankingLoss
          # ...
      balance_strategy: "scale"

  # Configure the classification head
  classification_head:
    loss:
      _target_: sat.loss.classification.CrossEntropyLoss
      # ...

  # Balance between the task heads
  mtl_balance_strategy: "uncertainty"
  mtl_balance_params:
    init_sigma: 1.0
```

### Monitoring Weight Evolution

Loss weights are automatically logged to TensorBoard during training, enabling visualization of balancing dynamics without any additional configuration.

For detailed documentation on loss balancing and multi-task learning integration, see:
- [docs/loss.md](docs/loss.md) - Comprehensive overview of the loss framework
- [docs/loss_weight_logging.md](docs/loss_weight_logging.md) - Guide to monitoring loss weights
- [docs/loss_optimization.md](docs/loss_optimization.md) - Details on optimized implementations
- [docs/performance_comparison.md](docs/performance_comparison.md) - Benchmark results and performance optimization recommendations

Exploratory Data Analysis (EDA) Framework
============================

SAT includes a comprehensive Exploratory Data Analysis (EDA) framework designed specifically for survival analysis datasets. This framework helps understand dataset characteristics, determine appropriate model configurations, and identify potential biases.

### Key Features

The EDA framework provides three main types of analysis:

1. **Distribution Analysis**
   - Fits parametric distributions (Weibull, LogNormal, LogLogistic) to event times
   - Evaluates goodness-of-fit using AIC/BIC metrics
   - Automatically generates DSM model configurations based on distribution analysis
   - Visualizes distribution fits against empirical data

2. **Censoring Analysis**
   - Quantifies censoring patterns and rates
   - Tests for informative censoring that could bias model training
   - Analyzes competing risks interactions when multiple event types are present
   - Creates visualizations of censoring patterns over time

3. **Covariate Analysis**
   - Analyzes feature distributions and their relationship to survival outcomes
   - Identifies potentially important risk factors using various statistical methods
   - Generates feature importance rankings through survival-specific methods
   - Creates stratified survival curves based on key covariates

### Usage

Run the EDA framework on a dataset:

```bash
python -m sat.eda dataset=metabric
```

Customize the analysis:

```bash
python -m sat.eda dataset=seer analysis.run_distribution_analysis=true analysis.run_censoring_analysis=true analysis.run_covariate_analysis=true
```

Export CSV files for external visualization tools (enabled by default):

```bash
python -m sat.eda dataset=metabric performance.export_csv=true
```

Disable CSV export if only visualizations are needed:

```bash
python -m sat.eda dataset=metabric performance.export_csv=false
```

### Configuration

The EDA framework uses Hydra configuration in `conf/eda.yaml`. Key configuration options:

```yaml
# Control which analyses to run
analysis:
  run_distribution_analysis: true
  run_censoring_analysis: true
  run_covariate_analysis: true

  # Distribution analysis settings
  distributions:
    - weibull
    - lognormal
    - loglogistic
  prefer_metric: bic

  # Censoring analysis settings
  censoring:
    alpha: 0.05
    plot_survival_curves: true

  # Covariate analysis settings
  covariates:
    top_n_features: 10
    importance_methods:
      - cox_ph
      - mutual_information
```

### Output

The EDA framework generates:

1. **Visualizations**: Plots for distribution fits, survival curves, feature importances, etc.
2. **JSON Reports**: Structured summaries of all analyses with statistics and recommendations
3. **Model Configurations**: Based on analysis results, generates optimal configuration files for DSM and other models

All outputs are saved to the configured output directory (`outputs/eda/{dataset}` by default).

For detailed information about the EDA framework, see [docs/eda.md](docs/eda.md).

**Note:** If you encounter an error about "cannot import name 'trapz' from 'scipy.integrate'", this is a compatibility issue between newer versions of scipy and lifelines. The framework includes an automatic fix for this issue, but you can also install specific compatible versions:
```
pip install lifelines==0.27.8 scipy==1.10.1
```

Model Evaluation Framework
============================

SAT provides robust frameworks for evaluating survival models through confidence interval estimation and cross-validation.

### Confidence Interval Estimation (CI)

The CI framework allows you to systematically evaluate model performance across multiple replications with statistical confidence.

**Key Features:**
- Runs the model pipeline multiple times with different random seeds
- Computes confidence intervals for key metrics (Brier score and C-index)
- Automatically terminates runs when sufficient statistical confidence is achieved
- Supports early stopping based on a predefined number of replications

**Statistical Methodology:**

The framework uses a rigorous approach to determine when sufficient statistical confidence has been achieved. It implements the following statistical test for each metric:

1. For each metric (Brier score and C-index), the framework tracks:
   - Sample mean (x̄)
   - Sample variance (S²)
   - Number of replications (n)

2. It computes the confidence interval half-length using the t-distribution:

   δ(n, α) = t(n-1, 1-α/2) × √(S²/n)

   Where:
   - t(n-1, 1-α/2) is the critical value of the t-distribution with n-1 degrees of freedom
   - α is the significance level (e.g., 0.05 for 95% confidence)

3. The framework checks if the relative error is within the desired precision:

   δ(n, α) / |x̄| ≤ γ'

   Where:
   - γ' is the adjusted relative error: γ' = γ/(1+γ)
   - γ is the desired precision/error margin

When this condition is met for both metrics, the framework concludes that sufficient statistical confidence has been achieved and terminates the replications (unless the minimum number of replications hasn't been reached yet).

**Usage:**
```bash
python -m sat.ci experiments=metabric/survival alpha=0.05 error=0.1 n=10
```

**Configuration Options:**
- `alpha`: Significance level for confidence intervals (default: 0.05)
- `error`: Desired precision/error margin (default: 0.1)
- `n`: Minimum number of replications to run (default: 10)
- `less_than_n`: Maximum number of replications to run (default: 10)
- `use_val`: Whether to use validation metrics instead of test metrics (default: false)

**Output:**
The CI framework generates a `metrics-pipeline-ci.json` file containing:
- Number of replications performed
- Mean, variance, and standard deviation for Brier score
- Mean, variance, and standard deviation for C-index (IPCW)

### K-Fold Cross-Validation (CV)

The CV framework implements k-fold cross-validation to assess model performance and robustness.

**Key Features:**
- Supports configurable k-fold cross-validation (default: 5-fold)
- Option to reuse existing data splits for reproducibility
- Automatically runs the full pipeline (data preparation, tokenizer training, label transformation, and fine-tuning) for each fold
- Computes aggregate statistics across all folds

**Statistical Consideration:**
It's important to note that in k-fold cross-validation, the validation sets are not independent samples because:
1. Each data point appears in exactly one validation fold
2. Training sets overlap significantly (each shares (k-2)/k of the data with other training sets)
3. This overlap creates dependencies between performance measurements on different folds

As a result, the variance and standard deviation reported across folds should not be directly interpreted as the variance of the performance estimate, and should not be used to construct confidence intervals as if the measurements were independent. These statistics are still useful for understanding the stability of the model across different data splits, but they require careful interpretation.

For more statistically rigorous variance estimation, consider using nested cross-validation or the CI framework with multiple independent training/test splits.

**Usage:**
```bash
python -m sat.cv experiments=metabric/survival cv.kfold=5 cv_kfold_reuse=false
```

**Configuration Options:**
- `cv.kfold`: Number of folds for cross-validation (default: 5)
- `cv_kfold_reuse`: Whether to reuse existing data splits (default: true)

**Output:**
The CV framework generates a `metrics-pipeline-cv.json` file containing:
- Number of folds evaluated
- Mean, variance, and standard deviation for Brier score across folds
- Mean, variance, and standard deviation for C-index (IPCW) across folds

Both frameworks integrate with MLflow for experiment tracking when `run_id` is provided.

Hyperparameter Tuning with Optuna
============================

SAT provides robust hyperparameter tuning capabilities using Optuna integrated with Hydra. This allows you to efficiently search for optimal hyperparameters while maintaining the Hydra configuration system.

### Key Features

- **Efficient Search**: Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler for efficient hyperparameter search
- **Pruning Support**: Automatically terminates poor-performing trials early to save computational resources
- **Study Persistence**: Stores optimization history in SQLite database for resilience to interruptions
- **Hydra Integration**: Leverages Hydra's configuration system for clean, modular parameter definitions
- **Visualization**: Generate optimization history and parameter importance plots

### Usage

Run hyperparameter tuning for a specific model configuration:

```bash
# Basic usage - specify both the experiment and sweep configuration
python -m sat.optuna_optimize experiments=metabric/survival sweep=metabric/survival

# Specify number of trials
python -m sat.optuna_optimize experiments=metabric/survival sweep=metabric/survival hydra.sweeper.n_trials=50

# Run with existing study (don't overwrite)
python -m sat.optuna_optimize experiments=metabric/survival sweep=metabric/survival optuna.study_overwrite=false

# Enable multirun for parallel trial execution
python -m sat.optuna_optimize experiments=metabric/survival sweep=metabric/survival --multirun
```

Note: All results, including the optimization database, will be stored in the `/data/optuna/` directory. Each experiment will create a subdirectory based on the study name.

### Available Search Spaces

Pre-configured search spaces are available for:

- **Survival Models**: `sweep=metabric/survival`
- **DeepHit Models**: `sweep=metabric/deephit`

### Creating Custom Search Spaces

To define a custom parameter search space, create a new configuration in `conf/sweep/<dataset>/`:

```yaml
# @package _global_

# This file defines the parameter search space for Optuna

# Study settings
hydra:
  sweeper:
    study_name: custom_optimization
    storage: sqlite:///${optuna_db}/custom_optimization.db

    # Define parameter search spaces
    params:
      learning_rate:
        type: float
        low: 1e-5
        high: 1e-3
        log: true

      # Add other parameters to tune
      # ...

# Optimization settings
optuna:
  metric: test_your_metric  # Metric to optimize
  direction: minimize  # or maximize
  study_overwrite: true  # Set to false to resume existing study

  # Optional pruning configuration
  pruner:
    type: median  # Options: median, percentile, threshold, hyperband, none
    n_startup_trials: 5  # Number of trials before pruning starts
    n_warmup_steps: 20  # Number of steps per trial before pruning starts
    interval_steps: 5  # Interval between pruning checks
```

For detailed documentation on parameter space configuration, see the [Optuna documentation](https://optuna.readthedocs.io/).

### Visualizing Results

After running optimization, you can visualize the results:

```bash
# Basic usage
python -m sat.optuna_visualize --db data/optuna/metabric_survival.db --study metabric_survival_opt

# Specify custom output directory
python -m sat.optuna_visualize --db data/optuna/metabric_deephit.db --study metabric_deephit_opt --output ./viz_results
```

This generates:
- Optimization history plots
- Parameter importance analysis
- Slice plots for top parameters
- Contour plots showing parameter interactions
- Summary statistics and best parameters
- CSV file with all trial data

MEDS Format Support
============================

SAT now supports the Medical Event Data Standard (MEDS) format, providing integration with standardized healthcare data for survival analysis.

### Key Features

- **MEDS Format Support**: Read Parquet files in the MEDS standard format
- **FEMR Integration**: Generate events and timelines using FEMR labelers
- **Multi-Event Support**: Handle competing risks with multiple event types
- **Flexible Configuration**: Customizable label definitions for various event types

### Using MEDS Data

1. **Install Dependencies**:
   Make sure you have the required dependencies:
   ```bash
   poetry add pyarrow fastparquet femr
   ```

2. **Configure Label Definitions**:
   Create a custom configuration file or modify the existing one at `conf/data/parse/meds.yaml`:

   ```yaml
   label_definitions:
     - name: mortality
       positive_class: true
       table_name: mortality
       time_field: days
       output_label_fields: ["days", "label_name"]
     - name: readmission
       positive_class: true
       table_name: readmissions
       time_field: days
       output_label_fields: ["days", "label_name"]
   ```

3. **Prepare the Data**:
   ```bash
   python -m sat.prepare_data experiments=meds/survival data_source=/path/to/your/meds/file.parquet dataset=your_meds_dataset
   ```

4. **Train the Tokenizer**:
   ```bash
   python -m sat.train_tokenizer experiments=meds/survival dataset=your_meds_dataset
   ```

5. **Train Label Transform**:
   ```bash
   python -m sat.train_labeltransform experiments=meds/survival dataset=your_meds_dataset
   ```

6. **Fine-tune a Model**:
   ```bash
   python -m sat.finetune experiments=meds/survival dataset=your_meds_dataset
   ```

### Example MEDS Data Processing

```python
from femr.datasets import DatasetBuilder
from femr.labelers import LabelFetcher

# Load MEDS data
builder = DatasetBuilder("path/to/meds_file.parquet")
dataset = builder.build()

# Define events to extract
label_definitions = [
    {
        "name": "mortality",
        "positive_class": True,
        "table_name": "mortality",
        "time_field": "days",
        "output_label_fields": ["days", "label_name"]
    }
]

# Create labels
label_fetcher = LabelFetcher(dataset)
mortality_labels = label_fetcher.get_labels(label_definitions[0])

# Use the labels in SAT survival analysis
# ...
```

For more information about MEDS format, visit [FEMR documentation](https://github.com/som-shahlab/femr).

To-Dos
============================
> See Issues
