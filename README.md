Folder Structure
============================

### Top-level directory layout

    .
    ├── conf                    # Hydra configuration files
    ├── data                    # Data files
    ├── envs                    # Conda environment files for Azure execution
    ├── sat                     # Python source files
    ├── project.toml            # project definitions/dependencies
    └── README.md

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

### Key Loss Functions

#### Survival Loss Functions
- **NLLPCHazard**: Negative log-likelihood loss for piece-wise constant hazard models
- **DeepHit Loss**: Combines likelihood, ranking, and calibration components for competing risks
- **Survival Focal Loss**: Down-weights easily predicted examples to focus on harder cases

#### Ranking Loss Functions
- **SampleRankingLoss**: Ensures proper ordering of different samples within the same event type
- **MultiEventRankingLoss**: Ranks different event types for the same sample in competing risks
- **ListMLE Losses**: Efficient list-based ranking losses that scale better than pairwise approaches

#### Auxiliary Loss Functions
- **Brier Score Loss**: Measures calibration of probability predictions
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
│       ├── SampleRankingLoss
│       └── BrierScore
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

To monitor loss balancing during training, use the `LossWeightLoggerCallback`:

```yaml
callbacks:
  - _target_: sat.transformers.callbacks.LossWeightLoggerCallback
    log_freq: 1
    prefix: "loss_weights"
    log_eval: true
    log_train: true
```

This logs weights to TensorBoard, enabling visualization of balancing dynamics.

For detailed documentation on loss balancing and multi-task learning integration, see:
- [docs/loss.md](docs/loss.md) - Comprehensive overview of the loss framework
- [docs/loss_weight_logging.md](docs/loss_weight_logging.md) - Guide to monitoring loss weights
- [docs/loss_optimization.md](docs/loss_optimization.md) - Details on optimized implementations

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

To-Dos
============================
> See Issues
