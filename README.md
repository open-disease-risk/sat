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

Survival Analysis Loss Functions
============================

SAT supports several loss functions for survival analysis:

### Standard Survival Loss Functions
- **Negative Log-Likelihood PCHazard**: The standard negative log-likelihood for piece-wise constant hazard models
- **DeepHit Loss**: A comprehensive loss function combining likelihood, ranking, and calibration components for competing risks
- **Survival Focal Loss**: Loss function that down-weights easily predicted survival outcomes to focus on harder cases, with support for multi-focal parameters

For detailed information about all available loss functions and their usage, see [docs/loss.md](docs/loss.md).

### DeepHit Loss Function
The DeepHit loss is based on the paper "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks" by Lee et al. (2018). It consists of three components:

1. **Likelihood Loss**: Maximizes the probability of observing the actual event times
2. **Ranking Loss**: Ensures proper ordering of survival probabilities
3. **Calibration Loss** (optional): Enforces agreement between predicted and empirical probabilities

Configuration example:
```yaml
task:
  transformer:
    sat-transformer:
      _target_: sat.loss.DeepHitLoss
      duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
      num_events: ${data.num_events}
      alpha: 0.5  # Weight for likelihood component
      beta: 0.5   # Weight for ranking component
      gamma: 0.0  # Weight for calibration component (optional)
      sigma: 0.1  # Scaling factor for ranking loss
```

Loss Balancing Framework
============================

SAT supports a flexible loss balancing framework for multi-objective training, especially useful for survival analysis which often requires balancing multiple loss components (likelihood, ranking, calibration, etc.).

### Balancing Strategies

The framework offers five different balancing strategies:

1. **Fixed Weighting** (`fixed`): Standard approach using predefined coefficients
   - Simple and predictable
   - Requires manual tuning of weights
   
2. **Scale Normalization** (`scale`): Dynamically normalizes losses based on their magnitudes
   - Automatically adapts to different loss scales
   - Prevents losses with larger scales from dominating
   - Uses exponential moving average (EMA) to track loss scales
   
3. **Gradient Normalization** (`grad`): Balances losses based on gradient magnitudes
   - Focuses on the rate of improvement rather than absolute values
   - Useful when losses have different optimization characteristics
   
4. **Homoscedastic Uncertainty Weighting** (`uncertainty`): Learns uncertainty parameters
   - Automatically learns optimal weighting based on loss variance
   - Based on principled probabilistic approach
   - Adapts during training through gradient descent
   
5. **Adaptive Weighting** (`adaptive`): Adjusts weights based on loss improvement trends
   - Gives more weight to tasks that are improving more slowly
   - Encourages balanced optimization across objectives
   - Detects when a loss component has stalled

### Usage in Configurations

Loss balancing can be configured in the YAML configuration files:

```yaml
# Example of MetaLoss with adaptive balancing
sat-transformer:
  _target_: sat.loss.MetaLoss
  losses:
    - _target_: sat.loss.SATNLLPCHazardLoss
      importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
      num_events: ${data.num_events}
    - _target_: sat.loss.SampleEventRankingLoss
      duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
      num_events: ${data.num_events}
  # Initial coefficients
  coeffs:
    - 1.0  # NLL weight
    - 0.5  # Ranking weight
  # Use adaptive balancing strategy
  balance_strategy: adaptive
  balance_params:
    alpha: 0.9
    window_size: 100
    adaptation_rate: 0.005
```

Predefined configurations can be found in `conf/tasks/losses/balancing.yaml` and can be included using Hydra's composition:

```yaml
# Reference a predefined balancing strategy
balance_strategy: ${losses.balancing.uncertainty.balance_strategy}
balance_params: ${losses.balancing.uncertainty.balance_params}
```

### Common Use Cases

1. **Handling unstable losses**: Use `scale` normalization to prevent sudden loss spikes from destabilizing training
2. **Multi-task learning**: Use `adaptive` or `uncertainty` to balance different tasks automatically
3. **Complex multi-component losses**: Use `grad` normalization when combining losses with different gradient characteristics
4. **Fine-tuning sensitivity**: Use `fixed` weighting with manually tuned coefficients for precise control

### Implementation Details

The balancing framework is implemented through:

- **Base `Loss` class**: All loss functions inherit balancing capabilities
- **`LossBalancer` hierarchy**: Each strategy is implemented as a separate balancer class
- **`MetaLoss` integration**: Full support for balancing multiple loss components

For examples of complex multi-component losses with different balancing strategies, see `conf/tasks/losses/balanced_multi_loss.yaml`.

### Monitoring Loss Weights with TensorBoard

You can monitor the evolution of loss weights during training by enabling loss weight logging:

1. Add the `loss_weight_logger` callback to your experiment configuration:

```yaml
defaults:
  - callbacks@callbacks: default
  - callbacks@callbacks: loss_weight_logger
```

2. Launch TensorBoard after training:

```bash
tensorboard --logdir logs/your-experiment-dir
```

3. Navigate to the "Scalars" section to visualize:
   - `loss_weights/train/weight_X`: Training loss weights over time
   - `loss_weights/eval/weight_X`: Evaluation loss weights over time

This allows you to observe how different balancing strategies behave and identify potential issues like weight oscillation or domination by a single loss component.

See [docs/loss_weight_logging.md](docs/loss_weight_logging.md) for detailed information on interpreting the weight logs and [docs/loss_optimization.md](docs/loss_optimization.md) for an overview of our multi-level approach to loss optimization.

To-Dos
============================
> See Issues
