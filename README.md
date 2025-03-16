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
          ├── callbacks/                # HF Training callbacks
          ├── data/                     # Data load/transform/parsing configuration
          ├── debug/                    # Configuration when debugging an execution
          ├── experiments/              # Experiment settings
          ├── hydra/                    # Hydra configuration
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

Loss Functions for Survival Analysis
============================

SAT provides a comprehensive suite of loss functions specifically designed for survival analysis tasks. These loss functions are located in the `sat.loss` module and are organized by their primary application:

### Core Loss Functions

#### Negative Log-Likelihood of Piecewise Constant Hazards (NLLPCHazard)

The NLLPCHazard loss is the primary loss function for most survival models. It leverages the piecewise constant hazard assumption to maximize the likelihood of the observed data:

$$\mathcal{L}_{\text{NLLPC}} = -\sum_{i=1}^{n} \delta_i \log h(t_i) + \int_{0}^{t_i} h(u) du$$

where:
- $\delta_i$ is the event indicator (1 if the event occurred, 0 if censored)
- $h(t)$ is the hazard function at time $t$
- $t_i$ is the observed time for individual $i$

```yaml
sat-transformer:
  _target_: sat.loss.survival.nllpc_hazard.SATNLLPCHazardLoss
  importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
  num_events: ${data.num_events}
```

#### Mismatch Loss

The Mismatch loss directly optimizes the difference between predicted mean survival times and observed times. It applies different penalties for censored and uncensored samples:

- For observed events: MSE between predicted mean lifetime and observed time
- For censored events: Hinge loss that penalizes when predicted mean lifetime < censoring time

$$\mathcal{L}_{\text{mismatch}} = \frac{1}{n_{obs}} \sum_{i: \delta_i=1} (\hat{\mu}_i - t_i)^2 + \frac{1}{n_{cens}} \sum_{i: \delta_i=0} \max(0, t_i - \hat{\mu}_i)$$

where $\hat{\mu}_i$ is the predicted mean lifetime.

```yaml
sat-transformer:
  _target_: sat.loss.survival.mismatch.MismatchLoss
  duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
  max_time: ${survival_max_time}
```

### Ranking-Based Losses

Ranking losses enforce the relative ordering of outcomes, which is particularly valuable in survival analysis where preserving the order of event times is often more important than exact time predictions. These losses can be used both as primary objectives or as regularizers when combined with other losses.

#### Sample Ranking Loss

Implements the DeepHit ranking loss that penalizes incorrectly ordered pairs of samples. This loss focuses on ensuring that patients who experience events earlier are given higher risk scores than those who experience events later or are censored.

$$\mathcal{L}_{\text{rank}} = \frac{1}{N_c N_e} \sum_{i=1}^{N_c} \sum_{j=1}^{N_e} \mathbf{1}_{T_i < T_j} \exp\left(\frac{S_i(T_i) - S_j(T_i)}{\sigma}\right)$$

where:
- $N_c$ and $N_e$ are the numbers of comparable pairs
- $T_i$ and $T_j$ are the times for samples $i$ and $j$
- $S_i(t)$ is the survival function for sample $i$ at time $t$
- $\sigma$ is a temperature parameter that controls the penalty magnitude

**Use as primary loss function:** When exact calibration of survival curves is less important than preserving ranking order (e.g., in medical risk stratification).

**Use as regularizer:** Combined with NLLPCHazard to improve discrimination between high and low-risk patients while maintaining calibration.

```yaml
sat-transformer:
  _target_: sat.loss.ranking.base_ranking.SampleRankingLoss
  duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
  importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
  sigma: ${ranking_loss_sigma}  # Lower values (e.g., 0.1) make the loss more sensitive to ranking errors
  num_events: ${data.num_events}
```

#### ListMLE Ranking Loss

Implements a list-wise ranking loss based on the ListMLE algorithm that optimizes the probability of the correct ordering. Unlike pairwise approaches, ListMLE optimizes the entire ordering at once, making it computationally more efficient for large datasets.

$$\mathcal{L}_{\text{ListMLE}} = -\sum_{i=1}^{n} \log \frac{\exp(\phi(x_i)/\tau)}{\sum_{j=i}^{n} \exp(\phi(x_j)/\tau)}$$

where:
- $\phi(x_i)$ is the score for instance $i$
- $\tau$ is the temperature parameter

The implementation provides both inter-subject ranking (ranking between different subjects for the same event) and intra-subject ranking (ranking different events for the same subject) controlled by `lambda_inter` and `lambda_intra` parameters.

**Use as primary loss function:** When computational efficiency is important and the model needs to learn from many examples simultaneously.

**Use as regularizer:** When you want to ensure consistent ordering of risks without the computational burden of pairwise comparisons.

```yaml
sat-transformer:
  _target_: sat.loss.ranking.listmle.ListMLERankingLoss
  temperature: 0.5  # Controls the sharpness of the ranking (lower = more strict)
  lambda_inter: 0.1  # Weight for inter-subject ranking component
  lambda_intra: 0.1  # Weight for intra-subject ranking component
  eps: 1e-5  # Numerical stability term
  importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
  num_events: ${data.num_events}
```

#### MultiEvent Ranking Loss

Extends the ranking loss concept to multiple competing risks, ensuring consistent ranking across different event types. This is particularly valuable in competing risks analysis where ordering of different event types needs to be preserved.

**Use as regularizer:** To improve discrimination in competing risks models by enforcing appropriate ordering between different event types.

```yaml
sat-transformer:
  _target_: sat.loss.ranking.base_ranking.MultiEventRankingLoss
  duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
  importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
  sigma: 0.1
  num_events: ${data.num_events}
```

#### SurvRNC Loss

Implements the Survival Rank-N-Contrast (SurvRNC) loss function that combines contrastive learning with ranking objectives. This approach brings ideas from self-supervised learning to survival analysis, creating more robust representations that capture relative risk ordering.

$$\mathcal{L}_{\text{SurvRNC}} = -\log \frac{\exp(s(x_i, x_i^+)/\tau)}{\sum_{j \in \mathcal{N}(i)} \exp(s(x_i, x_j)/\tau)}$$

where:
- $s(x_i, x_j)$ is the similarity score between samples $i$ and $j$
- $x_i^+$ is a positive example for sample $i$
- $\mathcal{N}(i)$ is the set of negative examples for sample $i$
- $\tau$ is the temperature parameter

**Use as primary loss function:** In scenarios with limited labeled data, as it can leverage unlabeled examples through contrastive learning principles.

**Use as regularizer:** To enhance representation learning in survival models, pushing similar risk profiles closer in the embedding space.

```yaml
sat-transformer:
  _target_: sat.loss.ranking.survrnc.SurvRNCLoss
  temperature: 0.1  # Controls sensitivity to similarities
  margin: 0.1  # Margin for the ranking component
  batch_samples: 16  # Number of samples used in each contrastive batch
  importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
  num_events: ${data.num_events}
```

#### RankNet and SOAP Loss

The package also implements other specialized ranking losses:

- **RankNetLoss**: Implements the RankNet algorithm for learning to rank, ideal for optimizing ranking metrics like concordance index directly.

- **SOAPLoss** (Statistically Optimal Accelerated Pairwise): A more efficient ranking loss which uses accelerated gradient methods for faster convergence in large datasets.

These losses can be particularly effective when ranking accuracy is the primary evaluation metric for your survival model.

### Regression Losses

Regression losses focus on predicting continuous values directly. In the context of survival analysis, these can be used to estimate time-to-event directly, predict specific quantiles of a survival distribution, or model other continuous outcomes in a multi-task setting.

#### L1/L2 Losses

Standard regression losses with special handling for censored data:

**L1 Loss (Mean Absolute Error)** is robust to outliers, making it suitable for skewed time-to-event distributions:

$$\mathcal{L}_{L1} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**L2 Loss (Mean Squared Error)** provides stronger gradients for optimization but is more sensitive to outliers:

$$\mathcal{L}_{L2} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Use as primary objective:** For direct time-to-event regression when traditional survival analysis approaches are not needed.

**Use in multi-task setting:** Can be combined with survival losses to simultaneously predict both the survival distribution and expected survival time or other continuous outcomes.

```yaml
sat-transformer-event-duration:
  _target_: sat.loss.regression.l1.L1Loss  # or sat.loss.regression.mse.MSELoss
  training_set: ${data.label_transform.save_dir}/transformed_train_labels.csv
  importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
  l1_type: margin  # or 'uncensored' - controls how censored data is handled
  reduction: "mean"  # Options: "mean", "sum", "none"
  num_events: ${data.num_events}
```

#### Quantile Loss 

The quantile loss enables predicting specific quantiles of the outcome distribution rather than just the mean or median, providing a more complete picture of the uncertainty in time-to-event predictions:

$$\mathcal{L}_{q}(y, \hat{y}) = \sum_{i=1}^{n} \rho_q(y_i - \hat{y}_i)$$

where $\rho_q(u) = u(q - \mathbf{1}_{u < 0})$ is the quantile loss function for quantile level $q$.

**Use as primary objective:** When you need to predict specific risk quantiles (e.g., pessimistic/optimistic scenarios) or want to model the entire conditional distribution of the time-to-event.

**Use in multi-task setting:** Can be combined with survival losses to provide both survival curves and specific quantile predictions of survival time.

```yaml
sat-transformer-event-duration:
  _target_: sat.loss.quantile.quantile.QuantileLoss
  quantiles: ${regression_quantiles}  # Can specify multiple quantiles like [0.1, 0.25, 0.5, 0.75, 0.9]
  importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
  training_set: ${data.label_transform.save_dir}/transformed_train_labels.csv
  l_type: margin  # Controls how censored data is handled
  num_events: ${data.num_events}
```

#### Smooth Quantile Loss

A differentiable approximation of the standard quantile loss that provides better optimization properties through continuous gradients. This addresses the non-differentiability issue of the standard quantile loss at the origin.

$$\mathcal{L}_{q}^{smooth}(u, \alpha) = \begin{cases}
 q u - \frac{\alpha}{2} u^2 & \text{if } 0 < u \leq \frac{q}{\alpha} \\
 \frac{q^2}{2\alpha} & \text{if } u > \frac{q}{\alpha} \\
 (q-1) u - \frac{\alpha}{2} u^2 & \text{if } \frac{q-1}{\alpha} \leq u \leq 0 \\
 \frac{(q-1)^2}{2\alpha} & \text{if } u < \frac{q-1}{\alpha}
\end{cases}$$

where $u = y - \hat{y}$ is the prediction error, $q$ is the quantile level, and $\alpha$ is the smoothing parameter.

**Use as primary objective:** When you need better convergence properties for quantile regression, especially with deeper networks or when using adaptive optimizers.

**Use in multi-task setting:** When combining with other losses that may create conflicting gradients, the smooth version helps stabilize optimization.

```yaml
sat-transformer-event-duration:
  _target_: sat.loss.quantile.smooth_quantile.SmoothQuantileLoss
  quantiles: ${regression_quantiles}  # Can specify multiple quantiles like [0.1, 0.25, 0.5, 0.75, 0.9]
  alpha: 0.5  # Smoothing parameter - controls the degree of smoothing (smaller = more smoothing)
  importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
  training_set: ${data.label_transform.save_dir}/transformed_train_labels.csv
  reduction: "mean"  # Options: "mean", "sum", "none"
  num_events: ${data.num_events}
```


### Classification Losses

Classification losses in the SAT framework can be used for individual event classification tasks or as part of a multi-task setup alongside survival analysis. These losses optimize the probability estimates for discrete event types or risk categories.

#### Cross-Entropy Loss

The standard cross-entropy loss for multi-class classification problems:

$$\mathcal{L}_{CE} = -\sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

where $y_{i,c}$ is the true class label (one-hot encoded) and $\hat{y}_{i,c}$ is the predicted probability for class $c$.

**Use as primary objective:** For discrete event classification when predicting which type of event will occur (without survival time).

**Use in multi-task setting:** For simultaneously predicting event occurrence and survival curves, improving model interpretability.

```yaml
sat-transformer-classification:
  _target_: sat.loss.classification.cross_entropy.CrossEntropyLoss
  weight: null  # Optional class weights for imbalanced datasets
  reduction: "mean"  # Options: "mean", "sum", "none"
  ignore_index: -100  # Value to ignore in target
  num_events: ${data.num_events}
```

#### Binary Cross-Entropy Loss

Specialized form of cross-entropy for binary classification problems:

$$\mathcal{L}_{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

**Use as primary objective:** For binary event classification (will the event occur or not).

**Use in multi-task setting:** For modeling both event occurrence probability and time-to-event distribution.

```yaml
sat-transformer-binary:
  _target_: sat.loss.classification.binary_cross_entropy.BCELoss
  weight: null  # Optional positive class weight
  reduction: "mean"  # Options: "mean", "sum", "none"
  pos_weight: null  # Weight for positive examples
  num_events: ${data.num_events}
```

#### Focal Loss

An extension of cross-entropy that down-weights well-classified examples, focusing training on hard examples:

$$\mathcal{L}_{Focal} = -\sum_{i=1}^{n} \sum_{c=1}^{C} (1 - \hat{y}_{i,c})^\gamma y_{i,c} \log(\hat{y}_{i,c})$$

where $\gamma$ is the focusing parameter.

**Use as primary objective:** For imbalanced classification problems where standard cross-entropy might focus too much on common classes.

```yaml
sat-transformer-classification:
  _target_: sat.loss.classification.focal.FocalLoss
  alpha: 0.25  # Class balance parameter
  gamma: 2.0  # Focusing parameter
  reduction: "mean"
  num_events: ${data.num_events}
```

### Combining Multiple Losses with MetaLoss

The MetaLoss class allows combining multiple loss functions with custom coefficients for a weighted sum. This is particularly useful for adding regularization terms to the primary loss:

```yaml
sat-transformer:
  _target_: sat.loss.meta.MetaLoss
  losses:
    - _target_: sat.loss.survival.nllpc_hazard.SATNLLPCHazardLoss
      importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
      num_events: ${data.num_events}
    - _target_: sat.loss.ranking.listmle.ListMLERankingLoss
      temperature: 0.5
      lambda_inter: 0.1
      lambda_intra: 0.1
      eps: 1e-5
      importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
      num_events: ${data.num_events}
  coeffs: [1.0, 0.1]  # Primary loss weight, regularizer weight
```

### Recommended Configurations

#### For Single-Event Survival Analysis

A strong baseline approach combines NLLPCHazard as the primary loss with a ranking-based regularizer:

```yaml
sat-transformer:
  _target_: sat.loss.meta.MetaLoss
  losses:
    - _target_: sat.loss.survival.nllpc_hazard.SATNLLPCHazardLoss
      importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
      num_events: ${data.num_events}
    - _target_: sat.loss.survival.ranking.SampleRankingLoss
      duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
      importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
      sigma: 0.1
      num_events: ${data.num_events}
  coeffs: [1.0, 0.05]  # Primary loss weight, regularizer weight
```

#### For Competing Risks Survival Analysis

For multiple competing events, combine NLLPCHazard with both sample ranking and multi-event ranking losses:

```yaml
sat-transformer:
  _target_: sat.loss.meta.MetaLoss
  losses:
    - _target_: sat.loss.survival.nllpc_hazard.SATNLLPCHazardLoss
      importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
      num_events: ${data.num_events}
    - _target_: sat.loss.survival.ranking.SampleRankingLoss
      duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
      importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
      sigma: 0.1
      num_events: ${data.num_events}
    - _target_: sat.loss.ranking.base_ranking.MultiEventRankingLoss
      duration_cuts: ${data.label_transform.save_dir}/duration_cuts.csv
      importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
      sigma: 0.1
      num_events: ${data.num_events}
  coeffs: [1.0, 0.05, 0.05]  # Primary, sample ranking, multi-event ranking
```


Loss Balancing Strategies
============================

Multi-objective training with complex loss functions is a common challenge in deep learning, especially in survival analysis where multiple components (likelihood, ranking, etc.) need to be balanced effectively. SAT provides a flexible loss balancing API that supports various strategies for optimizing multi-component losses. These strategies can be applied to any loss function in the library by specifying the `balance_strategy` parameter.

### Available Balancing Strategies

#### Standard Fixed Weighting (`none`)

**Description:** The simplest approach using fixed coefficients to weigh different loss components. Loss components are simply multiplied by predefined weights and summed.

**Pros:**
- Simple to implement and understand
- Deterministic behavior
- No overhead during training

**Cons:**
- Requires manual tuning of weights for each dataset/task
- Does not adapt to changing scales during training
- Can lead to one loss component dominating others if scales differ significantly

**Configuration:**
```yaml
sat-transformer:
  _target_: sat.loss.survival.dsm.DSMLoss
  balance_strategy: "none"  # Fixed weighting (default)
  # Other parameters...
```

**Citation:** This is the standard approach used in most deep learning frameworks.

#### Scale Normalization (`scale_normalization`)

**Description:** Dynamically adjusts weights based on the relative scales of different loss components. Uses an exponential moving average (EMA) to track the scale of each loss component and normalizes them accordingly.

**Pros:**
- Adapts to changing loss scales during training
- Prevents dominant loss components from overwhelming others
- Reduces the need for manual weight tuning

**Cons:**
- Introduces an additional hyperparameter (EMA momentum)
- May not be ideal when loss components should have different magnitudes

**Configuration:**
```yaml
sat-transformer:
  _target_: sat.loss.survival.dsm.DSMLoss
  balance_strategy: "scale_normalization"  # Scale normalization
  ema_momentum: 0.9  # Controls the EMA update rate (higher = slower adaptation)
  # Other parameters...
```

**Citation:** This approach is inspired by GradNorm (Chen et al., 2018) but focuses on loss scales rather than gradients.

#### Gradient Normalization (`grad_normalization`)

**Description:** Normalizes the gradients of different loss components to ensure they have similar contributions to parameter updates. This strategy directly addresses the impact of each loss component on learning.

**Pros:**
- Ensures balanced gradient flow from all loss components
- Works well for tasks with different convergence rates
- Preserves the original loss values for interpretability

**Cons:**
- Computationally more expensive (requires gradient calculations)
- May require careful tuning of the normalization factor

**Configuration:**
```yaml
sat-transformer:
  _target_: sat.loss.survival.dsm.DSMLoss
  balance_strategy: "grad_normalization"  # Gradient normalization
  # Other parameters...
```

**Citation:** Based on GradNorm (Chen et al., 2018): "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks" (ICML 2018).

#### Homoscedastic Uncertainty Weighting (`homoscedastic`)

**Description:** Learns the optimal weights based on the aleatoric uncertainty of each task/component. Loss components with higher uncertainty receive lower weights, allowing the model to focus more on predictable tasks.

**Pros:**
- Automatically learns optimal weights
- Theoretically grounded in uncertainty principles
- Adapts throughout training as task uncertainties change

**Cons:**
- Requires additional learnable parameters
- May struggle with highly correlated tasks

**Configuration:**
```yaml
sat-transformer:
  _target_: sat.loss.survival.dsm.DSMLoss
  balance_strategy: "homoscedastic"  # Homoscedastic uncertainty weighting
  # Other parameters...
```

**Citation:** Based on Kendall et al. (2018): "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (CVPR 2018).

#### Adaptive Loss Balancing (`adaptive`)

**Description:** Directly learns the optimal loss weights through backpropagation without relying on uncertainty estimates. The weights themselves become trainable parameters optimized alongside model parameters.

**Pros:**
- Highly flexible and adaptive
- No assumptions about the nature of the tasks
- Can discover complex relationships between loss components

**Cons:**
- More complex to implement
- May require regularization to prevent extreme weighting
- Potentially unstable in early training stages

**Configuration:**
```yaml
sat-transformer:
  _target_: sat.loss.survival.dsm.DSMLoss
  balance_strategy: "adaptive"  # Adaptive loss balancing
  # Other parameters...
```

**Citation:** Inspired by Liu et al. (2019): "End-to-End Multi-Task Learning with Attention" (CVPR 2019) and Guo et al. (2018): "Dynamic Task Prioritization for Multitask Learning" (ECCV 2018).

### Recommendations for Different Scenarios

1. **For initial experiments**: Start with `scale_normalization` as it provides a good balance between simplicity and effectiveness.

2. **For complex multi-component losses**: Use `adaptive` or `homoscedastic` to let the model learn the optimal weights.

3. **For computationally constrained environments**: Use `none` with manually tuned weights or `scale_normalization` as they have minimal computational overhead.

4. **For models with components that converge at different rates**: Use `grad_normalization` to ensure balanced learning across components.

5. **For ensembling or uncertainty estimation**: Try `homoscedastic` as it provides uncertainty estimates along with optimal weights.

### Example: Balancing DeepHit's Likelihood and Ranking Losses

```yaml
sat-transformer:
  _target_: sat.loss.survival.deephit.DeepHitLoss
  alpha: 0.2  # Initial weight between likelihood (0.8) and ranking (0.2)
  sigma: 0.1  # Temperature parameter for ranking component
  balance_strategy: "adaptive"  # Let the model learn optimal weights
  ema_momentum: 0.9  # Used with scale_normalization strategy
  importance_sample_weights: ${data.label_transform.save_dir}/imp_sample.csv
  num_events: ${data.num_events}
```

In this example, DeepHit's likelihood and ranking components will have their weights adaptively learned during training, starting from the 0.8/0.2 split specified by alpha but evolving as training progresses.


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

To-Dos
============================
> See Issues
