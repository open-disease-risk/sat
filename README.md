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

To-Dos
============================
> See Issues
