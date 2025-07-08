#!/bin/sh

TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 poetry run python -m sat.optuna_optimize experiments=metabric/survival sweep=survival  --multirun
TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 poetry run python -m sat.optuna_optimize experiments=metabric_numeric/survival sweep=survival  --multirun
TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 poetry run python -m sat.optuna_optimize experiments=hsa_synthetic/survival sweep=survival  --multirun

TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 poetry run python -m sat.optuna_optimize experiments=metabric/deephit sweep=survival  --multirun
TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 poetry run python -m sat.optuna_optimize experiments=metabric_numeric/deephit sweep=survival  --multirun
TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 poetry run python -m sat.optuna_optimize experiments=hsa_synthetic/deephit sweep=survival  --multirun

TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 poetry run python -m sat.optuna_optimize experiments=metabric/deephit_listmle sweep=survival  --multirun
TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 poetry run python -m sat.optuna_optimize experiments=metabric_numeric/deephit_listmle sweep=survival  --multirun
TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 poetry run python -m sat.optuna_optimize experiments=hsa_synthetic/deephit_listmle sweep=survival  --multirun

TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 poetry run python -m sat.optuna_optimize experiments=metabric/deephit_soap sweep=survival  --multirun
TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 poetry run python -m sat.optuna_optimize experiments=metabric_numeric/deephit_soap sweep=survival  --multirun
TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 poetry run python -m sat.optuna_optimize experiments=hsa_synthetic/deephit_soap sweep=survival  --multirun

TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 poetry run python -m sat.optuna_optimize experiments=metabric/deephit_ranknet sweep=survival  --multirun
TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 poetry run python -m sat.optuna_optimize experiments=metabric_numeric/deephit_ranknet sweep=survival  --multirun
TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 poetry run python -m sat.optuna_optimize experiments=hsa_synthetic/deephit_ranknet sweep=survival  --multirun
