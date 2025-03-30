"""Hyperparameter optimization script using Optuna with Hydra."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from logdecorator import log_on_start, log_on_end, log_on_error
from logging import DEBUG, ERROR

from sat.pipeline import _pipeline
from sat.utils import config, logging, rand

# Set up default logger
logger = logging.get_default_logger()


@rand.seed
def objective(cfg: DictConfig) -> float:
    """Run a single Optuna trial and return the objective value."""
    # Try to find the trial number in the config, or generate a new one if not found
    trial_number = None

    # Check places in the config where trial number might be stored
    if hasattr(cfg, "trial_number"):  # First check the direct trial_number
        trial_number = cfg.trial_number
    elif hasattr(cfg, "optuna_trial_number"):
        trial_number = cfg.optuna_trial_number
    elif hasattr(cfg, "trial_info") and hasattr(cfg.trial_info, "number"):
        trial_number = cfg.trial_info.number
    elif "hydra" in cfg and "job" in cfg.hydra and "trial_number" in cfg.hydra.job:
        trial_number = cfg.hydra.job.trial_number

    # If still not found, create a unique trial number based on timestamp and random digits
    if trial_number is None:
        import time
        import random

        timestamp = int(time.time())
        rand_digits = random.randint(1000, 9999)
        trial_number = timestamp + rand_digits

        logger.info(f"Generated new trial number: {trial_number}")

    dataset = cfg.dataset
    cfg.dataset = dataset + "_trial_" + str(trial_number)

    # Set up output directory for this trial
    trial_dir = Path(cfg.trainer.training_arguments.output_dir)
    os.makedirs(trial_dir, exist_ok=True)

    logger.info(
        f"Running objective function for trial #{trial_number} with output directory: {trial_dir}"
    )

    # Save the trial configuration
    try:
        with open(trial_dir / "trial_config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        logger.info(f"Saved trial configuration to {trial_dir / 'trial_config.yaml'}")
    except Exception as e:
        logger.error(f"Error saving trial configuration: {e}")

    # Get the metric to optimize from config
    metric_name = cfg.optuna.metric

    try:
        # Run the finetune function with the current configuration
        logger.info(f"Trial #{trial_number}: Starting training")
        val_metrics, test_metrics = _pipeline(cfg)
        print(val_metrics)
        logger.info(f"Trial #{trial_number}: Training completed")

        # Find the metric value
        metric_value = val_metrics.get(metric_name)

        if metric_value is None:
            logger.error(
                f"Trial #{trial_number}: Metric {metric_name} not found in results. Available metrics: "
                f"val={list(val_metrics.keys())}, test={list(test_metrics.keys())}"
            )
            # Return a default bad value
            return (
                float("-inf")
                if cfg.tasks.eval_metric_greater_is_better
                else float("inf")
            )

        # Log the result with trial number
        logger.info(
            f"Trial #{trial_number} completed with {metric_name} = {metric_value}"
        )

        return metric_value

    except Exception as e:
        # Log the error with trial number
        logger.error(f"Error during trial #{trial_number}: {e}")

        # Also log stack trace for debugging
        import traceback

        logger.error(f"Trial #{trial_number} stack trace: {traceback.format_exc()}")

        # Return a default bad value based on direction
        return (
            float("-inf") if cfg.tasks.eval_metric_greater_is_better else float("inf")
        )


@log_on_start(DEBUG, "Starting Optuna optimization...", logger=logger)
@log_on_error(
    ERROR,
    "Error during Optuna optimization: {e!r}",
    logger=logger,
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "Optuna optimization completed!", logger=logger)
@hydra.main(
    version_base=None, config_path="../conf", config_name="optuna_optimize.yaml"
)
def optimize(cfg: DictConfig) -> None:
    """Run Optuna hyperparameter optimization."""
    # Look for trial information in configuration
    trial_info = {}

    # Check for direct trial info in config
    if hasattr(cfg, "trial_number"):
        trial_info["trial_number"] = cfg.trial_number
    if hasattr(cfg, "optuna_trial_number"):
        trial_info["optuna_trial_number"] = cfg.optuna_trial_number

    # Check trial_info structure
    if hasattr(cfg, "trial_info"):
        for key in dir(cfg.trial_info):
            if not key.startswith("_"):
                trial_info[f"trial_info.{key}"] = getattr(cfg.trial_info, key)

    # Check hydra.job
    if "hydra" in cfg and "job" in cfg.hydra:
        for key in cfg.hydra.job:
            if "trial" in key.lower() or "num" == key:
                trial_info[f"hydra.job.{key}"] = cfg.hydra.job[key]

    # Log all trial information found
    if trial_info:
        logger.info(f"Found trial information: {trial_info}")
    else:
        logger.info("No trial information found in configuration")

    return objective(cfg)


if __name__ == "__main__":
    optimize()
