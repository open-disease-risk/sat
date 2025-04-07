"""Hyperparameter optimization script using Optuna with Hydra."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import os
import sys

from logging import DEBUG, ERROR
from pathlib import Path

import hydra
from glom import glom
from logdecorator import log_on_end, log_on_error, log_on_start
from omegaconf import DictConfig, OmegaConf

from sat.pipeline import _pipeline
from sat.utils import logging, rand

# Set up default logger
logger = logging.get_default_logger()


@rand.seed
def objective(cfg: DictConfig) -> float or tuple:
    """Run a single Optuna trial and return the objective value(s).

    For single-objective optimization, returns a float.
    For multi-objective optimization, returns a tuple of float values.
    The multi-objective optimization is handled by an appropriate sampler
    like NSGAIISampler which must be configured in the Hydra configuration.
    """
    # Try to find the trial number in the config, or generate a new one if not found
    trial_number = None

    # Log all override parameters from Optuna
    logger.info("Command line arguments:")
    logger.info(f"  {' '.join(sys.argv)}")

    # Extract parameters from sys.argv that might be from Optuna
    optuna_params = {}
    for arg in sys.argv:
        if "=" in arg and not arg.startswith("-"):
            key, value = arg.split("=", 1)
            optuna_params[key] = value

    if optuna_params:
        logger.info(f"Detected Optuna parameters from command line: {optuna_params}")

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
        import random
        import time

        timestamp = int(time.time())
        rand_digits = random.randint(1000, 9999)
        trial_number = timestamp + rand_digits

        logger.info(f"Generated new trial number: {trial_number}")

    modelname = cfg.modelname
    cfg.modelname = modelname + "_trial_" + str(trial_number)

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

    # Get the metrics to optimize from config
    # Handle both single metric (string) and multiple metrics (list) cases
    metric_names = cfg.optuna.metric
    metric_directions = (
        cfg.optuna.metric_direction if hasattr(cfg.optuna, "metric_direction") else None
    )

    # Convert single metric to list for consistent handling
    if isinstance(metric_names, str):
        metric_names = [metric_names]
        # Default to using eval_metric_greater_is_better for direction if not specified
        if metric_directions is None:
            metric_directions = [
                "maximize" if cfg.tasks.eval_metric_greater_is_better else "minimize"
            ]

    # Ensure metric_directions is a list
    if metric_directions is None:
        # Default all metrics to maximize if not specified
        metric_directions = ["maximize"] * len(metric_names)
    elif isinstance(metric_directions, str):
        metric_directions = [metric_directions]

    # Validate we have the same number of metrics and directions
    if len(metric_names) != len(metric_directions):
        logger.error(
            f"Number of metrics ({len(metric_names)}) doesn't match number of directions ({len(metric_directions)})"
        )
        # Default to bad values based on directions
        return [
            float("-inf") if direction == "maximize" else float("inf")
            for direction in metric_directions
        ]

    try:
        # Run the finetune function with the current configuration
        logger.info(f"Trial #{trial_number}: Starting training")
        val_metrics, test_metrics = _pipeline(cfg)
        print(val_metrics)
        logger.info(f"Trial #{trial_number}: Training completed")

        # Find the metric values
        metric_values = []
        missing_metrics = []

        for idx, metric_name in enumerate(metric_names):
            # Check if metric exists in validation
            if glom(val_metrics, metric_name, default=None) is not None:
                metric_values.append(glom(val_metrics, metric_name))
            else:
                missing_metrics.append(metric_name)
                # Add a bad value based on direction
                direction = metric_directions[idx]
                metric_values.append(
                    float("-inf") if direction == "maximize" else float("inf")
                )

        if missing_metrics:
            logger.error(
                f"Trial #{trial_number}: Metrics {missing_metrics} not found in results. Available metrics: "
                f"val={list(val_metrics.keys())}"
            )

        # Log the results with trial number
        metrics_str = ", ".join(
            [f"{name}={value}" for name, value in zip(metric_names, metric_values)]
        )
        logger.info(f"Trial #{trial_number} completed with {metrics_str}")

        # Log all metrics for clarity
        metrics_dict = dict(zip(metric_names, metric_values))
        logger.info(f"Trial #{trial_number} metrics: {metrics_dict}")

        # Store trial information for monitoring
        with open(trial_dir / "metrics.json", "w") as f:
            import json

            json.dump(metrics_dict, f, indent=2)

        # For multi-objective optimization, return a tuple of values
        # For single-objective, return a float
        if len(metric_values) > 1:
            logger.info(
                f"Multi-objective optimization: returning tuple of {len(metric_values)} values"
            )
            return tuple(metric_values)
        else:
            return float(metric_values[0])

    except Exception as e:
        # Log the error with trial number
        logger.error(f"Error during trial #{trial_number}: {e}")

        # Also log stack trace for debugging
        import traceback

        logger.error(f"Trial #{trial_number} stack trace: {traceback.format_exc()}")

        # Return appropriate bad values based on the optimization direction
        # For multi-objective, return a tuple of bad values
        if isinstance(metric_directions, list) and len(metric_directions) > 1:
            bad_values = [
                float("-inf") if d == "maximize" else float("inf")
                for d in metric_directions
            ]
            logger.info(
                f"Returning bad values for multi-objective optimization: {bad_values}"
            )
            return tuple(bad_values)
        else:
            # Single objective case
            direction = (
                metric_directions[0]
                if isinstance(metric_directions, list)
                else "maximize"
            )
            return float("-inf") if direction == "maximize" else float("inf")


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
    # Log the full configuration for debugging
    logger.info("Full configuration received by optimize function:")
    logger.info(OmegaConf.to_yaml(cfg))

    logger.debug("Ensure that we do not use CIs -- too costly")
    cfg.pipeline_use_ci = False

    # Check for Optuna sweep parameters
    logger.info("Checking for Optuna sweep parameters in the config:")

    # Check main Hydra sweeper config
    if "hydra" in cfg and "sweeper" in cfg.hydra:
        logger.info("Hydra sweeper configuration:")
        logger.info(OmegaConf.to_yaml(cfg.hydra.sweeper))

        if "params" in cfg.hydra.sweeper:
            logger.info("Found Hydra sweeper params:")
            logger.info(OmegaConf.to_yaml(cfg.hydra.sweeper.params))

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
