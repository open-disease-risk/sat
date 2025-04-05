"""
MoCo buffer size recommendation tool for SAT.

This script integrates with the Hydra configuration system to analyze datasets
and automatically recommend optimal MoCo buffer sizes based on dataset characteristics.

Usage:
    poetry run python -m sat.moco_recommend -cn experiments/metabric/survival
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple

from sat.utils.moco_buffer_estimator import (
    analyze_dataset_events,
    estimate_optimal_buffer_size,
    generate_moco_config,
    print_buffer_recommendations,
)
from sat.data import load
from sat.utils import logging

# Configure logging
logger = logging.get_default_logger()


def load_dataset_from_config(cfg: DictConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load the dataset based on Hydra configuration, following the approach in finetune.py.

    Args:
        cfg: Hydra configuration

    Returns:
        Tuple of (dataframe, dataset_info)
    """
    dataset_name = cfg.dataset
    logger.info(f"Loading dataset: {dataset_name}")

    # Dataset info to return
    dataset_info = {
        "name": dataset_name,
        "event_column": cfg.data.event_col,
        "time_column": cfg.data.duration_col,
        "multi_event": cfg.data.num_events > 1,
    }

    logger.info(f"Load data using {cfg.data.load}")
    dataset = hydra.utils.call(cfg.data.load)
    dataset = load.split_dataset(cfg.data, dataset)

    # Convert to pandas DataFrame for analysis
    if "train" in dataset:
        df = dataset["train"].to_pandas()
    else:
        # Use first available split
        first_split = list(dataset.keys())[0]
        df = dataset[first_split].to_pandas()

    logger.info(f"Loaded dataset with {len(df)} samples")

    return df, dataset_info


def extract_training_config(cfg: DictConfig) -> Dict[str, Any]:
    """
    Extract training configuration parameters from Hydra config.

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary with training parameters
    """
    training_config = {}

    # Extract batch size from trainer config
    if hasattr(cfg, "trainer") and hasattr(cfg.trainer, "training_arguments"):
        # Look in training arguments
        if hasattr(cfg.trainer.training_arguments, "per_device_train_batch_size"):
            training_config["batch_size"] = (
                cfg.trainer.training_arguments.per_device_train_batch_size
            )
        elif hasattr(cfg.trainer.training_arguments, "train_batch_size"):
            training_config["batch_size"] = (
                cfg.trainer.training_arguments.train_batch_size
            )
        else:
            # Default batch size
            training_config["batch_size"] = 32
            logger.warning(
                f"Batch size not found in training_arguments, using default: {training_config['batch_size']}"
            )
    elif hasattr(cfg, "trainer") and hasattr(cfg.trainer, "train_batch_size"):
        training_config["batch_size"] = cfg.trainer.train_batch_size
    elif hasattr(cfg, "trainer") and hasattr(cfg.trainer, "batch_size"):
        training_config["batch_size"] = cfg.trainer.batch_size
    else:
        # Default batch size
        training_config["batch_size"] = 32
        logger.warning(
            f"Batch size not found in config, using default: {training_config['batch_size']}"
        )

    # Extract minimum events per batch from config or use default
    if hasattr(cfg, "min_events_per_batch"):
        training_config["min_events_per_batch"] = cfg.min_events_per_batch
    else:
        training_config["min_events_per_batch"] = 10  # Reasonable default

    # Check for GPU usage (helps with memory considerations)
    if (
        hasattr(cfg, "trainer")
        and hasattr(cfg.trainer, "training_arguments")
        and hasattr(cfg.trainer.training_arguments, "cuda")
    ):
        training_config["accelerator"] = (
            "gpu" if cfg.trainer.training_arguments.cuda else "cpu"
        )
    elif hasattr(cfg, "trainer") and hasattr(cfg.trainer, "accelerator"):
        training_config["accelerator"] = cfg.trainer.accelerator
    else:
        training_config["accelerator"] = "cpu"

    return training_config


def analyze_batch_size_impact(
    dataset_stats: Dict[str, float],
    output_dir: str = "outputs",
    batch_sizes: Optional[List[int]] = None,
    min_events_per_batch: int = 10,
) -> pd.DataFrame:
    """
    Analyze and plot the impact of different batch sizes on buffer recommendations.

    Args:
        dataset_stats: Dataset statistics from analyze_dataset_events
        output_dir: Directory to save plots
        batch_sizes: List of batch sizes to analyze, or None for defaults
        min_events_per_batch: Minimum desired events per batch

    Returns:
        DataFrame with batch size analysis results
    """
    if batch_sizes is None:
        batch_sizes = [8, 16, 24, 32, 48, 64, 96, 128]

    logger.info(f"Analyzing impact of batch sizes: {batch_sizes}")

    # Storage for results
    buffer_sizes = []
    effective_batch_sizes = []
    buffer_ratios = []
    effective_events = []

    for batch_size in batch_sizes:
        # Get buffer recommendations
        config = estimate_optimal_buffer_size(
            num_samples=dataset_stats["total_samples"],
            censoring_rate=dataset_stats["censoring_rate"],
            batch_size=batch_size,
            min_events_per_batch=min_events_per_batch,
        )

        buffer_sizes.append(config["recommended_buffer_size"])
        effective_batch_sizes.append(config["effective_batch_size"])
        buffer_ratios.append(config["recommended_buffer_size"] / batch_size)

        events_per_batch = batch_size * (1 - dataset_stats["censoring_rate"])
        effective_events.append(events_per_batch * (1 + buffer_sizes[-1] / batch_size))

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot buffer size
    axes[0, 0].plot(batch_sizes, buffer_sizes, "o-", linewidth=2)
    axes[0, 0].set_xlabel("Batch Size")
    axes[0, 0].set_ylabel("Recommended Buffer Size")
    axes[0, 0].set_title("Buffer Size vs. Batch Size")
    axes[0, 0].grid(True)

    # Plot effective batch size
    axes[0, 1].plot(batch_sizes, effective_batch_sizes, "o-", linewidth=2)
    axes[0, 1].set_xlabel("Batch Size")
    axes[0, 1].set_ylabel("Effective Batch Size")
    axes[0, 1].set_title("Effective Batch Size vs. Batch Size")
    axes[0, 1].grid(True)

    # Plot buffer ratio
    axes[1, 0].plot(batch_sizes, buffer_ratios, "o-", linewidth=2)
    axes[1, 0].set_xlabel("Batch Size")
    axes[1, 0].set_ylabel("Buffer-to-Batch Ratio")
    axes[1, 0].set_title("Buffer Ratio vs. Batch Size")
    axes[1, 0].grid(True)

    # Plot effective events
    axes[1, 1].plot(batch_sizes, effective_events, "o-", linewidth=2)
    axes[1, 1].axhline(
        y=min_events_per_batch, color="r", linestyle="--", label="Target Events"
    )
    axes[1, 1].set_xlabel("Batch Size")
    axes[1, 1].set_ylabel("Events in Effective Batch")
    axes[1, 1].set_title("Effective Events vs. Batch Size")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "moco_batch_size_analysis.png")
    plt.savefig(output_path)
    logger.info(f"Saved batch size analysis plot to {output_path}")

    # Create summary table
    summary_table = pd.DataFrame(
        {
            "Batch Size": batch_sizes,
            "Buffer Size": buffer_sizes,
            "Effective Batch": effective_batch_sizes,
            "Buffer Ratio": buffer_ratios,
            "Effective Events": effective_events,
        }
    )

    summary_path = os.path.join(output_dir, "moco_batch_size_summary.csv")
    summary_table.to_csv(summary_path, index=False)
    logger.info(f"Saved batch size summary to {summary_path}")

    # Print summary
    logger.info("\nBatch Size Impact Summary:")
    logger.info("==========================")
    for i, batch_size in enumerate(batch_sizes):
        logger.info(
            f"Batch {batch_size:3d} → Buffer {buffer_sizes[i]:4d} → "
            f"Effective Batch {effective_batch_sizes[i]:4d} → "
            f"Events {effective_events[i]:.1f}"
        )

    return summary_table


@hydra.main(version_base=None, config_path="../conf", config_name="moco.yaml")
def moco_recommend(cfg: DictConfig) -> None:
    """
    run the MOCO recommendation.

    Args:
        cfg: Hydra configuration
    """
    logger.info("Starting MoCo buffer recommendation tool")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")

    # Load dataset using same approach as finetune.py
    df, dataset_info = load_dataset_from_config(cfg)

    # Extract training configuration
    training_config = extract_training_config(cfg)
    logger.info(f"Training configuration: {training_config}")

    # Analyze dataset
    logger.info("Analyzing dataset characteristics...")
    dataset_stats = analyze_dataset_events(
        dataframe=df,
        event_column=dataset_info["event_column"],
        time_column=dataset_info["time_column"],
        multi_event=dataset_info["multi_event"],
    )

    # Print dataset statistics
    logger.info(f"Dataset: {dataset_info['name']}")
    logger.info(f"Total samples: {dataset_stats['total_samples']}")
    logger.info(f"Censoring rate: {dataset_stats['censoring_rate']:.1%}")
    logger.info(f"Events: {dataset_stats['total_events']}")

    # Generate buffer recommendations
    recommendations = print_buffer_recommendations(
        dataset_stats=dataset_stats,
        training_config=training_config,
        output_format="markdown",
    )

    # Print recommendations
    logger.info("\nBuffer Size Recommendations:\n")
    for line in recommendations.split("\n"):
        logger.info(line)

    # Analyze batch size impact with configured batch sizes
    batch_sizes = (
        cfg.analyze_batch_sizes if hasattr(cfg, "analyze_batch_sizes") else None
    )
    min_events = training_config.get("min_events_per_batch", 10)

    analyze_batch_size_impact(
        dataset_stats=dataset_stats,
        output_dir=os.getcwd(),
        batch_sizes=batch_sizes,
        min_events_per_batch=min_events,
    )


if __name__ == "__main__":
    moco_recommend()
