"""Run the pipeline for survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import json
from logging import DEBUG, ERROR
from pathlib import Path
from typing import Any

import hydra
from logdecorator import log_on_end, log_on_error, log_on_start
from omegaconf import DictConfig

from sat.finetune import _finetune
from sat.utils import config, logging, statistics
from sat.utils.output import log_metrics_from_replications

logger = logging.get_default_logger()


def _ci(cfg: DictConfig) -> tuple[dict, dict]:
    # Skip test predictions during CI to save computation time
    cfg.compute_test_predictions = False

    # Always use validation metrics for model selection (ML best practice)
    metric_names = cfg.cv_ci_metrics.validation

    # Create OnlineStats objects for each configured metric
    online_stats = {metric: statistics.OnlineStats() for metric in metric_names}

    # Initialize test_metrics
    test_metrics = {}

    # Helper function to check if all metrics are confident
    def all_metrics_confident() -> bool:
        return all(
            statistics.isConfidentWithPrecision(stats, cfg.alpha, cfg.error)
            for stats in online_stats.values()
        )

    # Get the number of values (same for all metrics)
    def get_num_values() -> int:
        return next(iter(online_stats.values())).getNumValues()

    while (get_num_values() < cfg.n) or (not all_metrics_confident()):
        if get_num_values() >= cfg.less_than_n:
            logger.info(f"Stop the runs after {get_num_values()} runs")
            break

        cfg.replication = get_num_values()
        val_metrics, test_metrics = _finetune(cfg)

        # Always use validation metrics for model selection
        metrics = val_metrics

        # Push metrics to their respective OnlineStats objects
        for metric_name, stats in online_stats.items():
            if metric_name in metrics:
                stats.push(metrics[metric_name])
            else:
                logger.warning(f"Metric {metric_name} not found in results")

        logger.info(f"Finished replication run number {get_num_values()}")

    # Build results dictionary
    ci_results: dict[str, Any] = {
        "n": get_num_values(),
    }

    # Add statistics for each metric
    for metric_name, stats in online_stats.items():
        # Keep full metric name including prefix
        ci_results[metric_name] = {
            "mean": stats.mean(),
            "variance": stats.variance(),
            "sd": stats.standardDeviation(),
        }

    outDir = Path(f"{cfg.trainer.training_arguments.output_dir}/")
    outDir.mkdir(parents=True, exist_ok=True)

    with open(f"{outDir}/metrics-pipeline-ci.json", "w") as fp:
        json.dump(ci_results, fp, indent=4)

    # Save detailed CI metrics if needed
    if cfg.run_id != "":
        # Convert metrics to a prefixed format
        prefixed_metrics = log_metrics_from_replications(ci_results, "ci")

        # Save detailed metrics to a separate JSON file
        with open(f"{outDir}/ci_detailed_metrics.json", "w") as fp:
            json.dump(prefixed_metrics, fp, indent=4)

        logger.info(f"Saved detailed CI metrics to {outDir}/ci_detailed_metrics.json")

    return ci_results, test_metrics


@log_on_start(DEBUG, "Start the ci pipeline...", logger=logger)
@log_on_error(
    ERROR,
    "Error during the ci pipeline: {e!r}",
    logger=logger,
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "done!", logger=logger)
@hydra.main(version_base=None, config_path="../conf", config_name="ci.yaml")
def ci(cfg: DictConfig) -> None:
    config.Config()
    _ci(cfg)


if __name__ == "__main__":
    ci()
