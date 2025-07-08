"""Run the k-fold cross validation pipeline for survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import json
from logging import DEBUG, ERROR
from pathlib import Path

import hydra
from logdecorator import log_on_end, log_on_error, log_on_start
from omegaconf import DictConfig

from sat.finetune import _finetune
from sat.utils import config, logging, statistics

logger = logging.get_default_logger()


def _cv(cfg: DictConfig) -> tuple[dict, dict]:
    cfg.cv.k = cfg.cv_kfold
    logger.info(f"Run {cfg.cv.k}-fold cross validation")

    # Skip test predictions during CV to save computation time
    cfg.compute_test_predictions = False

    # Always use validation metrics for model selection (ML best practice)
    metric_names = cfg.cv_ci_metrics.validation

    # Create OnlineStats objects for each configured metric
    cv_metrics = {}
    for metric in metric_names:
        cv_metrics[metric] = statistics.OnlineStats()

    for fold in range(cfg.cv.k):
        cfg.replication = fold
        logger.info("Run training pipeline")
        val_metrics, test_metrics = _finetune(cfg)

        # Always use validation metrics for model selection
        metrics = val_metrics

        for metric, stat in cv_metrics.items():
            if metric in metrics:
                stat.push(metrics[metric])
            else:
                logger.warning(f"Metric {metric} not found in results")

        logger.info(f"Finished run of fold number {fold}")

    cv_results = {"n": cfg.cv.k}
    for metric, stat in cv_metrics.items():
        # Keep full metric name including prefix
        cv_results[metric] = {
            "mean": stat.mean(),
            "variance": stat.variance(),
            "sd": stat.standardDeviation(),
        }

    outDir = Path(f"{cfg.trainer.training_arguments.output_dir}/")
    outDir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Write CV statistics {outDir}")
    with open(f"{outDir}/metrics-pipeline-cv.json", "w") as fp:
        json.dump(cv_results, fp, indent=4)

    # Save additional metrics for replications if needed
    if cfg.run_id != "":
        # Convert metrics to a format suitable for JSON output
        prefixed_metrics = {}
        for k, v in cv_results.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    prefixed_metrics[f"cv_{k}_{subk}"] = subv
            else:
                prefixed_metrics[f"cv_{k}"] = v

        # Save detailed metrics to a separate JSON file
        with open(f"{outDir}/cv_detailed_metrics.json", "w") as fp:
            json.dump(prefixed_metrics, fp, indent=4)

        logger.info(f"Saved detailed CV metrics to {outDir}/cv_detailed_metrics.json")

    return cv_results, test_metrics


@log_on_start(DEBUG, "Start the cv pipeline...", logger=logger)
@log_on_error(
    ERROR,
    "Error during the cv pipeline: {e!r}",
    logger=logger,
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "done!", logger=logger)
@hydra.main(version_base=None, config_path="../conf", config_name="cv.yaml")
def cv(cfg: DictConfig) -> None:
    config.Config()
    _cv(cfg)


if __name__ == "__main__":
    cv()
