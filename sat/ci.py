"""Run the pipeline for survival analysis."""

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
from sat.utils.output import log_metrics_from_replications

logger = logging.get_default_logger()


def _ci(cfg: DictConfig) -> None:
    brier = statistics.OnlineStats()
    ipcw = statistics.OnlineStats()

    dataset = cfg.dataset
    while (
        (brier.getNumValues() < cfg.n)
        or (not statistics.isConfidentWithPrecision(brier, cfg.alpha, cfg.error))
        or (not statistics.isConfidentWithPrecision(ipcw, cfg.alpha, cfg.error))
    ):
        if brier.getNumValues() >= cfg.less_than_n:
            logger.info(f"Stop the runs after {brier.getNumValues()} runs")
            break

        cfg.dataset = dataset + "_" + str(brier.getNumValues())
        val_metrics, test_metrics = _finetune(cfg)

        metrics = val_metrics if cfg.use_val else test_metrics

        brier.push(metrics["test_brier_weighted_avg"])
        ipcw.push(metrics["test_ipcw_weighted_avg"])

        logger.info(f"Finished replication run number {brier.getNumValues()}")

    ci_results = {
        "n": brier.getNumValues(),
        "brier": {
            "mean": brier.mean(),
            "variance": brier.variance(),
            "sd": brier.standardDeviation(),
        },
        "ipcw": {
            "mean": ipcw.mean(),
            "variance": ipcw.variance(),
            "sd": ipcw.standardDeviation(),
        },
    }

    cfg.dataset = dataset
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
