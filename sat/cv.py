"""Run the k-fold cross validation pipeline for survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import json
from logging import DEBUG, ERROR
from pathlib import Path

import hydra
import pandas as pd
from logdecorator import log_on_end, log_on_error, log_on_start
from omegaconf import DictConfig

from sat.finetune import _finetune
from sat.prepare_data import _prepare_data
from sat.train_labeltransform import _train_labeltransform
from sat.train_tokenizer import _train_tokenizer
from sat.utils import config, logging, rand, statistics

logger = logging.get_default_logger()


@rand.reset_seed
def _pipeline(cfg: DictConfig) -> pd.DataFrame:
    if cfg.cv_kfold_reuse:
        logger.info("Reuse existing k-fold cross validation data")
    else:
        logger.info("Train label transformer")
        _train_labeltransform(cfg)
        logger.info("Train tokenizer")
        _train_tokenizer(cfg)
        logger.info("Run fine-tuning")

    val_metrics, test_metrics = _finetune(cfg)

    logger.debug("Serialize random number seed used for pipeline")
    with Path(f"{cfg.trainer.training_arguments.output_dir}/pipeline-seed.json").open(
        "w"
    ) as f:
        json.dump({"seed": cfg.seed}, f, ensure_ascii=False, indent=4)

    return val_metrics, test_metrics


def _cv(cfg: DictConfig) -> None:
    logger.info(f"Run {cfg.cv.kfold}-fold cross validation")

    if cfg.cv_kfold_reuse:
        logger.info("Reuse existing k-fold cross validation data")
    else:
        logger.info("Run data preparation")
        _prepare_data(cfg)

    dataset = cfg.dataset
    brier = statistics.OnlineStats()
    ipcw = statistics.OnlineStats()

    for i in range(cfg.cv.kfold):
        cfg.fold = str(i) + "_"
        logger.info("Run training pipeline")
        metrics, test_metrics = _pipeline(cfg)

        brier.push(metrics["test_brier_avg"])
        ipcw.push(metrics["test_ipcw_avg"])

        logger.info(f"Finished replication run number {i}")

    cv_results = {
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
