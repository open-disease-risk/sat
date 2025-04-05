"""Run the pipeline for survival analysis."""

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
from sat.utils import config, logging, rand

logger = logging.get_default_logger()


@rand.reset_seed
def _pipeline(cfg: DictConfig) -> pd.DataFrame:
    logger.info("Run data preparation")
    _prepare_data(cfg)
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


@log_on_start(DEBUG, "Start the pipeline...", logger=logger)
@log_on_error(
    ERROR,
    "Error during the pipeline: {e!r}",
    logger=logger,
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "done!", logger=logger)
@hydra.main(version_base=None, config_path="../conf", config_name="pipeline.yaml")
def pipeline(cfg: DictConfig) -> None:
    """Run the data preparation and traininig pipeline steps."""
    config.Config()
    _pipeline(cfg)


if __name__ == "__main__":
    pipeline()
