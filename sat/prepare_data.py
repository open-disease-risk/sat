"""Prepare data for survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import json
from logging import DEBUG, ERROR
from pathlib import Path

import hydra
from logdecorator import log_on_end, log_on_error, log_on_start
from omegaconf import DictConfig

from sat.utils import config, logging, rand

logger = logging.get_default_logger()


@rand.seed
def _prepare_data(cfg: DictConfig) -> None:
    dataModule = hydra.utils.instantiate(cfg.data.parse)
    dataModule.prepare()

    logger.debug("Serialize random number seed used for prepare_data")
    with Path(f"{cfg.data.parse.processed_dir}/prepare_data-seed.json").open("w") as f:
        json.dump({"seed": cfg.seed}, f, ensure_ascii=False, indent=4)


@log_on_start(DEBUG, "Start preparing the data...")
@log_on_error(
    ERROR,
    "Error during data preparation: {e!r}",
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "done!")
@hydra.main(version_base=None, config_path="../conf", config_name="prepare_data.yaml")
def prepare_data(cfg: DictConfig) -> None:
    config.Config()
    _prepare_data(cfg)


if __name__ == "__main__":
    prepare_data()
