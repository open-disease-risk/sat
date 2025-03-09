"""Train the `LabelTransform` to turn continuous time to event into discrete
intervals.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import hydra
import json
import pickle

import numpy as np
import pandas as pd

from logdecorator import log_on_start, log_on_end, log_on_error
from logging import DEBUG, ERROR
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


from sat.data import load
from sat.pycox.preprocessing.label_transforms import LabTransPCHazard
from sat.utils import config, logging, rand
from sat.transformers.feature_extractor import SAFeatureExtractor

logger = logging.get_default_logger()


@rand.seed
def _train_labeltransform(cfg: DictConfig) -> None:
    dataset = hydra.utils.call(cfg.data.load)
    dataset = load.split_dataset(cfg.data, dataset)

    if cfg.data.load.streaming:
        shuffled_dataset = dataset["train"].shuffle(
            buffer_size=cfg.data.label_transform.buffer_size
        )
    else:
        shuffled_dataset = dataset["train"].shuffle()

    train_data = next(
        shuffled_dataset.iter(batch_size=cfg.data.label_transform.buffer_size)
    )

    events = np.array(train_data[cfg.data.event_col]).ravel()
    durations = np.array(train_data[cfg.data.duration_col]).ravel()
    events_observed = events > 0
    if cfg.data.label_transform.event_type:
        events_observed = events == cfg.data.label_transform.event_type

    max_duration = np.max(train_data[cfg.data.duration_col])
    logger.debug(f"Max value of {cfg.data.duration_col}: {max_duration}")

    horizons = np.linspace(0, 1.0, cfg.data.label_transform.cuts + 1)[1:-1]
    logger.debug(f"Compute duration quantiles for {horizons}")
    times = np.quantile(durations[events_observed], horizons).tolist()
    cuts = np.array([0.0] + times + [max_duration])
    labtrans = LabTransPCHazard(cuts=cuts)

    logger.info("Fit the piece-wise constant hazard label transformer")
    labtrans.fit(durations, events)

    duration_cuts = pd.DataFrame({"cuts": labtrans.cuts})
    out_dir = Path(f"{cfg.data.label_transform.save_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Save the duration cuts: {labtrans.cuts} to {out_dir}/duration_cuts.csv"
    )
    duration_cuts.to_csv(
        Path(f"{out_dir}/duration_cuts.csv"),
        index=False,
        header=False,
    )

    dump_file = Path(f"{out_dir}/labtrans.pkl")

    with open(dump_file, "wb") as handle:
        logger.info(f"Write {dump_file}")
        pickle.dump(labtrans, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Computation of importance sampling weights
    logger.debug("Compute importance sampling weights")
    events = np.array(shuffled_dataset[cfg.data.event_col])
    events = events[:, np.newaxis] if events.ndim == 1 else events
    censored = (~events.any(1)).reshape(-1, 1)
    events = np.hstack((censored, events))
    imp_weights = events.sum(axis=0) / len(events)
    imp_weights = pd.DataFrame({"imp_samp": imp_weights})

    imp_weights.to_csv(
        Path(f"{out_dir}/imp_sample.csv"),
        index=False,
        header=False,
    )

    feature_extractor = SAFeatureExtractor(
        label_transform_path=str(dump_file),
        duration_col=cfg.data.duration_col,
        event_col=cfg.data.event_col,
        transformed_duration_cols=OmegaConf.to_object(
            cfg.data.transformed_duration_cols
        ),
    )
    logger.info(f"Save feature extractor to {dump_file}")
    feature_extractor.save_pretrained(out_dir)

    transformed_labels = Path(f"{out_dir}/transformed_train_labels.csv")
    logger.info(
        f"Transform training data and save for evaluation at {transformed_labels}"
    )

    transformed_data = feature_extractor(train_data)
    labels = pd.DataFrame(transformed_data["labels"])
    labels.to_csv(transformed_labels, index=False, header=False)

    logger.debug("Serialize random number seed used for prepare_data")
    with Path(f"{out_dir}/train_labeltransform-seed.json").open("w") as f:
        json.dump({"seed": cfg.seed}, f, ensure_ascii=False, indent=4)


@log_on_start(DEBUG, "Start training the transformer...")
@log_on_error(
    ERROR,
    "Error during training: {e!r}",
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "done!")
@hydra.main(
    version_base=None, config_path="../conf", config_name="train_labeltransform.yaml"
)
def train_labeltransform(cfg: DictConfig) -> None:
    config.Config()
    _train_labeltransform(cfg)


if __name__ == "__main__":
    train_labeltransform()
