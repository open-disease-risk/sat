"""Train a tokenizer"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import json
from logging import DEBUG, ERROR
from pathlib import Path

import hydra
from logdecorator import log_on_end, log_on_error, log_on_start
from omegaconf import DictConfig

from sat.data import splitter
from sat.utils import config, logging, rand

logger = logging.get_default_logger()


@rand.seed
def _train_tokenizer(cfg: DictConfig) -> None:
    ds_splitter = splitter.StreamingKFoldSplitter(
        id_field=cfg.data.id_col,
        k=None,
        val_ratio=cfg.data.validation_ratio,
        test_ratio=cfg.data.test_ratio,
        test_split_strategy="hash",
        split_names=cfg.data.splits,
    )
    dataset = ds_splitter.load_split(cfg=cfg.data.load)
    logger.info(f"Instantiate tokenizer {cfg.tokenizers.tokenizer}")
    tokenizer = hydra.utils.instantiate(cfg.tokenizers.tokenizer)

    if "pre_tokenizer" in cfg.tokenizers:
        logger.info(f"Instantiate pre-tokenizer: {cfg.tokenizers.pre_tokenizer}")
        tokenizer.pre_tokenizer = hydra.utils.instantiate(cfg.tokenizers.pre_tokenizer)
    if "post_processor" in cfg.tokenizers:
        logger.info(f"Instantiate post-processor: {cfg.tokenizers.post_processor}")
        tokenizer.post_processor = hydra.utils.instantiate(
            cfg.tokenizers.post_processor
        )
    if "normalizer" in cfg.tokenizers:
        logger.info(f"Instantiate normalizer: {cfg.tokenizers.normalizer}")
        tokenizer.normalizer = hydra.utils.instantiate(cfg.tokenizers.normalizer)

    trainer = hydra.utils.instantiate(cfg.tokenizers.trainer)

    def batch_iterator(dataset, column):
        logger.debug(f"Batch interate over {dataset}")
        for example in dataset.values():
            yield example[column]

    logger.info("Train the tokenizer...")
    tokenizer.train_from_iterator(
        batch_iterator(dataset, column=cfg.tokenizers.tokenize_column),
        trainer=trainer,
    )

    if "special_tokens" in cfg.tokenizers:
        logger.info(f"Set special tokens {cfg.tokenizers.special_tokens}")
        tokenizer.add_special_tokens(list(cfg.tokenizers.special_tokens.values()))
        trainer.special_tokens = list(cfg.tokenizers.special_tokens.values())

    if cfg.tokenizers.do_padding:
        logger.debug(f"Enable padding with {cfg.tokenizers.padding_args}")
        tokenizer.enable_padding(**cfg.tokenizers.padding_args)

    if cfg.tokenizers.do_truncation:
        logger.debug(f"Enable truncation with {cfg.tokenizers.truncation_args}")
        tokenizer.enable_truncation(**cfg.tokenizers.truncation_args)

    logger.info(f"Size of the vocabulary: {len(tokenizer.get_vocab().keys())}")
    logger.debug(f"{cfg.tokenizers.tokenizer_dir}")
    destination_path = Path(cfg.tokenizers.tokenizer_dir)
    destination_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Save tokenizer into {destination_path}")
    # fast_tokenizer.save_pretrained(destination_path)
    tokenizer.save(str(Path(f"{destination_path}/tokenizer.json")))

    logger.debug("Serialize random number seed used for train_tokenizer")
    with Path(f"{destination_path}/train_tokenizer-seed.json").open("w") as f:
        json.dump({"seed": cfg.seed}, f, ensure_ascii=False, indent=4)


@log_on_start(DEBUG, "Start training a tokenizer...")
@log_on_error(
    ERROR,
    "Error during training of a tokenizer: {e!r}",
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "done!")
@hydra.main(
    version_base=None, config_path="../conf", config_name="train_tokenizer.yaml"
)
def train_tokenizer(cfg: DictConfig) -> None:
    config.Config()
    _train_tokenizer(cfg)


if __name__ == "__main__":
    train_tokenizer()
