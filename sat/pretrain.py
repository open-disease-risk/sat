"""Preprocess data."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import hydra
import os
import sys

from logdecorator import log_on_start, log_on_end, log_on_error
from logging import DEBUG, ERROR

from omegaconf import DictConfig

from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from sat.data import load
from sat.utils import config, logging, rand

logger = logging.get_default_logger()


@rand.seed
def _pretrain(cfg: DictConfig) -> None:
    logger.debug(f"Loading tokenizer from {cfg.tokenizers.tokenizer_dir}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizers.tokenizer_dir)

    dataset = hydra.utils.call(cfg.data.load)
    dataset = load.split_dataset(cfg.data, dataset)

    tokenizer.enable_padding(
        length=cfg.tokenizers.max_seq_length,
        pad_to_multiple_of=cfg.tokenizers.pad_to_multiple_of,
    )
    tokenizer.enable_truncation(
        max_length=cfg.tokenizers.max_seq_length,
        direction=cfg.tokenizers.truncation_side,
    )

    def tokenize_function(element):
        return tokenizer(
            element[cfg.tokenizers.tokenize_column],
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    logger.debug(f"tokenized dataset: {tokenized_dataset}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=True, mlm_probability=0.15
    )

    logger.debug(tokenized_dataset["test"].column_names)
    tokenized_dataset = tokenized_dataset.remove_columns(
        [cfg.tokenizers.tokenize_column]
    )

    config = hydra.utils.instantiate(cfg.transformers.config)
    model = AutoModelForMaskedLM.from_config(config)
    logger.debug(f"Instantiated Model: {model}")
    args: TrainingArguments = hydra.utils.instantiate(cfg.trainer.training_arguments)
    logger.debug(f"Training Arguments: {args}")

    callbacks = []

    if "callbacks" in cfg:
        if cfg.callbacks:
            for callback in cfg.callbacks:
                cb = hydra.utils.instantiate(callback)
                logger.debug(f"Instantiated callback: {cb}")
                callbacks.append(cb)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model()


@log_on_start(DEBUG, "Start pretraining...")
@log_on_error(
    ERROR,
    "Error during pretraining: {e!r}",
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "done!")
@hydra.main(version_base=None, config_path="../conf", config_name="pretrain.yaml")
def pretrain(cfg: DictConfig) -> None:
    config.Config()
    _pretrain(cfg)


if __name__ == "__main__":
    pretrain()
