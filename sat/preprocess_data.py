"""Fine-tune a transformer model for survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import json
from logging import DEBUG, ERROR
from pathlib import Path

import hydra
from logdecorator import log_on_end, log_on_error, log_on_start
from omegaconf import DictConfig
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from sat.data import load
from sat.models.heads.embeddings import TokenEmbedding
from sat.transformers.feature_extractor import SAFeatureExtractor
from sat.utils import config, logging, rand

logger = logging.get_default_logger()


@rand.seed
def _preprocess_data(cfg: DictConfig) -> None:
    logger.debug(f"Loading tokenizer from {cfg.tokenizers.tokenizer_dir}")
    tokenizer = PreTrainedTokenizerFast(
        pad_token=cfg.tokenizers.pad_token,
        mask_token=cfg.tokenizers.mask_token,
        tokenizer_file=str(Path(f"{cfg.tokenizers.tokenizer_dir}/tokenizer.json")),
    )

    logger.debug(f"Loaded tokenizer: {tokenizer} with length {len(tokenizer)}")
    if cfg.token_emb == TokenEmbedding.BERT.value:
        # for classification, we want to prepend a classification token
        token_template = TemplateProcessing(
            single="$A [CLS]" if cfg.tokenizers.cls_model_type == "GPT" else "[CLS] $A",
            special_tokens=[("[CLS]", tokenizer.convert_tokens_to_ids("[CLS]"))],
        )
        tokenizer._tokenizer.post_processor = token_template

    dataset = hydra.utils.call(cfg.data.load)
    dataset = load.split_dataset(cfg.data, dataset)

    def tokenize_function(
        examples,
        tokenizer=None,
        data_key="code",
        max_length=512,
        padding="max_length",
        truncation=True,
        split_into_words=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    ):
        return tokenizer(
            text=examples[data_key],
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            is_split_into_words=split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )

    sa_features = SAFeatureExtractor.from_pretrained(cfg.data.label_transform.save_dir)
    # we need to set this path here, because in remote compute the mount path might have changed
    sa_features.label_transform_path = (
        cfg.data.label_transform.save_dir + "/labtrans.pkl"
    )

    def map_label(element):
        return sa_features(element)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "data_key": cfg.tokenizers.tokenize_column,
            "max_length": cfg.tokenizers.max_seq_length,
            "truncation": cfg.tokenizers.do_truncation,
            "split_into_words": cfg.tokenizers.is_split_into_words,
            "padding": cfg.tokenizers.padding_args.padding,
            "pad_to_multiple_of": cfg.tokenizers.padding_args.pad_to_multiple_of,
        },
    )
    logger.debug(f"Dataset columns: {tokenized_dataset.column_names}")
    mapped_labels_dataset = tokenized_dataset.map(map_label, batched=True)
    logger.debug(f"labels mapped in dataset: {mapped_labels_dataset}")

    logger.debug(f"Save dataset {mapped_labels_dataset}")
    mapped_labels_dataset.save_to_disk(
        f"{cfg.data.preprocess_outdir}", max_shard_size=cfg.max_shard_size
    )

    logger.debug("Serialize random number seed used for preprocess_data")
    with Path(f"{cfg.data.parse.processed_dir}/preprocess_data-seed.json").open(
        "w"
    ) as f:
        json.dump({"seed": cfg.seed}, f, ensure_ascii=False, indent=4)


@log_on_start(DEBUG, "Start finetuning...", logger=logger)
@log_on_error(
    ERROR,
    "Error during finetuning: {e!r}",
    logger=logger,
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "done!", logger=logger)
@hydra.main(
    version_base=None, config_path="../conf", config_name="preprocess_data.yaml"
)
def preprocess_data(cfg: DictConfig) -> None:
    config.Config()
    _preprocess_data(cfg)


if __name__ == "__main__":
    preprocess_data()
