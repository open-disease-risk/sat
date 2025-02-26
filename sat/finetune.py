"""Fine-tune a transformer model for survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import hydra
import json
import mlflow
import os
import torch

import pandas as pd

from logdecorator import log_on_start, log_on_end, log_on_error
from logging import DEBUG, ERROR
from omegaconf import DictConfig
from pathlib import Path

from datasets import load_from_disk

from tokenizers.processors import TemplateProcessing
from transformers import (
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from sat.utils import config, logging, rand, tokenizing
from sat.utils.output import write_output, log_metrics
from sat.data import load, collator
from sat.models.tasks import heads
from sat.models.tasks.config import TokenEmbedding
from sat.models.utils import get_device
from sat.transformers.feature_extractor import SAFeatureExtractor

logger = logging.get_default_logger()


@rand.seed
def _finetune(cfg: DictConfig) -> pd.DataFrame:
    device_str, device = get_device()
    logger.info(f"Running models on device: {device_str}")

    if "detect_anomalies" in cfg:
        logger.info(f"Set detect anomalies to {cfg.detect_anomalies}")
        torch.autograd.set_detect_anomaly(cfg.detect_anomalies)

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

    mtlConfig = hydra.utils.instantiate(cfg.tasks.config)
    model = heads.MTLForSurvival(mtlConfig)
    model.to(device)

    model.train()
    logging.log_gpu_utilization()

    if cfg.data.preprocess_data:
        mapped_labels_dataset = load_from_disk(cfg.data.preprocess_outdir)
    else:
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
                return_special_tokens_mask=True,
            )

        sa_features = SAFeatureExtractor.from_pretrained(
            cfg.data.label_transform.save_dir
        )
        # we need to set this path here, because in remote compute the mount path might have changed
        sa_features.label_transform_path = (
            cfg.data.label_transform.save_dir + "/labtrans.pkl"
        )

        def map_label(element):
            return sa_features(element)

        # Create cache directory for tokenized data if it doesn't exist
        cache_dir = (
            f"{cfg.data.cache_dir}/tokenized_dataset"
            if hasattr(cfg.data, "cache_dir")
            else None
        )
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Use caching for tokenized datasets to avoid redundant processing
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
            num_proc=4,  # Parallel processing for tokenization
        )

        logger.debug(f"Dataset columns: {tokenized_dataset.column_names}")
        # Create cache for label mapping
        labels_cache_dir = (
            f"{cfg.data.cache_dir}/labels_dataset"
            if hasattr(cfg.data, "cache_dir")
            else None
        )
        if labels_cache_dir:
            os.makedirs(labels_cache_dir, exist_ok=True)

        # Apply label mapping with caching
        mapped_labels_dataset = tokenized_dataset.map(
            map_label,
            batched=True,
            num_proc=4,  # Parallel processing for label mapping
        )

        # Process numerics if present
        if "numerics" in mapped_labels_dataset.column_names[cfg.data.splits[0]]:
            logger.debug("Numerics present, so processing padding/truncation")

            # Cache for numerics processing
            numerics_cache_dir = (
                f"{cfg.data.cache_dir}/numerics_processed"
                if hasattr(cfg.data, "cache_dir")
                else None
            )
            if numerics_cache_dir:
                os.makedirs(numerics_cache_dir, exist_ok=True)

            mapped_labels_dataset = mapped_labels_dataset.map(
                tokenizing.numerics_padding_and_truncation,
                fn_kwargs={
                    "max_seq_length": cfg.tokenizers.max_seq_length,
                    "truncation_direction": cfg.tokenizers.truncation_args.direction,
                    "padding_direction": cfg.tokenizers.padding_args.direction,
                    "token_emb": cfg.token_emb,
                },
                num_proc=4,  # Parallel processing for numerics padding/truncation
            )

        logger.debug(f"labels mapped in dataset: {mapped_labels_dataset}")

    metrics = {}
    if "metrics" in cfg.tasks:
        logger.info("Load the metrics...")
        logger.debug(f"{cfg.tasks.metrics}")
        metrics = hydra.utils.instantiate(cfg.tasks.metrics)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        preds_dict = {}
        idx = 1
        if model.is_survival:
            preds_dict["hazard"] = preds[1]
            preds_dict["risk"] = preds[2]
            preds_dict["survival"] = preds[3]
            idx = 4
        if model.is_regression:
            preds_dict["time_to_event"] = preds[idx]
            idx += 1
        if model.is_classification:
            preds_dict["event"] = preds[idx]

        metrics_dict = {}

        for metric in metrics:
            metric_results = metric.compute(preds_dict, labels)
            metrics_dict.update(metric_results)

        return metrics_dict

    callbacks = []
    if "callbacks" in cfg:
        callbacks = hydra.utils.instantiate(cfg.callbacks)

    args: TrainingArguments = hydra.utils.instantiate(cfg.trainer.training_arguments)
    logger.debug(f"Set random seed {args.seed} for HF training")
    args.seed = cfg.seed

    if device_str == "mps":
        logger.debug("Use MPS in training argumments")
        args.use_mps_device = True

    if cfg.do_sweep:
        args.output_dir = args.output_dir + "/" + cfg.model_dir
        logger.debug(f"Redirect the sweep output: {args.output_dir}")
    else:
        logger.debug(
            f"Append run ID {cfg.run_id} to output of training arguments {args.output_dir}"
        )
        args.output_dir = args.output_dir + "/" + cfg.run_id

    data_collator = collator.DefaultSATDataCollator(device=device_str)

    # Using the standard setup without custom accelerator since we downgraded accelerate to be compatible

    # Configure trainer kwargs
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": mapped_labels_dataset["train"],
        "eval_dataset": mapped_labels_dataset["valid"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
    }

    # Create trainer
    trainer = Trainer(**trainer_kwargs)

    logger.info("Start training")
    result = trainer.train()
    logging.log_summary(result)
    logger.info("Save model")
    trainer.save_model()

    logger.debug("Do predictions on validation set")
    valid_output = trainer.predict(test_dataset=mapped_labels_dataset["valid"])
    logger.debug("Do predictions on test set")
    output = trainer.predict(test_dataset=mapped_labels_dataset["test"])
    ids = mapped_labels_dataset["test"][cfg.data.id_col]
    events = mapped_labels_dataset["test"][cfg.data.event_col]
    durations = mapped_labels_dataset["test"][cfg.data.duration_col]

    write_output(
        output.predictions,
        output.metrics,
        cfg,
        trainer.args.output_dir,
        ids,
        events,
        durations,
        model,
    )

    logger.debug("Serialize random number seed used for finetuning")
    with Path(f"{trainer.args.output_dir}/finetune-seed.json").open("w") as f:
        json.dump({"seed": cfg.seed}, f, ensure_ascii=False, indent=4)

    if cfg.run_id != "" and not "multiple_replications" in cfg:
        with mlflow.start_run() as run:
            val_metrics = {
                k.replace("test_", "validation_"): v
                for k, v in valid_output.metrics.items()
            }
            log_metrics(metrics=val_metrics)
            log_metrics(metrics=output.metrics)

    return valid_output.metrics, output.metrics


@log_on_start(DEBUG, "Start finetuning...", logger=logger)
@log_on_error(
    ERROR,
    "Error during finetuning: {e!r}",
    logger=logger,
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "done!", logger=logger)
@hydra.main(version_base=None, config_path="../conf", config_name="finetune.yaml")
def finetune(cfg: DictConfig) -> None:
    config.Config()
    _finetune(cfg)


if __name__ == "__main__":
    finetune()
