"""Learning rate finder script for survival analysis models."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import hydra
import json
import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from logdecorator import log_on_start, log_on_end, log_on_error
from logging import DEBUG, ERROR
from omegaconf import DictConfig, OmegaConf

from datasets import load_from_disk

from tokenizers.processors import TemplateProcessing
from transformers import (
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from sat.utils import config, logging, rand, tokenizing
from sat.utils.lr_finder import find_optimal_lr
from sat.data import load, collator
from sat.models import heads
from sat.models.heads import TokenEmbedding
from sat.models.utils import get_device
from sat.transformers.feature_extractor import SAFeatureExtractor
from sat.transformers import trainer as satrain

logger = logging.get_default_logger()


@rand.seed
def _find_learning_rate(cfg: DictConfig) -> float:
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

    # Data loading logic - same as finetune.py
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
        # Disable multiprocessing to avoid subprocess errors
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
            num_proc=None,  # Disable parallel processing to avoid subprocess errors
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
        # Disable multiprocessing to avoid subprocess errors
        mapped_labels_dataset = tokenized_dataset.map(
            map_label,
            batched=True,
            num_proc=None,  # Disable parallel processing to avoid subprocess errors
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

            # Disable multiprocessing to avoid subprocess errors
            mapped_labels_dataset = mapped_labels_dataset.map(
                tokenizing.numerics_padding_and_truncation,
                fn_kwargs={
                    "max_seq_length": cfg.tokenizers.max_seq_length,
                    "truncation_direction": cfg.tokenizers.truncation_args.direction,
                    "padding_direction": cfg.tokenizers.padding_args.direction,
                    "token_emb": cfg.token_emb,
                },
                num_proc=None,  # Disable parallel processing to avoid subprocess errors
            )

        logger.debug(f"labels mapped in dataset: {mapped_labels_dataset}")

    # Simplified version of compute_metrics since we don't need metrics for LR finding
    def compute_metrics(eval_pred):
        return {}

    # Create a temporary output directory for LR finder
    lr_finder_output_dir = f"{cfg.lr_finder.save_plot_dir}"
    os.makedirs(lr_finder_output_dir, exist_ok=True)

    # Training arguments - simpler version for LR finding
    args: TrainingArguments = hydra.utils.instantiate(cfg.trainer.training_arguments)
    args.seed = cfg.seed
    # Force only a few steps for LR finding
    args.max_steps = cfg.lr_finder.num_steps
    args.logging_steps = max(1, min(5, cfg.lr_finder.num_steps // 10))

    # MPS device support
    if device_str == "mps":
        args.use_mps_device = True

    data_collator = collator.DefaultSATDataCollator(device=device_str)

    # Configure trainer kwargs
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": mapped_labels_dataset["train"],
        "eval_dataset": mapped_labels_dataset["valid"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    # Create trainer - use SATTrainer if specified
    if cfg.get(cfg.trainer.custom, False):
        trainer = satrain.SATTrainer(**trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    # Run the learning rate finder
    logger.info("Starting learning rate finder")
    # Construct the plot save path
    save_plot_path = None
    if cfg.lr_finder.save_plot:
        # Replace variables in the save path
        save_plot_path = os.path.join(
            cfg.lr_finder.save_plot_dir.replace("${dataset}", cfg.dataset).replace(
                "${modelname}", cfg.modelname
            ),
            "lr_finder_plot.png",
        )
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)

    # Run the LR finder
    best_lr = find_optimal_lr(
        trainer=trainer,
        model=model,
        start_lr=cfg.lr_finder.start_lr,
        end_lr=cfg.lr_finder.end_lr,
        num_steps=cfg.lr_finder.num_steps,
        save_plot_path=save_plot_path,
    )

    # Save the result to a file
    logger.info(f"Best learning rate found: {best_lr}")
    result_path = os.path.join(os.path.dirname(save_plot_path), "best_lr.json")
    with open(result_path, "w") as f:
        json.dump({"best_lr": best_lr}, f, indent=2)

    logger.info(f"Results saved to {result_path}")

    return best_lr


@log_on_start(DEBUG, "Starting learning rate finder...", logger=logger)
@log_on_error(
    ERROR,
    "Error during learning rate finding: {e!r}",
    logger=logger,
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "Learning rate finding completed!", logger=logger)
@hydra.main(version_base=None, config_path="../conf", config_name="lr_finder.yaml")
def find_lr(cfg: DictConfig) -> None:
    config.Config()

    best_lr = _find_learning_rate(cfg)

    # Print a clear message to the user about how to use this LR
    print("\n" + "=" * 80)
    print(f"RECOMMENDED LEARNING RATE: {best_lr:.8f}")
    print(
        f"To use this learning rate, set learning_rate={best_lr:.8f} in your training config"
    )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    find_lr()
