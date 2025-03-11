"""Fine-tune a transformer model for survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import hydra
import json
import os
import sys

from logdecorator import log_on_start, log_on_end, log_on_error
from logging import DEBUG, ERROR
from omegaconf import DictConfig
from pathlib import Path

from datasets import load_from_disk

from transformers import PreTrainedTokenizerFast
from transformers.pipelines import pipeline

from sat.utils import config, logging, rand
from sat.data import load
from sat.evaluate.evaluator import evaluator
from sat.models.tasks import heads
from sat.models.utils import get_device, load_model, compile_model
from sat.transformers.feature_extractor import SAFeatureExtractor

import sat.transformers.pipelines  # keep this import for pipeline registration to happen

logger = logging.get_default_logger()


@rand.seed
def _eval(cfg: DictConfig) -> None:
    device_str, device = get_device()
    logger.info(f"Running models on device: {device_str}")

    mtlConfig = hydra.utils.instantiate(cfg.tasks.config)
    model = heads.MTLForSurvival(mtlConfig)

    if "finetuned_transformer_dir" in cfg:
        logger.debug(f"Loading finetuned model from {cfg.finetuned_transformer_dir}")
        transformer_dir = cfg.finetuned_transformer_dir
    else:
        logger.debug(f"Loading model from {cfg.trainer.training_arguments.output_dir}")
        transformer_dir = cfg.trainer.training_arguments.output_dir

    model_path = Path(f"{transformer_dir}/pytorch_model.bin")
    logger.info(f"Load model: {model_path}")

    model = load_model(model_path, model)
    model.to(device)

    # Apply torch.compile if enabled in the config
    compile_config = {
        "use_compile": cfg.get("use_compile", False),
        "compile_mode": cfg.get("compile_mode", None),
        "compile_fullgraph": cfg.get("compile_fullgraph", False),
        "compile_backend": cfg.get("compile_backend", None),
        "dynamic_shapes": cfg.get("dynamic_shapes", False),
        "opt_level": cfg.get("opt_level", 2),
        "dynamo_cache": cfg.get("dynamo_cache", {}),
        "debug_options": cfg.get("debug_options", {}),
        "specialized_opts": cfg.get("specialized_opts", {}),
        "selective_compile": cfg.get("selective_compile", {}),
        "m_series_mac_defaults": cfg.get("m_series_mac_defaults", {"enabled": True}),
    }
    model = compile_model(model, config=compile_config)

    model.eval()

    sa_features = SAFeatureExtractor.from_pretrained(cfg.data.label_transform.save_dir)
    # we need to set this path here, because in remote compute the mount path might have changed
    sa_features.label_transform_path = (
        cfg.data.label_transform.save_dir + "/labtrans.pkl"
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(f"{cfg.tokenizers.tokenizer_dir}/tokenizer.json")),
        pad_token=cfg.tokenizers.pad_token,
        mask_token=cfg.tokenizers.mask_token,
    )

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

    logger.debug(
        f"""
        Load pipeline with

        model: {model}
        tokenizer: {tokenizer}
        """
    )

    sa_pipe = pipeline(
        "survival-analysis",
        model=model,
        tokenizer=tokenizer,
        tokenize_column=cfg.tokenizers.tokenize_column,
        max_length=cfg.tokenizers.max_seq_length,
        truncation=cfg.tokenizers.do_truncation,
        padding=cfg.tokenizers.padding_args.padding,
        pad_to_multiple_of=cfg.tokenizers.padding_args.pad_to_multiple_of,
        device=device,
    )

    if "bootstrap_sample_size" in cfg:
        bootstrap_sample_size = cfg.bootstrap_sample_size
    else:
        bootstrap_sample_size = None

    sa_eval = evaluator(
        "survival-analysis",
        num_threads=cfg.num_threads,
        size=bootstrap_sample_size,
    )

    logger.info("Load the metrics...")
    logger.debug(f"{cfg.tasks.metrics}")
    metrics = hydra.utils.instantiate(cfg.tasks.metrics)
    results = {}

    # Set batch processing configuration
    use_batch_pipeline = cfg.get("use_batch_pipeline", False)
    batch_size = cfg.get("batch_size", 32)

    for metric in metrics:
        metric_results = sa_eval.compute(
            model_or_pipeline=sa_pipe,
            data=mapped_labels_dataset["test"],
            metric=metric,
            label_mapping={},
            strategy="bootstrap",
            input_column=cfg.tokenizers.tokenize_column,
            label_column="labels",
            confidence_level=cfg.bootstrap_ci_level,
            n_resamples=cfg.bootstrap_samples,
            random_state=0,
            # Add batch processing parameters
            use_batch_pipeline=use_batch_pipeline,
            batch_size=batch_size,
        )
        results.update(metric_results)

    logger.debug(f"Evaluation results: {results}")
    logger.info(f"Store evaluation results at {transformer_dir}/bootstrap_metrics.json")
    with open(f"{transformer_dir}/bootstrap_metrics.json", "w") as outfile:
        json.dump(results, outfile, indent=4, cls=logging.NpEncoder)

    logger.debug("Serialize random number seed used for eval")
    with Path(f"{transformer_dir}/eval-seed.json").open("w") as f:
        json.dump({"seed": cfg.seed}, f, ensure_ascii=False, indent=4)


@log_on_start(DEBUG, "Start evaluating...", logger=logger)
@log_on_error(
    ERROR,
    "Error during evaluating: {e!r}",
    logger=logger,
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "done!", logger=logger)
@hydra.main(version_base=None, config_path="../conf", config_name="eval.yaml")
def eval(cfg: DictConfig) -> None:
    config.Config()
    _eval(cfg)


if __name__ == "__main__":
    eval()
