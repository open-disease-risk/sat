"""Fine-tune a transformer model for survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import json
from logging import DEBUG, ERROR
from pathlib import Path

import hydra
from datasets import load_from_disk
from logdecorator import log_on_end, log_on_error, log_on_start
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerFast
from transformers.pipelines import pipeline

from sat.data import load
from sat.evaluate.evaluator import evaluator
from sat.models.tasks import heads
from sat.models.utils import get_device, load_model
from sat.transformers.feature_extractor import SAFeatureExtractor
from sat.utils import config, logging, rand

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
