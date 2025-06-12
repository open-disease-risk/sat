"""Inference of a transformer model for survival analysis."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from logging import DEBUG, ERROR
from pathlib import Path

import hydra
from datasets import load_dataset
from logdecorator import log_on_end, log_on_error, log_on_start
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerFast
from transformers.pipelines import pipeline

from sat.models.tasks import heads
from sat.models.utils import get_device, load_model
from sat.utils import config, logging, rand
from sat.utils.output import write_interpolation, write_output

logger = logging.get_default_logger()


@rand.seed
def _infer(cfg: DictConfig) -> None:
    device_str, device = get_device()
    logger.info(f"Running models on device: {device_str}")

    mtlConfig = hydra.utils.instantiate(cfg.tasks.config)
    model = heads.MTLForSurvival(mtlConfig)

    model_path = Path(f"{cfg.trainer.training_arguments.output_dir}/pytorch_model.bin")
    logger.info(f"Load model: {model_path}")

    model = load_model(model_path, model)
    model.to(device)
    model.eval()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(f"{cfg.tokenizers.tokenizer_dir}/tokenizer.json")),
        pad_token=cfg.tokenizers.pad_token,
        mask_token=cfg.tokenizers.mask_token,
    )

    logger.debug("Load data")

    if cfg.infer_data:
        dataset = load_dataset(path="json", data_files={"infer": cfg.infer_data})
    else:
        logger.warn(
            "For inference a datafile has to be specified using the 'infer_data' config key"
        )
        return

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

    if cfg.select_id:
        dataset = tokenized_dataset["infer"].filter(
            lambda x: x[cfg.data.id_col] == cfg.select_id
        )
        ids = cfg.select_id
    else:
        dataset = tokenized_dataset["infer"]
        ids = dataset[cfg.data.id_col]

    output = sa_pipe(dataset)

    logger.info(f"Write prediction to {cfg.trainer.training_arguments.output_dir}")
    write_output(
        predictions=output,
        metrics=None,
        cfg=cfg,
        output_dir=cfg.trainer.training_arguments.output_dir,
        ids=ids,
        events=None,
        durations=None,
        model=model,
        prefix="inference_",
    )
    write_interpolation(
        cfg=cfg,
        predictions=output,
        ids=ids,
        output_dir=cfg.trainer.training_arguments.output_dir,
        is_survival=model.is_survival,
    )


@log_on_start(DEBUG, "Start inference...", logger=logger)
@log_on_error(
    ERROR,
    "Error during inference: {e!r}",
    logger=logger,
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "done!", logger=logger)
@hydra.main(version_base=None, config_path="../conf", config_name="infer.yaml")
def infer(cfg: DictConfig) -> None:
    config.Config()
    _infer(cfg)


if __name__ == "__main__":
    infer()
