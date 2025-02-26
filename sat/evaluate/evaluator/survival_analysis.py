"""Evaluator for Survival Analysis Tasks."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import numpy as np

from datasets import Dataset, load_dataset

from evaluate.evaluator.base import (
    EVALUATOR_COMPUTE_RETURN_DOCSTRING,
    EVALUTOR_COMPUTE_START_DOCSTRING,
    Evaluator,
)
from evaluate.evaluator.utils import DatasetColumn
from evaluate.module import EvaluationModule
from evaluate.utils.file_utils import add_end_docstrings, add_start_docstrings

from numbers import Number

from typing import Any, Callable, Dict, Optional, Union, List
from typing_extensions import Literal

from sat.utils import logging, statistics

logger = logging.get_default_logger()


TASK_DOCUMENTATION = r"""
    Examples:
    ```python
    >>> from evaluate import evaluator
    >>> from datasets import load_dataset
    >>> task_evaluator = evaluator("survival_analysis")
    >>> data = load_dataset("imdb", split="test[:2]")
    >>> results = task_evaluator.compute(
    >>>     model_or_pipeline="huggingface/prunebert-base-uncased-6-finepruned-w-distil-mnli",
    >>>     data=data,
    >>>     metric="accuracy",
    >>>     label_mapping={"LABEL_0": 0.0, "LABEL_1": 1.0},
    >>>     strategy="bootstrap",
    >>>     n_resamples=10,
    >>>     random_state=0
    >>> )
    ```
"""


class SurvivalAnalysisEvaluator(Evaluator):
    """
    Text classification evaluator.
    This text classification evaluator can currently be loaded from [`evaluator`] using the default task name
    `text-classification` or with a `"sentiment-analysis"` alias.
    Methods in this class assume a data format compatible with the [`TextClassificationPipeline`] - a single textual
    feature as input and a categorical label as output.
    """

    PIPELINE_KWARGS = {"truncation": True}

    def __init__(
        self,
        task="suvival-analysis",
        default_metric_name=None,
        num_threads=-1,
        size=None,
    ):
        super().__init__(task, default_metric_name=default_metric_name)
        self.num_threads = num_threads
        self.size = size

    def predictions_processor(self, predictions, label_mapping):
        """
        Process predictions more efficiently by avoiding multiple nested stack operations.
        """
        # Get dimensions from first prediction to allocate memory efficiently
        batch_size = len(predictions)
        if batch_size == 0:
            return {"predictions": np.array([])}

        sample = predictions[0]
        num_vars = 3  # hazard, risk, survival
        num_events = len(sample[1])
        duration_cuts = len(sample[1][0])

        # Pre-allocate output array with correct shape
        result = np.empty(
            (batch_size, num_vars, num_events, duration_cuts), dtype=np.float32
        )

        # Fill the array directly without nested stacks
        for b, instance in enumerate(predictions):
            for v in range(num_vars):
                # Copy data for each variable (hazard at idx 1, risk at idx 2, survival at idx 3)
                for e in range(num_events):
                    result[b, v, e, :] = instance[v + 1][e]

        # Squeeze unnecessary dimension
        result = result.squeeze(axis=3)

        return {"predictions": result}

    def prepare_data(
        self,
        data: Union[str, Dataset],
        input_column: str,
        label_column: str,
    ):
        if data is None:
            raise ValueError(
                "Please specify a valid `data` object - either a `str` with a name or a `Dataset` object."
            )

        self.check_required_columns(
            data, {"input_column": input_column, "label_column": label_column}
        )

        data = load_dataset(data) if isinstance(data, str) else data

        return {"references": np.array(data[label_column])}, DatasetColumn(
            data, input_column
        )

    @add_start_docstrings(EVALUTOR_COMPUTE_START_DOCSTRING)
    @add_end_docstrings(EVALUATOR_COMPUTE_RETURN_DOCSTRING, TASK_DOCUMENTATION)
    def compute(
        self,
        model_or_pipeline: Union[
            str,
            "Pipeline",
            Callable,
            "PreTrainedModel",
            "TFPreTrainedModel",  # noqa: F821
        ] = None,
        data: Union[str, Dataset] = None,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        metric: Union[str, EvaluationModule] = None,
        tokenizer: Optional[Union[str, "PreTrainedTokenizer"]] = None,  # noqa: F821
        feature_extractor: Optional[
            Union[str, "FeatureExtractionMixin"]
        ] = None,  # noqa: F821
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        device: int = None,
        random_state: Optional[int] = None,
        input_column: str = "text",
        label_column: str = "label",
        label_mapping: Optional[Dict[str, Number]] = None,
    ) -> Dict[str, float]:
        """
        input_column (`str`, *optional*, defaults to `"text"`):
            the name of the column containing the text feature in the dataset specified by `data`.
        second_input_column (`str`, *optional*, defaults to `None`):
            the name of the second column containing the text features. This may be useful for classification tasks
            as MNLI, where two columns are used.
        label_column (`str`, defaults to `"label"`):
            the name of the column containing the labels in the dataset specified by `data`.
        label_mapping (`Dict[str, Number]`, *optional*, defaults to `None`):
            We want to map class labels defined by the model in the pipeline to values consistent with those
            defined in the `label_column` of the `data` dataset.
        """

        result = {}

        data = self.load_data(data=data, subset=subset, split=split)

        metric_labels_dict, pipe_inputs = self.prepare_data(
            data=data,
            input_column=input_column,
            label_column=label_column,
        )

        pipe = self.prepare_pipeline(
            model_or_pipeline=model_or_pipeline,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=device,
        )
        metric = self.prepare_metric(metric)

        # Compute predictions
        predictions, perf_results = self.call_pipeline(pipe, pipe_inputs)
        metric_predictions_dict = self.predictions_processor(predictions, label_mapping)

        metric_inputs = {
            **metric_labels_dict,
            **metric_predictions_dict,
        }

        logger.debug(f"metric_inputs {metric_inputs}")

        # Compute metrics from references and predictions
        metric_results = self.compute_metric(
            metric=metric,
            metric_inputs=metric_inputs,
            strategy=strategy,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
        )

        result.update(metric_results)
        result.update(perf_results)

        return result

    def _compute_confidence_interval(
        self,
        metric,
        metric_inputs,
        metric_keys: List[str],
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        data = (metric_inputs["predictions"], metric_inputs["references"])
        dist = statistics.EmpiricalDistribution(data)

        def build_args_metric(metric=None, key=None):
            def args_metric(data):
                predictions, references = data
                logger.debug(
                    f"args_metric: predictions in build args: {predictions} and {references} for metric key {key}"
                )
                return metric.compute(predictions=predictions, references=references)[
                    key
                ]

            return args_metric

        stat_func_dict = {}
        theta_hat_dict = {}

        logger.debug("Build the statistical functions dictionary")
        for key in metric_keys:
            logger.debug(f"Add statistical metric {key} to the function dictionary")
            stat_func_dict[key] = build_args_metric(metric, key)
            logger.debug(f"Compute statistical metric {key} for the data")
            theta_hat_dict[key] = stat_func_dict[key](data)

        bootstrap_dict = statistics.boot_interval(
            dist,
            stat_func_dict,
            data,
            alpha=(1.0 - confidence_level) / 2.0,
            B=n_resamples,
            size=self.size,
            num_threads=self.num_threads,
            theta_hat=theta_hat_dict,
        )

        return bootstrap_dict
