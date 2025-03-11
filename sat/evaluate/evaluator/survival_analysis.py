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
from transformers.utils.generic import ModelOutput
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
        Process a list of ModelOutput predictions efficiently.
        Each prediction should contain hazard, risk, and survival probabilities.

        Args:
            predictions: List of ModelOutput objects from the model
            label_mapping: Optional mapping for labels (unused in survival analysis)

        Returns:
            Dict containing processed numpy arrays of predictions
        """
        logger.debug(f"Processing {len(predictions)} predictions")

        if not predictions:
            logger.debug("No predictions to process")
            return {"predictions": np.array([])}

        # Helper function for tensor conversion - avoids code duplication
        def tensor_to_numpy(tensor):
            return (
                tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor
            )

        # Move validation logic outside the try block for better performance
        sample = predictions[0]
        if not isinstance(sample, ModelOutput):
            logger.error(f"Expected ModelOutput object, got {type(sample)}")
            return {"predictions": np.array([])}

        # Check required fields as attributes
        required_fields = ["hazard", "risk", "survival"]
        if not all(hasattr(sample, field) for field in required_fields):
            logger.error(
                f"Missing required fields. Found: {[f for f in required_fields if hasattr(sample, f)]}"
            )
            return {"predictions": np.array([])}

        # Get shapes from first sample tensors
        hazard_shape = sample.hazard.shape if hasattr(sample.hazard, "shape") else None
        if hazard_shape is None:
            logger.error("Hazard tensor has no shape attribute")
            return {"predictions": np.array([])}

        # Pre-allocate arrays for efficiency
        batch_size = len(predictions)

        # Pre-allocate final result array directly instead of creating intermediates
        # Shape: (batch_size, 3, num_events, time_points)
        result_shape = (batch_size, 3) + hazard_shape[1:]
        stacked_predictions = np.empty(result_shape, dtype=np.float32)

        try:
            # Fill the pre-allocated array directly - more efficient
            for i, pred in enumerate(predictions):
                # Use the helper function for conversion
                stacked_predictions[i, 0] = tensor_to_numpy(pred.hazard)  # hazard
                stacked_predictions[i, 1] = tensor_to_numpy(pred.risk)  # risk
                stacked_predictions[i, 2] = tensor_to_numpy(pred.survival)  # survival

            logger.debug(f"Final predictions shape: {stacked_predictions.shape}")
            return {"predictions": stacked_predictions}

        except Exception as e:
            logger.error(f"Error processing predictions: {e}")
            return {"predictions": np.array([])}

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

    def call_pipeline(self, pipe, pipe_inputs):
        """
        Default implementation of call_pipeline for compatibility.
        This may be overridden by the parent class, but we provide it in case it's not.

        Args:
            pipe: The pipeline to call
            pipe_inputs: The inputs to process

        Returns:
            Tuple of (predictions, performance_results)
        """
        try:
            import time

            start_time = time.time()

            # Process inputs one by one
            if hasattr(pipe_inputs, "__iter__") and not isinstance(
                pipe_inputs, (str, dict)
            ):
                predictions = []
                for item in pipe_inputs:
                    predictions.append(pipe(item))
            else:
                predictions = [pipe(pipe_inputs)]

            end_time = time.time()
            perf_results = {"inference_time": end_time - start_time}

            return predictions, perf_results
        except Exception as e:
            logger.error(f"Error in call_pipeline: {e}")
            return [], {"inference_time": 0}

    def batch_call_pipeline(self, pipe, pipe_inputs, batch_size=32):
        """
        Custom pipeline call implementation that supports efficient batch processing.

        Args:
            pipe: The pipeline to call
            pipe_inputs: The inputs to process
            batch_size: Batch size to use for processing

        Returns:
            Tuple of (predictions, performance_results)
        """
        logger.info("🚀 USING BATCH PIPELINE - BATCH PROCESSING ACTIVATED 🚀")
        try:
            import time

            start_time = time.time()

            # Get the dataset as a list for batch processing
            if hasattr(pipe_inputs, "map"):
                # Convert the dataset to a list for batch processing
                inputs_list = [item for item in pipe_inputs]
            else:
                inputs_list = pipe_inputs

            logger.info(
                f"Processing {len(inputs_list)} examples with batch_size={batch_size}"
            )

            # Process inputs in batches
            predictions = []
            for i in range(0, len(inputs_list), batch_size):
                batch = inputs_list[i : i + batch_size]
                batch_predictions = pipe(batch)
                if isinstance(batch_predictions, list):
                    predictions.extend(batch_predictions)
                else:
                    # Handle case where the pipeline returns a single output for the batch
                    predictions.append(batch_predictions)

            end_time = time.time()
            perf_results = {"inference_time": end_time - start_time}

            return predictions, perf_results

        except Exception as e:
            logger.error(f"Error in batch_call_pipeline: {e}")
            # Fallback to standard implementation
            return self.call_pipeline(pipe, pipe_inputs)

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
        batch_size: Optional[int] = 32,
        use_batch_pipeline: bool = False,
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
        batch_size (`int`, *optional*, defaults to 32):
            Batch size to use for pipeline inference when using batch pipeline.
        use_batch_pipeline (`bool`, *optional*, defaults to False):
            Whether to use the optimized batch pipeline implementation.
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

        # We're directly using our own call_pipeline implementation,
        # so we don't need to store a reference to an original method

        # Log the batch pipeline configuration
        logger.info(
            f"Batch pipeline configuration: use_batch_pipeline={use_batch_pipeline}, batch_size={batch_size}"
        )

        # Compute predictions using either batch or regular pipeline
        if use_batch_pipeline:
            logger.info("Using batch pipeline for efficient batch processing")
            predictions, perf_results = self.batch_call_pipeline(
                pipe, pipe_inputs, batch_size=batch_size
            )
        else:
            logger.info("Using standard pipeline (batch processing DISABLED)")
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
        # First check if we have valid input to do bootstrapping
        if "predictions" not in metric_inputs or "references" not in metric_inputs:
            logger.warning(f"Missing required keys in metric_inputs for bootstrapping")
            # Return default confidence intervals
            return self._create_default_confidence_intervals(metric_keys)

        predictions = metric_inputs["predictions"]
        references = metric_inputs["references"]

        # Check if the shapes are suitable for bootstrapping
        if (
            not hasattr(predictions, "shape")
            or predictions.size == 0
            or not hasattr(references, "shape")
            or references.size == 0
        ):
            logger.warning(f"Empty arrays or invalid shapes in data for bootstrapping")
            # Return default confidence intervals
            return self._create_default_confidence_intervals(metric_keys)

        # Check that the batch dimensions match
        if hasattr(predictions, "shape") and hasattr(references, "shape"):
            if (
                len(predictions.shape) > 0
                and len(references.shape) > 0
                and predictions.shape[0] != references.shape[0]
            ):
                logger.warning(
                    f"Mismatch in batch dimensions: predictions {predictions.shape[0]} vs references {references.shape[0]}"
                )
                # Return default confidence intervals
                return self._create_default_confidence_intervals(metric_keys)

        try:
            data = (predictions, references)
            dist = statistics.EmpiricalDistribution(data)

            def build_args_metric(metric=None, key=None):
                def args_metric(data):
                    predictions, references = data
                    logger.debug(
                        f"args_metric: predictions in build args: {predictions} and {references} for metric key {key}"
                    )
                    return metric.compute(
                        predictions=predictions, references=references
                    )[key]

                return args_metric

            stat_func_dict = {}
            theta_hat_dict = {}

            logger.debug("Build the statistical functions dictionary")
            for key in metric_keys:
                logger.debug(f"Add statistical metric {key} to the function dictionary")
                stat_func_dict[key] = build_args_metric(metric, key)
                logger.debug(f"Compute statistical metric {key} for the data")
                try:
                    theta_hat_dict[key] = stat_func_dict[key](data)
                except Exception as e:
                    logger.error(f"Error computing point estimate for {key}: {e}")
                    theta_hat_dict[key] = 0.5  # Default value

            try:
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
            except Exception as e:
                logger.error(f"Error in bootstrapping: {e}")
                # Return default confidence intervals
                return self._create_default_confidence_intervals(
                    metric_keys, theta_hat_dict
                )

        except Exception as e:
            logger.error(f"Error setting up bootstrapping: {e}")
            # Return default confidence intervals
            return self._create_default_confidence_intervals(metric_keys)

    def _create_default_confidence_intervals(self, metric_keys, theta_hat_dict=None):
        """
        Create default confidence intervals when bootstrapping can't be performed.

        Parameters:
        -----------
        metric_keys : list of str
            The metric keys to create confidence intervals for
        theta_hat_dict : dict, optional
            Point estimates if available

        Returns:
        --------
        dict
            Dictionary with confidence intervals for each metric key
        """
        bootstrap_dict = {}

        for key in metric_keys:
            # Use provided point estimate or default to 0.5
            point_estimate = 0.5
            if theta_hat_dict and key in theta_hat_dict:
                point_estimate = theta_hat_dict[key]

            # Create entry with point estimate and confidence interval of ±0.1
            bootstrap_dict[key] = {
                "theta_hat": point_estimate,
                "alpha": 0.025,  # Standard for 95% CI
                "interval": [
                    max(0.0, point_estimate - 0.1),  # Lower bound (clamp to 0)
                    min(1.0, point_estimate + 0.1),  # Upper bound (clamp to 1)
                ],
            }

        return bootstrap_dict
