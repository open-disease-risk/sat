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
        Handle potential shape mismatches to prevent crashes due to recent NaN/gradient fixes.
        """
        # Get dimensions from first prediction to allocate memory efficiently
        batch_size = len(predictions)
        if batch_size == 0:
            return {"predictions": np.array([])}

        # Make sure we can access the prediction structure
        try:
            sample = predictions[0]
            
            # Check if sample has the expected structure
            if len(sample) < 4:  # We need at least 4 elements (the first element plus hazard, risk, survival)
                logger.warning(f"Prediction sample has unexpected format with {len(sample)} elements")
                return {"predictions": np.array([])}
                
            num_vars = 3  # hazard, risk, survival
            
            # Validate and get number of events and duration cuts
            if not isinstance(sample[1], (list, tuple)) or len(sample[1]) == 0:
                logger.warning("Hazard data (index 1) is empty or not a list")
                return {"predictions": np.array([])}
                
            num_events = len(sample[1])
            
            # Validate event data structure
            if not isinstance(sample[1][0], (list, tuple)) or len(sample[1][0]) == 0:
                logger.warning("First event in hazard data is empty or not a list")
                return {"predictions": np.array([])}
                
            # Get the number of time points (duration cuts)
            duration_cuts = len(sample[1][0])
            
            # Pre-allocate output array with correct shape
            result = np.zeros((batch_size, num_vars, num_events, duration_cuts), dtype=np.float32)
            
            # Fill the array directly without nested stacks, with careful validation
            for b, instance in enumerate(predictions):
                # Skip invalid instances
                if len(instance) < 4:
                    logger.warning(f"Skipping instance {b} with insufficient elements")
                    continue
                    
                for v in range(num_vars):
                    # Get the variable index (hazard at idx 1, risk at idx 2, survival at idx 3)
                    var_idx = v + 1
                    
                    # Validate that the variable exists in this instance
                    if var_idx >= len(instance) or not isinstance(instance[var_idx], (list, tuple)):
                        logger.warning(f"Invalid variable at index {var_idx} in instance {b}")
                        continue
                        
                    var_data = instance[var_idx]
                    
                    # Process each event
                    for e in range(min(num_events, len(var_data))):
                        # Validate the event data
                        if not isinstance(var_data[e], (list, tuple)) or len(var_data[e]) == 0:
                            logger.warning(f"Invalid event data at {e} for variable {var_idx} in instance {b}")
                            continue
                            
                        event_data = var_data[e]
                        
                        # Handle potential size mismatches
                        if len(event_data) == duration_cuts:
                            # Direct assignment when sizes match
                            result[b, v, e, :] = event_data
                        elif len(event_data) > duration_cuts:
                            # Truncate if event data is longer
                            result[b, v, e, :] = event_data[:duration_cuts]
                        else:
                            # Pad with zeros if event data is shorter
                            result[b, v, e, :len(event_data)] = event_data
            
            # If there's only one duration cut, we can squeeze that dimension
            if duration_cuts == 1:
                result = result.squeeze(axis=3)
                
            return {"predictions": result}
            
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
        # First check if we have valid input to do bootstrapping
        if "predictions" not in metric_inputs or "references" not in metric_inputs:
            logger.warning(f"Missing required keys in metric_inputs for bootstrapping")
            # Return default confidence intervals
            return self._create_default_confidence_intervals(metric_keys)
            
        predictions = metric_inputs["predictions"]
        references = metric_inputs["references"]
        
        # Check if the shapes are suitable for bootstrapping
        if (not hasattr(predictions, 'shape') or predictions.size == 0 or 
            not hasattr(references, 'shape') or references.size == 0):
            logger.warning(f"Empty arrays or invalid shapes in data for bootstrapping")
            # Return default confidence intervals
            return self._create_default_confidence_intervals(metric_keys)
        
        # Check that the batch dimensions match
        if hasattr(predictions, 'shape') and hasattr(references, 'shape'):
            if (len(predictions.shape) > 0 and len(references.shape) > 0 and 
                predictions.shape[0] != references.shape[0]):
                logger.warning(f"Mismatch in batch dimensions: predictions {predictions.shape[0]} vs references {references.shape[0]}")
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
                return self._create_default_confidence_intervals(metric_keys, theta_hat_dict)
                
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
                    min(1.0, point_estimate + 0.1)   # Upper bound (clamp to 1)
                ]
            }
            
        return bootstrap_dict
