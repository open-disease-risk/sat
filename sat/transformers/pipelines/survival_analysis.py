"""Pipeline for Survival Analysis Tasks"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from logging import DEBUG, ERROR
from typing import Dict

from logdecorator import log_on_end, log_on_error, log_on_start
from transformers.pipelines.base import PIPELINE_INIT_ARGS, GenericTensor, Pipeline
from transformers.utils import add_end_docstrings

from sat.utils import logging

logger = logging.get_default_logger()


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        return_all_scores (`bool`, *optional*, defaults to `False`):
            Whether to return all prediction scores or just the one of the predicted class.
        function_to_apply (`str`, *optional*, defaults to `"default"`):
            The function to apply to the model outputs in order to retrieve the scores. Accepts four different values:

            - `"default"`: if the model has a single label, will apply the sigmoid function on the output. If the model
              has several labels, will apply the softmax function on the output.
            - `"sigmoid"`: Applies the sigmoid function on the output.
            - `"softmax"`: Applies the softmax function on the output.
            - `"none"`: Does not apply any function on the output.
    """,
)
class SAPipeline(Pipeline):
    """ """

    @log_on_start(DEBUG, "Initialize the SAPipeline with {kwargs}", logger=logger)
    @log_on_error(
        ERROR,
        "Error during initialization: {e!r}",
        logger=logger,
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!", logger=logger)
    def __init__(self, **kwargs):
        self._tokenize_column = kwargs.pop("tokenize_column", "text")
        super().__init__(**kwargs)

    @log_on_start(DEBUG, "SAPipeline.__call__({args} and {kwargs})", logger=logger)
    @log_on_error(
        ERROR,
        "Error during __call__(): {e!r}",
        logger=logger,
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!", logger=logger)
    def __call__(self, *args, **kwargs):
        """
        Classify the text(s) given as inputs.

        Args:
            args (`str` or `List[str]` or `Dict[str]`, or `List[Dict[str]]`):
                One or several texts to classify. In order to use text pairs for your classification, you can send a
                dictionary containing `{"text", "text_pair"}` keys, or a list of those.
            top_k (`int`, *optional*, defaults to `1`):
                How many results to return.
            function_to_apply (`str`, *optional*, defaults to `"default"`):
                The function to apply to the model outputs in order to retrieve the scores. Accepts four different
                values:

                If this argument is not specified, then it will apply the following functions according to the number
                of labels:

                - If the model has a single label, will apply the sigmoid function on the output.
                - If the model has several labels, will apply the softmax function on the output.

                Possible values are:

                - `"sigmoid"`: Applies the sigmoid function on the output.
                - `"softmax"`: Applies the softmax function on the output.
                - `"none"`: Does not apply any function on the output.

        Return:
            A list or a list of list of `dict`: Each result comes as list of dictionaries with the following keys:

            - **label** (`str`) -- The label predicted.
            - **score** (`float`) -- The corresponding probability.

            If `top_k` is used, one such dictionary is returned per label.
        """
        result = super().__call__(*args, **kwargs)
        return result

    @log_on_start(DEBUG, "SAPipeline.preprocess({tokenizer_kwargs})", logger=logger)
    @log_on_error(
        ERROR,
        "Error during preprocessing: {e!r}",
        logger=logger,
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!", logger=logger)
    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        return_tensors = self.framework
        if isinstance(inputs, list):
            return self.tokenizer(
                inputs,
                return_tensors=return_tensors,
                **tokenizer_kwargs,
                is_split_into_words=True,
            )
        elif isinstance(inputs, str):
            return self.tokenizer(
                inputs, return_tensors=return_tensors, **tokenizer_kwargs
            )

        return self.tokenizer(
            inputs[self._tokenize_column],
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )

    @log_on_start(DEBUG, "SAPipeline._forward()", logger=logger)
    @log_on_error(
        ERROR,
        "Error during _forward: {e!r}",
        logger=logger,
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!", logger=logger)
    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    @log_on_start(DEBUG, "SAPipeline.postprocess()", logger=logger)
    @log_on_error(
        ERROR,
        "Error during postprocessing: {e!r}",
        logger=logger,
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!", logger=logger)
    def postprocess(self, model_outputs):
        return model_outputs

    @log_on_start(
        DEBUG, "SAPipeline._sanitize_parameters {tokenizer_kwargs}", logger=logger
    )
    @log_on_error(
        ERROR,
        "Error during sanitizing :-) {e!r}",
        logger=logger,
        on_exceptions=Exception,
        reraise=True,
    )
    @log_on_end(DEBUG, "done!", logger=logger)
    def _sanitize_parameters(self, **tokenizer_kwargs):
        preprocess_params = tokenizer_kwargs
        return preprocess_params, {}, {}
