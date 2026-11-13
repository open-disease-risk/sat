"""Configuration for task heads"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import abc
import json

from enum import Enum
from transformers import PretrainedConfig
from typing import List, Dict, Any

from transformers.models.bert.configuration_bert import BertConfig


class TokenEmbedding(Enum):
    """Determine what to do with the hidden states of the encoder layers."""

    SUM = 2
    AVG = 3
    CAT = 4
    BERT = 5
    ATTENTION = 6


class SentenceEmbedding(Enum):
    """Determine what to do to get sentence embeddings."""

    NONE = 1
    MAX = 2
    AVG = 3
    ATTENTION = 4
    HIERARCHICAL = 5


class BaseConfig(abc.ABC, PretrainedConfig):
    def __init__(
        self,
        loss_weight: float = 1.0,
        metric_train_sample_size: int = 5000,
        freeze_transformer: bool = False,
        initializer_range: int = 0.02,
        initializer: str = "kaiming_uniform",
        num_events: int = 1,
        loss: Dict[str, Any] = {},
        # Stability parameters for StableLinear
        grad_scale: float = 0.1,
        eps: float = 1e-8,
        max_norm: float = 1.0,
        init_gain: float = 0.01,
        weight_clamp_value: float = 0.1,
        noise_scale: float = 1e-5,
        nan_value: float = 0.0,
        posinf_value: float = 1.0,
        neginf_value: float = -1.0,
        output_min: float = -10.0,
        output_max: float = 10.0,
        # Additional parameters for StableMLPSequence
        weight_init_std: float = 0.02,
        softplus_beta: float = 1.0,
        leaky_relu_negative_slope: float = 0.01,
        use_residual: bool = True,
        layer_norm_eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.freeze_transformer = freeze_transformer
        self.loss_weight = loss_weight
        self.metric_train_sample_size = metric_train_sample_size
        self.initializer_range = initializer_range
        self.initializer = initializer
        self.task_specific_params: Dict[str, Any] = {}
        self.num_events = num_events
        self.loss = loss
        
        # Stability parameters
        self.grad_scale = grad_scale
        self.eps = eps
        self.max_norm = max_norm
        self.init_gain = init_gain
        self.weight_clamp_value = weight_clamp_value
        self.noise_scale = noise_scale
        self.nan_value = nan_value
        self.posinf_value = posinf_value
        self.neginf_value = neginf_value
        self.output_min = output_min
        self.output_max = output_max
        self.weight_init_std = weight_init_std
        self.softplus_beta = softplus_beta
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self.use_residual = use_residual
        self.layer_norm_eps = layer_norm_eps

    # deal with nested dictionaries
    def to_json_string(self, use_diff=True):
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (:obj:`bool`):
                If set to True, only the difference between the config instance and the default PretrainedConfig() is serialized to JSON string.

        Returns:
            :obj:`string`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        return (
            json.dumps(
                config_dict, default=lambda o: o.__dict__, indent=2, sort_keys=True
            )
            + "\n"
        )


class SatBertConfig(BertConfig):
    model_type = "sat-bert"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SurvivalConfig(BaseConfig):
    model_type = "sat-transformer"

    def __init__(
        self,
        num_inputs: int = 32,
        intermediate_size: int = 64,
        num_hidden_layers: int = 0,
        indiv_intermediate_size: int = 64,
        indiv_num_hidden_layers: int = 0,
        batch_norm: bool = True,
        hidden_dropout_prob: float = 0.05,
        bias: bool = True,
        max_time=400,
        duration_cuts=[],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_inputs = num_inputs
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.indiv_intermediate_size = indiv_intermediate_size
        self.indiv_num_hidden_layers = indiv_num_hidden_layers
        self.batch_norm = batch_norm
        self.hidden_dropout_prob = hidden_dropout_prob
        self.bias = bias
        self.max_time = max_time
        self.duration_cuts = duration_cuts


class EventClassificationTaskConfig(BaseConfig):
    model_type = "sat-transformer-event-classification"

    def __init__(
        self,
        num_inputs: int = 32,
        intermediate_size: int = 64,
        event_time_thr: float = 0.5,
        num_hidden_layers: int = 0,
        indiv_intermediate_size: int = 64,
        indiv_num_hidden_layers: int = 0,
        batch_norm: bool = True,
        hidden_dropout_prob: float = 0.05,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_inputs = num_inputs
        self.intermediate_size = intermediate_size
        self.event_time_thr = event_time_thr
        self.num_hidden_layers = num_hidden_layers
        self.indiv_intermediate_size = indiv_intermediate_size
        self.indiv_num_hidden_layers = indiv_num_hidden_layers
        self.batch_norm = batch_norm
        self.hidden_dropout_prob = hidden_dropout_prob
        self.bias = bias


class EventDurationTaskConfig(BaseConfig):
    model_type = "sat-transformer-event-duration"

    def __init__(
        self,
        num_inputs: int = 32,
        intermediate_size: int = 64,
        num_hidden_layers: int = 0,
        indiv_intermediate_size: int = 64,
        indiv_num_hidden_layers: int = 0,
        batch_norm: bool = True,
        hidden_dropout_prob: float = 0.05,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_inputs = num_inputs
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.indiv_intermediate_size = indiv_intermediate_size
        self.indiv_num_hidden_layers = indiv_num_hidden_layers
        self.batch_norm = batch_norm
        self.hidden_dropout_prob = hidden_dropout_prob
        self.bias = bias


class DistributionTaskConfig(BaseConfig):
    model_type = "sat-transformer-distribution"

    def __init__(
        self,
        num_inputs: int = 32,
        intermediate_size: int = 128,
        num_hidden_layers: int = 0,
        distribution_type: str = "Weibull",
        batch_norm: bool = True,
        hidden_dropout_prob: float = 0.1,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_inputs = num_inputs
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.distribution_type = distribution_type
        self.batch_norm = batch_norm
        self.hidden_dropout_prob = hidden_dropout_prob
        self.bias = bias


class MTLConfig(BaseConfig):
    model_type = "sat-mtl-transformer"

    def __init__(
        self,
        task_heads: List[PretrainedConfig] = None,
        intermediate_size: int = 64,
        num_hidden_layers: int = 0,
        batch_norm: bool = True,
        hidden_dropout_prob: float = 0.05,
        bias: bool = True,
        num_labels: int = 32,
        sentence_emb: SentenceEmbedding = SentenceEmbedding.NONE,
        token_emb: TokenEmbedding = TokenEmbedding.CAT,
        select_hidden_layers: List[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task_heads = task_heads
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.batch_norm = batch_norm
        self.hidden_dropout_prob = hidden_dropout_prob
        self.sentence_emb = sentence_emb
        self.token_emb = token_emb
        self.bias = bias
        self.num_labels = num_labels
        self.select_hidden_layers = select_hidden_layers
