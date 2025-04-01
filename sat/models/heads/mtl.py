"""MTL Task heads for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoModel

from sat.models.nets import SimpleMLP
from sat.utils import logging

from .base import BaseConfig, MTLTask
from .classification import EventClassificationTaskHead
from .embeddings import (
    SentenceEmbedder,
    SentenceEmbedding,
    TokenEmbedder,
    TokenEmbedding,
)
from .output import SAOutput
from .regression import EventDurationTaskHead
from .survival import SurvivalTaskHead
from .mensa import MENSATaskHead

logger = logging.get_default_logger()


class MTLConfig(BaseConfig):
    """Configuration for the MTLForSurvival model."""

    model_type = "mtl"

    def __init__(
        self,
        num_labels: int = 32,
        hidden_size: int = 768,
        intermediate_size: int = 64,
        num_hidden_layers: int = 0,
        batch_norm: bool = True,
        hidden_dropout_prob: float = 0.05,
        bias: bool = True,
        return_dict: bool = True,
        select_hidden_layers: Optional[List[int]] = None,
        token_emb: str = TokenEmbedding.BERT.value,
        sentence_emb: str = SentenceEmbedding.AVG.value,
        task_heads: List[Dict[str, Any]] = None,
        pretrained_params: Dict[str, Any] = None,
        mtl_balance_strategy: Optional[str] = None,
        mtl_balance_params: Dict[str, Any] = None,
        # BaseConfig parameters
        loss_weight: float = 1.0,
        metric_train_sample_size: int = 5000,
        freeze_transformer: bool = False,
        initializer_range: int = 0.02,
        initializer: str = "kaiming_uniform",
        num_events: int = 1,
        loss: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(
            loss_weight=loss_weight,
            metric_train_sample_size=metric_train_sample_size,
            freeze_transformer=freeze_transformer,
            initializer_range=initializer_range,
            initializer=initializer,
            num_events=num_events,
            loss=loss if loss is not None else {},
            **kwargs,
        )
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.batch_norm = batch_norm
        self.hidden_dropout_prob = hidden_dropout_prob
        self.bias = bias
        self.return_dict = return_dict
        self.select_hidden_layers = select_hidden_layers
        self.token_emb = token_emb
        self.sentence_emb = sentence_emb
        self.task_heads = task_heads if task_heads is not None else []
        self.pretrained_params = (
            pretrained_params if pretrained_params is not None else {}
        )
        self.mtl_balance_strategy = mtl_balance_strategy
        self.mtl_balance_params = (
            mtl_balance_params if mtl_balance_params is not None else {}
        )


class MTLForSurvival(MTLTask):
    config_class = MTLConfig
    base_model_prefix = "any"
    supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: MTLConfig):
        super().__init__(config)

        self.is_survival = False
        self.is_regression = False
        self.is_classification = False
        self.return_dict = config.return_dict

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Using configuration {config}")

        # Initialize transformer model
        self.transformer = None
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Using configuration {self.config}")
        if "pretrained_model_name_or_path" in self.config.pretrained_params:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Load model from pretrained {self.config.pretrained_params}"
                )
            self.transformer = AutoModel.from_pretrained(
                **self.config.pretrained_params
            )
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Load model from config {self.config.pretrained_params}")
            self.transformer = AutoModel.from_config(**self.config.pretrained_params)

        # Validate hidden layer selection if provided
        if self.config.select_hidden_layers:
            assert (
                min(self.config.select_hidden_layers) >= 0
            ), f"Layer indices in 'select_hidden_layers'= {self.config.select_hidden_layers} should be greater than or equal to 0"
            assert (
                self.config.select_hidden_layers[-1]
                <= self.transformer.config.num_hidden_layers
            ), f"Layer indices in 'select_hidden_layers'= {self.config.select_hidden_layers} should be less than or equal to the number of hidden layers {self.transformer.config.num_hidden_layers}"

        # Store transformer forward arguments
        self.forward_args = list(inspect.signature(self.transformer.forward).parameters)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Loaded transformer {self.transformer} supporting forward({self.forward_args})"
            )

        # Freeze transformer if specified
        if self.config.freeze_transformer:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Freeze the transformer")
            for name, param in self.transformer.base_model.named_parameters():
                param.requires_grad = False

        # Initialize embedding processors
        self.token_embedder = TokenEmbedder(
            hidden_size=self.config.hidden_size,
            token_emb_strategy=self.config.token_emb,
        )

        self.sentence_embedder = SentenceEmbedder(
            sentence_emb_strategy=self.config.sentence_emb,
        )

        # Calculate input features for shared network
        if self.config.token_emb == TokenEmbedding.CAT.value:
            if self.config.select_hidden_layers:
                self.in_features = self.config.hidden_size * (
                    len(self.config.select_hidden_layers)
                )
            else:
                self.in_features = self.config.hidden_size * (
                    self.transformer.config.num_hidden_layers + 1
                )
        else:
            self.in_features = self.config.hidden_size

        # Special case for token embeddings without sentence pooling
        if (self.config.token_emb != TokenEmbedding.BERT.value) and (
            self.config.sentence_emb == SentenceEmbedding.NONE.value
        ):
            self.in_features = self.in_features * self.config.num_features

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Number of input features to the shared MLP: {self.in_features}"
            )

        # Initialize shared network
        self.net = SimpleMLP(
            in_features=self.in_features,
            intermediate_size=self.config.intermediate_size,
            bias=self.config.bias,
            batch_norm=self.config.batch_norm,
            dropout=self.config.hidden_dropout_prob,
            num_hidden_layers=self.config.num_hidden_layers,
            out_features=self.config.num_labels,
        )

        # Initialize task heads
        self.heads = nn.ModuleList()
        for i, task_head_config in enumerate(self.config.task_heads):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{i}-th task configuration: {task_head_config}")
            model = AutoModel.from_config(task_head_config)
            if isinstance(model, SurvivalTaskHead) or isinstance(model, MENSATaskHead):
                if isinstance(model, MENSATaskHead):
                    logger.info("MENSA task head initialized")
                else:
                    logger.info("Survival task head initialized")
                self.is_survival = True
            elif isinstance(model, EventClassificationTaskHead):
                logger.info("Event classification task head initialized")
                self.is_classification = True
            elif isinstance(model, EventDurationTaskHead):
                logger.info("Event regression task head initialized")
                self.is_regression = True
            else:
                raise ValueError(f"Task head {task_head_config} not supported!")

            self.heads.append(model)

        # Initialize weights when created as a standalone model
        if self.__class__.__name__ == "MTLForSurvival":
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Standalone MTLForSurvival - initializing weights")
                # First initialize only shared network weights
                logger.debug("MTLTask: initializing shared network weights")
            for name, module in self.named_children():
                if name != "heads":  # Skip the heads module list
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Initializing MTL module: {name}")
                    self.initialize_module(module)

            # Then initialize each task head with its specialized initialization
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("MTLTask: initializing task head weights")
            for i, head in enumerate(self.heads):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Initializing task head {i}")
                head.post_init()
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("MTLForSurvival created as part of another model")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        place_of_service_ids: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
        visit_ids: Optional[torch.Tensor] = None,
        numerics: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SAOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # Filter arguments for transformer
        forward_dict = locals()
        del_args = [arg for arg in forward_dict if arg not in self.forward_args]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Remove unsupported arguments to the transformers forward function: {del_args}"
            )
        [forward_dict.pop(key) for key in del_args]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"""
                Score the transformer with
                - input ids {input_ids.shape}
                """
            )
            logger.debug(
                f"Forward dictionary for the backend transformer {forward_dict}"
            )

        # Always set output_hidden_states to True since we need them
        forward_dict["output_hidden_states"] = True

        # Get transformer outputs
        sequence_output = self.transformer(**forward_dict, return_dict=self.return_dict)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Got {sequence_output} from transformer")

        # Process hidden states through token embedder
        token_embeddings = self.token_embedder(
            sequence_output.hidden_states,
            sequence_output,
            self.config.select_hidden_layers,
            attention_mask,
        )

        # Create sentence embeddings from token embeddings
        sentence_embeddings = self.sentence_embedder(token_embeddings, attention_mask)

        # Get batch size and reshape for the shared network if needed
        batch_size = sentence_embeddings.shape[0]
        if len(sentence_embeddings.shape) > 2:
            # Flatten for input to shared network
            logits = self.net(sentence_embeddings.reshape((batch_size, -1)))
        else:
            logits = self.net(sentence_embeddings)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Score the task heads")

        # Process through task heads
        sa_output: SAOutput = None
        outputs = {}
        tte = None
        event = None
        loss = 0.0
        logits_tasks = []

        # Process all task heads
        for i, h in enumerate(self.heads):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Score {i}-th task head")
                logger.debug(f"head: {h}")

            outputs[i] = h(
                logits,
                labels,
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Got output {outputs[i]} from the {i}-th task head")

            logits_tasks.append(outputs[i].logits)
            if labels is not None:
                loss += outputs[i].loss * h.config.loss_weight

            # Save outputs based on task head type
            if isinstance(h, SurvivalTaskHead):
                sa_output = outputs[i]
            elif isinstance(h, EventDurationTaskHead):
                tte = outputs[i].predictions
            elif isinstance(h, EventClassificationTaskHead):
                event = outputs[i].predictions

        # Create combined output
        if sa_output:
            sa_output = SAOutput(
                loss=loss,
                logits=logits_tasks,
                hazard=sa_output.hazard,
                risk=sa_output.risk,
                survival=sa_output.survival,
                time_to_event=tte,
                event=event,
                hidden_states=sequence_output.hidden_states,
                attentions=sequence_output.attentions,
            )
        else:
            sa_output = SAOutput(
                loss=loss,
                logits=logits_tasks,
                hazard=None,
                risk=None,
                survival=None,
                time_to_event=tte,
                event=event,
                hidden_states=sequence_output.hidden_states,
                attentions=sequence_output.attentions,
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Return output {sa_output} from MTLForSurvival task head")

        return sa_output
