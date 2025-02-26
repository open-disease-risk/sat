"""Task heads for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import abc
import hydra
import inspect
import torch

import torch.nn.functional as F

from dataclasses import dataclass
from logging import DEBUG, ERROR

from torch import nn
from transformers import AutoModel
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from typing import Optional, Tuple, Union

from sat.models.nets import CauseSpecificNet, CauseSpecificNetCompRisk, SimpleMLP
from sat.models.tasks.config import (
    BaseConfig,
    SentenceEmbedding,
    SurvivalConfig,
    TokenEmbedding,
    EventClassificationTaskConfig,
    EventDurationTaskConfig,
    MTLConfig,
)
from sat.utils import logging

logger = logging.get_default_logger()


def pad_col(input, val=0, where="end"):
    """Adds a column of `val` at the start or end of `input` with optimized memory usage.

    This optimized version avoids unnecessary tensor allocations.
    """
    if len(input.shape) != 3:
        raise ValueError(f"Only works for `phi` tensor that is 3-D.")

    # Get shape for new tensor
    batch_size, num_events, seq_len = input.shape
    new_shape = (batch_size, num_events, seq_len + 1)

    # Create output tensor directly with correct size (avoids intermediate allocations)
    result = torch.zeros(new_shape, dtype=input.dtype, device=input.device)

    # Fill with data efficiently
    if where == "end":
        result[:, :, :seq_len].copy_(input)
        if val != 0:
            result[:, :, -1] = val
    elif where == "start":
        result[:, :, 1:].copy_(input)
        if val != 0:
            result[:, :, 0] = val
    else:
        raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

    return result


@dataclass
class SAOutput(ModelOutput):
    """
    Base class for outputs of survival analysis models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hazard: Optional[torch.FloatTensor] = None
    risk: Optional[torch.FloatTensor] = None
    survival: Optional[torch.FloatTensor] = None
    time_to_event: Optional[torch.FloatTensor] = None
    event: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class TaskOutput(ModelOutput):
    """
    Base class for outputs of survival analysis models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        predictions (`torch.FloatTensor` of shape `(batch_size, )`):
            Predictions of the task.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    predictions: torch.FloatTensor = None


class SurvivalPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            logger.debug("Initialize linear weights...")

            if self.config.initializer == "normal":
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif self.config.initializer == "xavier_uniform":
                torch.nn.init.xavier_uniform_(
                    module.weight, gain=torch.nn.init.calculate_gain("leaky_relu")
                )
            elif self.config.initializer == "xavier_normal":
                torch.nn.init.xavier_normal_(
                    module.weight, gain=torch.nn.init.calculate_gain("leaky_relu")
                )
            elif self.config.initializer == "kaiming_normal":
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif self.config.initializer == "kaiming_uniform":
                nn.init.kaiming_uniform_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            else:
                raise ValueError(
                    f"Initializer {self.config.initializer} not supported!"
                )

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            logger.debug("Initialize layer norm weights...")
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SurvivalTask(abc.ABC, SurvivalPreTrainedModel):
    def __init__(self, config: BaseConfig):
        super().__init__(config)


class SurvivalTaskHead(SurvivalTask):
    config_class = SurvivalConfig

    def __init__(self, config: SurvivalConfig):
        super().__init__(config)

        if self.config.num_events > 1:
            self.nets = CauseSpecificNetCompRisk(
                in_features=self.config.num_features,
                shared_intermediate_size=self.config.intermediate_size,
                shared_num_hidden_layers=self.config.num_hidden_layers,
                indiv_intermediate_size=self.config.indiv_intermediate_size,
                indiv_num_hidden_layers=self.config.indiv_num_hidden_layers,
                bias=self.config.bias,
                batch_norm=self.config.batch_norm,
                dropout=self.config.hidden_dropout_prob,
                out_features=self.config.num_labels,
                num_events=self.config.num_events,
            )
        else:
            self.nets = CauseSpecificNet(
                in_features=self.config.num_features,
                intermediate_size=self.config.intermediate_size,
                num_hidden_layers=self.config.num_hidden_layers,
                bias=self.config.bias,
                batch_norm=self.config.batch_norm,
                dropout=self.config.hidden_dropout_prob,
                out_features=self.config.num_labels,
                num_events=self.config.num_events,
            )

        # Initialize weights and apply final processing
        logger.debug("Post initialize in SurvivalTaskHead")
        self.post_init()

        loss = config.loss[config.model_type]
        logger.debug(f"Instantiate the loss {loss}")
        self.loss = hydra.utils.instantiate(loss)

    def forward(self, sequence_output, labels=None, **kwargs):
        logits = self.nets(sequence_output)  # num events x batch x duration cuts
        hazard = F.softplus(logits)
        hazard = pad_col(hazard, where="start")
        # Optimized tensor operations: fuse cumsum+mul+exp into a single operation
        surv = (
            -hazard.cumsum(dim=2)
        ).exp()  # More efficient than cumsum().mul(-1).exp()
        # Use in-place operation to create risk from survival
        risk = torch.ones_like(surv).sub_(
            surv
        )  # Equivalent to 1.0 - surv but more efficient

        output = SAOutput(
            loss=None,
            logits=logits,
            hazard=hazard,
            risk=risk,
            survival=surv,
            hidden_states=sequence_output,
        )

        if labels is not None:
            logger.debug(
                f"Computing loss with logits {logits[0].shape} and labels {labels.shape}"
            )
            output.loss = self.loss(
                output,
                labels,
            )

        return output


class EventClassificationTaskHead(SurvivalPreTrainedModel):
    config_class = EventClassificationTaskConfig

    def __init__(self, config: EventClassificationTaskConfig):
        super().__init__(config)

        if self.config.num_events > 1:
            self.nets = CauseSpecificNetCompRisk(
                in_features=self.config.num_features,
                shared_intermediate_size=self.config.intermediate_size,
                shared_num_hidden_layers=self.config.num_hidden_layers,
                indiv_intermediate_size=self.config.indiv_intermediate_size,
                indiv_num_hidden_layers=self.config.indiv_num_hidden_layers,
                bias=self.config.bias,
                batch_norm=self.config.batch_norm,
                dropout=self.config.hidden_dropout_prob,
                out_features=self.config.num_labels,
                num_events=self.config.num_events,
            )
        else:
            self.nets = CauseSpecificNet(
                in_features=self.config.num_features,
                intermediate_size=self.config.intermediate_size,
                num_hidden_layers=self.config.num_hidden_layers,
                bias=self.config.bias,
                batch_norm=self.config.batch_norm,
                dropout=self.config.hidden_dropout_prob,
                out_features=self.config.num_labels,
                num_events=self.config.num_events,
            )

        loss = config.loss[config.model_type]
        logger.debug(f"Instantiate the loss {loss}")
        self.loss = hydra.utils.instantiate(loss)

    def forward(self, sequence_output, labels=None, **kwargs):
        logits = self.nets(sequence_output)  # num events x batch x 1
        predictions = torch.sigmoid(logits)

        loss = None
        output = TaskOutput(loss=loss, logits=logits, predictions=predictions)
        if labels is not None:
            loss = self.loss(
                output,
                labels,
            )
            output.loss = loss

        return output


class EventDurationTaskHead(SurvivalPreTrainedModel):
    config_class = EventDurationTaskConfig

    def __init__(self, config: EventDurationTaskConfig):
        super().__init__(config)

        if self.config.num_events > 1:
            self.nets = CauseSpecificNetCompRisk(
                in_features=self.config.num_features,
                shared_intermediate_size=self.config.intermediate_size,
                shared_num_hidden_layers=self.config.num_hidden_layers,
                indiv_intermediate_size=self.config.indiv_intermediate_size,
                indiv_num_hidden_layers=self.config.indiv_num_hidden_layers,
                bias=self.config.bias,
                batch_norm=self.config.batch_norm,
                dropout=self.config.hidden_dropout_prob,
                out_features=self.config.num_labels,
                num_events=self.config.num_events,
            )
        else:
            self.nets = CauseSpecificNet(
                in_features=self.config.num_features,
                intermediate_size=self.config.intermediate_size,
                num_hidden_layers=self.config.num_hidden_layers,
                bias=self.config.bias,
                batch_norm=self.config.batch_norm,
                dropout=self.config.hidden_dropout_prob,
                out_features=self.config.num_labels,
                num_events=self.config.num_events,
            )

        loss = config.loss[config.model_type]
        logger.debug(f"Instantiate the loss {loss}")
        self.loss = hydra.utils.instantiate(loss)

    def forward(self, sequence_output, labels=None, **kwargs):
        logits = nn.ReLU()(self.nets(sequence_output))
        predictions = torch.squeeze(logits, dim=2)  # num events x batch x predictions

        loss = None
        output = TaskOutput(loss=loss, logits=logits, predictions=predictions)
        if labels is not None:
            loss = self.loss(
                output,
                labels,
            )
            output.loss = loss

        return output


class MTLForSurvival(SurvivalPreTrainedModel):
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

        logger.debug(f"Using configuration {config}")

        self.transformer = None
        logger.debug(f"Using configuration {self.config}")
        if "pretrained_model_name_or_path" in self.config.pretrained_params:
            logger.debug(f"Load model from pretrained {self.config.pretrained_params}")
            self.transformer = AutoModel.from_pretrained(
                **self.config.pretrained_params
            )
        else:
            logger.debug(f"Load model from config {self.config.pretrained_params}")
            self.transformer = AutoModel.from_config(**self.config.pretrained_params)

        if self.config.select_hidden_layers:
            assert (
                min(self.config.select_hidden_layers) >= 0
            ), f"Layer indices in 'select_hidden_layers'= {self.config.select_hidden_layers} should be greater than or equal to 0"
            assert (
                self.config.select_hidden_layers[-1]
                <= self.transformer.config.num_hidden_layers
            ), f"Layer indices in 'select_hidden_layers'= {self.config.select_hidden_layers} should be less than or equal to the number of hidden layers {self.transformer.config.num_hidden_layers}"

        self.forward_args = list(inspect.signature(self.transformer.forward).parameters)
        logger.debug(
            f"Loaded transformer {self.transformer} supporting forward({self.forward_args})"
        )

        if self.config.freeze_transformer:
            logger.debug("Freeze the transformer")
            for name, param in self.transformer.base_model.named_parameters():
                param.requires_grad = False

        # features for shared network before feeding into the task heads
        # is based on the pooling strategy for tokens and sentences
        if self.config.token_emb == TokenEmbedding.BERT.value:
            logger.debug(
                f"If BERT token embedding is chosen, sentence embeddings to NONE."
            )
            self.config.sentence_emb = SentenceEmbedding.NONE.value

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

        if (self.config.token_emb != TokenEmbedding.BERT.value) and (
            self.config.sentence_emb == SentenceEmbedding.NONE.value
        ):
            self.in_features = self.in_features * self.config.num_features

        logger.debug(f"Number of input features to the shared MLP: {self.in_features}")
        self.net = SimpleMLP(
            in_features=self.in_features,
            intermediate_size=self.config.intermediate_size,
            bias=self.config.bias,
            batch_norm=self.config.batch_norm,
            dropout=self.config.hidden_dropout_prob,
            num_hidden_layers=self.config.num_hidden_layers,
            out_features=self.config.num_labels,
        )

        self.heads = nn.ModuleList()
        # initialize the list of task heads
        for i, task_head_config in enumerate(self.config.task_heads):
            logger.debug(f"{i}-th task configuration: {task_head_config}")
            model = AutoModel.from_config(task_head_config)
            if isinstance(model, SurvivalTaskHead):
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

        # Initialize weights and apply final processing
        logger.debug("Post initialize in MTLForSurvival")
        self.post_init()

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
        forward_dict = locals()
        del_args = [arg for arg in forward_dict if arg not in self.forward_args]
        logger.debug(
            f"Remove unsupported arguments to the transformers forward function: {del_args}"
        )
        [forward_dict.pop(key) for key in del_args]

        logger.debug(
            f"""
            Score the transformer with
            - input ids {input_ids.shape}
            """
        )

        logger.debug(f"Forward dictionary for the backend transformer {forward_dict}")

        sequence_output = self.transformer(
            **forward_dict, return_dict=self.return_dict
        )  # batch x sentences x words x embedding size

        logger.debug(f"Got {sequence_output} from transformer")

        # layers x batches x tokens x features
        hidden_states = sequence_output.hidden_states
        token_embeddings = torch.stack(hidden_states, dim=0)
        # subset the layers
        if self.config.select_hidden_layers:
            logger.debug(f"Select hidden layers {self.config.select_hidden_layers}")
            token_embeddings = token_embeddings[
                self.config.select_hidden_layers, :, :, :
            ]
        # batch x tokens x layers x features
        token_embeddings = token_embeddings.permute(1, 2, 0, 3)
        logger.debug(
            f"Dimensions of token embeddings after layer selection {token_embeddings.shape}"
        )

        if self.config.token_emb == TokenEmbedding.AVG.value:
            logger.debug("Average the token embeddings")
            token_embeddings = torch.mean(token_embeddings, 2)
        elif self.config.token_emb == TokenEmbedding.SUM.value:
            logger.debug("Sum the token embeddings")
            token_embeddings = torch.sum(token_embeddings, 2)
        elif self.config.token_emb == TokenEmbedding.CAT.value:
            logger.debug("Concatenate the token embeddings")
            # split along the layer dimension
            layers = torch.tensor_split(
                token_embeddings, token_embeddings.shape[2], dim=2
            )
            token_embeddings = torch.cat(layers, dim=3).squeeze()
        elif self.config.token_emb == TokenEmbedding.BERT.value:
            logger.debug("Use Bert-pooler for the token embeddings")
            token_embeddings = sequence_output[1]
        else:
            logger.debug("No operation on the token embeddings")

        logger.debug(f"Dimensions of token embeddings {token_embeddings.shape}")
        sentence_embeddings = token_embeddings
        # we do not need sentence embedding if we do bert pooling
        if self.config.token_emb != TokenEmbedding.BERT.value:
            if self.config.sentence_emb == SentenceEmbedding.AVG.value:
                logger.debug("Average across tokens to produce sentence embeddings")
                sentence_embeddings = torch.mean(token_embeddings, 1)
            elif self.config.sentence_emb == SentenceEmbedding.MAX.value:
                logger.debug("Max across tokens to produce sentence embeddings")
                sentence_embeddings = torch.max(token_embeddings, 1).values
            else:
                logger.debug("No reductions to produce sentence embeddings")
                sentence_embeddings = token_embeddings

        logger.debug(f"Dimensions of sentence embeddings {sentence_embeddings.shape}")
        logger.debug(f"Number of input features to the shared MLP: {self.in_features}")
        logger.debug(f"Number of hidden layers: {self.config.num_hidden_layers}")

        batch_size = sentence_embeddings.shape[0]
        logits = self.net(sentence_embeddings.reshape((batch_size, -1)))
        logger.debug("Score the task heads")

        sa_output: SAOutput = None
        outputs = {}
        tte = None
        event = None
        loss = 0.0
        logits_tasks = []
        # consider the loss components as regularization terms except for the
        # survival analysis task head
        for i, h in enumerate(self.heads):
            logger.debug(f"Score {i}-th task head")
            logger.debug(f"head: {h}")
            outputs[i] = h(
                logits,
                labels,
            )
            logger.debug(f"Got output {outputs[i]} from the {i}-th task head")
            logits_tasks.append(outputs[i].logits)
            if labels is not None:
                loss += outputs[i].loss * h.config.loss_weight

            if isinstance(h, SurvivalTaskHead):
                sa_output = outputs[i]
            elif isinstance(h, EventDurationTaskHead):
                tte = outputs[i].predictions
            elif isinstance(h, EventClassificationTaskHead):
                event = outputs[i].predictions

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

        logger.debug(f"Return output {sa_output} from MTLForSurvival task head")
        return sa_output
