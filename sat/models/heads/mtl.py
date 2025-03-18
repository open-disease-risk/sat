"""MTL Task heads for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import inspect
import torch

from torch import nn
from transformers import AutoModel
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

from .base import SurvivalPreTrainedModel
from .output import SAOutput
from .survival import SurvivalTaskHead
from .classification import EventClassificationTaskHead
from .regression import EventDurationTaskHead

logger = logging.get_default_logger()


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
