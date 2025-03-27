"""MTL Task heads for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import inspect
import torch

from torch import nn
from transformers import AutoModel
from typing import Dict, List, Optional, Tuple, Union

from sat.loss.balancing import BalancingStrategy, LossBalancer
from sat.models.nets import SimpleMLP
from sat.utils import logging
from .embeddings import TokenEmbedder, SentenceEmbedder

from .base import BaseConfig, MTLTask, SentenceEmbedding, TokenEmbedding
from .output import SAOutput
from .survival import SurvivalTaskHead
from .dsm import DSMTaskHead
from .classification import EventClassificationTaskHead
from .regression import EventDurationTaskHead
from transformers import PretrainedConfig


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
        balance_strategy: Union[str, BalancingStrategy] = "fixed",
        balance_params: Optional[Dict] = None,
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
        self.balance_strategy = balance_strategy
        self.balance_params = balance_params or {}


logger = logging.get_default_logger()


class MTLForSurvival(MTLTask):
    config_class = MTLConfig
    base_model_prefix = "any"
    supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: MTLConfig):
        super().__init__(config)

        self.is_survival = False
        self.is_dsm = False
        self.is_regression = False
        self.is_classification = False
        self.return_dict = config.return_dict

        coeffs = []

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
            # Still collect initial coefficients for potential fixed weight balancing
            coeffs.append(task_head_config.loss_weight)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{i}-th task configuration: {task_head_config}")

            model = AutoModel.from_config(task_head_config)

            if isinstance(model, DSMTaskHead):
                logger.info("DSM task head initialized")
                self.is_dsm = True
            elif isinstance(model, SurvivalTaskHead):
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

        # Create loss balancer
        num_task_heads = len(self.heads)
        self.loss_balancer = LossBalancer.create(
            strategy=self.config.balance_strategy,
            num_losses=num_task_heads,
            coeffs=coeffs,  # Pass initial coefficients from task heads
            **self.config.balance_params,
        )
        logger.info(
            f"Using {self.config.balance_strategy} balancing strategy for MTL with {num_task_heads} tasks"
        )

        # Still register the initial coefficients for backward compatibility
        self.register_buffer("coeffs", torch.tensor(coeffs).to(torch.float32))

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
        losses = []
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
                losses.append(outputs[i].loss)

            # Save outputs based on task head type
            if isinstance(h, SurvivalTaskHead):
                sa_output = outputs[i]
            elif isinstance(h, DSMTaskHead):
                sa_output = outputs[i]
            elif isinstance(h, EventDurationTaskHead):
                tte = outputs[i].predictions
            elif isinstance(h, EventClassificationTaskHead):
                event = outputs[i].predictions

        # Balance losses using the loss balancer
        if labels is not None and losses:
            # Get the current training iteration from state if available
            iteration = getattr(self, "current_iteration", None)
            if hasattr(self, "current_iteration"):
                self.current_iteration += 1
            else:
                self.current_iteration = 0
                iteration = 0

            # Use the loss balancer to combine the losses
            total_loss = self.loss_balancer(losses, iteration)

            # Log balancing weights
            if logger.isEnabledFor(logging.DEBUG):
                weights = self.loss_balancer.get_weights()
                logger.debug(f"Current loss weights: {weights}")
        else:
            # Fallback to using fixed weights if no labels or losses
            total_loss = torch.tensor(0.0, device=sequence_output.device)

        # Create combined output
        if sa_output:
            # Extract DSM-specific attributes if present
            shape = getattr(sa_output, "shape", None)
            scale = getattr(sa_output, "scale", None)
            logits_g = getattr(sa_output, "logits_g", None)

            # Create SAOutput with all relevant fields
            sa_output = SAOutput(
                loss=total_loss,
                logits=logits_tasks,
                hazard=sa_output.hazard,
                risk=sa_output.risk,
                survival=sa_output.survival,
                time_to_event=tte,
                event=event,
                hidden_states=sequence_output.hidden_states,
                attentions=sequence_output.attentions,
                shape=shape,
                scale=scale,
                logits_g=logits_g,
            )
        else:
            sa_output = SAOutput(
                loss=total_loss,
                logits=logits_tasks,
                hazard=None,
                risk=None,
                survival=None,
                time_to_event=tte,
                event=event,
                hidden_states=sequence_output.hidden_states,
                attentions=sequence_output.attentions,
                shape=None,
                scale=None,
                logits_g=None,
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Return output {sa_output} from MTLForSurvival task head")

        return sa_output

    def get_loss_weights(self):
        """
        Return the current loss weights for logging.

        This method is called by the LossWeightLoggerCallback during training.

        Returns:
            List of current loss weights
        """
        if hasattr(self, "loss_balancer"):
            return self.loss_balancer.get_weights()
        elif hasattr(self, "coeffs"):
            return self.coeffs.cpu().tolist()
        else:
            return None
