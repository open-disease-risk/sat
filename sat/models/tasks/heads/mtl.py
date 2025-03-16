"""MTLForSurvival implementation for multi-task learning in survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import inspect
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
from typing import Optional, Tuple, Union

from . import embeddings
from .nets import StableMLPSequence
from .config import MTLConfig, SentenceEmbedding, TokenEmbedding
from sat.models.tasks.heads.base import SurvivalPreTrainedModel
from sat.models.tasks.heads.outputs import SAOutput
from sat.models.tasks.heads.survival_head import SurvivalTaskHead
from sat.models.tasks.heads.classification_head import EventClassificationTaskHead
from sat.models.tasks.heads.duration_head import EventDurationTaskHead
from sat.utils import logging

logger = logging.get_default_logger()


class MTLForSurvival(SurvivalPreTrainedModel):
    """Multi-task learning model for survival analysis.

    This model supports various task heads including survival analysis,
    event classification, and event duration prediction.
    """

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

        # Initialize the stable embedding processor
        self.embedding_processor = embeddings.StableEmbeddingProcessor(
            hidden_size=self.config.hidden_size,
            token_emb_type=self.config.token_emb,
            sentence_emb_type=self.config.sentence_emb,
            select_hidden_layers=self.config.select_hidden_layers,
            dropout_prob=self.config.hidden_dropout_prob,
            layer_norm=True,
        )

        # Calculate input features based on embedding processor configuration
        num_layers = self.transformer.config.num_hidden_layers
        self.in_features = self.embedding_processor.get_output_dim(num_layers)

        # Adjust for per-token output if not doing sentence pooling
        if (self.config.token_emb != TokenEmbedding.BERT.value) and (
            self.config.sentence_emb == SentenceEmbedding.NONE.value
        ):
            self.in_features = self.in_features * self.config.num_features

        logger.debug(f"Number of input features to the shared MLP: {self.in_features}")
        # Replace SimpleMLP with StableMLPSequence for consistency and better numerical stability
        # Use minimal architecture with a single hidden layer for shared representation
        self.net = StableMLPSequence(
            input_size=self.in_features,
            hidden_sizes=[
                self.config.intermediate_size
            ],  # Single hidden layer for shared representation
            output_size=self.config.num_labels,
            dropout_prob=(
                self.config.hidden_dropout_prob
                if self.config.hidden_dropout_prob
                else 0.1
            ),
            activation="gelu",
            bias=self.config.bias,
            use_residual=True,  # Enable residual connections for better gradient flow
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
        # Add support for gradient checkpointing to save memory during training
        use_gradient_checkpointing = (
            getattr(self.config, "gradient_checkpointing", False) and self.training
        )
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # Build transformer arguments more efficiently
        # Create a direct dictionary with only the needed parameters
        # This avoids unnecessary dictionary operations and list comprehensions
        forward_dict = {}
        for arg_name in self.forward_args:
            if (
                arg_name != "self"
                and arg_name in locals()
                and locals()[arg_name] is not None
            ):
                forward_dict[arg_name] = locals()[arg_name]

        logger.debug(
            f"Score the transformer with input_ids shape: {input_ids.shape if input_ids is not None else 'None'}"
        )

        # Use gradient checkpointing if enabled (saves memory during training)
        if use_gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # Extract input tensors in the order expected by the transformer
            inputs = [
                forward_dict.get(arg, None)
                for arg in self.forward_args
                if arg != "self"
            ]
            sequence_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.transformer),
                *inputs,
                use_reentrant=False,  # More memory efficient
            )
        else:
            # Regular forward pass without checkpointing
            sequence_output = self.transformer(
                **forward_dict, return_dict=self.return_dict
            )  # batch x sentences x words x embedding size

        logger.debug(f"Got {sequence_output} from transformer")

        # Process hidden states efficiently
        # layers x batches x tokens x features
        hidden_states = sequence_output.hidden_states

        # Pre-compute device for all tensors to ensure consistency
        device = hidden_states[0].device
        dtype = hidden_states[0].dtype

        # Process embeddings using the stable embedding processor
        sentence_embeddings = self.embedding_processor(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            pooler_output=sequence_output[1] if len(sequence_output) > 1 else None,
        )

        logger.debug(f"Dimensions of final embeddings {sentence_embeddings.shape}")

        # Compute net logits efficiently
        batch_size = sentence_embeddings.shape[0]

        # Reshape more efficiently - use view instead of reshape when possible
        # view() is more efficient when tensor is already contiguous
        if sentence_embeddings.is_contiguous():
            flattened_embeddings = sentence_embeddings.view(batch_size, -1)
        else:
            flattened_embeddings = sentence_embeddings.reshape(batch_size, -1)

        # Forward pass through the network
        logits = self.net(flattened_embeddings)
        logger.debug("Score the task heads")

        # Pre-allocate all needed variables
        sa_output = None
        outputs = (
            {}
        )  # Use dict for task outputs - more efficient than dynamic attribute lookup
        tte = None
        event = None
        loss = 0.0

        # Pre-allocate for logits_tasks to avoid repeated list append operations
        num_heads = len(self.heads)
        logits_tasks = [None] * num_heads

        # Process all task heads efficiently
        for i, h in enumerate(self.heads):
            logger.debug(f"Score {i}-th task head")

            # Process each head with logits and labels
            curr_output = h(logits, labels)
            outputs[i] = curr_output  # Store output by index

            # Store the logits directly in pre-allocated list
            logits_tasks[i] = curr_output.logits

            # Accumulate loss if labels provided - avoid repeated condition checks
            if labels is not None:
                loss += curr_output.loss * h.config.loss_weight

            # Determine head type once and store output data accordingly
            if isinstance(h, SurvivalTaskHead):
                sa_output = curr_output
            elif isinstance(h, EventDurationTaskHead):
                tte = curr_output.predictions
            elif isinstance(h, EventClassificationTaskHead):
                event = curr_output.predictions

        # Prepare common output parameters for both conditions
        output_params = {
            "loss": loss,
            "logits": logits_tasks,
            "time_to_event": tte,
            "event": event,
            "hidden_states": sequence_output.hidden_states,
            "attentions": sequence_output.attentions,
        }

        # Add survival-specific parameters only if available (avoids redundant code)
        if sa_output is not None:
            output_params.update(
                {
                    "hazard": sa_output.hazard,
                    "risk": sa_output.risk,
                    "survival": sa_output.survival,
                }
            )
        else:
            output_params.update(
                {
                    "hazard": None,
                    "risk": None,
                    "survival": None,
                }
            )

        # Create final output object
        sa_output = SAOutput(**output_params)

        logger.debug(f"Return output {sa_output} from MTLForSurvival task head")
        return sa_output
