"""EventClassificationTaskHead implementation for event classification tasks"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import hydra
import torch
from torch import nn

from .nets import StableMLPSequence
from .config import EventClassificationTaskConfig
from sat.models.tasks.heads.base import SurvivalPreTrainedModel
from sat.models.tasks.heads.outputs import TaskOutput
from sat.utils import logging

logger = logging.get_default_logger()


class EventClassificationTaskHead(SurvivalPreTrainedModel):
    """Task head for event classification prediction."""

    config_class = EventClassificationTaskConfig

    def __init__(self, config: EventClassificationTaskConfig):
        super().__init__(config)

        # Always use StableMLPSequence for improved numerical stability
        logger.info("Using StableMLPSequence for improved numerical stability")

        if self.config.num_events > 1:
            # Create a module list of event-specific MLPs
            self.nets = nn.ModuleList()

            # Shared network for all events
            self.shared_net = StableMLPSequence(
                input_size=self.config.num_features,
                hidden_sizes=[self.config.intermediate_size]
                * self.config.num_hidden_layers,
                output_size=self.config.intermediate_size,
                dropout_prob=(
                    self.config.hidden_dropout_prob
                    if self.config.hidden_dropout_prob
                    else 0.1
                ),
                activation="gelu",
                bias=self.config.bias,
                use_residual=True,
            )

            # Individual networks for each event
            for _ in range(self.config.num_events):
                net = StableMLPSequence(
                    input_size=self.config.num_features + self.config.intermediate_size,
                    hidden_sizes=[self.config.indiv_intermediate_size]
                    * self.config.indiv_num_hidden_layers,
                    output_size=self.config.num_labels,
                    dropout_prob=(
                        self.config.hidden_dropout_prob
                        if self.config.hidden_dropout_prob
                        else 0.1
                    ),
                    activation="gelu",
                    bias=self.config.bias,
                    output_activation="sigmoid",
                )
                self.nets.append(net)

            # Flag to identify as multi-event
            self.is_multi_event = True
        else:
            # Single event model
            self.nets = StableMLPSequence(
                input_size=self.config.num_features,
                hidden_sizes=[self.config.intermediate_size]
                * self.config.num_hidden_layers,
                output_size=self.config.num_labels,
                dropout_prob=(
                    self.config.hidden_dropout_prob
                    if self.config.hidden_dropout_prob
                    else 0.1
                ),
                activation="gelu",
                bias=self.config.bias,
                output_activation="sigmoid",  # For classification task
            )
            self.is_multi_event = False

        loss = config.loss[config.model_type]
        logger.debug(f"Instantiate the loss {loss}")
        self.loss = hydra.utils.instantiate(loss)

    def forward(self, sequence_output, labels=None, **kwargs):
        # Different handling for StableMLPSequence vs original implementation
        if hasattr(self, "is_multi_event") and self.is_multi_event:
            if hasattr(self, "shared_net"):
                # This is the StableMLPSequence implementation for multi-event
                # First get the shared features
                shared_features = self.shared_net(sequence_output)

                # Concatenate original features with shared features
                combined_features = torch.cat([sequence_output, shared_features], dim=1)

                # Apply each event-specific network and collect outputs
                batch_size = sequence_output.size(0)
                num_events = len(self.nets)
                num_labels = self.config.num_labels

                # Initialize tensor to store all event outputs
                logits = torch.zeros(
                    batch_size, num_events, num_labels, device=sequence_output.device
                )

                # Process each event network
                for i, net in enumerate(self.nets):
                    try:
                        # Apply the network and store results
                        event_logits = net(combined_features)
                        logits[:, i, :] = event_logits
                    except Exception as e:
                        logger.warning(f"Exception in event {i}: {e}")
                        # Keep zeros for this event on error
            else:
                # Original multi-event implementation
                logits = self.nets(sequence_output)
        else:
            # Single event model (either original or StableMLPSequence)
            logits = self.nets(sequence_output)
            # Reshape if needed for single event case
            if len(logits.shape) == 2 and self.config.num_events > 1:  # batch x labels
                logits = logits.unsqueeze(1)  # Add event dimension: batch x 1 x labels

        # Handle any NaN values in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.warning(
                f"NaN or Inf detected in logits: {torch.isnan(logits).sum()} NaNs, {torch.isinf(logits).sum()} Infs"
            )
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

        # Generate predictions with sigmoid
        predictions = torch.sigmoid(logits)

        # Create output object once
        output = TaskOutput(loss=None, logits=logits, predictions=predictions)

        # Calculate loss only if labels provided
        if labels is not None:
            # Apply loss with our updated implementation supporting loss balancing API
            output.loss = self.loss(output, labels)

        return output
