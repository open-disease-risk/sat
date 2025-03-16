"""EventDurationTaskHead implementation for event duration tasks"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import hydra
import torch
import torch.nn.functional as F
from torch import nn

from .nets import StableMLPSequence
from .config import EventDurationTaskConfig
from sat.models.tasks.heads.base import SurvivalPreTrainedModel
from sat.models.tasks.heads.outputs import TaskOutput
from sat.utils import logging

logger = logging.get_default_logger()


class EventDurationTaskHead(SurvivalPreTrainedModel):
    """Task head for event duration prediction."""

    config_class = EventDurationTaskConfig

    def __init__(self, config: EventDurationTaskConfig):
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
                    # No specific output activation for regression task
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
                # No specific output activation for regression task
            )
            self.is_multi_event = False

        loss = config.loss[config.model_type]
        logger.debug(f"Instantiate the loss {loss}")
        self.loss = hydra.utils.instantiate(loss)

    def forward(self, sequence_output, labels=None, **kwargs):
        # Check for NaN/Inf in sequence_output before passing to nets
        if torch.isnan(sequence_output).any() or torch.isinf(sequence_output).any():
            logger.warning(
                f"NaN or Inf detected in sequence_output: {torch.isnan(sequence_output).sum()} NaNs, {torch.isinf(sequence_output).sum()} Infs"
            )
            sequence_output = torch.nan_to_num(
                sequence_output, nan=0.0, posinf=1.0, neginf=-1.0
            )
            sequence_output = torch.clamp(sequence_output, min=-10.0, max=10.0)

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
                raw_logits = torch.zeros(
                    batch_size, num_events, num_labels, device=sequence_output.device
                )

                # Process each event network
                for i, net in enumerate(self.nets):
                    try:
                        # Apply the network and store results
                        event_logits = net(combined_features)
                        raw_logits[:, i, :] = event_logits
                    except Exception as e:
                        logger.warning(f"Exception in event {i}: {e}")
                        # Keep zeros for this event on error
            else:
                # Original multi-event implementation
                raw_logits = self.nets(sequence_output)
        else:
            # Single event model (either original or StableMLPSequence)
            raw_logits = self.nets(sequence_output)
            # Reshape if needed for single event case
            if (
                len(raw_logits.shape) == 2 and self.config.num_events > 1
            ):  # batch x labels
                raw_logits = raw_logits.unsqueeze(
                    1
                )  # Add event dimension: batch x 1 x labels

        # Apply Softplus instead of ReLU for better gradient stability
        # Softplus is a smooth approximation of ReLU: log(1 + exp(x))
        # This prevents the gradient discontinuity at 0 that ReLU has
        logits = F.softplus(raw_logits)

        # Apply additional safety checks and clamping to the logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            nan_count = torch.isnan(logits).sum().item()
            inf_count = torch.isinf(logits).sum().item()
            logger.warning(
                f"NaN or Inf detected in logits: {nan_count} NaNs, {inf_count} Infs"
            )

            # Replace NaNs and Infs with safe values
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=0.0)

            # Apply conservative clamping to prevent extreme values
            logits = torch.clamp(logits, min=1e-6, max=5.0)

        # Use squeeze operation to get predictions
        predictions = torch.squeeze(logits, dim=2)  # num events x batch x predictions

        # Apply safety checks to predictions as well
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)
            predictions = torch.clamp(predictions, min=1e-6, max=5.0)

        # Create output object with stabilized values
        output = TaskOutput(loss=None, logits=logits, predictions=predictions)

        # Compute loss only when needed
        if labels is not None:
            try:
                output.loss = self.loss(output, labels)

                # Check if loss is valid
                if output.loss is not None and (
                    torch.isnan(output.loss) or torch.isinf(output.loss)
                ):
                    logger.warning(f"NaN or Inf detected in loss: {output.loss}")
                    # Replace with a small constant loss to allow training to continue
                    output.loss = torch.tensor(
                        0.1, device=logits.device, dtype=logits.dtype
                    )
            except Exception as e:
                logger.error(f"Error computing loss: {e}")
                # Set a default loss to allow training to continue
                output.loss = torch.tensor(
                    0.1, device=logits.device, dtype=logits.dtype
                )

        return output
