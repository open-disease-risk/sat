"""SurvivalTaskHead implementation for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import hydra
import torch
import torch.nn.functional as F
from torch import nn

from .nets import StableMLPSequence, CauseSpecificNet, CauseSpecificNetCompRisk
from .config import SurvivalConfig
from sat.models.tasks.heads.base import SurvivalTask, pad_col
from sat.models.tasks.heads.outputs import SAOutput
from sat.utils import logging

logger = logging.get_default_logger()


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


class SurvivalTaskHeadNew(SurvivalTask):
    """Task head for survival analysis with proper handling of multi-event cases."""

    config_class = SurvivalConfig

    def __init__(self, config: SurvivalConfig):
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
                    output_activation="softplus",
                )
                self.nets.append(net)

            # Flag to identify as multi-event
            self.is_multi_event = True
        else:
            # Single event model
            self.nets = StableMLPSequence(
                input_size=self.config.num_features,
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
                output_activation="softplus",  # Ensure positive outputs for hazard function
            )
            self.is_multi_event = False

        # Initialize weights and apply final processing
        logger.debug("Post initialize in SurvivalTaskHead")
        self.post_init()

        loss = config.loss[config.model_type]
        logger.debug(f"Instantiate the loss {loss}")
        self.loss = hydra.utils.instantiate(loss)

    def forward(self, sequence_output, labels=None, **kwargs):
        # Different handling for StableMLPSequence vs original implementation
        if hasattr(self, "is_multi_event") and self.is_multi_event:
            if hasattr(self, "shared_net"):
                if (
                    torch.isnan(sequence_output).any()
                    or torch.isinf(sequence_output).any()
                ):
                    logger.warning(
                        f"NaN or Inf detected in sequence_output: {torch.isnan(sequence_output).sum()} NaNs, {torch.isinf(sequence_output).sum()} Infs"
                    )

                # This is the StableMLPSequence implementation for multi-event
                # First get the shared features
                shared_features = self.shared_net(sequence_output)
                if (
                    torch.isnan(shared_features).any()
                    or torch.isinf(shared_features).any()
                ):
                    logger.warning(
                        f"NaN or Inf detected in shared_features: {torch.isnan(shared_features).sum()} NaNs, {torch.isinf(shared_features).sum()} Infs"
                    )

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
                        if (
                            torch.isnan(event_logits).any()
                            or torch.isinf(event_logits).any()
                        ):
                            logger.warning(
                                f"NaN or Inf detected in event_logits: {torch.isnan(event_logits).sum()} NaNs, {torch.isinf(event_logits).sum()} Infs"
                            )
                        logits[:, i, :] = event_logits
                    except Exception as e:
                        logger.warning(f"Exception in event {i}: {e}")
                        # Keep zeros for this event on error
            else:
                # Original multi-event implementation
                logits = self.nets(sequence_output)  # batch x events x durations
        else:
            # Single event model (either original or StableMLPSequence)
            logits = self.nets(sequence_output)
            # Reshape if needed for single event case
            if len(logits.shape) == 2:  # batch x durations
                logits = logits.unsqueeze(
                    1
                )  # Add event dimension: batch x 1 x durations

        # Check if there are any NaN or Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.warning(
                f"NaN or Inf detected in logits: {torch.isnan(logits).sum()} NaNs, {torch.isinf(logits).sum()} Infs"
            )
            # Clean up NaN and Inf values
            # logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

        # Clamp logits to prevent extreme values
        # logits = torch.clamp(logits, min=-15.0, max=15.0)

        # Apply softplus activation with stability
        hazard = F.softplus(logits)

        # Add padding column more efficiently
        hazard = pad_col(hazard, where="start")

        # Check for any issues in hazard
        if torch.isnan(hazard).any() or torch.isinf(hazard).any():
            logger.warning(
                f"NaN or Inf detected in hazard after softplus: {torch.isnan(hazard).sum()} NaNs, {torch.isinf(hazard).sum()} Infs"
            )
            hazard = torch.nan_to_num(hazard, nan=1e-6, posinf=10.0, neginf=1e-6)

        # Compute cumulative sum with explicit clamping to prevent overflow
        cum_hazard = hazard.cumsum(dim=2)
        # cum_hazard = torch.clamp(
        #    cum_hazard, min=0.0, max=20.0
        # )  # Prevent extreme values before negation

        # Negate and compute survival probabilities with careful exp
        neg_cum_hazard = -cum_hazard  # Separate operation for clarity

        # Apply exponentiation with safety measures
        surv = torch.exp(neg_cum_hazard)  # Use non-inplace version for better stability

        # Clamp survival probabilities to valid range
        surv = torch.clamp(surv, min=1e-7, max=1.0)  # Ensure valid probability range

        # Compute risk safely
        # risk = torch.clamp(1.0 - surv, min=1e-7, max=1.0)
        risk = 1.0 - surv

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
