"""Deep Survival Machines Task Head for survival analysis"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

from sat.models.nets import MLP, CauseSpecificNet, CauseSpecificNetCompRisk
from sat.models.parameter_nets import (
    ParamCauseSpecificNet,
    ParamCauseSpecificNetCompRisk,
)
from sat.utils import logging

from .survival import SurvivalConfig
from .base import SurvivalTask
from .output import SAOutput  # Using SAOutput directly instead of a custom DSMOutput
from .utils import pad_col  # For padding hazard with zeros at the start

logger = logging.get_default_logger()


class DSMConfig(SurvivalConfig):
    """Configuration for Deep Survival Machines head."""

    model_type = "sat-transformer"  # Match the key used in loss config yaml

    def __init__(
        self,
        num_mixtures: int = 4,
        distribution: str = "weibull",
        temp: float = 1000.0,
        discount: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mixtures = num_mixtures
        self.distribution = distribution
        self.temp = temp  # Temperature for gumbel softmax
        self.discount = discount  # Discount factor for censoring loss


class DSMTaskHead(SurvivalTask):
    """
    Deep Survival Machines implementation for survival analysis.

    DSM models the survival function as a mixture of parametric distributions,
    typically Weibull distributions. It uses a neural network to learn the parameters
    of these distributions from covariates.

    Key features:
    - Mixture of distributions for flexibility in modeling survival times
    - Learns latent representations from covariates
    - Handles right-censoring through careful loss formulation
    - Can model competing risks with multiple event types
    """

    config_class = DSMConfig

    def __init__(self, config: DSMConfig):
        super().__init__(config)

        self.num_mixtures = config.num_mixtures
        self.distribution = config.distribution
        self.temp = config.temp
        self.discount = config.discount

        # Following the SurvivalTaskHead pattern:
        # - Use ParamCauseSpecificNet for single event
        # - Use ParamCauseSpecificNetCompRisk for multiple events
        if self.config.num_events > 1:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Using ParamCauseSpecificNetCompRisk for {self.config.num_events} events"
                )

            # For multi-event, use the CompRisk parameter network
            self.nets = ParamCauseSpecificNetCompRisk(
                in_features=self.config.num_features,
                shared_intermediate_size=self.config.intermediate_size,
                shared_num_hidden_layers=self.config.num_hidden_layers,
                indiv_intermediate_size=self.config.indiv_intermediate_size,
                indiv_num_hidden_layers=self.config.indiv_num_hidden_layers,
                bias=self.config.bias,
                batch_norm=self.config.batch_norm,
                dropout=self.config.hidden_dropout_prob,
                num_mixtures=self.num_mixtures,
                num_events=self.config.num_events,
            )
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Using ParamCauseSpecificNet for single event")

            # For single event, use the simpler parameter network
            self.nets = ParamCauseSpecificNet(
                in_features=self.config.num_features,
                intermediate_size=self.config.intermediate_size,
                num_hidden_layers=self.config.num_hidden_layers,
                bias=self.config.bias,
                batch_norm=self.config.batch_norm,
                dropout=self.config.hidden_dropout_prob,
                num_mixtures=self.num_mixtures,
                num_events=self.config.num_events,
            )

        # Initialize weights
        if self.__class__.__name__ == "DSMTaskHead":
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Standalone DSMTaskHead - initializing weights")
            self.post_init()
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "DSMTaskHead created as part of MTL - will be initialized by MTL"
                )

        # Instantiate loss function if available
        self.loss = None
        if hasattr(config, "loss") and config.model_type in config.loss:
            loss = config.loss[config.model_type]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Instantiate the loss {loss}")
            try:
                self.loss = hydra.utils.instantiate(loss)
            except Exception as e:
                logger.error(f"Failed to instantiate loss: {str(e)}")
                logger.warning("Continuing without loss function")

    def _compute_survival_function(self, time_points, shape, scale, logits_g):
        """
        Compute survival function for given time points and distribution parameters.

        Args:
            time_points: Tensor of time points at which to evaluate survival [batch_size, num_time_points]
            shape: Shape parameters [batch_size, num_mixtures]
            scale: Scale parameters [batch_size, num_mixtures]
            logits_g: Mixture weight logits [batch_size, num_mixtures]

        Returns:
            Survival function values [batch_size, num_time_points]
        """
        batch_size, num_time_points = time_points.shape
        device = time_points.device

        # Compute mixture weights with gumbel softmax
        if self.training:
            weights = F.gumbel_softmax(logits_g, tau=self.temp, hard=False, dim=1)
        else:
            weights = F.softmax(logits_g, dim=1)

        # Create output tensor
        survival = torch.zeros(batch_size, num_time_points, device=device)

        # Expand parameters for broadcasting
        # shape: [batch_size, num_mixtures, 1]
        shape_expanded = shape.unsqueeze(2)
        # scale: [batch_size, num_mixtures, 1]
        scale_expanded = scale.unsqueeze(2)
        # weights: [batch_size, num_mixtures, 1]
        weights_expanded = weights.unsqueeze(2)
        # time_points: [batch_size, 1, num_time_points]
        time_expanded = time_points.unsqueeze(1)

        if self.distribution.lower() == "weibull":
            # Weibull survival function: exp(-(t/scale)^shape)
            survival_per_mixture = torch.exp(
                -torch.pow(time_expanded / scale_expanded, shape_expanded)
            )
        elif self.distribution.lower() == "lognormal":
            # Log-normal survival function: 1 - CDF of normal distribution
            z = (torch.log(time_expanded) - scale_expanded) / shape_expanded
            survival_per_mixture = 0.5 - 0.5 * torch.erf(
                z / torch.sqrt(torch.tensor(2.0, device=device))
            )
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

        # Compute weighted sum of individual distributions
        # [batch_size, num_mixtures, num_time_points] -> [batch_size, num_time_points]
        survival = torch.sum(weights_expanded * survival_per_mixture, dim=1)

        return survival

    def _compute_hazard_function(self, time_points, shape, scale, logits_g):
        """
        Compute hazard function from the survival function.

        Args:
            time_points: Tensor of time points [batch_size, num_time_points]
            shape: Shape parameters [batch_size, num_mixtures]
            scale: Scale parameters [batch_size, num_mixtures]
            logits_g: Mixture weight logits [batch_size, num_mixtures]

        Returns:
            Hazard function values [batch_size, num_time_points-1]
        """
        batch_size, num_time_points = time_points.shape
        device = time_points.device

        # Compute survival function at each time point
        survival = self._compute_survival_function(time_points, shape, scale, logits_g)

        # Calculate hazard: -d/dt log(S(t)) ≈ -(log(S(t+dt)) - log(S(t))) / dt
        # We take log(S(t+dt)) - log(S(t)) instead of S(t+dt) - S(t) to stabilize computation
        # This approximates instantaneous hazard by discrete hazard over small intervals

        # Get S(t) and S(t+dt)
        survival_t = survival[:, :-1]  # S(t)
        survival_t_dt = survival[:, 1:]  # S(t+dt)

        # Time differences for each interval
        dt = time_points[:, 1:] - time_points[:, :-1]

        # Avoid division by zero or negative values
        eps = 1e-7
        survival_t = torch.clamp(survival_t, min=eps)
        survival_t_dt = torch.clamp(survival_t_dt, min=eps)

        # Compute log survival
        log_surv_t = torch.log(survival_t)
        log_surv_t_dt = torch.log(survival_t_dt)

        # Compute discrete hazard
        discrete_hazard = -(log_surv_t_dt - log_surv_t) / dt

        # Ensure non-negative hazard
        hazard = F.softplus(discrete_hazard)

        return hazard

    def forward(self, sequence_output, labels=None, **kwargs):
        batch_size = sequence_output.shape[0]
        device = sequence_output.device

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DSM forward input shape: {sequence_output.shape}")

        # Generate time points for evaluating the survival function
        # These should match the duration cuts used in training
        if hasattr(self.loss, "duration_cuts"):
            time_points = (
                self.loss.duration_cuts.to(device).unsqueeze(0).expand(batch_size, -1)
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Using duration cuts from loss, shape: {time_points.shape}"
                )
        else:
            # If duration cuts aren't available, create a reasonable range
            # This is just a fallback - ideally we'd use the actual cuts
            time_points = (
                torch.linspace(0.1, 10.0, steps=20, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using fallback time points, shape: {time_points.shape}")

        # Compute parameters using our specialized parameter networks
        shape, scale, logits_g = self.nets(sequence_output)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Parameter shapes - shape: {shape.shape}, scale: {scale.shape}, logits_g: {logits_g.shape}"
            )

        # Results for all events
        all_survivals = []
        all_hazards = []
        all_risks = []

        # For multi-event case, process each event separately
        # For single-event case, we have shape [batch, 1, num_mixtures] so will process once
        num_events = shape.size(1)

        for event_idx in range(num_events):
            # Extract parameters for this event
            event_shape = shape[:, event_idx, :]
            event_scale = scale[:, event_idx, :]
            event_logits_g = logits_g[:, event_idx, :]

            # Compute survival and hazard functions
            survival = self._compute_survival_function(
                time_points, event_shape, event_scale, event_logits_g
            )
            hazard = self._compute_hazard_function(
                time_points, event_shape, event_scale, event_logits_g
            )

            # Compute risk (1 - survival)
            risk = 1.0 - survival

            # Store results for this event
            all_survivals.append(survival)
            all_hazards.append(hazard)
            all_risks.append(risk)

        # Stack along event dimension [batch, events, ...]
        stacked_survival = torch.stack(all_survivals, dim=1)
        stacked_hazard = torch.stack(all_hazards, dim=1)
        # Pad hazard with zeros at the start to match survival dimensions
        stacked_hazard = pad_col(stacked_hazard, val=0.0, where="start")
        stacked_risk = torch.stack(all_risks, dim=1)

        # Create output container with complete set of fields
        output = SAOutput(
            loss=None,
            logits=torch.ones_like(
                stacked_survival
            ),  # Provide dummy logits to match the expected format
            hazard=stacked_hazard,
            risk=stacked_risk,
            survival=stacked_survival,
            hidden_states=sequence_output,
            attentions=None,
            time_to_event=None,
            event=None,
            shape=shape,  # Already has the right shape [batch, events, num_mixtures]
            scale=scale,
            logits_g=logits_g,
        )

        # Compute loss if labels are provided
        if labels is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Computing DSM loss with labels {labels.shape}")
            output.loss = self.loss(output, labels)

        return output
