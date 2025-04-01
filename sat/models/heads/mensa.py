"""MENSA Task Head for multi-event survival analysis"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

from sat.models.parameter_nets import MENSAParameterNet
from sat.utils import logging

from .base import SurvivalTask
from .output import SAOutput
from .survival import SurvivalConfig
from .utils import pad_col  # For padding hazard with zeros at the start

logger = logging.get_default_logger()


class MENSAConfig(SurvivalConfig):
    """Configuration for MENSA head."""

    model_type = "sat-mensa"  # Match the key used in loss config yaml

    def __init__(
        self,
        num_mixtures: int = 4,
        distribution: str = "weibull",
        event_dependency: bool = True,
        temp: float = 1000.0,
        discount: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mixtures = num_mixtures
        self.distribution = distribution
        self.event_dependency = event_dependency
        self.temp = temp  # Temperature for gumbel softmax
        self.discount = discount  # Discount factor for censoring loss


class MENSATaskHead(SurvivalTask):
    """
    Multi-Event Neural Survival Analysis (MENSA) implementation.

    MENSA models survival functions as mixtures of Weibull distributions
    with explicit dependencies between different event types.

    Based on the paper: "MENSA: Multi-Event Neural Survival Analysis" (2024)

    Key features:
    - Explicit modeling of dependencies between event types
    - SELU activations for more stable training
    - Weibull mixture model for flexible survival distributions
    - Specifically designed for multi-event scenarios
    """

    config_class = MENSAConfig

    def __init__(self, config: MENSAConfig):
        super().__init__(config)

        self.num_mixtures = config.num_mixtures
        self.distribution = config.distribution
        self.event_dependency = config.event_dependency
        self.temp = config.temp
        self.discount = config.discount

        # Always use MENSAParameterNet
        self.nets = MENSAParameterNet(
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
            event_dependency=self.event_dependency,
            activation=nn.SELU,  # MENSA uses SELU activation
        )

        # Initialize weights
        if self.__class__.__name__ == "MENSATaskHead":
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Standalone MENSATaskHead - initializing weights")
            self.post_init()
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "MENSATaskHead created as part of MTL - will be initialized by MTL"
                )

        # Instantiate loss function if available
        loss = config.loss["survival"]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Instantiate the loss {loss}")
        self.loss = hydra.utils.instantiate(loss)

    def _compute_survival_function(self, time_points, shape, scale, logits_g):
        """
        Compute survival function for given time points and distribution parameters.

        Args:
            time_points: Tensor of time points at which to evaluate survival [batch_size, num_time_points]
            shape: Shape parameters [batch_size, num_events, num_mixtures]
            scale: Scale parameters [batch_size, num_events, num_mixtures]
            logits_g: Mixture weight logits [batch_size, num_events, num_mixtures]

        Returns:
            Survival function values [batch_size, num_events, num_time_points]
        """
        batch_size, num_time_points = time_points.shape
        num_events = shape.size(1)
        device = time_points.device

        # Initialize survival tensor
        survival = torch.zeros(batch_size, num_events, num_time_points, device=device)

        # Process each event separately
        for event_idx in range(num_events):
            # Get parameters for this event
            event_shape = shape[:, event_idx, :]  # [batch_size, num_mixtures]
            event_scale = scale[:, event_idx, :]  # [batch_size, num_mixtures]
            event_logits_g = logits_g[:, event_idx, :]  # [batch_size, num_mixtures]

            # Compute mixture weights with gumbel softmax
            if self.training:
                weights = F.gumbel_softmax(
                    event_logits_g, tau=self.temp, hard=False, dim=1
                )
            else:
                weights = F.softmax(event_logits_g, dim=1)

            # Expand parameters for broadcasting
            # shape: [batch_size, num_mixtures, 1]
            shape_expanded = event_shape.unsqueeze(2)
            # scale: [batch_size, num_mixtures, 1]
            scale_expanded = event_scale.unsqueeze(2)
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
            event_survival = torch.sum(weights_expanded * survival_per_mixture, dim=1)

            # Store in the appropriate event slot
            survival[:, event_idx, :] = event_survival

        return survival

    def _compute_hazard_function(self, time_points, shape, scale, logits_g):
        """
        Compute hazard function from the survival function.

        Args:
            time_points: Tensor of time points [batch_size, num_time_points]
            shape: Shape parameters [batch_size, num_events, num_mixtures]
            scale: Scale parameters [batch_size, num_events, num_mixtures]
            logits_g: Mixture weight logits [batch_size, num_events, num_mixtures]

        Returns:
            Hazard function values [batch_size, num_events, num_time_points-1]
        """
        batch_size, num_time_points = time_points.shape
        num_events = shape.size(1)
        device = time_points.device

        # Compute survival function at each time point
        survival = self._compute_survival_function(time_points, shape, scale, logits_g)

        # Initialize hazard tensor
        hazard = torch.zeros(batch_size, num_events, num_time_points - 1, device=device)

        # Process each event separately
        for event_idx in range(num_events):
            # Get survival for this event
            event_survival = survival[:, event_idx, :]  # [batch_size, num_time_points]

            # Get S(t) and S(t+dt)
            survival_t = event_survival[:, :-1]  # S(t)
            survival_t_dt = event_survival[:, 1:]  # S(t+dt)

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
            event_hazard = F.softplus(discrete_hazard)

            # Store in the appropriate event slot
            hazard[:, event_idx, :] = event_hazard

        return hazard

    def forward(self, sequence_output, labels=None, **kwargs):
        """
        Forward pass for MENSA.

        Args:
            sequence_output: Output from transformer or features [batch_size, num_features]
            labels: Reference labels (optional) [batch_size, 4*num_events]
            **kwargs: Additional arguments

        Returns:
            SAOutput: Object containing survival, hazard, risk and other outputs
        """
        batch_size = sequence_output.shape[0]
        device = sequence_output.device

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"MENSA forward input shape: {sequence_output.shape}")

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
            time_points = (
                torch.linspace(0.1, 10.0, steps=20, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using fallback time points, shape: {time_points.shape}")

        # Get Weibull parameters from MENSA parameter network
        shape, scale, logits_g = self.nets(sequence_output)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Parameter shapes - shape: {shape.shape}, scale: {scale.shape}, logits_g: {logits_g.shape}"
            )

        # If using event dependency, store the dependency matrix for later analysis
        dependency_matrix = None
        if self.event_dependency and hasattr(self.nets, "event_dependency_matrix"):
            dependency_matrix = F.softmax(self.nets.event_dependency_matrix, dim=1)

        # Compute survival and hazard functions
        survival = self._compute_survival_function(time_points, shape, scale, logits_g)
        hazard = self._compute_hazard_function(time_points, shape, scale, logits_g)

        # Compute risk (1 - survival)
        risk = 1.0 - survival

        # Pad hazard with zeros at the start to match survival dimensions
        hazard = pad_col(hazard, val=0.0, where="start")

        # Create output container with complete set of fields
        output = SAOutput(
            loss=None,
            logits=torch.ones_like(
                survival
            ),  # Provide dummy logits to match the expected format
            hazard=hazard,
            risk=risk,
            survival=survival,
            hidden_states=sequence_output,
            attentions=None,
            time_to_event=None,
            event=None,
            shape=shape,  # [batch, events, num_mixtures]
            scale=scale,  # [batch, events, num_mixtures]
            logits_g=logits_g,  # [batch, events, num_mixtures]
            event_dependency_matrix=dependency_matrix,  # Store dependency matrix if available
        )

        # Compute loss if labels are provided
        if labels is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Computing MENSA loss with labels {labels.shape}")
            output.loss = self.loss(output, labels)

        return output
