"""MENSA Task Head for multi-event survival analysis"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

from sat.models.parameter_nets import MENSAParameterNet
from sat.utils import logging
from sat.distributions import WeibullDistribution, LogNormalDistribution
from sat.distributions import WeibullMixtureDistribution, LogNormalMixtureDistribution

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
        use_expert_priors: bool = False,
        constrain_params: bool = False,
        event_types: list = None,
        dependency_regularization: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mixtures = num_mixtures
        self.distribution = distribution
        self.event_dependency = event_dependency
        self.temp = temp  # Temperature for gumbel softmax
        self.discount = discount  # Discount factor for censoring loss
        self.use_expert_priors = use_expert_priors  # Whether to use expert knowledge
        self.constrain_params = (
            constrain_params  # Whether to apply parameter constraints
        )
        self.event_types = event_types  # Optional list of event type names
        self.dependency_regularization = (
            dependency_regularization  # Weight for dependency regularization
        )


class MENSATaskHead(SurvivalTask):
    """
    Multi-Event Neural Survival Analysis (MENSA) implementation.

    MENSA models survival functions as mixtures of parametric distributions
    with explicit dependencies between different event types.

    Based on the paper: "MENSA: Multi-Event Neural Survival Analysis" (2024)

    Key features:
    - Explicit modeling of dependencies between event types
    - SELU activations for more stable training
    - Weibull or LogNormal mixture models for flexible survival distributions
    - Specifically designed for multi-event scenarios
    - Support for expert knowledge incorporation via informative priors and constraints

    Expert knowledge references:
    - Crowder, M. J. (2001). "Classical Competing Risks," Chapman and Hall/CRC, pp. 75-108.
    - Kleinbaum & Klein (2012). "Survival Analysis: A Self-Learning Text," Springer, Chapter 9.
    - Ibrahim, Chen & Sinha (2001). "Bayesian Survival Analysis," Springer, Chapter 3.4 (Prior Elicitation).
    - Christensen et al. (2011). "Bayesian Ideas and Data Analysis," CRC Press, Chapter 11.
    """

    config_class = MENSAConfig

    def __init__(self, config: MENSAConfig):
        super().__init__(config)

        self.num_mixtures = config.num_mixtures
        self.distribution = config.distribution
        self.event_dependency = config.event_dependency
        self.temp = config.temp
        self.discount = config.discount
        self.use_expert_priors = config.use_expert_priors
        self.constrain_params = config.constrain_params
        self.event_types = config.event_types
        self.dependency_regularization = config.dependency_regularization

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
        Uses distribution classes for numerical stability with expert knowledge.

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

            # Process logits differently for training vs inference
            if self.training:
                # Apply gumbel softmax to create differentiable one-hot vectors
                processed_logits = F.gumbel_softmax(
                    event_logits_g, tau=self.temp, hard=False, dim=1
                )
            else:
                # Just pass through the original logits for inference
                processed_logits = event_logits_g

            # Extract event type if available for the specified event
            event_type = None
            if self.event_types is not None and event_idx < len(self.event_types):
                event_type = self.event_types[event_idx]

            # Create the appropriate mixture distribution based on distribution type
            if self.distribution.lower() == "weibull":
                mixture_dist = WeibullMixtureDistribution(
                    event_shape,
                    event_scale,
                    processed_logits,
                    constrain_shape=self.constrain_params,
                    event_type=event_type,
                    use_expert_priors=self.use_expert_priors,
                )
            elif self.distribution.lower() == "lognormal":
                mixture_dist = LogNormalMixtureDistribution(
                    event_scale,  # For LogNormal, this is loc
                    event_shape,  # For LogNormal, this is scale
                    processed_logits,
                    constrain_params=self.constrain_params,
                    event_type=event_type,
                    use_expert_priors=self.use_expert_priors,
                )
            else:
                raise ValueError(f"Unsupported distribution: {self.distribution}")

            # Compute survival function using the mixture distribution
            event_survival = mixture_dist.survival_function(time_points)

            # Store in the appropriate event slot
            survival[:, event_idx, :] = event_survival

        return survival

    def _compute_hazard_function(self, time_points, shape, scale, logits_g):
        """
        Compute hazard function from the survival function.
        Uses distribution classes for numerical stability with expert knowledge.

        Args:
            time_points: Tensor of time points [batch_size, num_time_points]
            shape: Shape parameters [batch_size, num_events, num_mixtures]
            scale: Scale parameters [batch_size, num_events, num_mixtures]
            logits_g: Mixture weight logits [batch_size, num_events, num_mixtures]

        Returns:
            Hazard function values [batch_size, num_events, num_time_points]
        """
        batch_size, num_time_points = time_points.shape
        num_events = shape.size(1)
        device = time_points.device

        # Initialize hazard tensor - use same dimensions as time_points for consistent shape
        hazard = torch.zeros(batch_size, num_events, num_time_points, device=device)

        # Process each event separately
        for event_idx in range(num_events):
            # Get parameters for this event
            event_shape = shape[:, event_idx, :]  # [batch_size, num_mixtures]
            event_scale = scale[:, event_idx, :]  # [batch_size, num_mixtures]
            event_logits_g = logits_g[:, event_idx, :]  # [batch_size, num_mixtures]

            # Process logits differently for training vs inference
            if self.training:
                # Apply gumbel softmax to create differentiable one-hot vectors
                processed_logits = F.gumbel_softmax(
                    event_logits_g, tau=self.temp, hard=False, dim=1
                )
            else:
                # Just pass through the original logits for inference
                processed_logits = event_logits_g

            # Extract event type if available for the specified event
            event_type = None
            if self.event_types is not None and event_idx < len(self.event_types):
                event_type = self.event_types[event_idx]

            # Create the appropriate mixture distribution based on distribution type
            if self.distribution.lower() == "weibull":
                mixture_dist = WeibullMixtureDistribution(
                    event_shape,
                    event_scale,
                    processed_logits,
                    constrain_shape=self.constrain_params,
                    event_type=event_type,
                    use_expert_priors=self.use_expert_priors,
                )
            elif self.distribution.lower() == "lognormal":
                mixture_dist = LogNormalMixtureDistribution(
                    event_scale,  # For LogNormal, this is loc
                    event_shape,  # For LogNormal, this is scale
                    processed_logits,
                    constrain_params=self.constrain_params,
                    event_type=event_type,
                    use_expert_priors=self.use_expert_priors,
                )
            else:
                raise ValueError(f"Unsupported distribution: {self.distribution}")

            # Compute hazard function using the mixture distribution directly
            event_hazard = mixture_dist.hazard_function(time_points)

            # Apply softplus to ensure non-negative hazard and clamp extreme values
            event_hazard = F.softplus(event_hazard)
            event_hazard = torch.clamp(event_hazard, max=1e3)

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
        if hasattr(self.loss, "duration_cuts") and self.loss.duration_cuts is not None:
            # Ensure duration cuts are positive and sorted
            eps = 1e-7
            duration_cuts = self.loss.duration_cuts.to(device)

            # Add small epsilon to ensure strictly positive values
            if duration_cuts.min() <= 0:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Found non-positive values in duration cuts, adding epsilon"
                    )
                duration_cuts = torch.clamp(duration_cuts, min=eps)

            time_points = duration_cuts.unsqueeze(0).expand(batch_size, -1)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Using duration cuts from loss, shape: {time_points.shape}, range: [{time_points.min():.5f}, {time_points.max():.5f}]"
                )
        else:
            # If duration cuts aren't available, create a reasonable range
            time_points = (
                torch.linspace(0.1, 10.0, steps=20, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Using fallback time points, shape: {time_points.shape}, range: [{time_points.min():.5f}, {time_points.max():.5f}]"
                )

        # Get distribution parameters from MENSA parameter network
        shape, scale, logits_g = self.nets(sequence_output)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Parameter shapes - shape: {shape.shape}, scale: {scale.shape}, logits_g: {logits_g.shape}"
            )

        # If using event dependency, store the dependency matrix for later analysis
        dependency_matrix = None
        if self.event_dependency and hasattr(self.nets, "event_dependency_matrix"):
            dependency_matrix = F.softmax(self.nets.event_dependency_matrix, dim=1)

        # Ensure time points have good numeric properties
        # We need to ensure the first time point is strictly positive
        if time_points.shape[1] > 0 and time_points[:, 0].min() < 1e-5:
            # If first time point is very close to zero, replace it with a small value
            # This avoids numerical issues at t=0 for hazard calculations
            safe_time_points = time_points.clone()
            safe_time_points[:, 0] = torch.max(
                safe_time_points[:, 0], torch.tensor(1e-5, device=device)
            )
        else:
            safe_time_points = time_points

        # Special case: if any time point is too close to another one, add a small offset
        # This prevents division issues in hazard calculation
        if safe_time_points.shape[1] > 1:
            time_diffs = safe_time_points[:, 1:] - safe_time_points[:, :-1]
            min_diff = 1e-6
            for i in range(1, safe_time_points.shape[1]):
                if (time_diffs[:, i - 1] < min_diff).any():
                    safe_time_points[:, i] = torch.max(
                        safe_time_points[:, i], safe_time_points[:, i - 1] + min_diff
                    )

        # Log time points for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Time points after safety adjustment: range [{safe_time_points.min():.8f}, {safe_time_points.max():.8f}]"
            )

        # Compute survival and hazard functions with improved numerical stability
        survival = self._compute_survival_function(
            safe_time_points, shape, scale, logits_g
        )

        # Compute hazard using the same time points as survival to ensure consistent dimensions
        hazard = self._compute_hazard_function(safe_time_points, shape, scale, logits_g)

        # Compute risk (1 - survival)
        risk = 1.0 - survival

        # Ensure survival, hazard, and risk all have the same dimensions
        if hazard.shape != survival.shape:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    f"Dimension mismatch: hazard {hazard.shape}, survival {survival.shape}"
                )
            # Create new hazard tensor with correct shape
            correct_hazard = torch.zeros_like(survival)
            # Copy as much as possible from original hazard
            min_time_dim = min(hazard.shape[2], survival.shape[2])
            correct_hazard[:, :, :min_time_dim] = hazard[:, :, :min_time_dim]
            hazard = correct_hazard

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
            event_types=self.event_types,  # Pass event types for expert knowledge
            use_expert_priors=self.use_expert_priors,  # Pass expert priors flag
            constrain_params=self.constrain_params,  # Pass parameter constraint flag
        )

        # Compute loss if labels are provided
        if labels is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Computing MENSA loss with labels {labels.shape}")
            output.loss = self.loss(output, labels)

        return output