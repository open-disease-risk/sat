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
from sat.distributions import WeibullDistribution, LogNormalDistribution
from sat.distributions import WeibullMixtureDistribution, LogNormalMixtureDistribution

from .base import SurvivalTask
from .output import SAOutput  # Using SAOutput directly instead of a custom DSMOutput
from .survival import SurvivalConfig
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
        use_expert_priors: bool = False,
        constrain_params: bool = False,
        event_types: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mixtures = num_mixtures
        self.distribution = distribution
        self.temp = temp  # Temperature for gumbel softmax
        self.discount = discount  # Discount factor for censoring loss
        self.use_expert_priors = use_expert_priors  # Whether to use expert knowledge
        self.constrain_params = (
            constrain_params  # Whether to apply parameter constraints
        )
        self.event_types = event_types  # Optional list of event type names


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
    - Support for expert knowledge incorporation via informative priors and constraints

    Original DSM paper:
    - Nagpal et al. (2021). "Deep Survival Machines: Fully Parametric Survival Regression and Representation Learning for Censored Data with Competing Risks." IEEE Journal of Biomedical and Health Informatics, 25(8), 3163-3175.

    Expert knowledge references:
    - Cox, C. (2008). "The generalized F distribution: An umbrella for parametric survival analysis." Statistics in Medicine, 27(21), 4301-4312.
    - Collett, D. (2015). "Modelling Survival Data in Medical Research," Chapman and Hall/CRC, Chapter 6.
    - Ibrahim, Chen & Sinha (2001). "Bayesian Survival Analysis," Springer, Chapter 3.4 (Prior Elicitation).
    - Lambert, Collett, Kimber & Johnson (2004). "Parametric accelerated failure time models with random effects and an application to kidney transplant survival." Statistics in Medicine, 23(20), 3177-3192.
    """

    config_class = DSMConfig

    def __init__(self, config: DSMConfig):
        super().__init__(config)

        self.num_mixtures = config.num_mixtures
        self.distribution = config.distribution
        self.temp = config.temp
        self.discount = config.discount
        self.use_expert_priors = config.use_expert_priors
        self.constrain_params = config.constrain_params
        self.event_types = config.event_types

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
        loss = config.loss["survival"]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Instantiate the loss {loss}")
        self.loss = hydra.utils.instantiate(loss)

    def _compute_survival_function(
        self, time_points, shape, scale, logits_g, event_idx=0
    ):
        """
        Compute survival function for given time points and distribution parameters.
        Uses distribution classes for numerical stability.

        Args:
            time_points: Tensor of time points at which to evaluate survival [batch_size, num_time_points]
            shape: Shape parameters [batch_size, num_mixtures]
            scale: Scale parameters [batch_size, num_mixtures]
            logits_g: Mixture weight logits [batch_size, num_mixtures]
            event_idx: Optional event index for event-specific priors

        Returns:
            Survival function values [batch_size, num_time_points]
        """
        batch_size, num_time_points = time_points.shape

        # For training, use gumbel softmax for the mixture weights
        if self.training:
            # Apply gumbel softmax to create differentiable one-hot vectors
            processed_logits = F.gumbel_softmax(
                logits_g, tau=self.temp, hard=False, dim=1
            )
        else:
            # Just pass through the original logits for inference
            processed_logits = logits_g

        # Extract event type if available for the specified event
        event_type = None
        if self.event_types is not None and event_idx < len(self.event_types):
            event_type = self.event_types[event_idx]

        # Create the appropriate mixture distribution based on distribution type
        if self.distribution.lower() == "weibull":
            mixture_dist = WeibullMixtureDistribution(
                shape,
                scale,
                processed_logits,
                constrain_shape=self.constrain_params,
                event_type=event_type,
                use_expert_priors=self.use_expert_priors,
            )
        elif self.distribution.lower() == "lognormal":
            mixture_dist = LogNormalMixtureDistribution(
                scale,  # For LogNormal, this is loc
                shape,  # For LogNormal, this is scale
                processed_logits,
                constrain_params=self.constrain_params,
                event_type=event_type,
                use_expert_priors=self.use_expert_priors,
            )
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

        # Compute survival function using the mixture distribution
        survival = mixture_dist.survival_function(time_points)

        return survival

    def _compute_hazard_function(
        self, time_points, shape, scale, logits_g, event_idx=0
    ):
        """
        Compute hazard function from the survival function.
        Uses distribution classes for numerical stability.

        Args:
            time_points: Tensor of time points [batch_size, num_time_points]
            shape: Shape parameters [batch_size, num_mixtures]
            scale: Scale parameters [batch_size, num_mixtures]
            logits_g: Mixture weight logits [batch_size, num_mixtures]
            event_idx: Optional event index for event-specific priors

        Returns:
            Hazard function values [batch_size, num_time_points-1]
        """
        batch_size, num_time_points = time_points.shape

        # For training, use gumbel softmax for the mixture weights
        if self.training:
            # Apply gumbel softmax to create differentiable one-hot vectors
            processed_logits = F.gumbel_softmax(
                logits_g, tau=self.temp, hard=False, dim=1
            )
        else:
            # Just pass through the original logits for inference
            processed_logits = logits_g

        # Extract event type if available for the specified event
        event_type = None
        if self.event_types is not None and event_idx < len(self.event_types):
            event_type = self.event_types[event_idx]

        # Create the appropriate mixture distribution based on distribution type
        if self.distribution.lower() == "weibull":
            mixture_dist = WeibullMixtureDistribution(
                shape,
                scale,
                processed_logits,
                constrain_shape=self.constrain_params,
                event_type=event_type,
                use_expert_priors=self.use_expert_priors,
            )
        elif self.distribution.lower() == "lognormal":
            mixture_dist = LogNormalMixtureDistribution(
                scale,  # For LogNormal, this is loc
                shape,  # For LogNormal, this is scale
                processed_logits,
                constrain_params=self.constrain_params,
                event_type=event_type,
                use_expert_priors=self.use_expert_priors,
            )
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

        # Ensure time points are strictly positive for hazard calculations
        eps = 1e-5
        time_safe = torch.clamp(time_points, min=eps)

        # Get hazard function from the distribution using the safe time points
        hazard = mixture_dist.hazard_function(time_safe)

        # Apply softplus to ensure non-negative hazard and clamp to prevent extreme values
        hazard = F.softplus(hazard)
        hazard = torch.clamp(hazard, min=0.0, max=1e3)

        # Check for any NaN or Inf values and replace them
        hazard = torch.nan_to_num(hazard, nan=0.0, posinf=1e3, neginf=0.0)

        return hazard

    def forward(self, sequence_output, labels=None, **kwargs):
        batch_size = sequence_output.shape[0]
        device = sequence_output.device

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DSM forward input shape: {sequence_output.shape}")

        # Generate time points for evaluating the survival function
        # These should match the duration cuts used in training
        eps = 1e-7
        if hasattr(self.loss, "duration_cuts") and self.loss.duration_cuts is not None:
            # Ensure duration cuts are positive and sorted
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
            # This is just a fallback - ideally we'd use the actual cuts
            time_points = (
                torch.linspace(0.1, 10.0, steps=20, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Using fallback time points, shape: {time_points.shape}, range: [{time_points.min():.5f}, {time_points.max():.5f}]"
                )

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

            # Compute survival and hazard functions with event_idx for expert knowledge
            survival = self._compute_survival_function(
                time_points, event_shape, event_scale, event_logits_g, event_idx
            )
            hazard = self._compute_hazard_function(
                time_points, event_shape, event_scale, event_logits_g, event_idx
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
