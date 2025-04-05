"""
Utility functions for working with survival distributions.

This module provides factory functions and other utilities to create and
manipulate survival distributions, including methods to incorporate expert knowledge
and statistical theory into survival models through:

1. Informative priors on distribution parameters
2. Domain-specific parameter constraints
3. Regularization approaches based on survival analysis literature
4. Event-specific parameter initialization
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from typing import Dict, List, Optional, Union

import torch

from .base import MixtureDistribution, SurvivalDistribution
from .lognormal import LogNormalDistribution, LogNormalMixtureDistribution
from .weibull import WeibullDistribution, WeibullMixtureDistribution


def create_distribution(
    distribution_type: str,
    params: Dict[str, torch.Tensor],
    is_mixture: bool = True,
    constrain_params: bool = False,
    event_type: Optional[str] = None,
    use_expert_priors: bool = False,
) -> Union[SurvivalDistribution, MixtureDistribution]:
    """
    Factory function to create survival distributions with optional expert knowledge.

    Args:
        distribution_type: Type of distribution ('weibull' or 'lognormal')
        params: Dictionary of distribution parameters:
            - For Weibull: 'shape', 'scale', 'logits' (if mixture)
            - For LogNormal: 'loc', 'scale', 'logits' (if mixture)
        is_mixture: Whether to create a mixture distribution
        constrain_params: Whether to apply statistical constraints to parameters
        event_type: Optional event type for specific priors
        use_expert_priors: Whether to use expert knowledge for initialization

    Returns:
        A SurvivalDistribution or MixtureDistribution instance

    Raises:
        ValueError: If the distribution type is not supported
    """
    distribution_type = distribution_type.lower()

    if distribution_type == "weibull":
        if is_mixture:
            if "shape" not in params or "scale" not in params or "logits" not in params:
                raise ValueError(
                    "WeibullMixtureDistribution requires 'shape', 'scale', and 'logits' parameters"
                )

            return WeibullMixtureDistribution(
                shape=params["shape"],
                scale=params["scale"],
                logits=params["logits"],
                constrain_shape=constrain_params,
                event_type=event_type,
                use_expert_priors=use_expert_priors,
            )
        else:
            if "shape" not in params or "scale" not in params:
                raise ValueError(
                    "WeibullDistribution requires 'shape' and 'scale' parameters"
                )

            return WeibullDistribution(
                shape=params["shape"],
                scale=params["scale"],
                constrain_shape=constrain_params,
                event_type=event_type,
                use_expert_priors=use_expert_priors,
            )

    elif distribution_type == "lognormal":
        if is_mixture:
            if "loc" not in params or "scale" not in params or "logits" not in params:
                raise ValueError(
                    "LogNormalMixtureDistribution requires 'loc', 'scale', and 'logits' parameters"
                )

            return LogNormalMixtureDistribution(
                loc=params["loc"],
                scale=params["scale"],
                logits=params["logits"],
                constrain_params=constrain_params,
                event_type=event_type,
                use_expert_priors=use_expert_priors,
            )
        else:
            if "loc" not in params or "scale" not in params:
                raise ValueError(
                    "LogNormalDistribution requires 'loc' and 'scale' parameters"
                )

            return LogNormalDistribution(
                loc=params["loc"],
                scale=params["scale"],
                constrain_params=constrain_params,
                event_type=event_type,
                use_expert_priors=use_expert_priors,
            )

    else:
        raise ValueError(
            f"Unsupported distribution type: {distribution_type}. "
            f"Supported types are 'weibull' and 'lognormal'."
        )


def create_dsm_distribution(
    shape: torch.Tensor,
    scale: torch.Tensor,
    logits_g: torch.Tensor,
    distribution_type: str = "weibull",
    event_idx: int = 0,
    event_types: Optional[List[str]] = None,
    constrain_params: bool = False,
    use_expert_priors: bool = False,
) -> MixtureDistribution:
    """
    Create a mixture distribution for Deep Survival Machines (DSM) model
    with optional expert knowledge incorporation.

    Args:
        shape: Shape/location parameters [batch_size, num_events, num_mixtures]
        scale: Scale parameters [batch_size, num_events, num_mixtures]
        logits_g: Mixture weight logits [batch_size, num_events, num_mixtures]
        distribution_type: Type of distribution ('weibull' or 'lognormal')
        event_idx: Event index to extract parameters for
        event_types: Optional list of event type names for event-specific priors
        constrain_params: Whether to apply statistical constraints to parameters
        use_expert_priors: Whether to use expert knowledge for initialization

    Returns:
        A MixtureDistribution for the specified event
    """
    # Extract parameters for the specified event
    event_shape = shape[:, event_idx, :]  # [batch_size, num_mixtures]
    event_scale = scale[:, event_idx, :]  # [batch_size, num_mixtures]
    event_logits_g = logits_g[:, event_idx, :]  # [batch_size, num_mixtures]

    # Extract event type if provided
    event_type = None
    if event_types is not None and event_idx < len(event_types):
        event_type = event_types[event_idx]

    # Create parameter dictionary based on distribution type
    if distribution_type.lower() == "weibull":
        params = {"shape": event_shape, "scale": event_scale, "logits": event_logits_g}
    elif distribution_type.lower() == "lognormal":
        params = {
            "loc": event_shape,  # For LogNormal, shape represents location
            "scale": event_scale,
            "logits": event_logits_g,
        }
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")

    # Create and return distribution with expert knowledge if requested
    return create_distribution(
        distribution_type,
        params,
        is_mixture=True,
        constrain_params=constrain_params,
        event_type=event_type,
        use_expert_priors=use_expert_priors,
    )


def create_conditional_dsm_distribution(
    shape: torch.Tensor,
    scale: torch.Tensor,
    logits_g: torch.Tensor,
    dependency_matrix: Optional[torch.Tensor] = None,
    event_idx: int = 1,
    observed_events: Optional[torch.Tensor] = None,
    observed_times: Optional[torch.Tensor] = None,
    distribution_type: str = "weibull",
    event_types: Optional[List[str]] = None,
    constrain_params: bool = False,
    use_expert_priors: bool = False,
) -> MixtureDistribution:
    """
    Create a conditional mixture distribution for the MENSA model
    with optional expert knowledge incorporation.

    Args:
        shape: Shape/location parameters [batch_size, num_events, num_mixtures]
        scale: Scale parameters [batch_size, num_events, num_mixtures]
        logits_g: Mixture weight logits [batch_size, num_events, num_mixtures]
        dependency_matrix: Event dependency matrix [num_events, num_events]
        event_idx: Event index to create distribution for (target event)
        observed_events: Binary indicators for observed events [batch_size, num_events]
        observed_times: Times of observed events [batch_size, num_events]
        distribution_type: Type of distribution ('weibull' or 'lognormal')
        event_types: Optional list of event type names for event-specific priors
        constrain_params: Whether to apply statistical constraints to parameters
        use_expert_priors: Whether to use expert knowledge for initialization

    Returns:
        A conditional MixtureDistribution for the specified event
    """
    # If no dependency matrix or observed events, create regular distribution
    if dependency_matrix is None or observed_events is None or observed_times is None:
        return create_dsm_distribution(
            shape,
            scale,
            logits_g,
            distribution_type,
            event_idx,
            event_types,
            constrain_params,
            use_expert_priors,
        )

    # Extract parameters for the specified event
    event_shape = shape[:, event_idx, :].clone()  # [batch_size, num_mixtures]
    event_scale = scale[:, event_idx, :].clone()  # [batch_size, num_mixtures]
    event_logits_g = logits_g[:, event_idx, :].clone()  # [batch_size, num_mixtures]

    batch_size, num_events = observed_events.shape

    # Extract event type if provided
    event_type = None
    if event_types is not None and event_idx < len(event_types):
        event_type = event_types[event_idx]

    # Apply clinically-informed dependency adjustments (based on expert knowledge)
    # This adjusts parameters based on dependencies between events and observed times
    for prev_event_idx in range(num_events):
        if prev_event_idx == event_idx:
            continue

        # Get dependency weight
        dependency_weight = dependency_matrix[event_idx, prev_event_idx]

        # Apply expert knowledge to dependency weight if requested
        if use_expert_priors and event_types is not None:
            prev_event_type = (
                event_types[prev_event_idx]
                if prev_event_idx < len(event_types)
                else None
            )

            # Adjust dependency weight based on clinical knowledge
            if event_type == "cancer" and prev_event_type == "treatment_complications":
                # Treatment complications often influence cancer survival (stronger dependency)
                dependency_weight = dependency_weight * 1.2
            elif event_type == "heart_disease" and prev_event_type == "hypertension":
                # Hypertension strongly influences heart disease risk
                dependency_weight = dependency_weight * 1.5

        # Adjust only for observed previous events
        mask = observed_events[:, prev_event_idx] == 1
        if not torch.any(mask):
            continue

        # Apply adjustment based on dependency weight and observed time
        observed_time = observed_times[mask, prev_event_idx].unsqueeze(
            1
        )  # [filtered_batch, 1]

        # Smaller observed time should lead to smaller predicted time
        # We adjust scale proportionally to observed time
        if distribution_type.lower() == "weibull":
            # For Weibull, scale parameter directly affects the expected time
            scale_adjustment = torch.exp(
                -dependency_weight * torch.log(observed_time + 1e-7)
            )
            event_scale[mask] = event_scale[mask] * scale_adjustment

            # Apply expert knowledge to shape parameter if requested
            if use_expert_priors and event_type is not None:
                prev_event_type = (
                    event_types[prev_event_idx]
                    if prev_event_idx < len(event_types)
                    else None
                )

                # For certain event combinations, adjust the shape parameter
                if (
                    event_type == "cancer"
                    and prev_event_type == "treatment_complications"
                ):
                    # If complication occurs early, cancer often has increasing hazard
                    early_complication_mask = observed_time < 2.0  # Early complications
                    if torch.any(early_complication_mask):
                        # Increase shape to model increasing hazard (for those with early complications)
                        shape_adjustment = 1.2  # Increase shape by 20%
                        local_mask = mask.clone()
                        local_mask[mask] = early_complication_mask.squeeze()
                        event_shape[local_mask] = (
                            event_shape[local_mask] * shape_adjustment
                        )

        else:  # lognormal
            # For LogNormal, loc parameter shifts the distribution
            loc_adjustment = -dependency_weight * torch.log(observed_time + 1e-7)
            event_shape[mask] = event_shape[mask] + loc_adjustment

    # Create parameter dictionary based on distribution type
    if distribution_type.lower() == "weibull":
        params = {"shape": event_shape, "scale": event_scale, "logits": event_logits_g}
    elif distribution_type.lower() == "lognormal":
        params = {
            "loc": event_shape,  # For LogNormal, shape represents location
            "scale": event_scale,
            "logits": event_logits_g,
        }
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")

    # Create and return distribution with expert knowledge
    return create_distribution(
        distribution_type,
        params,
        is_mixture=True,
        constrain_params=constrain_params,
        event_type=event_type,
        use_expert_priors=use_expert_priors,
    )


def apply_informative_prior(
    distribution_params: Dict[str, torch.Tensor],
    distribution_type: str,
    event_type: Optional[str] = None,
    demographic_info: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Apply informative priors to distribution parameters based on expert knowledge
    and survival analysis literature.

    This function incorporates statistical knowledge about specific event types
    or patient demographics to guide parameter initialization.

    Args:
        distribution_params: Dictionary of distribution parameters
        distribution_type: Type of distribution ('weibull' or 'lognormal')
        event_type: Type of event (e.g., 'cancer', 'heart_disease', etc.)
        demographic_info: Dictionary of demographic information

    Returns:
        Updated distribution parameters with expert knowledge incorporated
    """
    params = {k: v.clone() for k, v in distribution_params.items()}

    if distribution_type.lower() == "weibull":
        # Apply Weibull priors based on event type
        if event_type == "cancer":
            # For cancer events, Weibull shape often falls between 1-3
            # indicating increasing hazard with time
            shape_prior_mean = 1.5
            shape_prior_std = 0.5

            # Initialize shapes with slight bias toward increasing hazard
            if "shape" in params:
                shape = params["shape"]
                # Nudge shape values toward domain knowledge while preserving learned patterns
                shape_prior = torch.ones_like(shape) * shape_prior_mean
                # Blend current values with prior (70% current, 30% prior)
                params["shape"] = 0.7 * shape + 0.3 * shape_prior
                # Ensure shape > 1 for increasing hazard (cancer progression typically)
                params["shape"] = torch.maximum(
                    params["shape"], torch.ones_like(params["shape"]) * 1.05
                )

        elif event_type == "heart_disease":
            # For heart disease, Weibull shape is typically close to 1
            # indicating more constant hazard
            shape_prior_mean = 1.1
            shape_prior_std = 0.3

            if "shape" in params:
                shape = params["shape"]
                shape_prior = torch.ones_like(shape) * shape_prior_mean
                # Blend current values with prior
                params["shape"] = 0.75 * shape + 0.25 * shape_prior

        elif event_type == "infectious_disease":
            # For infectious disease, high initial risk followed by decreasing hazard
            # Shape < 1 indicates decreasing hazard
            shape_prior_mean = 0.8
            shape_prior_std = 0.2

            if "shape" in params:
                shape = params["shape"]
                shape_prior = torch.ones_like(shape) * shape_prior_mean
                # Blend current values with prior
                params["shape"] = 0.7 * shape + 0.3 * shape_prior
                # Ensure shape < 1 for decreasing hazard (after acute infection)
                params["shape"] = torch.minimum(
                    params["shape"], torch.ones_like(params["shape"]) * 0.95
                )

        # Apply demographic-specific adjustments if available
        if demographic_info is not None and "age" in demographic_info:
            age = demographic_info["age"]
            # Older patients typically have shorter survival times
            if "scale" in params:
                # Scale inversely proportional to age
                age_factor = torch.exp(
                    -0.01 * (age - 50) / 10
                )  # Normalized to 1.0 at age 50
                age_factor = torch.clamp(age_factor, 0.7, 1.3)
                params["scale"] = params["scale"] * age_factor.unsqueeze(-1)

    elif distribution_type.lower() == "lognormal":
        # Apply LogNormal priors based on event type
        if event_type == "cancer":
            # Cancer survival times often follow log-normal with specific parameters
            loc_prior_mean = 2.0  # log(e^2) ≈ 7.4 months median survival
            scale_prior_mean = 1.2  # Higher variance in cancer outcomes

            if "loc" in params:
                loc = params["loc"]
                loc_prior = torch.ones_like(loc) * loc_prior_mean
                # Blend current values with prior
                params["loc"] = 0.8 * loc + 0.2 * loc_prior

            if "scale" in params:
                scale = params["scale"]
                scale_prior = torch.ones_like(scale) * scale_prior_mean
                # Blend current values with prior
                params["scale"] = 0.8 * scale + 0.2 * scale_prior

        elif event_type == "chronic_disease":
            # Chronic disease often has longer tail
            loc_prior_mean = 3.0  # log(e^3) ≈ 20 months median survival
            scale_prior_mean = 1.5  # Higher variance

            if "loc" in params:
                loc = params["loc"]
                loc_prior = torch.ones_like(loc) * loc_prior_mean
                # Blend current values with prior
                params["loc"] = 0.8 * loc + 0.2 * loc_prior

            if "scale" in params:
                scale = params["scale"]
                scale_prior = torch.ones_like(scale) * scale_prior_mean
                # Blend current values with prior
                params["scale"] = 0.8 * scale + 0.2 * scale_prior

        # Apply demographic-specific adjustments
        if demographic_info is not None:
            if "age" in demographic_info:
                age = demographic_info["age"]
                # Older patients typically have shorter survival times
                if "loc" in params:
                    # Decrease loc for older patients (shift distribution left)
                    age_shift = -0.02 * (age - 50) / 10  # Small shift based on age
                    params["loc"] = params["loc"] + age_shift.unsqueeze(-1)

            if "treatment" in demographic_info:
                treatment = demographic_info["treatment"]
                # Treatment typically extends survival
                if "loc" in params and torch.any(treatment > 0):
                    # Increase loc for patients receiving treatment
                    treatment_shift = 0.3 * treatment  # Positive shift for treatment
                    params["loc"] = params["loc"] + treatment_shift.unsqueeze(-1)

    return params


def calculate_regularization_weight(
    distribution_params: Dict[str, torch.Tensor], distribution_type: str
) -> torch.Tensor:
    """
    Calculate a regularization weight based on the distribution parameters and statistical theory.

    This function penalizes implausible parameter combinations based on survival analysis theory.

    Args:
        distribution_params: Dictionary of distribution parameters
        distribution_type: Type of distribution ('weibull' or 'lognormal')

    Returns:
        Regularization weight (higher means more regularization needed)
    """
    if distribution_type.lower() == "weibull":
        if "shape" in distribution_params and "scale" in distribution_params:
            shape = distribution_params["shape"]
            scale = distribution_params["scale"]

            # Penalize extreme shape values - very high shape is implausible
            shape_penalty = torch.mean(
                torch.where(
                    shape > 5.0, torch.square(shape - 5.0), torch.zeros_like(shape)
                )
            )

            # Penalize extreme scale values - should align with reasonable survival times
            scale_penalty = torch.mean(
                torch.where(
                    scale > 100.0,
                    torch.square(torch.log(scale) - torch.log(torch.tensor(100.0))),
                    torch.zeros_like(scale),
                )
            )

            # Combined regularization weight
            return shape_penalty + 0.1 * scale_penalty

    elif distribution_type.lower() == "lognormal":
        if "loc" in distribution_params and "scale" in distribution_params:
            loc = distribution_params["loc"]
            scale = distribution_params["scale"]

            # Penalize extreme loc values - should align with reasonable survival times
            loc_penalty = torch.mean(
                torch.where(
                    torch.abs(loc) > 5.0,
                    torch.square(loc - torch.sign(loc) * 5.0),
                    torch.zeros_like(loc),
                )
            )

            # Penalize extreme scale values - high uncertainty is often implausible
            scale_penalty = torch.mean(
                torch.where(
                    scale > 2.0, torch.square(scale - 2.0), torch.zeros_like(scale)
                )
            )

            # Combined regularization weight
            return loc_penalty + scale_penalty

    # Default regularization weight
    return torch.tensor(0.0)


def apply_event_specific_constraints(
    distribution_params: Dict[str, torch.Tensor],
    event_idx: int,
    num_events: int,
    distribution_type: str,
    event_types: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Apply event-specific parameter constraints based on clinical knowledge.

    Different event types have different expected hazard patterns which can
    be encoded in the distribution parameters.

    Args:
        distribution_params: Dictionary of distribution parameters
        event_idx: Index of the current event
        num_events: Total number of events
        distribution_type: Type of distribution ('weibull' or 'lognormal')
        event_types: Optional list of event type names for more specific constraints

    Returns:
        Updated distribution parameters with event-specific constraints applied
    """
    params = {k: v.clone() for k, v in distribution_params.items()}

    # If we have named event types, use those
    event_type = None
    if event_types is not None and event_idx < len(event_types):
        event_type = event_types[event_idx]

    # Apply general constraints based on distribution type
    if distribution_type.lower() == "weibull":
        if "shape" in params:
            shape = params["shape"]

            # Apply event-specific shape constraints
            if event_type == "cancer_recurrence":
                # Cancer recurrence typically has increasing hazard (shape > 1)
                params["shape"] = torch.maximum(shape, torch.ones_like(shape) * 1.1)
            elif event_type == "treatment_complications":
                # Treatment complications often have decreasing hazard (shape < 1)
                params["shape"] = torch.minimum(shape, torch.ones_like(shape) * 0.9)
            elif event_type == "long_term_survival":
                # Long-term survival often has constant hazard (shape ≈ 1)
                params["shape"] = 0.9 * shape + 0.1 * torch.ones_like(shape)

    elif distribution_type.lower() == "lognormal":
        if "scale" in params:
            scale = params["scale"]

            # Apply event-specific scale constraints
            if event_type == "cancer_recurrence":
                # Cancer recurrence typically has lower variance
                params["scale"] = torch.minimum(scale, torch.ones_like(scale) * 1.0)
            elif event_type == "treatment_complications":
                # Treatment complications often have higher variance
                params["scale"] = torch.maximum(scale, torch.ones_like(scale) * 0.8)

    return params


def create_clinically_informed_distribution(
    distribution_type: str,
    params: Dict[str, torch.Tensor],
    event_type: Optional[str] = None,
    demographic_info: Optional[Dict[str, torch.Tensor]] = None,
    is_mixture: bool = True,
) -> Union[SurvivalDistribution, MixtureDistribution]:
    """
    Create a survival distribution with parameters informed by clinical knowledge.

    This is an enhanced version of create_distribution that incorporates expert
    knowledge into the distribution parameters.

    Args:
        distribution_type: Type of distribution ('weibull' or 'lognormal')
        params: Dictionary of distribution parameters
        event_type: Type of event (e.g., 'cancer', 'heart_disease')
        demographic_info: Dictionary of demographic information
        is_mixture: Whether to create a mixture distribution

    Returns:
        A clinically informed SurvivalDistribution or MixtureDistribution
    """
    # Apply informative priors based on clinical knowledge
    informed_params = apply_informative_prior(
        params, distribution_type, event_type, demographic_info
    )

    # Create distribution with informed parameters
    return create_distribution(distribution_type, informed_params, is_mixture)
