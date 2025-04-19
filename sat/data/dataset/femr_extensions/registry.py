"""Registry for FEMR extensions.

This module provides a registry system for labelers and featurizers,
allowing them to be dynamically instantiated by Hydra.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"


import logging
from typing import Dict, Type, Any, Optional, List

from femr.labelers import Labeler

# Import and register all built-in labelers
from sat.data.dataset.femr_extensions.labelers import (
    SurvivalLabeler,
    CompetingRiskLabeler,
    CustomEventLabeler,
)

logger = logging.getLogger(__name__)


class BaseClass:
    pass


class Registry:
    """Base class for component registries."""

    _registry: Dict[str, Type] = {}
    _base_class: Type = BaseClass

    @classmethod
    def register(cls, name: str, component_class: Type) -> Type:
        """Register a component class with the given name.

        Args:
            name: Identifier for the component
            component_class: The class to register

        Returns:
            The registered class (for decorator use)
        """
        # Validate that the class extends the base class
        if not issubclass(component_class, cls._base_class):
            raise TypeError(
                f"{component_class.__name__} must be a subclass of {cls._base_class.__name__}"
            )

        cls._registry[name] = component_class
        logger.debug(f"Registered {cls.__name__} component: {name}")
        return component_class

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """Get a component class by name.

        Args:
            name: Identifier for the component

        Returns:
            The component class if found, None otherwise
        """
        return cls._registry.get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available component types.

        Returns:
            List of registered component names
        """
        return list(cls._registry.keys())

    @classmethod
    def instantiate(cls, config: Dict[str, Any]) -> Any:
        """Instantiate a component from config.

        Args:
            config: Configuration dictionary with 'type' field

        Returns:
            Instance of the component

        Raises:
            ValueError: If type is not specified or not found
        """
        if "type" not in config:
            raise ValueError(f"Configuration must include 'type' field: {config}")

        component_type = config["type"]
        component_class = cls.get(component_type)

        if component_class is None:
            available = cls.list_available()
            raise ValueError(
                f"Unknown component type '{component_type}'. Available types: {available}"
            )

        # Filter out the 'type' field and pass the rest as kwargs
        kwargs = {k: v for k, v in config.items() if k != "type"}

        # Instantiate the component
        try:
            return component_class(**kwargs)
        except Exception as e:
            logger.error(f"Error instantiating {component_type}: {e}")
            raise


class LabelerRegistry(Registry):
    """Registry for labelers."""

    _base_class = Labeler

LabelerRegistry.register("survival", SurvivalLabeler)
LabelerRegistry.register("competing_risk", CompetingRiskLabeler)
LabelerRegistry.register("custom", CustomEventLabeler)


# Define resolver functions for Hydra
def resolve_labeler(**config) -> Any:
    """Resolver function for Hydra to instantiate labelers.

    Args:
        config: Configuration dictionary

    Returns:
        Instantiated labeler
    """
    return LabelerRegistry.instantiate(config)


# External API to register custom components
def register_labeler(name: str, labeler_class: Type = None) -> Type:
    """Register a custom labeler class.

    Args:
        name: Identifier for the labeler
        labeler_class: The labeler class to register

    Returns:
        The registered labeler class or decorator function
    """
    if labeler_class is None:
        # Used as a decorator
        def decorator(cls):
            return LabelerRegistry.register(name, cls)

        return decorator
    else:
        # Used as a function
        return LabelerRegistry.register(name, labeler_class)


# Example usage for decorators:
# @register_labeler("my_custom_labeler")
# class MyCustomLabeler(LABELER_BASE):
#     pass
#
# @register_featurizer("my_custom_featurizer")
# class MyCustomFeaturizer(FEATURIZER_BASE):
#     pass
