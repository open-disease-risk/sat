"""Base classes for the heads for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import abc
import torch

from torch import nn
from transformers.modeling_utils import PreTrainedModel

from sat.utils import logging

from .config import BaseConfig

logger = logging.get_default_logger()


class BaseTask(PreTrainedModel):
    """Base class for all task models."""

    def post_init(self):
        """Override the post_init method to apply ownership-based initialization"""
        # Instead of using apply on the entire model, we'll initialize only direct children
        for name, module in self.named_children():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Initializing module {name} directly owned by {self.__class__.__name__}"
                )
            # Initialize this module with the task's specific initialization
            self.initialize_module(module)

    def initialize_module(self, module):
        """Initialize a module and its direct parameters, but NOT its sub-modules"""
        # Initialize parameters directly owned by this module
        for name, param in module.named_parameters(recurse=False):
            if "weight" in name:
                if isinstance(module, nn.Linear):
                    self._init_linear_weight(module)
                elif isinstance(module, nn.LayerNorm):
                    self._init_layernorm_weight(module)
            elif "bias" in name:
                if isinstance(module, nn.Linear):
                    self._init_linear_bias(module)
                elif isinstance(module, nn.LayerNorm):
                    self._init_layernorm_bias(module)

        # Recursively initialize child modules with the same task's initializer
        for child_name, child_module in module.named_children():
            self.initialize_module(child_module)

    def _init_linear_weight(self, module):
        """Initialize linear layer weights based on config"""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{self.__class__.__name__} initializing linear weights: {module}"
            )

        if self.config.initializer == "normal":
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif self.config.initializer == "xavier_uniform":
            torch.nn.init.xavier_uniform_(
                module.weight, gain=torch.nn.init.calculate_gain("leaky_relu")
            )
        elif self.config.initializer == "xavier_normal":
            torch.nn.init.xavier_normal_(
                module.weight, gain=torch.nn.init.calculate_gain("leaky_relu")
            )
        elif self.config.initializer == "kaiming_normal":
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="leaky_relu"
            )
        elif self.config.initializer == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                module.weight, mode="fan_out", nonlinearity="leaky_relu"
            )
        else:
            raise ValueError(f"Initializer {self.config.initializer} not supported!")

    def _init_linear_bias(self, module):
        """Initialize linear layer bias"""
        if module.bias is not None:
            module.bias.data.zero_()

    def _init_layernorm_weight(self, module):
        """Initialize layernorm weights"""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{self.__class__.__name__} initializing layernorm weights: {module}"
            )
        module.weight.data.fill_(1.0)

    def _init_layernorm_bias(self, module):
        """Initialize layernorm bias"""
        module.bias.data.zero_()

    def _init_weights(self, module):
        """
        This is kept for compatibility with the PreTrainedModel class, but
        in our implementation we prefer the more explicit post_init() method
        that uses the initialize_module approach above.
        """
        if isinstance(module, nn.Linear):
            self._init_linear_weight(module)
            self._init_linear_bias(module)
        elif isinstance(module, nn.LayerNorm):
            self._init_layernorm_weight(module)
            self._init_layernorm_bias(module)


class SurvivalTask(abc.ABC, BaseTask):
    """Base class for survival analysis tasks."""

    supports_gradient_checkpointing = True

    def __init__(self, config: BaseConfig):
        super().__init__(config)

    def _init_linear_weight(self, module):
        """Survival-specific linear weight initialization with special handling for multi-event cases"""
        # Special initialization for output layer of survival networks
        if hasattr(self, "nets") and hasattr(module, "weight"):
            # Check if this is a multi-event model
            is_multi_event = (
                hasattr(self.config, "num_events") and self.config.num_events > 1
            )

            # Check if this is the final output layer
            # Check if this is the final output layer - safely
            is_final_output = False
            if hasattr(self.nets, "event_nets"):
                try:
                    is_final_output = any(
                        hasattr(net, "net") and module is net.net[-1]
                        for net in self.nets.event_nets
                    )
                except (AttributeError, TypeError):
                    pass

            if is_final_output:
                # For multi-event models, initialize with zeros to ensure stable starting point
                if is_multi_event:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"SurvivalTask: initializing multi-event final output layer with zeros: {module}"
                        )
                    nn.init.zeros_(module.weight)
                    # Note: we'll also handle the bias in _init_linear_bias
                else:
                    # For single event, use a small positive value
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"SurvivalTask: initializing single-event final output layer: {module}"
                        )
                    nn.init.constant_(module.weight, 0.01)
                return

            # For hidden layers in multi-event models, use a more conservative initialization
            elif is_multi_event:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"SurvivalTask: conservative init for hidden layer in multi-event model: {module}"
                    )
                # Reduce variance of initialization to prevent exploding gradients
                if self.config.initializer == "normal":
                    # Use a smaller standard deviation
                    module.weight.data.normal_(
                        mean=0.0, std=self.config.initializer_range * 0.5
                    )
                elif self.config.initializer in ["kaiming_normal", "kaiming_uniform"]:
                    # Use fan_in mode for more conservative initialization
                    if self.config.initializer == "kaiming_normal":
                        nn.init.kaiming_normal_(
                            module.weight, mode="fan_in", nonlinearity="relu"
                        )
                    else:
                        nn.init.kaiming_uniform_(
                            module.weight, mode="fan_in", nonlinearity="relu"
                        )
                    # Scale down the weights further
                    with torch.no_grad():
                        module.weight.data.mul_(0.5)
                else:
                    # Use default but scale down
                    super()._init_linear_weight(module)
                    with torch.no_grad():
                        module.weight.data.mul_(0.5)
                return

        # Default initialization for other layers
        super()._init_linear_weight(module)

    def _init_linear_bias(self, module):
        """Survival-specific linear bias initialization with special handling for multi-event cases"""
        if hasattr(self, "nets") and module.bias is not None:
            # Check if this is a multi-event model
            is_multi_event = (
                hasattr(self.config, "num_events") and self.config.num_events > 1
            )

            # Check if this is the final output layer
            is_final_output = False
            if hasattr(self.nets, "event_nets"):
                try:
                    # Check for CauseSpecificNet pattern (event_nets with net member)
                    is_final_output = any(
                        hasattr(net, "net") and module is net.net[-1] 
                        for net in self.nets.event_nets
                    )
                    
                    # Check for SimpleCompRiskNet pattern (event_nets with direct Linear layers)
                    if not is_final_output:
                        is_final_output = any(module is net for net in self.nets.event_nets)
                except (AttributeError, TypeError):
                    pass
                    
            if is_final_output:
                if is_multi_event:
                    # For multi-event final layer, initialize bias to small negative values
                    # This ensures initial hazard values will be small after softplus
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"SurvivalTask: initializing multi-event final layer bias to -2.0: {module}"
                        )
                    nn.init.constant_(module.bias, -2.0)
                else:
                    # For single-event models, use a smaller negative value
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"SurvivalTask: initializing single-event final layer bias: {module}"
                        )
                    nn.init.constant_(module.bias, -1.0)
                return

        # For all other cases, use default initialization (zeros)
        super()._init_linear_bias(module)


class RegressionTask(abc.ABC, BaseTask):
    """Base class for regression tasks."""

    supports_gradient_checkpointing = True

    def __init__(self, config: BaseConfig):
        super().__init__(config)

    def _init_linear_weight(self, module):
        """Regression-specific linear weight initialization"""
        # Special initialization for output layer of regression networks
        if hasattr(self, "nets") and hasattr(module, "weight"):
            # Check if this is the final output layer
            is_final_output = False
            if hasattr(self.nets, "event_nets"):
                try:
                    # Check for CauseSpecificNet pattern (event_nets with net member)
                    is_final_output = any(
                        hasattr(net, "net") and module is net.net[-1] 
                        for net in self.nets.event_nets
                    )
                    
                    # Check for SimpleCompRiskNet pattern (event_nets with direct Linear layers)
                    if not is_final_output:
                        is_final_output = any(module is net for net in self.nets.event_nets)
                except (AttributeError, TypeError):
                    pass
                    
            if is_final_output:
                # Output layer initialization for regression with ReLU
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"RegressionTask: initializing final output layer: {module}"
                    )
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                return

        # Default initialization for other layers
        super()._init_linear_weight(module)

    def _init_linear_bias(self, module):
        """Regression-specific linear bias initialization"""
        # Special initialization for output layer of regression networks
        if hasattr(self, "nets") and module.bias is not None:
            # Check if this is the final output layer
            is_final_output = False
            if hasattr(self.nets, "event_nets"):
                try:
                    # Check for CauseSpecificNet pattern (event_nets with net member)
                    is_final_output = any(
                        hasattr(net, "net") and module is net.net[-1] 
                        for net in self.nets.event_nets
                    )
                    
                    # Check for SimpleCompRiskNet pattern (event_nets with direct Linear layers)
                    if not is_final_output:
                        is_final_output = any(module is net for net in self.nets.event_nets)
                except (AttributeError, TypeError):
                    pass
                    
            if is_final_output:
                # Initialize bias to positive value for ReLU
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"RegressionTask: initializing final output bias: {module}"
                    )
                nn.init.constant_(module.bias, 0.1)
                return

        # Default initialization for other layers
        super()._init_linear_bias(module)


class ClassificationTask(abc.ABC, BaseTask):
    """Base class for classification tasks."""

    supports_gradient_checkpointing = True

    def __init__(self, config: BaseConfig):
        super().__init__(config)

    def _init_linear_weight(self, module):
        """Classification-specific linear weight initialization"""
        # Special initialization for output layer of classification networks
        if hasattr(self, "nets") and hasattr(module, "weight"):
            # Check if this is the final output layer
            is_final_output = False
            if hasattr(self.nets, "event_nets"):
                try:
                    # Check for CauseSpecificNet pattern (event_nets with net member)
                    is_final_output = any(
                        hasattr(net, "net") and module is net.net[-1] 
                        for net in self.nets.event_nets
                    )
                    
                    # Check for SimpleCompRiskNet pattern (event_nets with direct Linear layers)
                    if not is_final_output:
                        is_final_output = any(module is net for net in self.nets.event_nets)
                except (AttributeError, TypeError):
                    pass
                    
            if is_final_output:
                # Output layer initialization for sigmoid activation
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"ClassificationTask: initializing final output layer: {module}"
                    )
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                return

        # Default initialization for other layers
        super()._init_linear_weight(module)

    def _init_linear_bias(self, module):
        """Classification-specific linear bias initialization"""
        # Special initialization for output layer of classification networks
        if hasattr(self, "nets") and module.bias is not None:
            # Check if this is the final output layer
            is_final_output = False
            if hasattr(self.nets, "event_nets"):
                try:
                    # Check for CauseSpecificNet pattern (event_nets with net member)
                    is_final_output = any(
                        hasattr(net, "net") and module is net.net[-1] 
                        for net in self.nets.event_nets
                    )
                    
                    # Check for SimpleCompRiskNet pattern (event_nets with direct Linear layers)
                    if not is_final_output:
                        is_final_output = any(module is net for net in self.nets.event_nets)
                except (AttributeError, TypeError):
                    pass
                    
            if is_final_output:
                # Initialize bias to zero for balanced sigmoid
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"ClassificationTask: initializing final output bias: {module}"
                    )
                nn.init.zeros_(module.bias)
                return

        # Default initialization for other layers
        super()._init_linear_bias(module)


class MTLTask(abc.ABC, BaseTask):
    """Base class for multi-task learning tasks."""

    supports_gradient_checkpointing = True

    def __init__(self, config: BaseConfig):
        super().__init__(config)

    def post_init(self):
        """Override post_init for MTL to handle both shared layers and sub-tasks

        Note: This method might be called in two scenarios:
        1. When MTLTask is first created - in this case, initialization is handled in __init__
        2. When it's called explicitly later - in this case, we should do the initialization

        To avoid double-initialization, we check if we're called from __init__ or externally
        """
        # Check if this is being called from outside __init__
        # In that case, do the initialization
        import inspect

        caller_frame = inspect.currentframe().f_back
        caller_name = caller_frame.f_code.co_name

        if caller_name != "__init__":
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "MTLTask.post_init called from outside __init__, doing initialization"
                )
            # First initialize only the shared networks that MTL directly owns
            for name, module in self.named_children():
                # Skip the heads module list - these will be initialized separately
                if name != "heads":
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"MTLTask: initializing shared module {name}")
                    self.initialize_module(module)

            # Now initialize each task head with its own specific initialization
            if hasattr(self, "heads"):
                for i, head in enumerate(self.heads):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"MTLTask: initializing task head {i} using its own initializer"
                        )
                    head.post_init()
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "MTLTask.post_init called from __init__, skipping (already initialized)"
                )
            # No need to do anything, initialization was already done in __init__

    def _init_linear_weight(self, module):
        """MTL-specific linear weight initialization"""
        # Special initialization for MTL shared network
        if hasattr(self, "net") and hasattr(module, "weight"):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"MTLTask: initializing shared network linear layer: {module}"
                )
            nn.init.kaiming_uniform_(
                module.weight, mode="fan_out", nonlinearity="leaky_relu"
            )
            return

        # Default initialization for other layers
        super()._init_linear_weight(module)

    def _init_linear_bias(self, module):
        """MTL-specific linear bias initialization"""
        # Special initialization for MTL shared network
        if hasattr(self, "net") and module.bias is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"MTLTask: initializing shared network bias: {module}")
            nn.init.zeros_(module.bias)
            return

        # Default initialization for other layers
        super()._init_linear_bias(module)
