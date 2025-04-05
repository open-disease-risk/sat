"""Parameter network implementations for survival models"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torch.nn as nn
import torch.nn.functional as F

from sat.utils import logging

logger = logging.get_default_logger()


class ParamCauseSpecificNet(nn.Module):
    """
    Network for computing distribution parameters in single-event case.
    Similar to CauseSpecificNet but outputs distribution parameters.
    """

    def __init__(
        self,
        in_features,
        intermediate_size,
        num_hidden_layers,
        num_mixtures,
        batch_norm=True,
        dropout=None,
        bias=True,
        activation=nn.LeakyReLU,
        num_events=1,
    ):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.num_events = num_events

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Creating ParamCauseSpecificNet with in_features={in_features}, "
                f"intermediate_size={intermediate_size}, num_mixtures={num_mixtures}"
            )

        # Feature extraction network (same as CauseSpecificNet)
        self.feature_net = nn.Sequential()

        # First layer (in_features -> intermediate_size)
        self.feature_net.append(nn.Linear(in_features, intermediate_size, bias=bias))
        if batch_norm:
            self.feature_net.append(nn.BatchNorm1d(intermediate_size))
        self.feature_net.append(activation())
        if dropout is not None and dropout > 0:
            self.feature_net.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.feature_net.append(
                nn.Linear(intermediate_size, intermediate_size, bias=bias)
            )
            if batch_norm:
                self.feature_net.append(nn.BatchNorm1d(intermediate_size))
            self.feature_net.append(activation())
            if dropout is not None and dropout > 0:
                self.feature_net.append(nn.Dropout(dropout))

        # Parameter networks
        self.shape_net = nn.Linear(intermediate_size, num_mixtures)
        self.scale_net = nn.Linear(intermediate_size, num_mixtures)
        self.mixture_net = nn.Linear(intermediate_size, num_mixtures)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, in_features]

        Returns:
            Tuple of (shape, scale, mixture) parameters, each of shape [batch_size, num_mixtures] or
            [batch_size, num_events, num_mixtures] for multi-event case
        """
        # Extract features
        features = self.feature_net(x)

        # Compute shape, scale, and mixture parameters
        shape = F.softplus(self.shape_net(features)) + 0.01
        scale = F.softplus(self.scale_net(features)) + 0.01
        logits_g = self.mixture_net(features)

        # Add event dimension for consistency with multi-event case
        if self.num_events == 1:
            shape = shape.unsqueeze(1)
            scale = scale.unsqueeze(1)
            logits_g = logits_g.unsqueeze(1)

        return shape, scale, logits_g


class ParamCauseSpecificNetCompRisk(nn.Module):
    """
    Network for computing distribution parameters in multi-event case.
    Similar to CauseSpecificNetCompRisk but outputs distribution parameters.
    """

    def __init__(
        self,
        in_features,
        shared_intermediate_size,
        shared_num_hidden_layers,
        indiv_intermediate_size,
        indiv_num_hidden_layers,
        num_mixtures,
        batch_norm=True,
        dropout=None,
        bias=True,
        activation=nn.LeakyReLU,
        num_events=1,
    ):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.num_events = num_events

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Creating ParamCauseSpecificNetCompRisk with in_features={in_features}, "
                f"shared_intermediate_size={shared_intermediate_size}, "
                f"indiv_intermediate_size={indiv_intermediate_size}, "
                f"num_events={num_events}, num_mixtures={num_mixtures}"
            )

        # Create shared network - similar to CauseSpecificNetCompRisk
        self.shared_net = nn.Sequential()

        # First layer of shared network
        self.shared_net.append(
            nn.Linear(in_features, shared_intermediate_size, bias=bias)
        )
        if batch_norm:
            self.shared_net.append(nn.BatchNorm1d(shared_intermediate_size))
        self.shared_net.append(activation())
        if dropout is not None and dropout > 0:
            self.shared_net.append(nn.Dropout(dropout))

        # Hidden layers of shared network
        for _ in range(shared_num_hidden_layers - 1):
            self.shared_net.append(
                nn.Linear(shared_intermediate_size, shared_intermediate_size, bias=bias)
            )
            if batch_norm:
                self.shared_net.append(nn.BatchNorm1d(shared_intermediate_size))
            self.shared_net.append(activation())
            if dropout is not None and dropout > 0:
                self.shared_net.append(nn.Dropout(dropout))

        # Event-specific networks
        self.event_nets = nn.ModuleList()

        for event_idx in range(num_events):
            # Create event-specific network
            event_net = nn.ModuleDict()

            # Feature extraction part
            feature_layers = []

            # First layer of event network
            feature_layers.append(
                nn.Linear(shared_intermediate_size, indiv_intermediate_size, bias=bias)
            )
            if batch_norm:
                feature_layers.append(nn.BatchNorm1d(indiv_intermediate_size))
            feature_layers.append(activation())
            if dropout is not None and dropout > 0:
                feature_layers.append(nn.Dropout(dropout))

            # Hidden layers of event network
            for _ in range(indiv_num_hidden_layers - 1):
                feature_layers.append(
                    nn.Linear(
                        indiv_intermediate_size, indiv_intermediate_size, bias=bias
                    )
                )
                if batch_norm:
                    feature_layers.append(nn.BatchNorm1d(indiv_intermediate_size))
                feature_layers.append(activation())
                if dropout is not None and dropout > 0:
                    feature_layers.append(nn.Dropout(dropout))

            event_net["features"] = nn.Sequential(*feature_layers)

            # Parameter layers
            event_net["shape"] = nn.Linear(indiv_intermediate_size, num_mixtures)
            event_net["scale"] = nn.Linear(indiv_intermediate_size, num_mixtures)
            event_net["mixture"] = nn.Linear(indiv_intermediate_size, num_mixtures)

            # Initialize event network differently for each event (to break symmetry)
            if num_events > 1:
                with torch.no_grad():
                    # Slightly different initialization for each event
                    scale_factor = 0.9 + 0.2 * event_idx
                    event_net["shape"].weight.data *= scale_factor
                    event_net["scale"].weight.data *= scale_factor

                    # Different bias initialization
                    if event_net["shape"].bias is not None:
                        event_net["shape"].bias.data.fill_(0.1 * event_idx)
                    if event_net["scale"].bias is not None:
                        event_net["scale"].bias.data.fill_(0.2 * event_idx)
                    if event_net["mixture"].bias is not None:
                        event_net["mixture"].bias.data.fill_(-0.1 * event_idx)

            self.event_nets.append(event_net)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, in_features]

        Returns:
            Tuple of (shape, scale, mixture) parameters, each of shape [batch_size, num_events, num_mixtures]
        """
        # Pass through shared network
        shared_features = self.shared_net(x)

        # Compute parameters for each event
        all_shapes = []
        all_scales = []
        all_logits_g = []

        for event_idx, event_net in enumerate(self.event_nets):
            # Extract event-specific features
            event_features = event_net["features"](shared_features)

            # Compute parameters
            shape = F.softplus(event_net["shape"](event_features)) + 0.01
            scale = F.softplus(event_net["scale"](event_features)) + 0.01
            logits_g = event_net["mixture"](event_features)

            all_shapes.append(shape)
            all_scales.append(scale)
            all_logits_g.append(logits_g)

        # Stack along event dimension
        stacked_shape = torch.stack(all_shapes, dim=1)
        stacked_scale = torch.stack(all_scales, dim=1)
        stacked_logits_g = torch.stack(all_logits_g, dim=1)

        return stacked_shape, stacked_scale, stacked_logits_g


class MENSAParameterNet(nn.Module):
    """
    MENSA-specific parameter network that models dependencies between events.

    Based on the paper: "MENSA: Multi-Event Neural Survival Analysis" (2024)

    Key features:
    - Uses SELU activation for better numerical stability
    - Explicit modeling of event dependencies via a dependency matrix
    - Specialized parameter initialization for distribution stability
    - Proper parameter range constraints to prevent gradient issues
    """

    def __init__(
        self,
        in_features,
        shared_intermediate_size,
        shared_num_hidden_layers,
        indiv_intermediate_size,
        indiv_num_hidden_layers,
        num_mixtures,
        batch_norm=True,
        dropout=None,
        bias=True,
        activation=nn.SELU,  # MENSA uses SELU activation
        num_events=1,
        event_dependency=True,
    ):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.num_events = num_events
        self.event_dependency = event_dependency and num_events > 1

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Creating MENSAParameterNet with in_features={in_features}, "
                f"shared_intermediate_size={shared_intermediate_size}, "
                f"indiv_intermediate_size={indiv_intermediate_size}, "
                f"num_events={num_events}, num_mixtures={num_mixtures}, "
                f"event_dependency={event_dependency}"
            )

        # Create shared network
        self.shared_net = nn.Sequential()

        # First layer of shared network
        self.shared_net.append(
            nn.Linear(in_features, shared_intermediate_size, bias=bias)
        )
        if batch_norm:
            self.shared_net.append(nn.BatchNorm1d(shared_intermediate_size))
        self.shared_net.append(activation())
        if dropout is not None and dropout > 0:
            self.shared_net.append(nn.Dropout(dropout))

        # Hidden layers of shared network
        for _ in range(shared_num_hidden_layers - 1):
            self.shared_net.append(
                nn.Linear(shared_intermediate_size, shared_intermediate_size, bias=bias)
            )
            if batch_norm:
                self.shared_net.append(nn.BatchNorm1d(shared_intermediate_size))
            self.shared_net.append(activation())
            if dropout is not None and dropout > 0:
                self.shared_net.append(nn.Dropout(dropout))

        # If event_dependency is True, create dependency matrix
        if self.event_dependency:
            # Create event dependency matrix - learnable parameter
            self.event_dependency_matrix = nn.Parameter(
                torch.zeros(num_events, num_events)
            )
            # Initialize matrix with proper values to ensure numerical stability
            with torch.no_grad():
                # Initialize diagonal elements to higher values (3.0) to start with more independence
                # This ensures the softmax will give ~95% weight to the diagonal
                self.event_dependency_matrix.fill_diagonal_(3.0)

                # Add small random noise to off-diagonal elements to break symmetry
                # but keep values low enough that initial dependencies are minimal
                mask = ~torch.eye(num_events, dtype=torch.bool)
                self.event_dependency_matrix.data[mask] = (
                    torch.randn(num_events, num_events)[mask] * 0.01
                )

        # Event-specific networks
        self.event_nets = nn.ModuleList()

        for event_idx in range(num_events):
            # Create event-specific network
            event_net = nn.ModuleDict()

            # Feature extraction part
            feature_layers = []

            # First layer of event network
            feature_layers.append(
                nn.Linear(shared_intermediate_size, indiv_intermediate_size, bias=bias)
            )
            if batch_norm:
                feature_layers.append(nn.BatchNorm1d(indiv_intermediate_size))
            feature_layers.append(activation())
            if dropout is not None and dropout > 0:
                feature_layers.append(nn.Dropout(dropout))

            # Hidden layers of event network
            for _ in range(indiv_num_hidden_layers - 1):
                feature_layers.append(
                    nn.Linear(
                        indiv_intermediate_size, indiv_intermediate_size, bias=bias
                    )
                )
                if batch_norm:
                    feature_layers.append(nn.BatchNorm1d(indiv_intermediate_size))
                feature_layers.append(activation())
                if dropout is not None and dropout > 0:
                    feature_layers.append(nn.Dropout(dropout))

            event_net["features"] = nn.Sequential(*feature_layers)

            # Parameter layers
            event_net["shape"] = nn.Linear(indiv_intermediate_size, num_mixtures)
            event_net["scale"] = nn.Linear(indiv_intermediate_size, num_mixtures)
            event_net["mixture"] = nn.Linear(indiv_intermediate_size, num_mixtures)

            # Initialize parameters differently for each event
            self._initialize_event_parameters(event_net, event_idx)

            self.event_nets.append(event_net)

    def _initialize_event_parameters(self, event_net, event_idx):
        """
        Initialize parameters for an event network to break symmetry.

        Uses He initialization for weights and strategic bias initialization
        for better convergence in mixture models.
        """
        with torch.no_grad():
            # MENSA paper uses specialized initialization for SELU activation
            # Use He (kaiming) initialization for weights
            nn.init.kaiming_normal_(event_net["shape"].weight, nonlinearity="linear")
            nn.init.kaiming_normal_(event_net["scale"].weight, nonlinearity="linear")
            nn.init.kaiming_normal_(event_net["mixture"].weight, nonlinearity="linear")

            # Different bias initialization for each event to break symmetry
            if event_net["shape"].bias is not None:
                event_net["shape"].bias.data.fill_(0.1 * event_idx)
            if event_net["scale"].bias is not None:
                event_net["scale"].bias.data.fill_(0.2 * event_idx)
            if event_net["mixture"].bias is not None:
                event_net["mixture"].bias.data.fill_(-0.1 * event_idx)

    def forward(self, x):
        """
        Forward pass through the MENSA parameter network.

        Args:
            x: Input tensor of shape [batch_size, in_features]

        Returns:
            Tuple of (shape, scale, mixture) parameters, each of shape
            [batch_size, num_events, num_mixtures]
        """
        # Pass through shared network
        shared_features = self.shared_net(x)

        # Compute parameters for each event
        all_shapes = []
        all_scales = []
        all_logits_g = []

        # For each event, compute parameters
        for event_idx, event_net in enumerate(self.event_nets):
            # Extract event-specific features
            event_features = event_net["features"](shared_features)

            # Compute parameters for this event with improved numerical stability
            # Use appropriate minimum values for distribution parameters
            shape = F.softplus(event_net["shape"](event_features)) + 0.1  # Min 0.1
            scale = F.softplus(event_net["scale"](event_features)) + 0.5  # Min 0.5

            # Ensure values are in a reasonable range to prevent gradient issues
            shape = torch.clamp(shape, min=0.1, max=100.0)
            scale = torch.clamp(scale, min=0.5, max=1000.0)

            # Regular logits for mixture weights (no constraints needed)
            logits_g = event_net["mixture"](event_features)

            all_shapes.append(shape)
            all_scales.append(scale)
            all_logits_g.append(logits_g)

        # Stack along event dimension
        stacked_shape = torch.stack(all_shapes, dim=1)  # [batch, events, mixtures]
        stacked_scale = torch.stack(all_scales, dim=1)  # [batch, events, mixtures]
        stacked_logits_g = torch.stack(all_logits_g, dim=1)  # [batch, events, mixtures]

        # Apply event dependencies if enabled
        if self.event_dependency:
            # Apply temperature to dependency matrix to control softmax sharpness
            # Temperature of 1.0 gives balanced importance to dependencies
            temp = 1.0

            # Normalize the dependency matrix using softmax along rows
            # This creates a probability distribution over events for each event
            dependency_weights = F.softmax(self.event_dependency_matrix / temp, dim=1)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Dependency weights: {dependency_weights}")

            # For each event, adjust parameters based on dependencies
            adjusted_shape = torch.zeros_like(stacked_shape)
            adjusted_scale = torch.zeros_like(stacked_scale)

            # Apply weighted dependencies to each parameter with stability safeguards
            for i in range(self.num_events):
                for j in range(self.num_events):
                    # Apply weighted influence from event j to event i
                    weight = dependency_weights[i, j]

                    # Skip negligible influences to improve numerical stability and speed
                    if weight > 1e-6:
                        adjusted_shape[:, i, :] += weight * stacked_shape[:, j, :]
                        adjusted_scale[:, i, :] += weight * stacked_scale[:, j, :]

            # Apply final clamping to ensure reasonable parameter ranges
            adjusted_shape = torch.clamp(adjusted_shape, min=0.1, max=100.0)
            adjusted_scale = torch.clamp(adjusted_scale, min=0.5, max=1000.0)

            # Return adjusted parameters, but keep original mixture weights
            # This allows the mixture components to still be event-specific
            return adjusted_shape, adjusted_scale, stacked_logits_g

        # If no dependency modeling, return original parameters
        return stacked_shape, stacked_scale, stacked_logits_g
