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
