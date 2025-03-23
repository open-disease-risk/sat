"""Building blocks for networks"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torchtuples as tt

from logging import DEBUG, ERROR
from torch import nn

from sat.utils import logging

logger = logging.get_default_logger()


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        intermediate_size,
        out_features,
        num_hidden_layers,
        bias=True,
        batch_norm=True,
        dropout=0.0,
        activation=nn.LeakyReLU,
    ):
        super().__init__()
        layers = []

        # flatten all dimensions except batch
        layers.append(nn.Flatten())

        # arrange hidden layers with the first one accepting the in_features
        for i in range(num_hidden_layers):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Creating hidden linear layer number {i}")
            layers.append(
                nn.Linear(
                    in_features if i == 0 else intermediate_size,
                    intermediate_size,
                    bias,
                )
            )
            if batch_norm:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Add batch norm")
                layers.append(nn.BatchNorm1d(intermediate_size))
            layers.append(activation())
            if dropout:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Add dropout")
                layers.append(nn.Dropout(dropout))

        if out_features > 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Creating output linear layer number")

            layers.append(
                nn.Linear(
                    in_features if num_hidden_layers == 0 else intermediate_size,
                    out_features,
                    bias,
                )
            )
        self.linears = nn.Sequential(*layers)

    def forward(self, input):
        out = self.linears(input)
        return out


class SimpleMLP(nn.Module):
    """Simple network structure for competing risks."""

    def __init__(
        self,
        in_features,
        intermediate_size,
        num_hidden_layers,
        out_features,
        batch_norm=True,
        dropout=None,
        bias=True,
        activation=nn.LeakyReLU,
    ):
        super().__init__()
        self.mlp = tt.practical.MLPVanilla(
            in_features=in_features,
            num_nodes=[intermediate_size] * num_hidden_layers,
            out_features=out_features,
            batch_norm=batch_norm,
            dropout=dropout,
            activation=activation,
            output_bias=bias,
        )

    def forward(self, input):
        out = self.mlp(input)
        return out


class CauseSpecificNet(nn.Module):
    def __init__(
        self,
        in_features,
        intermediate_size,
        num_hidden_layers,
        out_features,
        batch_norm=True,
        dropout=None,
        bias=True,
        activation=nn.LeakyReLU,
        num_events=1,
    ):
        super().__init__()
        self.out_features = out_features  # Store out_features as class attribute

        # Define a more conservative weight initialization function
        # This matches the one in CauseSpecificNetCompRisk for consistency
        def conservative_init(w):
            if num_events > 1:
                # For multi-event case, use a smaller scale factor to prevent
                # large activation values that can cause numerical instability
                scale_factor = 0.5  # Reduce the variance of the initialization
                fan_mode = "fan_in"  # Initialize based on input size for stability
                nn.init.kaiming_normal_(w, a=0.0, mode=fan_mode, nonlinearity="relu")
                with torch.no_grad():
                    w.mul_(scale_factor)  # Scale down the weights
            else:
                # For single event case, use standard initialization
                nn.init.kaiming_normal_(w, nonlinearity="relu")

            # Apply additional clipping to prevent extreme initial values
            with torch.no_grad():
                w.clamp_(-0.1, 0.1)  # Clip initial weights to a reasonable range

        # Create the event-specific networks
        self.event_nets = nn.ModuleList()
        for i in range(num_events):
            net = tt.practical.MLPVanilla(
                in_features=in_features,
                num_nodes=[intermediate_size] * num_hidden_layers,
                out_features=out_features,
                batch_norm=batch_norm,
                dropout=dropout,
                activation=activation,
                output_bias=bias,
                w_init_=conservative_init,  # Use our custom initialization
            )

            # For multi-event case, initialize the final output layer to small values
            if num_events > 1:
                with torch.no_grad():
                    # Get the output layer (last module)
                    output_layer = net.net[-1]
                    if isinstance(output_layer, nn.Linear):
                        # Initialize output weights with small values
                        nn.init.zeros_(output_layer.weight)
                        # Initialize bias to small negative values
                        if output_layer.bias is not None:
                            nn.init.constant_(output_layer.bias, -1.0)

            self.event_nets.append(net)

    def forward(self, input):
        # Fast path for common single event case
        if len(self.event_nets) == 1:
            return self.event_nets[0](input).unsqueeze(1)

        # Optimize multi-event case with torch.stack
        # This is more memory efficient by avoiding pre-allocation and indexing
        outputs = [net(input) for net in self.event_nets]
        return torch.stack(outputs, dim=1)


class CauseSpecificNetCompRisk(nn.Module):
    def __init__(
        self,
        in_features,
        indiv_intermediate_size,
        indiv_num_hidden_layers,
        shared_intermediate_size,
        shared_num_hidden_layers,
        out_features,
        batch_norm=True,
        dropout=None,
        bias=True,
        activation=nn.LeakyReLU,
        num_events=1,
    ):
        super().__init__()
        self.out_features = out_features  # Store out_features as class attribute
        self.shared_intermediate_size = shared_intermediate_size  # Store for clarity

        # Define a more conservative weight initialization function
        # Using He initialization with a smaller scale factor for multi-event case
        def conservative_init(w):
            if num_events > 1:
                # For multi-event case, use a smaller scale factor to prevent
                # large activation values that can cause numerical instability
                scale_factor = 0.5  # Reduce the variance of the initialization
                fan_mode = "fan_in"  # Initialize based on input size for stability
                nn.init.kaiming_normal_(w, a=0.0, mode=fan_mode, nonlinearity="relu")
                with torch.no_grad():
                    w.mul_(scale_factor)  # Scale down the weights
            else:
                # For single event case, use standard initialization
                nn.init.kaiming_normal_(w, nonlinearity="relu")

            # Apply additional clipping to prevent extreme initial values
            with torch.no_grad():
                w.clamp_(-0.1, 0.1)  # Clip initial weights to a reasonable range

        # Create the shared MLP with custom initialization
        self.shared_mlp = tt.practical.MLPVanilla(
            in_features=in_features,
            num_nodes=[shared_intermediate_size] * shared_num_hidden_layers,
            out_features=shared_intermediate_size,
            batch_norm=batch_norm,
            dropout=dropout,
            activation=activation,
            output_bias=bias,
            w_init_=conservative_init,  # Use our custom initialization
        )

        # Create the per-event networks with custom initialization
        self.event_nets = nn.ModuleList()
        for i in range(num_events):
            net = tt.practical.MLPVanilla(
                in_features=in_features + shared_intermediate_size,
                num_nodes=[indiv_intermediate_size] * indiv_num_hidden_layers,
                out_features=out_features,
                batch_norm=batch_norm,
                dropout=dropout,
                activation=activation,
                output_bias=bias,
                w_init_=conservative_init,  # Use our custom initialization
            )

            # For multi-event case, initialize the final output layer close to zero
            # This helps ensure the initial hazard rates are small and stable
            if num_events > 1:
                with torch.no_grad():
                    # Get the output layer (last module)
                    output_layer = net.net[-1]
                    if isinstance(output_layer, nn.Linear):
                        # Initialize output weights with small values
                        nn.init.zeros_(output_layer.weight)
                        # Initialize with a slight bias toward negative values
                        # This will produce initially small hazard values after softplus
                        if output_layer.bias is not None:
                            nn.init.constant_(output_layer.bias, -1.0)

            self.event_nets.append(net)

    def forward(self, input):
        # Compute shared features once
        shared_features = self.shared_mlp(input)

        # Create combined input for all networks
        combined = torch.cat([input, shared_features], dim=1)  # residual connections

        # Fast path for single event case (common scenario)
        if len(self.event_nets) == 1:
            return self.event_nets[0](combined).unsqueeze(1)

        # Optimize multi-event case with torch.stack
        # This avoids pre-allocation, indexing, and improves memory efficiency
        outputs = [net(combined) for net in self.event_nets]
        return torch.stack(outputs, dim=1)
