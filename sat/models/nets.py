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

        # flatten all dimensions except batch - more efficient for subsequent operations
        layers.append(nn.Flatten())

        # arrange hidden layers with the first one accepting the in_features
        for i in range(num_hidden_layers):
            logger.debug(f"Creating hidden linear layer number {i}")
            layers.append(
                nn.Linear(
                    in_features if i == 0 else intermediate_size,
                    intermediate_size,
                    bias,
                )
            )
            if batch_norm:
                logger.debug("Add batch norm")
                layers.append(nn.BatchNorm1d(intermediate_size))

            # Use functional activation for more flexibility and efficiency
            if activation == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif activation == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(activation())

            if dropout:
                logger.debug("Add dropout")
                layers.append(nn.Dropout(dropout))

        if out_features > 0:
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
        # Ensure input is contiguous for more efficient processing
        if not input.is_contiguous():
            input = input.contiguous()
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
        self.out_features = out_features
        self.num_events = num_events

        # Create networks all at once instead of in a loop for better initialization efficiency
        self.event_nets = nn.ModuleList(
            [
                tt.practical.MLPVanilla(
                    in_features=in_features,
                    num_nodes=[intermediate_size] * num_hidden_layers,
                    out_features=out_features,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    activation=activation,
                    output_bias=bias,
                )
                for _ in range(num_events)
            ]
        )

    def forward(self, input):
        # Ensure input is contiguous for more efficient processing
        if not input.is_contiguous():
            input = input.contiguous()

        # Fast path for the common single event case
        if self.num_events == 1:
            return self.event_nets[0](input).unsqueeze(1)

        # Pre-allocate output tensor with correct size for multiple events
        batch_size = input.shape[0]
        out = torch.empty(
            batch_size,
            self.num_events,
            self.out_features,
            device=input.device,
            dtype=input.dtype,
        )

        # Process each event network - parallelize if beneficial
        for i, net in enumerate(self.event_nets):
            out[:, i, :] = net(input)

        return out


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
        self.in_features = in_features
        self.shared_intermediate_size = shared_intermediate_size
        self.out_features = out_features
        self.num_events = num_events

        # Initialize shared MLP for feature extraction
        self.shared_mlp = tt.practical.MLPVanilla(
            in_features=in_features,
            num_nodes=[shared_intermediate_size] * shared_num_hidden_layers,
            out_features=shared_intermediate_size,
            batch_norm=batch_norm,
            dropout=dropout,
            activation=activation,
            output_bias=bias,
        )

        # Create event networks efficiently as a list comprehension
        self.event_nets = nn.ModuleList(
            [
                tt.practical.MLPVanilla(
                    in_features=in_features + shared_intermediate_size,
                    num_nodes=[indiv_intermediate_size] * indiv_num_hidden_layers,
                    out_features=out_features,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    activation=activation,
                    output_bias=bias,
                )
                for _ in range(num_events)
            ]
        )

    def forward(self, input):
        # Ensure input is contiguous for better performance with subsequent operations
        if not input.is_contiguous():
            input = input.contiguous()

        # Compute shared features once for all event networks
        shared_features = self.shared_mlp(input)

        # Efficient residual connection with pre-allocation and contiguity enforcement
        combined = torch.cat([input, shared_features], dim=1)

        # Fast path for single event case (common scenario)
        if self.num_events == 1:
            return self.event_nets[0](combined).unsqueeze(1)

        # Optimize batch processing for multiple events with pre-allocated tensor
        batch_size = combined.shape[0]
        out = torch.empty(
            batch_size,
            self.num_events,
            self.out_features,
            device=input.device,
            dtype=input.dtype,
        )

        # Process each event network efficiently
        for i, net in enumerate(self.event_nets):
            out[:, i, :] = net(combined)

        return out
