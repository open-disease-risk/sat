"""Building blocks for networks"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
import torchtuples as tt

from logging import DEBUG, ERROR
from torch import nn
import torch.nn.functional as F

from sat.utils import logging

logger = logging.get_default_logger()


class StableLinear(nn.Linear):
    """
    Custom Linear layer with extreme gradient stability measures.

    This layer includes multiple stability measures to prevent numerical issues during training:
    1. Orthogonal initialization with small gain to maintain controlled weight magnitudes
    2. Weight scaling during forward/backward passes to prevent exploding gradients
    3. Noise addition during training to break symmetry and prevent vanishing gradients
    4. Input and output value clamping to prevent extreme activations
    5. NaN and Inf detection and replacement

    Parameters
    ----------
    in_features : int
        Size of each input sample
    out_features : int
        Size of each output sample
    bias : bool, default=True
        If set to False, the layer will not learn an additive bias
    grad_scale : float, default=0.1
        Scale factor for gradients, smaller values provide more stability but slower learning
    eps : float, default=1e-8
        Small constant to prevent division by zero
    max_norm : float, default=1.0
        Maximum norm for gradient clipping
    init_gain : float, default=0.01
        Gain for orthogonal weight initialization, smaller values create more stable initialization
    weight_clamp_value : float, default=0.1
        Maximum absolute value for initial weights
    noise_scale : float, default=1e-5
        Scale of noise to add during training to break symmetry
    nan_value : float, default=0.0
        Value to replace NaNs with during forward pass
    posinf_value : float, default=1.0
        Value to replace positive infinities with
    neginf_value : float, default=-1.0
        Value to replace negative infinities with
    output_min : float, default=-10.0
        Minimum value for output clamping
    output_max : float, default=10.0
        Maximum value for output clamping
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        grad_scale=0.1,
        eps=1e-8,
        max_norm=1.0,
        init_gain=0.01,
        weight_clamp_value=0.1,
        noise_scale=1e-5,
        nan_value=0.0,
        posinf_value=1.0,
        neginf_value=-1.0,
        output_min=-10.0,
        output_max=10.0,
    ):
        super().__init__(in_features, out_features, bias)
        self.grad_scale = grad_scale
        self.eps = eps
        self.max_norm = max_norm
        self.init_gain = init_gain
        self.weight_clamp_value = weight_clamp_value
        self.noise_scale = noise_scale
        self.nan_value = nan_value
        self.posinf_value = posinf_value
        self.neginf_value = neginf_value
        self.output_min = output_min
        self.output_max = output_max
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize with orthogonal weights with configurable gain
        nn.init.orthogonal_(self.weight.data, gain=self.init_gain)
        # Clip weights to prevent extreme values using configurable clamp value
        with torch.no_grad():
            self.weight.data.clamp_(-self.weight_clamp_value, self.weight_clamp_value)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        # Check for NaN in input with configurable replacement values
        if torch.isnan(input).any() or torch.isinf(input).any():
            input = torch.nan_to_num(
                input,
                nan=self.nan_value,
                posinf=self.posinf_value,
                neginf=self.neginf_value,
            )

        # Apply a scaled version of weight for numerical stability
        scaled_weight = self.weight * self.grad_scale

        # Compute output with scaled weight
        if self.bias is not None:
            output = F.linear(input, scaled_weight, self.bias * self.grad_scale)
        else:
            output = F.linear(input, scaled_weight, None)

        # Apply noise to break symmetry with configurable scale
        if self.training:
            noise = torch.randn_like(output) * self.noise_scale
            output = output + noise

        # Rescale output to account for the grad_scale
        output = output / self.grad_scale

        # Safe clipping of output values with configurable bounds
        output = torch.clamp(output, self.output_min, self.output_max)

        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, grad_scale={self.grad_scale}, "
            f"init_gain={self.init_gain}, weight_clamp={self.weight_clamp_value}, "
            f"noise_scale={self.noise_scale}, output_range=[{self.output_min}, {self.output_max}]"
        )


class StableMLPVanilla(nn.Module):
    """A stable version of the MLPVanilla from torchtuples using StableLinear layers"""

    def __init__(
        self,
        in_features,
        num_nodes,
        out_features,
        batch_norm=True,
        dropout=None,
        activation=nn.ReLU,
        output_activation=None,
        output_bias=True,
        grad_scale=0.1,
    ):
        super().__init__()

        self.batch_norm = batch_norm
        self.dropout = dropout
        self.output_activation = output_activation
        self.output_bias = output_bias
        self.grad_scale = grad_scale

        # Create a list of layers
        self.net = nn.ModuleList()

        # Add input layer
        if len(num_nodes) > 0:
            self.net.append(
                StableLinear(
                    in_features,
                    num_nodes[0],
                    bias=not batch_norm,
                    grad_scale=grad_scale,
                )
            )
            if batch_norm:
                self.net.append(nn.BatchNorm1d(num_nodes[0], eps=1e-4))
            if activation:
                self.net.append(activation())
            if dropout:
                self.net.append(nn.Dropout(dropout))

            # Add hidden layers
            for i in range(len(num_nodes) - 1):
                self.net.append(
                    StableLinear(
                        num_nodes[i],
                        num_nodes[i + 1],
                        bias=not batch_norm,
                        grad_scale=grad_scale,
                    )
                )
                if batch_norm:
                    self.net.append(nn.BatchNorm1d(num_nodes[i + 1], eps=1e-4))
                if activation:
                    self.net.append(activation())
                if dropout:
                    self.net.append(nn.Dropout(dropout))

            # Add output layer
            self.net.append(
                StableLinear(
                    num_nodes[-1], out_features, bias=output_bias, grad_scale=grad_scale
                )
            )
        else:
            # Direct connection from input to output when no hidden layers
            self.net.append(
                StableLinear(
                    in_features, out_features, bias=output_bias, grad_scale=grad_scale
                )
            )

        if output_activation:
            self.net.append(output_activation())

    def forward(self, x):
        # Handle NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            x = torch.clamp(x, min=-10.0, max=10.0)

        # Apply each layer in sequence with safety checks
        for layer in self.net:
            try:
                x = layer(x)
                # Check after each layer and fix any NaN/Inf
                if torch.isnan(x).any() or torch.isinf(x).any():
                    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                    x = torch.clamp(x, min=-10.0, max=10.0)
            except Exception as e:
                logger.warning(f"Exception in StableMLPVanilla layer: {e}")
                # Return safe output on error
                return torch.zeros(
                    x.size(0), self.net[-1].out_features, device=x.device, dtype=x.dtype
                )

        return x


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


class SimpleMLPSequence(nn.Module):
    """A simplified MLP implementation with the same interface as StableMLPSequence.

    This implementation provides the same API as StableMLPSequence but without the stability
    measures. It's useful for comparison and for cases where the extreme stability measures
    are not required.

    Parameters
    ----------
    input_size : int
        Size of each input sample
    hidden_sizes : list of int
        Sizes of each hidden layer
    output_size : int
        Size of each output sample
    dropout_prob : float, default=0.1
        Dropout probability to prevent overfitting
    activation : str, default="gelu"
        Activation function to use ("gelu", "relu", "leaky_relu", "silu", "swish")
    bias : bool, default=True
        Whether to use bias in linear layers
    use_residual : bool, default=False
        Whether to use residual connections
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        dropout_prob=0.1,
        activation="gelu",
        bias=True,
        output_activation=None,
        use_residual=False,
        # Unused parameters kept for API compatibility
        layer_norm_eps=1e-12,
        grad_scale=0.1,
        weight_init_std=0.02,
        softplus_beta=1.0,
        nan_value=0.0,
        posinf_value=1.0,
        neginf_value=-1.0,
        output_clamp_min=-10.0,
        output_clamp_max=10.0,
        leaky_relu_negative_slope=0.01,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_residual = use_residual
        self.output_size = output_size

        # Setup activation function
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(
                negative_slope=leaky_relu_negative_slope, inplace=True
            )
        elif activation == "silu" or activation == "swish":
            act_fn = nn.SiLU(inplace=True)
        else:
            # Default to GELU
            act_fn = nn.GELU()

        # Setup output activation if specified
        if output_activation == "sigmoid":
            self.out_activation = nn.Sigmoid()
        elif output_activation == "tanh":
            self.out_activation = nn.Tanh()
        elif output_activation == "softplus":
            self.out_activation = nn.Softplus(beta=softplus_beta)
        else:
            self.out_activation = None

        # Build layers
        sizes = [input_size] + hidden_sizes

        for i in range(len(sizes) - 1):
            layer = nn.Sequential(
                nn.Linear(sizes[i], sizes[i + 1], bias=bias),
                nn.BatchNorm1d(sizes[i + 1]) if i < len(sizes) - 2 else nn.Identity(),
                act_fn,
                nn.Dropout(dropout_prob),
            )
            self.layers.append(layer)

        # Output layer
        self.output_layer = nn.Linear(sizes[-1], output_size, bias=bias)

    def forward(self, x):
        # No need for input LayerNorm in this simplified version
        hidden = x

        # Process through hidden layers
        for i, layer in enumerate(self.layers):
            if self.use_residual and hidden.shape == layer(hidden).shape:
                hidden = hidden + layer(hidden)
            else:
                hidden = layer(hidden)

        # Output layer
        output = self.output_layer(hidden)

        # Apply output activation if specified
        if self.out_activation is not None:
            output = self.out_activation(output)

        return output

    def extra_repr(self):
        return f"layers={len(self.layers)}, output_size={self.output_size}, use_residual={self.use_residual}"


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
        self.event_nets = nn.ModuleList()
        for _ in range(num_events):
            net = tt.practical.MLPVanilla(
                in_features=in_features,
                num_nodes=[intermediate_size] * num_hidden_layers,
                out_features=out_features,
                batch_norm=batch_norm,
                dropout=dropout,
                activation=activation,
                output_bias=bias,
            )
            self.event_nets.append(net)

    def forward(self, input):
        out = [net(input) for net in self.event_nets]
        out = torch.stack(out, dim=1)
        # Optimize for the common single event case
        if len(self.event_nets) == 1:
            return self.event_nets[0](input).unsqueeze(1)

        # More efficient batch processing for multiple events
        batch_size = input.shape[0]
        out = torch.empty(
            batch_size,
            len(self.event_nets),
            self.event_nets[0].out_features,
            device=input.device,
            dtype=input.dtype,
        )

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
        self.shared_mlp = tt.practical.MLPVanilla(
            in_features=in_features,
            num_nodes=[shared_intermediate_size] * shared_num_hidden_layers,
            out_features=shared_intermediate_size,
            batch_norm=batch_norm,
            dropout=dropout,
            activation=activation,
            output_bias=bias,
        )
        self.event_nets = nn.ModuleList()
        for _ in range(num_events):
            net = tt.practical.MLPVanilla(
                in_features=in_features + shared_intermediate_size,
                num_nodes=[indiv_intermediate_size] * indiv_num_hidden_layers,
                out_features=out_features,
                batch_norm=batch_norm,
                dropout=dropout,
                activation=activation,
                output_bias=bias,
            )
            self.event_nets.append(net)

    def forward(self, input):
        out = self.shared_mlp(input)
        out = torch.cat([input, out], dim=1)  # residual connections
        out = [net(out) for net in self.event_nets]
        out = torch.stack(out, dim=1)
        # Compute shared features once
        shared_features = self.shared_mlp(input)

        # More efficient residual connection with pre-allocation
        combined = torch.cat([input, shared_features], dim=1)  # residual connections

        # Batch processing for event networks instead of list comprehension
        if len(self.event_nets) == 1:
            # Optimization for single event case (common scenario)
            out = self.event_nets[0](combined).unsqueeze(1)
        else:
            # More efficient than list comprehension for multiple events
            batch_size = combined.shape[0]
            out = torch.empty(
                batch_size,
                len(self.event_nets),
                self.event_nets[0].out_features,
                device=input.device,
                dtype=input.dtype,
            )
            for i, net in enumerate(self.event_nets):
                out[:, i, :] = net(combined)

        return out


import collections


class StableMLPSequence(nn.Module):
    """
    A highly stable MLP implementation with proper layer normalization and residual connections.

    This implementation follows best practices for numerical stability:
    - LayerNorm for normalization (more stable than BatchNorm)
    - Careful weight initialization
    - GELU activation (smoother gradients than ReLU)
    - Proper ordering: linear → norm → activation → dropout
    - NaN/Inf checking during forward pass
    - Residual connections to improve gradient flow

    Parameters
    ----------
    input_size : int
        Size of each input sample
    hidden_sizes : list of int
        Sizes of each hidden layer
    output_size : int
        Size of each output sample
    dropout_prob : float, default=0.1
        Dropout probability to prevent overfitting
    activation : str, default="gelu"
        Activation function to use ("gelu", "relu", "leaky_relu", "silu", "swish")
    layer_norm_eps : float, default=1e-12
        Small constant for layer normalization numerical stability
    bias : bool, default=True
        Whether to use bias in linear layers
    output_activation : str, default=None
        Activation function for output layer ("sigmoid", "tanh", "softplus")
    use_residual : bool, default=True
        Whether to use residual connections
    grad_scale : float, default=0.1
        Scale factor for gradients in StableLinear layers
    weight_init_std : float, default=0.02
        Standard deviation for weight initialization
    softplus_beta : float, default=1.0
        Beta parameter for softplus activation (when output_activation="softplus")
    nan_value : float, default=0.0
        Value to replace NaNs with during forward pass
    posinf_value : float, default=1.0
        Value to replace positive infinities with
    neginf_value : float, default=-1.0
        Value to replace negative infinities with
    output_clamp_min : float, default=-10.0
        Minimum value for output clamping
    output_clamp_max : float, default=10.0
        Maximum value for output clamping
    leaky_relu_negative_slope : float, default=0.01
        Negative slope for LeakyReLU activation
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        dropout_prob=0.1,
        activation="gelu",
        layer_norm_eps=1e-12,
        bias=True,
        output_activation=None,
        use_residual=True,
        # Stability parameters
        grad_scale=0.1,
        weight_init_std=0.02,
        softplus_beta=1.0,
        nan_value=0.0,
        posinf_value=1.0,
        neginf_value=-1.0,
        output_clamp_min=-10.0,
        output_clamp_max=10.0,
        leaky_relu_negative_slope=0.01,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_layernorm = nn.LayerNorm(input_size, eps=layer_norm_eps)
        self.use_residual = use_residual
        self.output_size = output_size

        # Store stability parameters
        self.grad_scale = grad_scale
        self.weight_init_std = weight_init_std
        self.softplus_beta = softplus_beta
        self.nan_value = nan_value
        self.posinf_value = posinf_value
        self.neginf_value = neginf_value
        self.output_clamp_min = output_clamp_min
        self.output_clamp_max = output_clamp_max

        # Setup activation function
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(
                negative_slope=leaky_relu_negative_slope, inplace=True
            )
        elif activation == "silu" or activation == "swish":
            act_fn = nn.SiLU(inplace=True)
        else:
            # Default to GELU for best stability
            act_fn = nn.GELU()

        # Setup output activation if specified
        if output_activation == "sigmoid":
            self.out_activation = nn.Sigmoid()
        elif output_activation == "tanh":
            self.out_activation = nn.Tanh()
        elif output_activation == "softplus":
            self.out_activation = nn.Softplus(beta=softplus_beta)  # Configurable beta
        else:
            self.out_activation = None

        # Build layers with proper normalization and residual connections
        sizes = [input_size] + hidden_sizes

        for i in range(len(sizes) - 1):
            # Create layer block
            layer_dict = collections.OrderedDict()

            # Linear transformation with configurable stability parameters
            layer_dict[f"linear_{i}"] = StableLinear(
                sizes[i],
                sizes[i + 1],
                bias=bias,
                grad_scale=grad_scale,
                nan_value=nan_value,
                posinf_value=posinf_value,
                neginf_value=neginf_value,
                output_min=output_clamp_min,
                output_max=output_clamp_max,
            )

            # Normalization (after linear, before activation)
            layer_dict[f"layernorm_{i}"] = nn.LayerNorm(
                sizes[i + 1], eps=layer_norm_eps
            )

            # Activation
            layer_dict[f"activation_{i}"] = act_fn

            # Dropout
            layer_dict[f"dropout_{i}"] = nn.Dropout(dropout_prob)

            # Add as sequential block
            self.layers.append(nn.Sequential(layer_dict))

        # Output layer (separate from hidden layers) with stability parameters
        self.output_layer = StableLinear(
            sizes[-1],
            output_size,
            bias=bias,
            grad_scale=grad_scale,
            nan_value=nan_value,
            posinf_value=posinf_value,
            neginf_value=neginf_value,
            output_min=output_clamp_min,
            output_max=output_clamp_max,
        )

        # Initialize weights carefully
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights with controlled variance to prevent extreme values"""
        if isinstance(module, (nn.Linear, StableLinear)):
            # Use configurable std for initialization
            module.weight.data.normal_(mean=0.0, std=self.weight_init_std)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        """Forward pass with comprehensive stability measures"""
        # Handle batch or sequence input
        orig_shape = x.shape
        if len(orig_shape) > 2:
            # Flatten all dimensions except batch
            x = x.reshape(orig_shape[0], -1)

        # Initial normalization
        x = self.input_layernorm(x)

        # Process through hidden layers with residual connections
        prev_x = x
        for i, layer in enumerate(self.layers):
            # Apply layer
            new_x = layer(prev_x)

            # Apply residual connection if shapes match and enabled
            if self.use_residual and prev_x.shape == new_x.shape:
                new_x = new_x + prev_x

            # Check for NaN/Inf and handle with configurable values
            if torch.isnan(new_x).any() or torch.isinf(new_x).any():
                new_x = torch.nan_to_num(
                    new_x,
                    nan=self.nan_value,
                    posinf=self.posinf_value,
                    neginf=self.neginf_value,
                )
                # Clamp with configurable bounds
                new_x = torch.clamp(
                    new_x, min=self.output_clamp_min, max=self.output_clamp_max
                )
                logger.warning(f"NaN/Inf detected in StableMLPSequence layer {i}")

            # Update for next layer
            prev_x = new_x

        # Apply output layer
        try:
            x = self.output_layer(prev_x)

            # Apply output activation if specified
            if self.out_activation is not None:
                x = self.out_activation(x)

            # Final safety check with configurable values
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(
                    x,
                    nan=self.nan_value,
                    posinf=self.posinf_value,
                    neginf=self.neginf_value,
                )
                # Clamp with configurable bounds
                x = torch.clamp(x, min=self.output_clamp_min, max=self.output_clamp_max)
        except Exception as e:
            logger.warning(f"Exception in StableMLPSequence output layer: {e}")
            # Create safe dummy output if error occurs
            x = torch.zeros(
                prev_x.size(0),
                self.output_size,
                device=prev_x.device,
                dtype=prev_x.dtype,
            )

        return x

    def extra_repr(self) -> str:
        """Add stability parameters to the string representation"""
        return (
            f"input_size={self.input_layernorm.normalized_shape[0]}, "
            f"output_size={self.output_size}, "
            f"use_residual={self.use_residual}, "
            f"grad_scale={self.grad_scale}, "
            f"weight_init_std={self.weight_init_std}, "
            f"output_clamp=[{self.output_clamp_min}, {self.output_clamp_max}]"
        )
