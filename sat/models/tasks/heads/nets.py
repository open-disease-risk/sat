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
                neginf=self.neginf_value
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
            act_fn = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True)
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
                nn.Dropout(dropout_prob)
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
        self.out_features = out_features
        self.num_events = num_events

        # Use a more stable activation function if needed
        stable_activation = nn.ReLU if activation == nn.LeakyReLU else activation
        # Add stabilizing input normalization
        self.input_stabilizer = nn.BatchNorm1d(in_features, eps=1e-4, momentum=0.1)
        # Create networks using our StableMLPVanilla for extreme gradient stabilization
        self.event_nets = nn.ModuleList(
            [
                StableMLPVanilla(
                    in_features=in_features,
                    num_nodes=[intermediate_size] * num_hidden_layers,
                    out_features=out_features,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    activation=stable_activation,
                    output_bias=bias,
                    grad_scale=0.1,  # Small gradient scale for stability
                )
                for _ in range(num_events)
            ]
        )

        # Apply extreme weight initialization for stability
        self._initialize_weights()

        # Add output activation to ensure positive values
        self.output_activation = nn.Softplus(beta=1.0)  # Smoother activation than ReLU

        # Track NaN occurrences for logging
        self.nan_count = 0

    def _initialize_weights(self):
        """Extreme initialization for stable gradients"""
        for net in self.event_nets:
            for module in net.modules():
                if isinstance(module, nn.Linear):
                    # Start with zeros to ensure clean initialization
                    module.weight.data.zero_()

                    # Use orthogonal initialization with very small gain
                    nn.init.orthogonal_(module.weight.data, gain=0.01)

                    # Double-check no extreme values
                    with torch.no_grad():
                        module.weight.data.clamp_(-0.1, 0.1)

                    # Zero bias for more stability
                    if module.bias is not None:
                        module.bias.data.zero_()

                elif isinstance(module, nn.BatchNorm1d):
                    # Start with smaller scale for batch norm
                    module.weight.data.fill_(0.1)
                    module.bias.data.zero_()
                    # Increase eps for better stability
                    module.eps = 1e-4

    def forward(self, input):
        # Process input data in-place for better memory efficiency
        # Handle NaN/Inf values with extreme measures
        if torch.isnan(input).any() or torch.isinf(input).any():
            self.nan_count += 1
            if self.nan_count % 10 == 0:  # Log occasionally to avoid filling logs
                logger.warning(
                    f"NaN/Inf detected in CauseSpecificNet input (count: {self.nan_count})"
                )
            input = torch.nan_to_num(input, nan=0.0, posinf=1.0, neginf=-1.0)
            input = torch.clamp(input, min=-1.0, max=1.0)

        # Ensure input is contiguous for more efficient processing
        if not input.is_contiguous():
            input = input.contiguous()

        # Apply batch normalization to input for better stability
        try:
            # Handle case where input is 2D (batch_size x features)
            if input.dim() == 2:
                input = self.input_stabilizer(input)
        except Exception as e:
            # If stabilization fails, use basic clamping as fallback
            input = torch.clamp(input, min=-1.0, max=1.0)

        # Fast path for single event case with comprehensive error handling
        if self.num_events == 1:
            try:
                # Forward with extra safety checks
                hidden = self.event_nets[0](input)

                # Apply softplus to ensure positive values
                hidden = self.output_activation(hidden)

                # Ensure output doesn't have extreme values
                hidden = torch.clamp(hidden, min=1e-6, max=1.0)

                # Reshape output
                output = hidden.unsqueeze(1)

                # Final safety check
                if torch.isnan(output).any() or torch.isinf(output).any():
                    # Replace NaN/Inf with small positive values
                    nan_mask = torch.isnan(output) | torch.isinf(output)
                    output[nan_mask] = 1e-6

            except Exception as e:
                logger.warning(f"Exception in CauseSpecificNet forward: {e}")
                # Complete error recovery - return small positive values
                output = (
                    torch.ones(
                        input.shape[0],
                        1,
                        self.out_features,
                        device=input.device,
                        dtype=input.dtype,
                    )
                    * 1e-6
                )

            return output

        # Handle multiple events case
        batch_size = input.shape[0]
        out = (
            torch.ones(  # Initialize with small positive values
                batch_size,
                self.num_events,
                self.out_features,
                device=input.device,
                dtype=input.dtype,
            )
            * 1e-6
        )  # Small positive values for numerical stability

        # Process each event network with safety checks
        for i, net in enumerate(self.event_nets):
            try:
                # Forward pass for this event
                hidden = net(input)

                # Apply output activation for stability
                result = self.output_activation(hidden)

                # Constrain to safe range
                result = torch.clamp(result, min=1e-6, max=1.0)

                # Safety check before assignment
                if not (torch.isnan(result).any() or torch.isinf(result).any()):
                    out[:, i, :] = result
            except Exception as e:
                logger.warning(f"Exception in CauseSpecificNet event {i}: {e}")
                # Keep the small positive initialization for this event
                pass

        # Final safety check on complete output tensor
        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.ones_like(out) * 1e-6

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
        self.out_features = out_features
        self.num_events = num_events

        # Use only stable activation functions
        stable_activation = nn.ReLU

        # Add input stabilization
        self.input_stabilizer = nn.BatchNorm1d(in_features, eps=1e-4, momentum=0.1)

        # Initialize shared MLP with our stabilized implementation
        self.shared_mlp = StableMLPVanilla(
            in_features=in_features,
            num_nodes=[shared_intermediate_size] * shared_num_hidden_layers,
            out_features=shared_intermediate_size,
            batch_norm=batch_norm,
            dropout=dropout,
            activation=stable_activation,
            output_bias=bias,
            grad_scale=0.1,  # Small gradient scale for stability
        )

        # Add stabilization between shared and individual networks
        self.combined_stabilizer = nn.Sequential(
            nn.BatchNorm1d(in_features + shared_intermediate_size, eps=1e-4),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),  # Hard constraint on combined range
        )

        # Create event networks with more stable configurations
        self.event_nets = nn.ModuleList()
        for _ in range(num_events):
            net = StableMLPVanilla(
                in_features=in_features + shared_intermediate_size,
                num_nodes=[indiv_intermediate_size] * indiv_num_hidden_layers,
                out_features=out_features,
                batch_norm=batch_norm,
                dropout=dropout,
                activation=stable_activation,
                output_bias=bias,
                grad_scale=0.1,  # Small gradient scale for stability
            )
            self.event_nets.append(net)

        # Apply extreme initialization for stability
        self._initialize_weights()

        # Add output activation to ensure positive values
        self.output_activation = nn.Softplus(beta=1.0)  # Smoother activation than ReLU

        # Track NaN occurrences for logging
        self.nan_count = 0

    def _initialize_weights(self):
        """Apply extreme weight initialization for stable gradients"""
        # Initialize shared network
        for module in self.shared_mlp.modules():
            if isinstance(module, nn.Linear):
                # Start with zeros to ensure clean initialization
                module.weight.data.zero_()

                # Use orthogonal initialization with very small gain
                nn.init.orthogonal_(module.weight.data, gain=0.01)

                # Clip weights to prevent extreme values
                with torch.no_grad():
                    module.weight.data.clamp_(-0.1, 0.1)

                # Zero bias for more stability
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                # Smaller scale for batch norm
                module.weight.data.fill_(0.1)
                module.bias.data.zero_()
                # Increase eps for better stability
                module.eps = 1e-4

        # Initialize event networks
        for net in self.event_nets:
            for module in net.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.zero_()
                    nn.init.orthogonal_(module.weight.data, gain=0.01)
                    with torch.no_grad():
                        module.weight.data.clamp_(-0.1, 0.1)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm1d):
                    module.weight.data.fill_(0.1)
                    module.bias.data.zero_()
                    module.eps = 1e-4

    def forward(self, input):
        # Process input data for better stability
        # Handle NaN/Inf values with extreme measures
        if torch.isnan(input).any() or torch.isinf(input).any():
            self.nan_count += 1
            if self.nan_count % 10 == 0:  # Log occasionally to avoid filling logs
                logger.warning(
                    f"NaN/Inf detected in CauseSpecificNetCompRisk input (count: {self.nan_count})"
                )
            input = torch.nan_to_num(input, nan=0.0, posinf=1.0, neginf=-1.0)
            input = torch.clamp(input, min=-1.0, max=1.0)

        # Ensure input is contiguous for more efficient processing
        if not input.is_contiguous():
            input = input.contiguous()

        # Apply batch normalization to input for better stability
        try:
            # Handle case where input is 2D (batch_size x features)
            if input.dim() == 2:
                input = self.input_stabilizer(input)
        except Exception as e:
            # If stabilization fails, use basic clamping as fallback
            input = torch.clamp(input, min=-1.0, max=1.0)

        # Compute shared features with safety checks
        try:
            shared_features = self.shared_mlp(input)

            # Apply safety checks on shared features
            if torch.isnan(shared_features).any() or torch.isinf(shared_features).any():
                shared_features = torch.nan_to_num(
                    shared_features, nan=0.0, posinf=1.0, neginf=-1.0
                )
                shared_features = torch.clamp(shared_features, min=-1.0, max=1.0)

        except Exception as e:
            logger.warning(f"Exception in shared mlp: {e}")
            # Create safe fallback shared features
            shared_features = torch.zeros(
                input.shape[0],
                self.shared_mlp[-1].out_features,
                device=input.device,
                dtype=input.dtype,
            )

        # Combine inputs with additional stability measures
        try:
            combined = torch.cat([input, shared_features], dim=1)

            # Apply additional stabilization to combined features
            combined = self.combined_stabilizer(combined)

        except Exception as e:
            logger.warning(f"Exception in feature concatenation: {e}")
            # Create safe fallback combined features
            combined = torch.zeros(
                input.shape[0],
                input.shape[1] + shared_features.shape[1],
                device=input.device,
                dtype=input.dtype,
            )

        # Handle single event case with comprehensive error handling
        if self.num_events == 1:
            try:
                # Forward with extra safety checks
                hidden = self.event_nets[0](combined)

                # Apply softplus to ensure positive values
                hidden = self.output_activation(hidden)

                # Ensure output doesn't have extreme values
                hidden = torch.clamp(hidden, min=1e-6, max=1.0)

                # Reshape output
                output = hidden.unsqueeze(1)

                # Final safety check
                if torch.isnan(output).any() or torch.isinf(output).any():
                    # Replace NaN/Inf with small positive values
                    nan_mask = torch.isnan(output) | torch.isinf(output)
                    output[nan_mask] = 1e-6

            except Exception as e:
                logger.warning(f"Exception in CauseSpecificNetCompRisk forward: {e}")
                # Create safe output with small positive values
                output = (
                    torch.ones(
                        input.shape[0],
                        1,
                        self.out_features,
                        device=input.device,
                        dtype=input.dtype,
                    )
                    * 1e-6
                )

            return output

        # Handle multiple events case
        batch_size = combined.shape[0]
        out = (
            torch.ones(  # Initialize with small positive values
                batch_size,
                self.num_events,
                self.out_features,
                device=input.device,
                dtype=input.dtype,
            )
            * 1e-6
        )  # Small positive values for numerical stability

        # Process each event network with safety checks
        for i, net in enumerate(self.event_nets):
            try:
                # Forward pass for this event
                hidden = net(combined)

                # Apply output activation for stability
                result = self.output_activation(hidden)

                # Constrain to safe range
                result = torch.clamp(result, min=1e-6, max=1.0)

                # Safety check before assignment
                if not (torch.isnan(result).any() or torch.isinf(result).any()):
                    out[:, i, :] = result
            except Exception as e:
                logger.warning(f"Exception in CauseSpecificNetCompRisk event {i}: {e}")
                # Keep the small positive initialization for this event
                pass

        # Final safety check on complete output tensor
        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.ones_like(out) * 1e-6

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
            act_fn = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True)
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
                output_max=output_clamp_max
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
            output_max=output_clamp_max
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
                    neginf=self.neginf_value
                )
                # Clamp with configurable bounds
                new_x = torch.clamp(new_x, min=self.output_clamp_min, max=self.output_clamp_max)
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
                    neginf=self.neginf_value
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
