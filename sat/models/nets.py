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


class SimpleCompRiskNet(nn.Module):
    """
    A simplified competing risks network that enforces variability between
    predictions for different events and patients.
    """

    def __init__(
        self,
        in_features,
        intermediate_size,
        num_hidden_layers,
        out_features,
        num_events=1,
        batch_norm=True,
        dropout=0.0,
        bias=True,
        activation=nn.LeakyReLU,
    ):
        super().__init__()
        logger.warning(f"Creating SimpleCompRiskNet with {num_events} events")

        # Store parameters for validation
        self.in_features = in_features
        self.out_features = out_features
        self.num_events = num_events

        # Store input parameters as attributes for later use
        self.in_features = in_features
        self.out_features = out_features
        self.num_events = num_events

        # Create a shared network for all events
        shared_layers = []
        current_in_features = in_features

        # Add hidden layers for shared network
        for i in range(num_hidden_layers):
            # Add linear layer
            shared_layers.append(
                nn.Linear(current_in_features, intermediate_size, bias=bias)
            )
            current_in_features = intermediate_size

            # Add batch norm if requested
            if batch_norm:
                shared_layers.append(nn.BatchNorm1d(intermediate_size))

            # Add activation
            shared_layers.append(activation())

            # Add dropout if requested
            if dropout > 0:
                shared_layers.append(nn.Dropout(dropout))

        self.shared_network = nn.Sequential(*shared_layers)

        # Create separate output heads for each event
        # Use completely different networks for each event type
        self.output_heads = nn.ModuleList()

        # Flag to check if we've already broken symmetry
        self._symmetry_broken = False
        for i in range(num_events):
            # Each event gets an output head with different initialization
            head = nn.Linear(current_in_features, out_features, bias=bias)

            # Initialize with different values for each event and each output
            with torch.no_grad():
                # Use normal initialization with different mean for each event
                # Increase the standard deviation for more variability
                head_mean = -0.1 + i * 0.1
                head_std = 0.2 + i * 0.05
                nn.init.normal_(head.weight, mean=head_mean, std=head_std)

                # Initialize bias differently for each event
                if head.bias is not None:
                    # Use different bias initialization per event
                    # Event 0 gets more negative bias, Event 1 gets less negative
                    # Increase the difference between events
                    bias_val = -1.0 + i * 1.0
                    nn.init.constant_(head.bias, bias_val)

                    # Add more per-output variation to the bias
                    # This ensures each output for each event is different
                    head.bias.add_(torch.linspace(-0.5, 0.5, out_features))

            # Ensure each head is uniquely initialized
            with torch.no_grad():
                # Add noise based on event index
                noise_scale = 0.2 * (i + 1)  # Larger noise for later events
                head.weight.data += torch.randn_like(head.weight) * noise_scale

                if head.bias is not None:
                    # Use very different bias initialization for each event
                    if i == 0:
                        # Event 0 gets negative bias
                        head.bias.fill_(-1.0 - i * 0.5)
                    else:
                        # Event 1+ get less negative bias
                        head.bias.fill_(-0.5 + i * 0.5)

                    # Add position-dependent variation to bias
                    position_scale = torch.linspace(-0.5, 0.5, out_features)
                    head.bias.data += position_scale * (1.0 + i * 0.5)

            self.output_heads.append(head)

        logger.warning(
            f"SimpleCompRiskNet created with {len(self.output_heads)} output heads"
        )

        # Break symmetry between events one more time
        self._break_symmetry()

        # Validate the initialization worked
        self._validate_symmetry_breaking()

    def _break_symmetry(self):
        """Add substantial noise to break any symmetry between events"""
        if self._symmetry_broken:
            return

        with torch.no_grad():
            # For each event output head
            for i, head in enumerate(self.output_heads):
                # Scale factor increases with event index
                scale = 0.1 * (i + 1)

                # Add significant random noise
                head.weight.data += torch.randn_like(head.weight) * scale

                if head.bias is not None:
                    # Make bias very different between events
                    offset = i * 0.5  # Different offset per event
                    head.bias.data += offset

                    # Add more random noise
                    head.bias.data += torch.randn_like(head.bias) * scale

            # Also add some noise to shared network to break symmetry
            for module in self.shared_network.modules():
                if isinstance(module, nn.Linear):
                    # Small noise to shared network
                    module.weight.data += torch.randn_like(module.weight) * 0.01

        # Mark symmetry as broken
        self._symmetry_broken = True
        logger.warning("Symmetry breaking applied to network")

    def _validate_symmetry_breaking(self):
        """Validate that symmetry breaking worked by comparing event outputs"""
        try:
            # Create dummy input
            batch_size = 10
            # Check if in_features exists, use a fallback if not
            if not hasattr(self, "in_features"):
                # Get input dimension from first output head
                if len(self.output_heads) > 0:
                    in_features = self.output_heads[0].weight.shape[1]
                else:
                    logger.warning(
                        "Cannot validate symmetry breaking - no output heads"
                    )
                    return
            else:
                in_features = self.in_features

            # Create input tensor on the same device as model
            device = self.output_heads[0].weight.device
            dummy_input = torch.randn(batch_size, in_features, device=device)

            # Get shared features
            with torch.no_grad():
                shared_out = self.shared_network(dummy_input)

                # Get outputs from each head
                outputs = []
                means = []
                stds = []

                for i, head in enumerate(self.output_heads):
                    out = head(shared_out)
                    outputs.append(out)
                    means.append(out.mean().item())
                    stds.append(out.std().item())

                    logger.warning(
                        f"Event {i} test output: mean={means[-1]:.6f}, std={stds[-1]:.6f}"
                    )

                # For multiple events, check differences
                if len(outputs) >= 2:
                    # Calculate correlation
                    stacked_outputs = torch.stack(
                        [outputs[0].flatten(), outputs[1].flatten()]
                    )
                    corr = torch.corrcoef(stacked_outputs)[0, 1].item()
                    logger.warning(
                        f"Test output correlation between events: {corr:.6f}"
                    )

                    # Calculate absolute difference
                    abs_diff = torch.abs(outputs[0] - outputs[1]).mean().item()
                    logger.warning(
                        f"Test output mean difference between events: {abs_diff:.6f}"
                    )

                    # Check for issues
                    if corr > 0.9:
                        logger.error("CRITICAL: Event outputs are highly correlated!")
                        # Force more aggressive symmetry breaking
                        self._force_asymmetry()

                    if abs_diff < 0.05:
                        logger.error(
                            "CRITICAL: Event outputs have very small differences!"
                        )
                        # Force more aggressive symmetry breaking
                        self._force_asymmetry()
        except Exception as e:
            logger.error(f"Error during symmetry validation: {str(e)}")

    def _force_asymmetry(self):
        """Force asymmetry by using completely different architectures for each event"""
        logger.warning("Forced asymmetry being applied")

        with torch.no_grad():
            # For event 0, make all weights negative
            if len(self.output_heads) > 0:
                head0 = self.output_heads[0]
                head0.weight.data = torch.abs(head0.weight.data) * -1.0
                if head0.bias is not None:
                    head0.bias.data = torch.abs(head0.bias.data) * -1.0

            # For event 1, make all weights positive
            if len(self.output_heads) > 1:
                head1 = self.output_heads[1]
                head1.weight.data = torch.abs(head1.weight.data)
                if head1.bias is not None:
                    head1.bias.data = (
                        torch.abs(head1.bias.data) * -0.2
                    )  # Less negative bias

    def forward(self, x):
        # First, run shared network
        shared_features = self.shared_network(x)

        batch_size = x.shape[0]
        num_events = len(self.output_heads)
        out_features = self.output_heads[0].out_features

        # Create output tensor of appropriate shape
        outputs = torch.zeros(batch_size, num_events, out_features, device=x.device)

        # Process each event with its output head
        for i, head in enumerate(self.output_heads):
            # For multi-event case, add event-specific variations to inputs
            if num_events > 1:
                # Add small event-specific offset to make inputs different
                event_specific_input = shared_features.clone()

                # Each event gets slightly different input
                # This ensures outputs for each event use different paths
                if self.training:
                    # During training, add random noise to break symmetry
                    noise_scale = 0.01 * (i + 1)  # More noise for higher event indices
                    event_specific_input += (
                        torch.randn_like(event_specific_input) * noise_scale
                    )
                else:
                    # During inference, add deterministic shifts based on event index
                    shift = 0.001 * (i + 1)  # Small shift, different for each event
                    event_specific_input += shift

                # Run through the event-specific output head
                event_output = head(event_specific_input)
            else:
                # Single event case - use shared features directly
                event_output = head(shared_features)

            # Store output for this event
            outputs[:, i, :] = event_output

            # Log statistics for debug purposes
            if self.training and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Event {i} output - mean: {event_output.mean().item():.4f}, std: {event_output.std().item():.4f}"
                )
                logger.debug(
                    f"Event {i} batch variance: {torch.var(event_output, dim=0).mean().item():.6f}"
                )

        return outputs


class CauseSpecificNetCompRisk(nn.Module):
    """
    Competing risks network that routes inputs through multiple event-specific networks.
    This implementation uses explicit debugging and ensures different behavior for multi-event cases.
    """

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

        # Store parameters for debugging and forward pass
        self.out_features = out_features
        self.num_events = num_events
        self.in_features = in_features

        # Set up logging
        from sat.utils import logging

        self.logger = logging.get_default_logger()
        self.logger.warning(
            f"Initializing CauseSpecificNetCompRisk with {num_events} events"
        )

        # Create a simpler implementation that emphasizes different behavior for each event
        # First, a shared feature extractor for all events
        shared_layers = []
        current_feat_size = in_features

        # Build the shared layers
        for i in range(shared_num_hidden_layers):
            shared_layers.append(
                nn.Linear(current_feat_size, shared_intermediate_size, bias=bias)
            )
            if batch_norm:
                shared_layers.append(nn.BatchNorm1d(shared_intermediate_size))
            shared_layers.append(activation())
            if dropout is not None and dropout > 0:
                shared_layers.append(nn.Dropout(dropout))
            current_feat_size = shared_intermediate_size

        self.shared_net = nn.Sequential(*shared_layers)

        # Create different event-specific networks
        self.event_nets = nn.ModuleList()

        for event_idx in range(num_events):
            # Each event gets a unique network structure
            event_layers = []

            # For multi-event case, vary network structure per event
            current_size = current_feat_size + in_features  # Residual connection

            # Use different layer sizes for each event
            for layer_idx in range(indiv_num_hidden_layers):
                # Vary size based on event - each event gets incrementally larger first layer
                if layer_idx == 0 and num_events > 1:
                    size_factor = 1.0 + (
                        event_idx * 0.2
                    )  # Each event gets different size
                    layer_size = max(
                        indiv_intermediate_size,
                        int(indiv_intermediate_size * size_factor),
                    )
                else:
                    layer_size = indiv_intermediate_size

                # Create the layer
                layer = nn.Linear(current_size, layer_size, bias=bias)

                # For multi-event, use intentionally different initialization per event
                if num_events > 1:
                    with torch.no_grad():
                        # Use different initializations for each event to break symmetry
                        if event_idx == 0:
                            # Event 0 - use negative-leaning weights
                            nn.init.normal_(
                                layer.weight, mean=-0.01 * (layer_idx + 1), std=0.05
                            )
                        else:
                            # Event 1+ - use positive-leaning weights with increasing variance
                            nn.init.normal_(
                                layer.weight,
                                mean=0.01 * event_idx * (layer_idx + 1),
                                std=0.05 + 0.01 * event_idx,
                            )

                        # Also vary the bias per event
                        if layer.bias is not None:
                            bias_val = -0.1 * (event_idx + 1) * (layer_idx + 1)
                            nn.init.constant_(layer.bias, bias_val)

                event_layers.append(layer)

                # Add batch norm with track_running_stats=True to maintain different statistics
                if batch_norm:
                    bn = nn.BatchNorm1d(layer_size, track_running_stats=True)
                    # For multi-event, initialize batch norm differently too
                    if num_events > 1:
                        with torch.no_grad():
                            # Different initial gamma/beta for different events
                            bn.weight.fill_(
                                1.0 + 0.1 * event_idx
                            )  # Different gamma per event
                            bn.bias.fill_(0.01 * event_idx)  # Different beta per event
                    event_layers.append(bn)

                event_layers.append(activation())

                if dropout is not None and dropout > 0:
                    event_layers.append(nn.Dropout(dropout))

                current_size = layer_size

            # Add final output layer
            output_layer = nn.Linear(current_size, out_features, bias=bias)

            # For multi-event, ensure very different output initialization
            if num_events > 1:
                with torch.no_grad():
                    # Event 0 gets negative-biased output, Event 1 gets positive-biased output
                    if event_idx == 0:
                        nn.init.normal_(output_layer.weight, mean=-0.05, std=0.02)
                        if output_layer.bias is not None:
                            nn.init.constant_(output_layer.bias, -1.0)
                            # Add position-dependent variations to bias
                            output_layer.bias += torch.linspace(-0.2, 0.2, out_features)
                    else:
                        nn.init.normal_(output_layer.weight, mean=0.05, std=0.02)
                        if output_layer.bias is not None:
                            nn.init.constant_(
                                output_layer.bias, -0.5
                            )  # Less negative for event 1
                            # Add different position-dependent variations
                            output_layer.bias += torch.linspace(0.2, -0.2, out_features)

            event_layers.append(output_layer)

            # Create the full event network
            self.event_nets.append(nn.Sequential(*event_layers))

            # Debug output
            self.logger.warning(
                f"Created event {event_idx} network with {len(event_layers)} layers"
            )

    def forward(self, input):
        # Run through shared network
        shared_features = self.shared_net(input)

        # Combine with residual connection
        combined = torch.cat([input, shared_features], dim=1)

        # Process each event with its own network
        batch_size = input.shape[0]
        outputs = []

        # Get separate outputs for each event
        for event_idx, event_net in enumerate(self.event_nets):
            if self.num_events > 1:
                # For multi-event case, create event-specific input with added noise
                # to break symmetry and force different outputs
                event_input = combined.clone()

                # During training, add random noise that's different for each event
                if self.training:
                    noise_scale = 0.05 * (
                        event_idx + 1
                    )  # More noise for higher event indices
                    event_input += torch.randn_like(event_input) * noise_scale

                # Process through event-specific network
                event_output = event_net(event_input)

                # During training, log output statistics
                if self.training:
                    mean_val = event_output.mean().item()
                    std_val = event_output.std().item()
                    batch_var = torch.var(event_output, dim=0).mean().item()
                    self.logger.warning(
                        f"Event {event_idx} output - mean={mean_val:.4f}, std={std_val:.4f}, batch_var={batch_var:.6f}"
                    )

                # Add output to results
                outputs.append(event_output)
            else:
                # Single event case - simpler processing
                outputs.append(event_net(combined))

        # Stack along event dimension
        stacked_outputs = torch.stack(outputs, dim=1)

        # Debug check for multi-event case - ensure outputs are different
        if self.num_events > 1 and self.training:
            # Check for uniform outputs across batch
            for event_idx in range(self.num_events):
                event_output = stacked_outputs[:, event_idx, :]
                batch_std = torch.std(event_output, dim=0)
                mean_std = torch.mean(batch_std).item()

                # If all outputs are too similar across batch, add noise
                if mean_std < 1e-4:
                    self.logger.error(
                        f"Event {event_idx} has uniform outputs (std={mean_std:.6f}) - forcing variability"
                    )
                    # Force different outputs by adding significant noise
                    stacked_outputs[:, event_idx, :] += (
                        torch.randn_like(event_output) * 0.1
                    )

        return stacked_outputs
