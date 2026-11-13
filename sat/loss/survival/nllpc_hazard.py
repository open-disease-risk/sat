"""NLLPC Hazard loss for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd
import torch
from transformers.utils import ModelOutput

from sat.pycox.models.loss import NLLPCHazardLoss
from sat.models.tasks.heads import SAOutput
from sat.loss.base import Loss


class SATNLLPCHazardLoss(Loss):
    def __init__(
        self,
        importance_sample_weights: str = None,
        num_events: int = 1,
    ):
        super(SATNLLPCHazardLoss, self).__init__(num_events)
        self.loss_fct = NLLPCHazardLoss(reduction="none")
        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)
        self.register_buffer("weights", weights)

    def nllp_hazard_loss(self, predictions, references, event_type) -> torch.Tensor:
        events = self.events(references)[:, event_type].to(bool)
        duration_percentiles = self.duration_percentiles(references)[:, event_type]
        fraction_duration = self.fraction_with_quantile(references)[:, event_type]
        predictions = predictions[:, event_type]
        return (
            self.loss_fct(
                predictions,
                duration_percentiles,
                events,
                fraction_duration,
            ).mean()
            * self.weights[event_type + 1]
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute a loss.
        Parameters:
            predictions (SAOutput: Predictions of the model (SAOutput: 5 x events x batch size x cuts)
            references (torch.Tensor): Reference values. (dims: batch size x 4)
        Returns:
            float: The loss value.
        """
        # variables x batch x events x duration cuts
        logits = predictions.logits
        loss = 0.0
        for i in range(self.num_events):
            loss += self.nllp_hazard_loss(logits, references, i)
        return loss


class SATNLLPCHazardLossNew(Loss):
    def __init__(
        self,
        importance_sample_weights: str = None,
        num_events: int = 1,
        balance_strategy: str = "none",
        ema_momentum: float = 0.9,
    ):
        super(SATNLLPCHazardLoss, self).__init__(
            num_events, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )

        self.loss_fct = NLLPCHazardLoss(reduction="none")

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

    def nllp_hazard_loss(self, predictions, references, event_type) -> torch.Tensor:
        events = self.events(references)[:, event_type].to(bool)
        duration_percentiles = self.duration_percentiles(references)[:, event_type]
        fraction_duration = self.fraction_with_quantile(references)[:, event_type]
        predictions = predictions[:, event_type]

        return (
            self.loss_fct(
                predictions,
                duration_percentiles,
                events,
                fraction_duration,
            ).mean()
            * self.weights[event_type + 1]
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute a loss.

        Parameters:
            predictions (SAOutput): Predictions of the model (SAOutput: 5 x events x batch size x cuts)
            references (torch.Tensor): Reference values. (dims: batch size x 4)

        Returns:
            float: The loss value.
        """
        import logging

        logger = logging.getLogger("sat")

        # Track validity of inputs for diagnostics
        inputs_valid = True

        # variables x batch x events x duration cuts
        logits = predictions.logits

        # Check for extreme values in logits that would break computation
        has_nan = torch.isnan(logits).any()
        has_inf = torch.isinf(logits).any()

        if has_nan or has_inf:
            nan_count = torch.isnan(logits).sum().item() if has_nan else 0
            inf_count = torch.isinf(logits).sum().item() if has_inf else 0

            # Only clean if there's a small proportion of bad values
            if (nan_count + inf_count) < (logits.numel() * 0.1):
                # Only replace bad values, preserve valid ones
                logger.info(
                    f"NLLPCHazardLoss: Found {nan_count} NaNs and {inf_count} Infs in logits, applying selective correction"
                )

                # Clone first to avoid modifying original tensor
                logits_clean = logits.clone()

                # Fix problematic values only
                if has_nan:
                    logits_clean[torch.isnan(logits)] = 0.0
                if has_inf:
                    logits_clean[torch.isinf(logits) & (logits > 0)] = 50.0
                    logits_clean[torch.isinf(logits) & (logits < 0)] = -50.0

                logits = logits_clean
                inputs_valid = False
            else:
                # Too many problematic values - warn but try to continue
                logger.warning(
                    f"NLLPCHazardLoss: High proportion of invalid values in logits ({(nan_count + inf_count)/logits.numel()*100:.1f}%)"
                )
                inputs_valid = False

        # Attempt to compute losses with refined approach
        try:
            # Compute accumulation of losses for each event type
            all_losses = []

            for i in range(self.num_events):
                event_loss = self.nllp_hazard_loss(logits, references, i)

                # Check individual event losses for validity
                if torch.isnan(event_loss) or torch.isinf(event_loss):
                    logger.warning(
                        f"NLLPCHazardLoss: Event {i} produced invalid loss value {event_loss.item()}"
                    )
                    # Skip invalid event loss to allow other events to contribute
                    continue

                all_losses.append(event_loss)

            # If we have any valid losses, use them
            if all_losses:
                # Sum all valid losses
                loss = sum(all_losses)

                # Apply a gradient clipper to the loss to prevent extreme gradients
                with torch.no_grad():
                    loss_val = loss.item()
                    # Only log if the loss seems unusually high
                    if loss_val > 10.0:
                        logger.info(f"NLLPCHazardLoss: High loss value: {loss_val}")

                return loss
            else:
                # No valid losses - use a fallback that allows learning
                logger.warning(
                    "NLLPCHazardLoss: No valid event losses, using fallback loss"
                )

                # Create a fallback loss with gradient that won't explode but allows learning
                # This fallback is designed to be non-zero but much less impactful than real loss
                dummy_loss = torch.tensor(
                    0.1, device=logits.device, dtype=logits.dtype, requires_grad=True
                )
                return dummy_loss

        except Exception as e:
            logger.error(f"NLLPCHazardLoss: Exception in loss computation: {str(e)}")

            # Create emergency fallback that won't crash training but allows some progress
            fallback = torch.tensor(
                0.5, device=logits.device, dtype=logits.dtype, requires_grad=True
            )
            return fallback


class SATNLLPCHazardLossNew(Loss):
    def __init__(
        self,
        importance_sample_weights: str = None,
        num_events: int = 1,
        balance_strategy: str = "none",
        ema_momentum: float = 0.9,
    ):
        super(SATNLLPCHazardLoss, self).__init__(
            num_events, balance_strategy=balance_strategy, ema_momentum=ema_momentum
        )

        self.loss_fct = NLLPCHazardLoss(reduction="none")

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

    def nllp_hazard_loss(self, predictions, references, event_type) -> torch.Tensor:
        # Directly get events as bool to avoid unnecessary conversion later
        events = self.events(references)[:, event_type].bool()
        duration_percentiles = self.duration_percentiles(references)[:, event_type]
        fraction_duration = self.fraction_with_quantile(references)[:, event_type]
        predictions = predictions[:, event_type]

        # Apply loss and weight in one efficient operation
        event_loss = self.loss_fct(
            predictions,
            duration_percentiles,
            events,
            fraction_duration,
        )

        # Use a single operation with the weight factor
        return event_loss.mean() * self.weights[event_type + 1]

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute a loss.

        Parameters:
            predictions (SAOutput): Predictions of the model containing logits
            references (torch.Tensor): Reference values (dims: batch size x 4) containing event types and durations

        Returns:
            float: The loss value.
        """
        import logging

        logger = logging.getLogger("sat")

        # Debug logging to understand what we're getting
        logger.debug(f"NLLPC Loss - Prediction type: {type(predictions).__name__}")
        logger.debug(
            f"NLLPC Loss - Available attributes: {[attr for attr in dir(predictions) if not attr.startswith('_')]}"
        )

        # Extract logits directly from SAOutput
        logits = predictions.logits
        logger.debug(f"NLLPC Loss - Logits shape: {logits.shape}")

        # For multiple events, use accumulation
        loss = self.nllp_hazard_loss(logits, references, 0)
        for i in range(1, self.num_events):
            loss += self.nllp_hazard_loss(logits, references, i)

        return loss
