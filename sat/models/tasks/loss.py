"""Loss functions for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import pandas as pd

import torch
import torch.nn.functional as F

from torch import nn
from transformers.utils import ModelOutput
from typing import List

from sat.pycox.models.loss import NLLPCHazardLoss
from sat.utils import logging
from sat.models.tasks.heads import SAOutput, TaskOutput
from sat.utils.km import KaplanMeierArea

logger = logging.get_default_logger()


class Loss(nn.Module):
    """Base class for losses."""

    def __init__(self, num_events: int = 1):
        super(Loss, self).__init__()
        self.num_events = num_events

    def durations(self, references: torch.Tensor):
        return references[
            :, (3 * self.num_events) : (3 * self.num_events + self.num_events)
        ].float()

    def duration_percentiles(self, references: torch.Tensor):
        return references[:, 0 : self.num_events].long()

    def events(self, references: torch.Tensor):
        return references[
            :, (1 * self.num_events) : (1 * self.num_events + self.num_events)
        ].long()

    def fraction_with_quantile(self, references: torch.Tensor):
        return references[
            :, (2 * self.num_events) : (2 * self.num_events + self.num_events)
        ].float()

    def survivals(self, predictions):
        surv = []
        for logits in predictions:
            hazard = F.softplus(logits)
            surv.append(hazard.cumsum(1).mul(-1).exp()[:, :-1])
        return surv


class MetaLoss(Loss):
    """A loss class that linearly combines multiple loss components."""

    def __init__(self, losses: list, coeffs: list):
        super(MetaLoss, self).__init__()

        self.losses = nn.ModuleList(losses)
        self.register_buffer("coeffs", torch.tensor(coeffs).to(torch.float32))

    def forward(
        self, predictions: ModelOutput, references: torch.Tensor
    ) -> torch.Tensor:
        l = 0.0
        for i, loss in enumerate(self.losses):
            l += self.coeffs[i] * loss(predictions, references)

        return l


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
            # Apply clamping to ensure weights are in a reasonable range for numerical stability
            weights = torch.tensor(df.weights.values).to(torch.float32)
            # Clamp weights to prevent extreme values that could cause numerical issues
            weights = torch.clamp(weights, min=1e-4, max=1e4)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

    def nllp_hazard_loss(self, predictions, references, event_type) -> torch.Tensor:
        events = self.events(references)[:, event_type].to(bool)
        duration_percentiles = self.duration_percentiles(references)[:, event_type]
        fraction_duration = self.fraction_with_quantile(references)[:, event_type]
        predictions = predictions[:, event_type]

        # Apply the loss function with error handling
        try:
            loss_vals = self.loss_fct(
                predictions,
                duration_percentiles,
                events,
                fraction_duration,
            )
            
            # Check for NaN or Inf values and replace them
            invalid_mask = torch.isnan(loss_vals) | torch.isinf(loss_vals)
            if invalid_mask.any():
                logger.warning(f"Found {invalid_mask.sum().item()} invalid loss values. Replacing with zeros.")
                loss_vals[invalid_mask] = 0.0
                
            # Apply weight with numerical stability
            weight = torch.clamp(self.weights[event_type + 1], min=1e-4, max=1e4)
            return loss_vals.mean() * weight
            
        except Exception as e:
            logger.error(f"Error in nllp_hazard_loss: {e}")
            # Return a small loss value to allow training to continue
            # Create a tensor that requires grad to avoid breaking the autograd graph
            return torch.tensor(1e-4, device=predictions.device, requires_grad=True)

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

        # Check if logits contain NaN or Inf values
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.warning("NaN or Inf values detected in logits. Clipping values...")
            # Replace NaN with zeros and clip any infinite values
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        loss = 0.0
        for i in range(self.num_events):
            event_loss = self.nllp_hazard_loss(logits, references, i)
            
            # Verify that the loss is valid before adding it
            if not torch.isnan(event_loss) and not torch.isinf(event_loss):
                loss += event_loss
            else:
                logger.warning(f"Invalid loss for event type {i}. Using default value.")
                # Add a small constant to allow training to continue
                # Create a tensor that requires grad to avoid breaking the autograd graph
                default_loss = torch.tensor(1e-4, device=logits.device, requires_grad=True)
                loss += default_loss

        # Final check for NaN/Inf in the total loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error("Total loss is NaN or Inf. Returning a default value.")
            # Create a tensor that requires grad to avoid breaking the autograd graph
            return torch.tensor(1e-4, device=logits.device, requires_grad=True)
            
        return loss


class RankingLoss(Loss):
    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        sigma: float = 1.0,
        num_events: int = 1,
    ):
        super(RankingLoss, self).__init__(num_events)

        self.sigma = sigma

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

        df = pd.read_csv(duration_cuts, header=None, names=["cuts"])
        self.register_buffer(
            "duration_cuts", torch.Tensor(df.cuts.values)
        )  # tn duration cut points

    def ranking_loss(
        self, events, durations, survival, hazard, weights
    ) -> torch.Tensor:
        device = events.device
        n = events.shape[0]
        e = events.shape[1]

        I = events.to(bool)
        I_censored = ~I  # censored indicator (n x e)

        T = self.duration_cuts.expand(n, e, -1)  # duration cut points (n x e x tn)
        indexSmaller = self.duration_cuts.view(1, 1, -1) <= durations.unsqueeze(2)
        t0Index = (
            torch.sum(indexSmaller, dim=2) - 1
        )  # left boundary of time interval (n x e)
        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)
        t1Index = t0Index + 1  # right boundary of time interval (n x e)

        # if we run out of bounds, we match t0Index this means that dT will be
        # zero and causes NaNs in hstar, which we need to fix
        fixOOB = t1Index == len(self.duration_cuts)
        t1Index[fixOOB] = len(self.duration_cuts) - 1

        T0 = torch.gather(
            T, 2, t0Index
        )  # left boundary of time interval for all events i and time constraints j (n x e x e)
        T1 = torch.gather(
            T, 2, t1Index
        )  # right boundary of time interval for all events i and time constraints j (n x e)

        SatT0 = torch.gather(survival, 2, t0Index)  # survival at T0 (n x e x e)
        SatT1 = torch.gather(survival, 2, t1Index)  # survival at T1 (n x e x e)
        hstar = torch.gather(hazard, 2, t0Index)  # hazard at T0 (n x e x e)

        dT = T1 - T0

        # when dT is zero or negative we know that the time duration for an
        # observation is greater or equal to the maximum duration cut. So, we
        # need to use the corresponding hazard rather than interpolating. Since
        # we initialized hstar with the hazard at T0, we only need to take care
        # of the valid interpolations below:
        positive_mask = torch.gt(dT, 0.0)
        hstar[positive_mask] = torch.div(
            torch.log(0.000001 + SatT0[positive_mask])
            - torch.log(0.000001 + SatT1[positive_mask]),
            (dT[positive_mask]),
        )  # solve for hazard given the survival at T0 and T1 (n x e x e)

        SatT = SatT0 * torch.exp(
            -(durations.unsqueeze(1).repeat(1, e, 1) - T0) * hstar
        )  # solve for survival at time t (n x e x e)

        # compute an epsilon time to be subtracted from t in order to compute
        # the survival at t-epsilon for when the event occurred for sample i and
        # j
        t_epsilon = (
            self.duration_cuts[-1] - self.duration_cuts[0]
        ) / self.duration_cuts[-1]
        TMinus = torch.nn.functional.relu(
            durations.unsqueeze(1).repeat(1, e, 1) - t_epsilon
        )
        SatTMinus = SatT0 * torch.exp(
            -(TMinus - T0) * hstar
        )  # solve for survival at time t-epsilon (n x e x e)

        # get the n inner diagonals of e x e and repeat column-wise
        diag_S = torch.diagonal(SatT, dim1=-2, dim2=-1).unsqueeze(2).repeat(1, 1, e)
        diag_S2 = (
            torch.diagonal(SatTMinus, dim1=-2, dim2=-1).unsqueeze(2).repeat(1, 1, e)
        )

        dS1 = diag_S - torch.transpose(
            SatT, 1, 2
        )  # dS_{ij} = S_{i}(T_{i}) - S_{j}(T_{i}) (n x e x e)
        dS2 = SatTMinus - torch.transpose(
            diag_S2, 1, 2
        )  # dS_{ij} = S_{i}(T_{j}-1) - S_{j}(T_{j}-1) (n x e x e)
        dS3 = SatT - torch.transpose(
            diag_S, 1, 2
        )  # dS_{ij} = S_{i}(T_{j}) - S_{j}(T_{j}) (n x e x e)

        # A_{nij}=1 if t_i < t_j and A_{ij}=0 if t_i >= t_j
        #              and A_{ij}=1 when event occured for subject i (n x e x e)
        A1 = I.unsqueeze(2).repeat(1, 1, e).float() * torch.nn.functional.relu(
            torch.sign(
                durations.unsqueeze(1).repeat(1, e, 1)
                - durations.unsqueeze(2).repeat(1, 1, e)
            )
        )
        A2 = (
            A1 * I.unsqueeze(1).repeat(1, e, 1).float()
        )  # and A_{ij}=1 when event occured for subject j (n x e x e)
        A3 = (
            A1 * I_censored.unsqueeze(1).repeat(1, e, 1).float()
        )  # and A_{ij}=1 when subject j is censored (n x e x e)

        eta = torch.mean(
            weights
            * (
                A1 * torch.exp(dS1 / self.sigma)
                + A2 * torch.exp(dS2 / self.sigma)
                + A3 * torch.exp(dS3 / self.sigma)
            ),
        )

        return eta  # (1 x 1)


class SampleRankingLoss(RankingLoss):
    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        sigma: float = 1.0,
        num_events: int = 1,
    ):
        super(SampleRankingLoss, self).__init__(
            duration_cuts, importance_sample_weights, sigma, num_events
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute the deephit ranking loss.

        Parameters:
            predictions (SAOutput: Predictions of the model (SAOutput: 5 x events x batch size x cuts)
            references (torch.Tensor): Reference values. (dims: batch size x 4)

        Returns:
            float: The loss value.
        """
        events = self.events(references).permute(1, 0)
        n = events.shape[0]
        e = events.shape[1]
        eta = self.ranking_loss(
            events,
            self.durations(references).permute(1, 0),
            predictions.survival.permute(1, 0, 2),
            predictions.hazard.permute(1, 0, 2),
            self.weights[1:].unsqueeze(1).unsqueeze(2).repeat(1, e, e),
        )

        return eta


class MultiEventRankingLoss(RankingLoss):
    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        sigma: float = 1.0,
        num_events: int = 1,
    ):
        super(MultiEventRankingLoss, self).__init__(
            duration_cuts, importance_sample_weights, sigma, num_events
        )

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute the deephit ranking loss.

        Parameters:
            predictions (SAOutput: Predictions of the model (SAOutput: 5 x events x batch size x cuts)
            references (torch.Tensor): Reference values. (dims: batch size x 4)

        Returns:
            float: The loss value.
        """
        events = self.events(references)
        n = events.shape[0]
        e = events.shape[1]
        eta = self.ranking_loss(
            events,
            self.durations(references),
            predictions.survival,
            predictions.hazard,
            self.weights[1:]
            .unsqueeze(0)
            .unsqueeze(2)
            .repeat(1, 1, e)
            .expand(n, -1, -1),
        )
        return eta


class MSELoss(Loss):
    """MSE Loss."""

    def __init__(
        self,
        training_set: str,
        importance_sample_weights: str = None,
        l2_type: str = "uncensored",
        num_events: int = 1,
    ):
        super(MSELoss, self).__init__(num_events)

        self.l2_type = l2_type
        self.kms: List[KaplanMeierArea] = []

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

        if l2_type == "margin":
            if training_set is None:
                raise ValueError(
                    "If 'margin' is chosen, training set values must be included."
                )

            logger.debug("Train the Kaplan Meier Curves")
            # read training data into pandas dataframe with given columns
            df = pd.read_csv(training_set, header=0)
            for event in range(self.num_events):
                training_event_times = df["duration"]
                training_event_indicators = df["event"] == event + 1
                self.kms.append(
                    KaplanMeierArea(training_event_times, training_event_indicators)
                )

    def mse(
        self,
        predictions,
        references,
        event_type: int,
    ) -> torch.Tensor:
        device = references.device
        event_indicators = self.events(references)[:, event_type].to(bool)
        durations = self.durations(references)[:, event_type]
        predictions = predictions[:, event_type]

        if self.l2_type == "uncensored":
            scores = durations[event_indicators] - predictions[event_indicators]
            loss = torch.mean(torch.square(scores))
        elif self.l2_type == "margin":
            censor_times = durations[~event_indicators]
            weights = torch.Tensor(
                1.0 - self.kms[event_type].predict(censor_times.detach().cpu().numpy())
            ).to(device)
            best_guesses = torch.Tensor(
                self.kms[event_type].best_guess(censor_times.detach().cpu().numpy())
            ).to(device)

            scores = torch.empty_like(predictions)
            scores[event_indicators] = (
                durations[event_indicators] - predictions[event_indicators]
            )
            scores[~event_indicators] = weights * (
                best_guesses - predictions[~event_indicators]
            )
            weighted_multiplier = torch.ones(1).to(device) / (
                torch.sum(event_indicators) + torch.sum(weights)
            )
            loss = (weighted_multiplier * torch.mean(torch.square(scores)))[0]
        else:
            raise ValueError("L2 type must be either 'uncensored' or 'margin'.")

        return self.weights[event_type + 1] * loss

    def forward(
        self, predictions: TaskOutput, references: torch.Tensor
    ) -> torch.Tensor:
        predictions = predictions.predictions

        loss = 0.0
        for i in range(self.num_events):
            loss += self.mse(predictions, references, i)

        return loss


class QuantileLoss(Loss):
    """Quantile Loss for Quantile Regression"""

    def __init__(
        self,
        quantiles: List[float],
        training_set: str,
        importance_sample_weights: str = None,
        l_type: str = "uncensored",
    ):
        super(QuantileLoss, self).__init__(num_events)

        self.l_type = l_type
        self.quantiles = quantiles
        self.num_events = num_events
        self.kms: List[KaplanMeierArea] = []

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

        if self.l_type == "margin":
            if training_set is None:
                raise ValueError(
                    "If 'margin' is chosen, training set values must be included."
                )

            logger.debug("Train the Kaplan Meier Curves")
            # read training data into pandas dataframe with given columns
            df = pd.read_csv(training_set, header=0)
            for event in range(self.num_events):
                training_event_times = df["duration"]
                training_event_indicators = df["event"] == event + 1
                self.kms.append(
                    KaplanMeierArea(training_event_times, training_event_indicators)
                )

    def quantile_loss(self, predictions, references, event_type) -> torch.Tensor:
        device = references.device

        event_indicator = self.events(references)[:, event_type].to(bool)
        durations = self.durations(references)[:, event_type]
        predictions = predictions[:, event_type]

        if self.l_type == "uncensored":
            scores = torch.zeros_like(durations[event_indicators])
            for i, q in enumerate(self.quantiles):
                errors = (
                    durations[event_indicators] - predictions[event_indicators][:, i]
                )
                scores += torch.max((q - 1) * errors, q * errors)

            loss = torch.mean(scores)
        elif self.l_type == "margin":
            censor_times = durations[~event_indicators]
            weights = torch.Tensor(
                1.0 - self.kms[event_type].predict(censor_times.detach().cpu().numpy())
            ).to(device)
            best_guesses = torch.Tensor(
                self.kms[event_type].best_guess(censor_times.detach().cpu().numpy())
            ).to(device)

            scores = torch.zeros_like(durations)
            errors = torch.empty_like(durations)
            for i, q in enumerate(self.quantiles):
                errors[event_indicators] = (
                    durations[event_indicators] - predictions[event_indicators][:, i]
                )
                errors[~event_indicators] = weights * (
                    best_guesses - predictions[~event_indicators][:, i]
                )
                scores += torch.max((q - 1) * errors, q * errors)

            weighted_multiplier = torch.ones(1).to(device) / (
                torch.sum(event_indicators) + torch.sum(weights)
            )
            loss = (weighted_multiplier * torch.mean(scores))[0]
        else:
            raise ValueError("L type must be either 'uncensored' or 'margin'.")

        return self.weights[event_type + 1] * loss

    def forward(
        self, predictions: TaskOutput, references: torch.Tensor
    ) -> torch.Tensor:
        predictions = predictions.predictions

        loss = 0.0
        for i in range(self.num_events):
            loss += self.quantile_loss(predictions, references, i)

        return loss


class CrossEntropyLoss(Loss):
    """Cross Entropy Loss."""

    def __init__(
        self,
        event_time_thr: float,
        training_set: str,
        l_type: str = "margin",
        importance_sample_weights: str = None,
    ):
        super(CrossEntropyLoss, self).__init__(num_events)

        self.loss_func = nn.BCELoss(reduction="none")
        self.num_events = num_events
        self.event_time_thr = event_time_thr
        self.l_type = l_type
        self.kms: List[KaplanMeierArea] = []

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

        if l_type == "margin":
            if training_set is None:
                raise ValueError(
                    "If 'margin' is chosen, training set values must be included."
                )

            logger.debug("Train the Kaplan Meier Curves")
            # read training data into pandas dataframe with given columns
            df = pd.read_csv(training_set, header=0)
            for event in range(self.num_events):
                training_event_times = df["duration"]
                training_event_indicators = df["event"] == event + 1
                self.kms.append(
                    KaplanMeierArea(training_event_times, training_event_indicators)
                )

    def ce(self, predictions, references, event_type) -> torch.Tensor:
        device = references.device
        durations = self.durations(references)[:, event_type]
        event_indicator = self.events(references)[:, event_type].to(bool)

        if self.l_type == "uncensored":
            # survival times less than T
            event_occurred = durations <= self.event_time_thr

            # The classification target is 1 if the event occurred and where
            # survival times are < T, Otherwise 0.
            targets = torch.logical_and(event_occurred, event_indicator).float()

            # for classification we only care about observations with the
            # event or when survival times are > T
            relevant_for_classification = torch.logical_or(
                ~event_occurred, event_indicator
            )

            # subset predictions and targets to what is relevant for
            # classification
            preds = predictions[:, event_type][relevant_for_classification].squeeze()
            targets = targets[relevant_for_classification]

            logger.debug(
                f"Compute loss between predictions {preds.shape} and targets {targets.shape}"
            )

            loss = self.weights[event_type + 1] * torch.mean(
                self.loss_func(preds, targets)
            )
        elif self.l_type == "margin":
            censor_times = durations[~event_indicator]
            weights = torch.Tensor(
                1.0 - self.kms[event_type].predict(censor_times.detach().cpu().numpy())
            ).to(device)
            best_guesses = torch.Tensor(
                self.kms[event_type].best_guess(censor_times.detach().cpu().numpy())
            ).to(device)

            # survival times less than T accounting also for best guesses
            event_occurred = torch.empty_like(durations).bool()
            event_occurred[event_indicator] = (
                durations[event_indicator] <= self.event_time_thr
            )
            event_occurred[~event_indicator] = best_guesses <= self.event_time_thr

            # The classification target is 1 if the event occurred and where
            # survival times are < T, Otherwise 0.
            targets = torch.logical_and(event_occurred, event_indicator).float()

            # for classification we only care about observations with the
            # event or when survival times are > T
            relevant_for_classification = torch.logical_or(
                ~event_occurred, event_indicator
            )

            # subset predictions and targets to what is relevant for
            # classification
            preds = predictions[:, event_type][relevant_for_classification].squeeze()
            targets = targets[relevant_for_classification]

            logger.debug(
                f"Compute loss between predictions {preds.shape} and targets {targets.shape}"
            )

            losses = self.loss_func(preds, targets)
            losses[~event_indicator] = weights * losses[~event_indicator]

            weighted_multiplier = torch.tensor(1.0).to(device) / (
                torch.sum(event_indicator) + torch.sum(weights)
            )

            loss = (
                self.weights[event_type + 1] * weighted_multiplier * torch.mean(losses)
            )

            return loss

    def forward(
        self, predictions: TaskOutput, references: torch.Tensor
    ) -> torch.Tensor:
        predictions = predictions.predictions

        loss = 0.0
        for event in range(self.num_events):
            loss += self.ce(predictions, references, event)

        return loss


class L1Loss(Loss):
    """L1 loss"""

    def __init__(
        self,
        training_set: str,
        importance_sample_weights: str = None,
        l1_type: str = "hinge",
        num_events: int = 1,
    ):
        super(L1Loss, self).__init__(num_events)

        self.l1_type = l1_type
        self.kms: List[KaplanMeierArea] = []

        # load the importance sampling weights if not None
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)

        self.register_buffer("weights", weights)

        if l1_type == "margin":
            if training_set is None:
                raise ValueError(
                    "If 'margin' is chosen, training set values must be included."
                )

            logger.debug("Train the Kaplan Meier Curves")
            # read training data into pandas dataframe with given columns
            df = pd.read_csv(training_set, header=0)
            for event in range(self.num_events):
                training_event_times = df["duration"]
                training_event_indicators = df["event"] == event + 1
                self.kms.append(
                    KaplanMeierArea(training_event_times, training_event_indicators)
                )

    def l1(
        self,
        predictions,
        references,
        event_type: int,
    ) -> torch.Tensor:
        device = references.device

        event_indicators = self.events(references)[:, event_type].to(bool)
        durations = self.durations(references)[:, event_type]
        predictions = predictions[:, event_type]

        if self.l1_type == "uncensored":
            scores = durations[event_indicators] - predictions[event_indicators]
            loss = torch.mean(torch.abs(scores))
        elif self.l1_type == "hinge":
            scores = durations - predictions
            scores[~event_indicators] = torch.maximum(
                scores[~event_indicators], torch.zeros_like(scores[~event_indicators])
            )
            loss = torch.mean(torch.abs(scores))
        elif self.l1_type == "margin":
            censor_times = durations[~event_indicators]
            weights = torch.Tensor(
                1.0 - self.kms[event_type].predict(censor_times.detach().cpu().numpy())
            ).to(device)
            best_guesses = torch.Tensor(
                self.kms[event_type].best_guess(censor_times.detach().cpu().numpy())
            ).to(device)

            scores = torch.empty_like(predictions)
            scores[event_indicators] = (
                durations[event_indicators] - predictions[event_indicators]
            )
            scores[~event_indicators] = weights * (
                best_guesses - predictions[~event_indicators]
            )
            weighted_multiplier = torch.ones(1).to(device) / (
                torch.sum(event_indicators) + torch.sum(weights)
            )
            loss = (weighted_multiplier * torch.sum(torch.abs(scores)))[0]
        else:
            raise ValueError("L1 type must be either 'hinge' or 'margin'.")

        return self.weights[event_type + 1] * loss

    def forward(
        self, predictions: TaskOutput, references: torch.Tensor
    ) -> torch.Tensor:
        predictions = predictions.predictions

        loss = 0.0
        for event in range(self.num_events):
            loss += self.l1(predictions, references, event)

        return loss


# not good practice -- I know -- but this is going to be refactored
# class ImprovedSurvtraceLoss(Loss):
#     """Improved Survtrace loss"""

#     def __init__(
#         self,
#         importance_sample_weights: str,
#     ):
#         device_str, device = get_device()
#         # load the importance sampling weights
#         df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
#         self.importance_sample_weights = (
#             torch.tensor(df.weights.values).to(torch.float32).to(device)
#         )

#         self.loss_fct = NLLPCHazardLoss(reduction="none")

#     def loss(self, predictions: SAOutput, references: torch.Tensor) -> float:
#         """Compute the NLLPC hazard loss.

#         This function implements a loss for both competing and single event
#         cases. It iterates through the events and masks every other event as
#         censored in turn.

#         Parameters:
#             predictions (SAOutput: Predictions of the model (SAOutput: 5 x events x batch size x cuts)
#             references (torch.Tensor): Reference values. (dims: batch size x 4)

#         Returns:
#             float: The loss value.

#         """
#         logits = predictions.logits
#         device_str, device = get_device()
#         num_events = len(logits)
#         events = self.events(references)

#         loss = 0.0
#         for i in range(num_events):
#             cond = torch.logical_or((events == (i + 1)), (events == 0))

#             loss += (
#                 self.loss_fct(
#                     logits[i] * cond[:, None],
#                     self.duration_percentiles(references) * cond,
#                     (events == (i + 1)),
#                     self.fraction_with_quantile(references) * cond,
#                 )
#                 * 1.0
#                 / (
#                     self.importance_sample_weights[events]
#                     * (
#                         num_events * (events == 0).to(torch.int)
#                         + (events != 0).to(torch.int)
#                     )
#                 )
#             ).mean()

#         return loss


# TODO: This needs to be refactored into the current loss template
class MismatchLoss(Loss):
    """Mismatch Loss Class."""

    def __init__(
        self,
        duration_cuts: torch.Tensor,
        max_time: float,
    ):
        super(MismatchLoss, self).__init__()
        self.duration_cuts = duration_cuts
        self.max_time = torch.tensor(max_time)

    def mean_lifetime(self, predictions, references) -> torch.Tensor:
        device = references.device
        num_events = predictions.shape[1]
        # The next two lines compute the expected event time, we consider that the
        # surv probability goes to zero at max_time
        time_intervals = torch.cat(
            (self.duration_cuts, self.max_time.unsqueeze(0)), 0
        ) - torch.cat((torch.tensor(0).to(device).unsqueeze(0), self.duration_cuts), 0)

        surv = self.survivals(predictions)
        mean_lifetimes = torch.zeros(num_events, surv[0].shape[0]).to(device)
        dummy = self.duration_cuts.expand(surv[0].shape).to(device)

        for i in range(num_events):
            mean_lifetimes[i, :] = (
                torch.sum(
                    time_intervals
                    * (
                        torch.cat(
                            (
                                surv[i],
                                torch.tensor(0)
                                .to(device)
                                .expand(dummy.shape[0])
                                .view(-1, 1),
                            ),
                            1,
                        )
                        + torch.cat(
                            (
                                torch.tensor(1)
                                .to(device)
                                .expand(dummy.shape[0])
                                .view(-1, 1),
                                surv[i],
                            ),
                            1,
                        )
                    ),
                    dim=1,
                )
                / 2
            )

            return mean_lifetimes

    def mismatch_loss(self, references, mean_lifetimes) -> torch.Tensor:
        device = references.device
        duration = self.durations(references)
        events = self.events(references)

        # Finding the first event, and cases where we have mismatch
        est_event = torch.argmin(mean_lifetimes, dim=0) + 1
        mismatch = (est_event != events) & (events != 0)

        # The following variables are defined to help mismatch loss computation
        mean_life_temp = mean_lifetimes[:, mismatch]
        est_event_temp = est_event[mismatch]
        event_temp = events[mismatch]
        mean_life_event = torch.zeros(event_temp.shape[0]).to(device)
        mean_life_est = torch.zeros(event_temp.shape[0]).to(device)

        for i in range(event_temp.shape[0]):
            mean_life_event[i] = mean_life_temp[
                event_temp[i] - 1, i
            ]  # Estimated time of actual event
            mean_life_est[i] = mean_life_temp[
                est_event_temp[i] - 1, i
            ]  # Estimated event time of wrong estimated event

        mismatch_loss = torch.mean(
            nn.ReLU()(duration[mismatch] - mean_life_est)
            + nn.ReLU()(mean_life_event - duration[mismatch])
            + mean_life_event
            - mean_life_est
        )

        return mismatch_loss

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute the mismatch loss.

        This function implements SurvTrace loss for both competing and single event cases
        Parameters:
            predictions (SAOutput: Predictions of the model (SAOutput: 5 x events x batch size x cuts)
            references (torch.Tensor): Reference values. (dims: batch size x 4)

        Returns:
            float: The loss value.
        """
        logits = predictions.logits
        mean_lifetimes = self.mean_lifetime(logits, references)
        logger.debug(f"Mean lifetimes {mean_lifetimes}")
        loss = self.mismatch_loss(references, mean_lifetimes)

        return loss
