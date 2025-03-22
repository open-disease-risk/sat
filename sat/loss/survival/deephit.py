"""DeepHit loss components for survival analysis with competing risks."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Union, List, Tuple

from ..balancing import BalancingStrategy
from sat.models.heads import SAOutput
from sat.utils import logging

from ..base import Loss

logger = logging.get_default_logger()


class DeepHitLikelihoodLoss(Loss):
    """
    Likelihood loss component of DeepHit.
    
    Computes the negative log-likelihood for competing risks survival data,
    considering both event and censored observations.
    """
    
    def __init__(
        self,
        num_events: int = 1,
        importance_sample_weights: Optional[str] = None,
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
    ):
        """
        Initialize DeepHitLikelihoodLoss.
        
        Args:
            num_events: Number of competing events
            importance_sample_weights: Optional path to CSV file with importance weights
            balance_strategy: Strategy for balancing loss components
            balance_params: Additional parameters for the balancing strategy
        """
        super(DeepHitLikelihoodLoss, self).__init__(
            num_events=num_events,
            balance_strategy=balance_strategy,
            balance_params=balance_params
        )
        
        # Load importance sampling weights if provided
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)
            
        self.register_buffer("weights", weights)
    
    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the negative log-likelihood loss component.
        
        Args:
            predictions: Model predictions (SAOutput with survival probabilities)
            references: Ground truth references
            
        Returns:
            Negative log-likelihood loss
        """
        batch_size = predictions.logits.shape[0]
        
        # Extract event information
        events = self.events(references)        # [batch_size, num_events]
        duration_idx = self.duration_percentiles(references)  # [batch_size, num_events]
        
        # Create masks for each event type
        event_masks = []
        for i in range(self.num_events):
            # Check if event i occurred
            event_occurred = events[:, i] == 1
            event_masks.append(event_occurred)
        
        # Get hazard and survival values from predictions
        hazard = predictions.hazard  # [batch_size, num_events, num_time_bins]
        survival = predictions.survival  # [batch_size, num_events, num_time_bins+1]
        
        # Compute negative log-likelihood for uncensored subjects
        uncensored_loss = 0.0
        num_uncensored = 0
        
        for i in range(self.num_events):
            # Get mask for subjects with event type i
            mask = event_masks[i]
            
            if mask.sum() > 0:
                # Get indices of event times for subjects with event type i
                time_idx = duration_idx[mask, i]
                
                # Get hazard at event time for event type i
                event_hazards = hazard[mask, i, :]
                event_hazard_at_t = torch.gather(event_hazards, 1, time_idx.unsqueeze(1)).squeeze(1)
                
                # Get survival up to event time for all event types (including i)
                event_survival = survival[mask, :, :]
                
                # For the specific event type i, get survival right before event
                event_survival_before_t = torch.gather(
                    event_survival[:, i, :], 
                    1, 
                    time_idx.unsqueeze(1)
                ).squeeze(1)
                
                # For all other event types, get survival at event time
                other_events_survival = torch.ones_like(event_hazard_at_t)
                for j in range(self.num_events):
                    if j != i:
                        other_event_survival_at_t = torch.gather(
                            event_survival[:, j, :], 
                            1, 
                            (time_idx + 1).unsqueeze(1)  # +1 because survival includes time 0
                        ).squeeze(1)
                        other_events_survival *= other_event_survival_at_t
                
                # Probability of event i at time t
                prob_i_t = event_hazard_at_t * event_survival_before_t * other_events_survival
                
                # Add small constant for numerical stability
                prob_i_t = torch.clamp(prob_i_t, min=1e-7)
                
                # Negative log-likelihood
                event_nll = -torch.log(prob_i_t)
                
                # Apply weights if provided
                if self.weights is not None:
                    event_nll = event_nll * self.weights[i+1]  # +1 to skip first weight
                
                uncensored_loss += event_nll.sum()
                num_uncensored += mask.sum()
        
        # Compute negative log-likelihood for censored subjects
        censored_mask = torch.all(events == 0, dim=1)
        censored_loss = 0.0
        
        if censored_mask.sum() > 0:
            # For censored subjects, get the last observed time
            censored_times = torch.max(duration_idx[censored_mask], dim=1)[0]
            
            # Get overall survival probability at censoring time for all event types
            censored_survival = survival[censored_mask]
            
            overall_survival = torch.ones(censored_mask.sum(), device=survival.device)
            for i in range(self.num_events):
                # Get survival at censoring time (+1 for indexing since survival includes time 0)
                surv_i_at_censor = torch.gather(
                    censored_survival[:, i, :], 
                    1, 
                    (censored_times + 1).unsqueeze(1)
                ).squeeze(1)
                overall_survival *= surv_i_at_censor
            
            # Add small constant for numerical stability
            overall_survival = torch.clamp(overall_survival, min=1e-7)
            
            # Negative log-likelihood for censored subjects
            censor_nll = -torch.log(overall_survival)
            
            # Apply weight for censored observations
            if self.weights is not None:
                censor_nll = censor_nll * self.weights[0]  # First weight is for censored
                
            censored_loss = censor_nll.sum()
        
        # Combine losses and normalize
        total_subjects = batch_size
        total_loss = (uncensored_loss + censored_loss) / total_subjects
        
        return total_loss


class DeepHitRankingLoss(Loss):
    """
    Ranking loss component of DeepHit.
    
    Computes the ranking loss to ensure proper ordering of survival probabilities,
    penalizing when subjects with earlier events have higher survival probabilities
    than those with later events.
    """
    
    def __init__(
        self,
        duration_cuts: str,
        sigma: float = 0.1,
        num_events: int = 1,
        importance_sample_weights: Optional[str] = None,
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
    ):
        """
        Initialize DeepHitRankingLoss.
        
        Args:
            duration_cuts: Path to CSV file containing duration cut points for discretization
            sigma: Scaling factor for ranking loss (smaller values = sharper differences)
            num_events: Number of competing events
            importance_sample_weights: Optional path to CSV file with importance weights
            balance_strategy: Strategy for balancing loss components
            balance_params: Additional parameters for the balancing strategy
        """
        super(DeepHitRankingLoss, self).__init__(
            num_events=num_events,
            balance_strategy=balance_strategy,
            balance_params=balance_params
        )
        
        self.sigma = sigma
        
        # Load duration cut points
        df = pd.read_csv(duration_cuts, header=None, names=["cuts"])
        self.register_buffer("duration_cuts", torch.tensor(df.cuts.values, dtype=torch.float32))
        self.num_time_bins = len(df.cuts)
        
        # Load importance sampling weights if provided
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)
            
        self.register_buffer("weights", weights)
    
    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the ranking loss component.
        
        Args:
            predictions: Model predictions (SAOutput with survival probabilities)
            references: Ground truth references
            
        Returns:
            Ranking loss component
        """
        batch_size = predictions.logits.shape[0]
        survival = predictions.survival  # [batch_size, num_events, num_time_bins+1]
        
        # Extract event information
        events = self.events(references)        # [batch_size, num_events]
        durations = self.durations(references)  # [batch_size, num_events]
        
        # Create indicator matrix for each event type
        rank_loss = 0.0
        
        for event_type in range(self.num_events):
            # Get indices of subjects with this event type
            event_occurred = events[:, event_type] == 1
            
            if event_occurred.sum() == 0:
                continue
                
            # Get event times for subjects with this event type
            event_times = durations[event_occurred, event_type]
            
            # For each subject with this event type, compare with all other subjects
            for i, time_i in enumerate(event_times):
                # Create risk indicator: 1 if subject j's time > subject i's time
                risk_indicator = (durations[:, event_type] > time_i).float()
                
                # Compute survival at time_i for all subjects for this event type
                # Find closest time bin index for time_i
                time_bin_idx = torch.searchsorted(self.duration_cuts, time_i)
                # Ensure index is within bounds (max index should be num_time_bins)
                time_bin_idx = torch.clamp(time_bin_idx, max=self.num_time_bins-1)
                
                # Get survival at time_i for all subjects
                # +1 because survival includes time 0, but make sure it's within bounds
                survival_idx = torch.min(time_bin_idx + 1, torch.tensor(survival.size(2)-1, device=time_bin_idx.device))
                all_survival_at_t = survival[:, event_type, survival_idx]
                
                # Subject i's survival
                i_idx = torch.where(event_occurred)[0][i]
                i_survival_at_t = all_survival_at_t[i_idx]
                
                # Check for higher risk predictions for subject i compared to others who should be lower risk
                survival_diff = i_survival_at_t - all_survival_at_t
                
                # Apply exponential scaling with sigma
                exp_diff = torch.exp(survival_diff / self.sigma)
                
                # Multiply by risk indicator to only consider valid comparisons
                valid_comparisons = risk_indicator * exp_diff
                
                # Apply weight for this event type
                if self.weights is not None:
                    valid_comparisons = valid_comparisons * self.weights[event_type+1]
                
                # Sum over all comparisons
                rank_loss += valid_comparisons.sum()
        
        # Normalize by number of subjects
        rank_loss = rank_loss / batch_size if batch_size > 0 else 0.0
        
        return rank_loss


class DeepHitCalibrationLoss(Loss):
    """
    Calibration loss component of DeepHit.
    
    Computes the mean squared error between predicted event probabilities
    and actual event indicators at specific time points.
    """
    
    def __init__(
        self,
        duration_cuts: str,
        eval_times: Optional[List[float]] = None,
        num_events: int = 1,
        importance_sample_weights: Optional[str] = None,
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
    ):
        """
        Initialize DeepHitCalibrationLoss.
        
        Args:
            duration_cuts: Path to CSV file containing duration cut points for discretization
            eval_times: Optional list of specific times to evaluate calibration
            num_events: Number of competing events
            importance_sample_weights: Optional path to CSV file with importance weights
            balance_strategy: Strategy for balancing loss components
            balance_params: Additional parameters for the balancing strategy
        """
        super(DeepHitCalibrationLoss, self).__init__(
            num_events=num_events,
            balance_strategy=balance_strategy,
            balance_params=balance_params
        )
        
        # Load duration cut points
        df = pd.read_csv(duration_cuts, header=None, names=["cuts"])
        self.register_buffer("duration_cuts", torch.tensor(df.cuts.values, dtype=torch.float32))
        self.num_time_bins = len(df.cuts)
        
        # Store evaluation times if provided
        if eval_times is not None:
            self.register_buffer("eval_times", torch.tensor(eval_times, dtype=torch.float32))
            self.eval_time_indices = []
            for t in eval_times:
                idx = torch.searchsorted(self.duration_cuts, t).item()
                if idx >= self.num_time_bins:
                    idx = self.num_time_bins - 1
                self.eval_time_indices.append(idx)
            self.eval_time_indices = torch.tensor(self.eval_time_indices)
        else:
            self.eval_times = None
            self.eval_time_indices = None
        
        # Load importance sampling weights if provided
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)
            
        self.register_buffer("weights", weights)
    
    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the calibration loss component.
        
        Args:
            predictions: Model predictions (SAOutput with survival probabilities)
            references: Ground truth references
            
        Returns:
            Calibration loss component
        """
        batch_size = predictions.logits.shape[0]
        survival = predictions.survival  # [batch_size, num_events, num_time_bins+1]
        
        # If no evaluation times provided, use all duration cut points
        if self.eval_time_indices is None:
            eval_time_indices = torch.arange(self.num_time_bins, device=survival.device)
        else:
            eval_time_indices = self.eval_time_indices.to(survival.device)
        
        # Extract event information
        events = self.events(references)        # [batch_size, num_events]
        durations = self.durations(references)  # [batch_size, num_events]
        
        calibration_loss = 0.0
        num_comparisons = 0
        
        for event_type in range(self.num_events):
            for t_idx in eval_time_indices:
                # Get time value (ensure t_idx is within bounds)
                t_idx_safe = min(t_idx.item(), len(self.duration_cuts)-1)
                t = self.duration_cuts[t_idx_safe]
                
                # Create binary indicator: 1 if subject had event of this type before time t
                event_before_t = ((events[:, event_type] == 1) & (durations[:, event_type] <= t)).float()
                
                # Get predicted probability of event before time t
                # 1 - S(t) for this event type
                # +1 because survival includes time 0, but ensure it's within bounds
                survival_idx = min(t_idx_safe + 1, survival.size(2)-1)
                pred_prob = 1.0 - survival[:, event_type, survival_idx]
                
                # Square difference between actual and predicted
                squared_diff = (event_before_t - pred_prob) ** 2
                
                # Apply weight for this event type
                if self.weights is not None:
                    squared_diff = squared_diff * self.weights[event_type+1]
                
                calibration_loss += squared_diff.sum()
                num_comparisons += batch_size
        
        # Normalize by number of comparisons
        calibration_loss = calibration_loss / num_comparisons if num_comparisons > 0 else 0.0
        
        return calibration_loss