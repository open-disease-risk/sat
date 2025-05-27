import torch 
import pandas as pd

from ..base import Loss
from sat.models.heads import SAOutput


class IntraEventRankingLoss(Loss):
    def __init__(
        self,
        duration_cuts: str,
        importance_sample_weights: str = None,
        sigma: float = 1.0,
        num_events: int = 1,
    ):
        super(IntraEventRankingLoss, self).__init__(num_events)

        self.sigma = sigma

        # Load importance sampling weights (optional)
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events)

        self.register_buffer("weights", weights)

        # Load time cut points
        df = pd.read_csv(duration_cuts, header=None, names=["cuts"])
        self.register_buffer("duration_cuts", torch.tensor(df.cuts.values, dtype=torch.float32))
        self.num_time_bins = len(df.cuts)
        
        
        
    def ranking_loss(
        self, events, durations, survival, hazard, weights
    ) -> torch.Tensor:
        """
        Efficient implementation of mismatching loss with vectorized operations.

        Args:
            events: Event indicators (dims: num_events x batch_size)
            durations: Event times (dims: num_events x batch_size)
            survival: Survival probabilities (dims:  num_events x batch_size x (num_time_bins+1))
            hazard: Hazard values (dims: num_events x batch_size x num_time_bins)
            weights: Importance weights (dims: batch_size x num_events-1 x num_events or None)

        Returns:
            torch.Tensor: The computed loss value
        """
        device = events.device
        e, n = events.shape  # num_events, batch_size
        tn = hazard.shape[2]  # num_time_bins

        # create event mask once
        I = events.to(bool)
        I_censored = ~I  # e x n
        
        # Initialize duration cut points tensor efficiently
        T = self.duration_cuts.to(device).expand(n, e, -1)  # (n x e x tn)

        # Compute indices for time intervals - done once and reused
        durations_expanded = durations.unsqueeze(2)  # (n x e x 1)
        cuts_expanded = self.duration_cuts.to(device).view(1, 1, -1)  # (1 x 1 x tn)
        indexSmaller = cuts_expanded <= durations_expanded  # (n x e x tn)

        # Calculate left and right boundary indices for each event
        # t0Index: last index where cut is smaller than duration
        t0Index = torch.sum(indexSmaller, dim=2) - 1  # (e x n)

        # Fix negative indices (for durations smaller than all cuts)
        # t1Index: first index where cut is greater than duration
        t0Index = torch.clamp(t0Index, min=0)

        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)  # (e x e x n)
        t1Index = t0Index + 1  # (e x e x n)

        # Fix out of bounds indices
        max_idx = len(self.duration_cuts) - 1
        fixOOB = t1Index >= len(self.duration_cuts) # (e x e x n)
        t1Index[fixOOB] = max_idx

        # Gather time values efficiently
        T0 = torch.gather(T, 2, t0Index)  # (e x e x n)
        T1 = torch.gather(T, 2, t1Index)  # (e x e x n)

        # Gather survival and hazard values efficiently
        SatT0 = torch.gather(survival, 2, t0Index)  # (e x e x n)
        SatT1 = torch.gather(survival, 2, t1Index)  # (e x e x n)
        hstar = torch.gather(hazard, 2, t0Index)  # (e x e x n)

        # Calculate time differences
        dT = T1 - T0  # (e x e x n)

        # Handle interpolation for hazard
        positive_mask = torch.gt(dT, 0.0)

        # Use masked operations to avoid unnecessary calculations
        if positive_mask.any():
            # Add small epsilon for numerical stability
            epsilon = 1e-6
            log_SatT0 = torch.log(SatT0[positive_mask] + epsilon)
            log_SatT1 = torch.log(SatT1[positive_mask] + epsilon)
            hstar[positive_mask] = (log_SatT0 - log_SatT1) / dT[positive_mask]

        # Calculate survival at specific times, SatT[i,j,k] is the survival of event i at time of event_j for observation k
        durations_tiled = durations.unsqueeze(1).repeat(1, e, 1)  # (e x e x n)
        SatT = SatT0 * torch.exp(-(durations_tiled - T0) * hstar)  # (e x e x n)

        # Calculate epsilon time for survival computation
        # TODO: Question: if duration_cuts is zero and duration_cuts[-1] is a small number(under 10), then TMinus will be far away from durations_tilled
        t_epsilon = (
            self.duration_cuts[-1] - self.duration_cuts[0]
        ) / self.duration_cuts[-1]
        TMinus = torch.nn.functional.relu(durations_tiled - t_epsilon)  # (e x e x n)

        # Calculate survival at t-epsilon
        SatTMinus = SatT0 * torch.exp(-(TMinus - T0) * hstar)  # (e x e x n)

        # Extract diagonals efficiently(The diagonal is sat at t0)
        diag_S = (
            torch.diagonal(SatT, dim1=0, dim2=1).permute(1, 0).unsqueeze(0).repeat(e, 1, 1)
        )  # (e x e x n)
        diag_S2 = (
            torch.diagonal(SatTMinus, dim1=0, dim2=1).permute(1, 0).unsqueeze(0).repeat(e, 1, 1)
        )  # (e x e x n)

        # Calculate survival differences
        SatT_T = torch.transpose(SatT, 0, 1)  # (e x e x n)
        diag_S2_T = torch.transpose(diag_S2, 0, 1)  # (e x e x n)
        diag_S_T = torch.transpose(diag_S, 0, 1)  # (e x e x n)

        
        dS1 = diag_S - SatT_T  # (e x e x n)
        dS2 = SatTMinus - diag_S2_T  # (e x e x n)
        dS3 = SatT - diag_S_T  # (e x e x n)

        # Create comparison masks efficiently
        durations_i = durations.unsqueeze(1).repeat(1, e, 1)  # (e x e x n)
        durations_j = durations.unsqueeze(0).repeat(e, 1, 1)  # (e x e x n)
        comp = torch.sign(durations_i - durations_j)  # (e x e x n)
        
        # Mismatch mask (Here, Only consider i > j -> event i should be before event j)
        lower_mask = torch.tril(torch.ones_like(durations_i[:, :, 0]), diagonal=1).unsqueeze(-1)
        mask = (comp > 0) & (lower_mask.bool())
        comp = comp.masked_fill(mask, 0) # (e x e x n)
        

        # Apply ReLU to keep only positive values
        comp_pos = torch.nn.functional.relu(comp)  # (e x e x n)
        
        
        
        # Create event masks efficiently
        I_expanded = I.unsqueeze(0).repeat(e, 1, 1).float()  # (e x e x n)
        I_T = I.unsqueeze(1).repeat(1, e, 1).float()  # (e x e x n)
        I_censored_T = I_censored.unsqueeze(1).repeat(1, e, 1).float()  # (e x e x n)

        # Create ranking pair masks efficiently
        # A1: event i and event j are both observed, and we consider the event i before event j
        # A2: using TMinus to calculate the loss
        # A3: event j is censored, event i is observed
        A1 = I_expanded * comp_pos  # (n x e x e)
        A2 = A1 * I_T  # (n x e x e)
        A3 = A1 * I_censored_T  # (n x e x e)

        # Traditional loss using only exponential scaling
        loss_dS1 = torch.exp(dS1 / self.sigma)
        loss_dS2 = torch.exp(dS2 / self.sigma)
        loss_dS3 = torch.exp(dS3 / self.sigma)

        # Apply weights if provided
        if weights is not None:
            # Ensure weights have appropriate device
            weights = weights.to(device)
            loss_term = weights * (A1 * loss_dS1 + A2 * loss_dS2 + A3 * loss_dS3)
        else:
            loss_term = A1 * loss_dS1 + A2 * loss_dS2 + A3 * loss_dS3

        # Calculate mean efficiently
        # Count number of non-zero elements for proper normalization
        num_valid = torch.sum((A1 + A2 + A3) > 0)

        # Return zero tensor with gradient if no valid comparisons
        return torch.sum(loss_term) / num_valid if num_valid > 0 else torch.tensor(0.0, device=device, requires_grad=True)


    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """
        Compute the ranking loss by permuting tensors to compare observations.

        Parameters:
            predictions (SAOutput): Predictions of the model with survival probabilities
            references (torch.Tensor): Reference values (dims: batch size x 4 * num_events)

        Returns:
            torch.Tensor: The loss value
        """
        # Permute the dimensions to change from [batch, events] to [events, batch]
        # This allows comparing different observations with the same event type
        events = self.events(references).permute(1, 0)
        e, n = events.shape  # Batch size after permutation

        # Create weight tensor if needed - permuted to match the new tensor orientation
        weights_expanded = None
        if self.weights is not None:
            # Skip the first weight (censoring) and use only event weights
            weights_expanded = self.weights[1:].to(references.device)
            # Expand to match the expected dimensions with the permuted orientation
            weights_expanded = (
                weights_expanded.unsqueeze(1).unsqueeze(2).repeat(1, e, n) # (e x e x n)
            )

        # Use the vectorized ranking loss from the parent class
        # with permuted tensors to compare observations instead of events
        return self.ranking_loss(
            events,
            self.durations(references).permute(1, 0),
            predictions.survival.permute(1, 0, 2),
            predictions.hazard.permute(1, 0, 2),
            weights_expanded,
        )

        


