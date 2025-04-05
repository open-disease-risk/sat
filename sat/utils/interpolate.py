"""Interpolating hazard rates"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"


from typing import Tuple

import torch
from torch import nn


class Interpolator(nn.Module):
    def __init__(self, cut_points: torch.FloatTensor, grid_points: int = 10):
        super(Interpolator, self).__init__()
        self.cut_points = cut_points
        self.grid_points = grid_points

    def forward(
        self, hazard: torch.FloatTensor, survival: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Interpolate the survival function at `grid_points`."""
        device = hazard.device

        if hazard is None:
            raise ValueError("No hazard function to interpolate.")

        hazard = hazard
        survival = survival

        n = hazard.shape[0]
        # times at which to interpolate the survival function
        ts = torch.linspace(
            self.cut_points[0],
            self.cut_points[-1],
            (self.cut_points.shape[0] - 1) * self.grid_points,
            device=device,
        )

        T = self.cut_points.expand(n, -1)

        mask = self.cut_points.unsqueeze(0) <= ts.unsqueeze(1)
        t0Index = torch.sum(mask, dim=1) - 1  # left boundary of time intervals (ts)
        t1Index = torch.sum(mask, dim=1)  # right boundary of time intervals (ts)

        fixOOB = t1Index == self.cut_points.shape[0]
        t1Index[fixOOB] = self.cut_points.shape[0] - 1

        T0 = torch.index_select(self.cut_points, dim=0, index=t0Index).expand(
            n, -1
        )  # time at T0Index (n x ts)
        T1 = torch.index_select(self.cut_points, dim=0, index=t1Index).expand(
            n, -1
        )  # time at T1Index (n x ts)

        SatT0 = torch.index_select(
            survival, dim=1, index=t0Index
        )  # survival at T0 (n x ts)
        SatT1 = torch.index_select(
            survival, dim=1, index=t1Index
        )  # survival at T1 (n x ts)

        hatT0 = torch.index_select(hazard, dim=1, index=t0Index)
        dT = T1 - T0

        hstar = torch.div(
            torch.log(0.000001 + SatT0) - torch.log(0.000001 + SatT1),
            (dT),
        )  # solve for hazard given the survival at T0 and T1 (n x ts)

        # when dT is zero or negative we know that the time duration for an
        # observation is greater or equal to the maximum duration cut. So, we
        # need to use the corresponding hazard rather than interpolating
        negative_mask = torch.le(dT, 0.0)
        if torch.any(negative_mask):
            hstar[negative_mask] = hatT0[negative_mask]

        SatT = SatT0 * torch.exp(
            -(ts.reshape(1, -1).expand(n, -1) - T0) * hstar
        )  # solve for survival at time t (n x ts)

        return ts, hstar, SatT
