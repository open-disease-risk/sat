"""Test the loss functionality."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import unittest
import torch

events = torch.tensor(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
)

durations = torch.tensor(
    [
        [33.5497, 44.9450, 52.9705],
        [41.7243, 33.5752, 5.6912],
        [47.2839, 23.6867, 14.4613],
        [81.8689, 29.9798, 34.2786],
        [31.4856, 54.5854, 8.9012],
        [63.8081, 65.0721, 37.0763],
        [0.8597, 31.1869, 39.0517],
        [90.4055, 24.1355, 12.2456],
    ]
)

hazard = torch.tensor(
    [
        [
            [0.0000, 0.7499, 0.7573, 0.5667, 0.3400],
            [0.0000, 0.2152, 0.7564, 0.3471, 0.2860],
            [0.0000, 0.4641, 0.4854, 0.0488, 0.1614],
        ],
        [
            [0.0000, 0.9417, 0.7911, 0.1852, 0.0637],
            [0.0000, 0.5227, 0.6552, 0.1906, 0.2647],
            [0.0000, 0.9427, 0.4955, 0.7755, 0.9287],
        ],
        [
            [0.0000, 0.9885, 0.2216, 0.7868, 0.2136],
            [0.0000, 0.8153, 0.1046, 0.9159, 0.9111],
            [0.0000, 0.8708, 0.8820, 0.8580, 0.0475],
        ],
        [
            [0.0000, 0.7937, 0.7597, 0.8319, 0.9429],
            [0.0000, 0.0637, 0.0363, 0.0388, 0.9893],
            [0.0000, 0.5969, 0.2212, 0.4174, 0.9721],
        ],
        [
            [0.0000, 0.6706, 0.8313, 0.8086, 0.9726],
            [0.0000, 0.9978, 0.4443, 0.4320, 0.8237],
            [0.0000, 0.0347, 0.3580, 0.4140, 0.8433],
        ],
        [
            [0.0000, 0.6478, 0.0373, 0.8160, 0.1342],
            [0.0000, 0.3275, 0.4752, 0.8146, 0.2459],
            [0.0000, 0.9899, 0.5558, 0.6715, 0.9468],
        ],
        [
            [0.0000, 0.4254, 0.8707, 0.4908, 0.4262],
            [0.0000, 0.4023, 0.6601, 0.4823, 0.1233],
            [0.0000, 0.8618, 0.9668, 0.1375, 0.0577],
        ],
        [
            [0.0000, 0.5526, 0.8428, 0.0011, 0.8259],
            [0.0000, 0.3710, 0.6289, 0.0732, 0.7696],
            [0.0000, 0.6532, 0.9453, 0.9594, 0.4427],
        ],
    ]
)

survival = hazard.cumsum(2).mul(-1).exp()

duration_cuts_1 = torch.tensor([0, 20, 40, 60, 80])
duration_cuts_2 = torch.tensor([0, 20, 40, 60, 80, 100])

weights = torch.tensor([0.4, 0.3, 0.2, 0.1])
n = 8
e = 3


class TestRankingLossFunctionality(unittest.TestCase):
    def test_T(self):
        T = duration_cuts_1.expand(n, e, -1)
        self.assertTrue(torch.all(T.eq(duration_cuts_1)))

    def test_IndexSmaller(self):
        durationShape = durations.unsqueeze(2).shape
        self.assertEqual(durationShape[0], 8)
        self.assertEqual(durationShape[1], 3)
        self.assertEqual(durationShape[2], 1)

        durationCutShape = duration_cuts_1.view(1, 1, -1).shape
        self.assertEqual(durationCutShape[0], 1)
        self.assertEqual(durationCutShape[1], 1)
        self.assertEqual(durationCutShape[2], 5)

        indexSmaller = duration_cuts_1.view(1, 1, -1) <= durations.unsqueeze(2)
        for n in range(indexSmaller.shape[0]):
            for e in range(indexSmaller.shape[1]):
                for tn in range(indexSmaller.shape[2]):
                    self.assertEqual(
                        indexSmaller[n, e, tn], duration_cuts_1[tn] <= durations[n, e]
                    )

    def test_t0Index(self):
        indexSmaller = duration_cuts_1.view(1, 1, -1) <= durations.unsqueeze(2)
        t0Index = torch.sum(indexSmaller, dim=2) - 1
        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)
        for n in range(t0Index.shape[0]):
            for e1 in range(t0Index.shape[1]):
                for e2 in range(t0Index.shape[2]):
                    if t0Index[n, e1, e2] > 0:
                        self.assertTrue(indexSmaller[n, e2, t0Index[n, e1, e2] - 1])
                    self.assertTrue(indexSmaller[n, e2, t0Index[n, e1, e2]])
                    if t0Index[n, e1, e2] < duration_cuts_1.shape[0] - 1:
                        self.assertFalse(indexSmaller[n, e2, t0Index[n, e1, e2] + 1])

    def test_t1Index(self):
        indexSmaller = duration_cuts_1.view(1, 1, -1) <= durations.unsqueeze(2)
        t0Index = torch.sum(indexSmaller, dim=2) - 1
        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)
        t1Index = t0Index + 1
        self.assertEqual(torch.sum(t1Index == len(duration_cuts_1)), 6)

        fixOOB = t1Index == len(duration_cuts_1)
        t1Index[fixOOB] = len(duration_cuts_1) - 1
        self.assertEqual(torch.sum(t1Index == len(duration_cuts_1)), 0)

    def test_T0(self):
        T = duration_cuts_1.expand(n, e, -1)
        indexSmaller = duration_cuts_1.view(1, 1, -1) <= durations.unsqueeze(2)
        t0Index = torch.sum(indexSmaller, dim=2) - 1
        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)
        T0 = torch.gather(
            T, 2, t0Index
        )  # left boundary of time interval for all events i and time constraints j (n x e x e)
        for o in range(T0.shape[0]):
            self.assertTrue(torch.all(T0[o, 0] == T0[o]).item())

    def test_T1(self):
        T = duration_cuts_1.expand(n, e, -1)
        indexSmaller = duration_cuts_1.view(1, 1, -1) <= durations.unsqueeze(2)
        t0Index = torch.sum(indexSmaller, dim=2) - 1
        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)
        t1Index = t0Index + 1  # right boundary of time interval (n x e)

        # if we run out of bounds, we match t0Index this means that dT will be
        # zero and causes NaNs in hstar, which we need to fix
        fixOOB = t1Index == len(duration_cuts_1)
        t1Index[fixOOB] = len(duration_cuts_1) - 1
        T0 = torch.gather(
            T, 2, t0Index
        )  # left boundary of time interval for all events i and time constraints j (n x e x e)
        T1 = torch.gather(
            T, 2, t1Index
        )  # right boundary of time interval for all events i and time constraints j (n x e)
        for o in range(T1.shape[0]):
            self.assertTrue(torch.all(T1[o, 0] == T1[o]).item())

        self.assertTrue(torch.all(T0 <= T1))

        dT = T1 - T0
        self.assertTrue(torch.all(dT >= 0))

    def test_S0_S1(self):
        T = duration_cuts_1.expand(n, e, -1)
        indexSmaller = duration_cuts_1.view(1, 1, -1) <= durations.unsqueeze(2)
        t0Index = torch.sum(indexSmaller, dim=2) - 1
        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)
        t1Index = t0Index + 1  # right boundary of time interval (n x e)

        # if we run out of bounds, we match t0Index this means that dT will be
        # zero and causes NaNs in hstar, which we need to fix
        fixOOB = t1Index == len(duration_cuts_1)
        t1Index[fixOOB] = len(duration_cuts_1) - 1
        T0 = torch.gather(
            T, 2, t0Index
        )  # left boundary of time interval for all events i and time constraints j (n x e x e)
        T1 = torch.gather(
            T, 2, t1Index
        )  # right boundary of time interval for all events i and time constraints j (n x e)
        SatT0 = torch.gather(survival, 2, t0Index)  # survival at T0 (n x e x e)
        SatT1 = torch.gather(survival, 2, t1Index)  # survival at T1 (n x e x e)

        self.assertTrue(torch.all(SatT0 >= SatT1))

    def test_SatT(self):
        T = duration_cuts_1.expand(n, e, -1)
        indexSmaller = duration_cuts_1.view(1, 1, -1) <= durations.unsqueeze(2)
        t0Index = torch.sum(indexSmaller, dim=2) - 1
        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)
        t1Index = t0Index + 1  # right boundary of time interval (n x e)

        # if we run out of bounds, we match t0Index this means that dT will be
        # zero and causes NaNs in hstar, which we need to fix
        fixOOB = t1Index == len(duration_cuts_1)
        t1Index[fixOOB] = len(duration_cuts_1) - 1
        T0 = torch.gather(
            T, 2, t0Index
        )  # left boundary of time interval for all events i and time constraints j (n x e x e)
        T1 = torch.gather(
            T, 2, t1Index
        )  # right boundary of time interval for all events i and time constraints j (n x e)
        SatT0 = torch.gather(survival, 2, t0Index)  # survival at T0 (n x e x e)
        SatT1 = torch.gather(survival, 2, t1Index)  # survival at T1 (n x e x e)
        hstar = torch.gather(hazard, 2, t0Index)
        dT = T1 - T0
        positive_mask = torch.gt(dT, 0.0)
        hstar[positive_mask] = torch.div(
            torch.log(0.000001 + SatT0[positive_mask])
            - torch.log(0.000001 + SatT1[positive_mask]),
            (dT[positive_mask]),
        )  # solve for hazard given the survival at T0 and T1 (n x e x e)

        SatT = SatT0 * torch.exp(
            -(durations.unsqueeze(1).repeat(1, e, 1) - T0) * hstar
        )  # solve for survival at time t (n x e x e)

        self.assertTrue(torch.all(SatT0 >= SatT))
        self.assertTrue(torch.all(SatT[positive_mask] >= SatT1[positive_mask]))
        # extrapolation beyond the last duration cut
        self.assertTrue(torch.all(SatT[~positive_mask] <= SatT1[~positive_mask]))

    def test_SatTMinus(self):
        T = duration_cuts_1.expand(n, e, -1)
        indexSmaller = duration_cuts_1.view(1, 1, -1) <= durations.unsqueeze(2)
        t0Index = torch.sum(indexSmaller, dim=2) - 1
        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)
        t1Index = t0Index + 1  # right boundary of time interval (n x e)

        # if we run out of bounds, we match t0Index this means that dT will be
        # zero and causes NaNs in hstar, which we need to fix
        fixOOB = t1Index == len(duration_cuts_1)
        t1Index[fixOOB] = len(duration_cuts_1) - 1
        T0 = torch.gather(
            T, 2, t0Index
        )  # left boundary of time interval for all events i and time constraints j (n x e x e)
        T1 = torch.gather(
            T, 2, t1Index
        )  # right boundary of time interval for all events i and time constraints j (n x e)
        SatT0 = torch.gather(survival, 2, t0Index)  # survival at T0 (n x e x e)
        SatT1 = torch.gather(survival, 2, t1Index)  # survival at T1 (n x e x e)
        hstar = torch.gather(hazard, 2, t0Index)
        dT = T1 - T0
        positive_mask = torch.gt(dT, 0.0)
        hstar[positive_mask] = torch.div(
            torch.log(0.000001 + SatT0[positive_mask])
            - torch.log(0.000001 + SatT1[positive_mask]),
            (dT[positive_mask]),
        )  # solve for hazard given the survival at T0 and T1 (n x e x e)

        SatT = SatT0 * torch.exp(
            -(durations.unsqueeze(1).repeat(1, e, 1) - T0) * hstar
        )  # solve for survival at time t (n x e x e)
        t_epsilon = (duration_cuts_1[-1] - duration_cuts_1[0]) / duration_cuts_1[-1]
        TMinus = torch.nn.functional.relu(
            durations.unsqueeze(1).repeat(1, e, 1) - t_epsilon
        )
        SatTMinus = SatT0 * torch.exp(
            -(TMinus - T0) * hstar
        )  # solve for survival at time t-epsilon (n x e x e)

        self.assertTrue(torch.all(SatT0 >= SatTMinus))
        self.assertTrue(torch.all(SatTMinus >= SatT))

    def test_dS1(self):
        T = duration_cuts_1.expand(n, e, -1)
        indexSmaller = duration_cuts_1.view(1, 1, -1) <= durations.unsqueeze(2)
        t0Index = torch.sum(indexSmaller, dim=2) - 1
        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)
        t1Index = t0Index + 1  # right boundary of time interval (n x e)

        # if we run out of bounds, we match t0Index this means that dT will be
        # zero and causes NaNs in hstar, which we need to fix
        fixOOB = t1Index == len(duration_cuts_1)
        t1Index[fixOOB] = len(duration_cuts_1) - 1
        T0 = torch.gather(
            T, 2, t0Index
        )  # left boundary of time interval for all events i and time constraints j (n x e x e)
        T1 = torch.gather(
            T, 2, t1Index
        )  # right boundary of time interval for all events i and time constraints j (n x e)
        SatT0 = torch.gather(survival, 2, t0Index)  # survival at T0 (n x e x e)
        SatT1 = torch.gather(survival, 2, t1Index)  # survival at T1 (n x e x e)
        hstar = torch.gather(hazard, 2, t0Index)
        dT = T1 - T0
        positive_mask = torch.gt(dT, 0.0)
        hstar[positive_mask] = torch.div(
            torch.log(0.000001 + SatT0[positive_mask])
            - torch.log(0.000001 + SatT1[positive_mask]),
            (dT[positive_mask]),
        )  # solve for hazard given the survival at T0 and T1 (n x e x e)

        SatT = SatT0 * torch.exp(
            -(durations.unsqueeze(1).repeat(1, e, 1) - T0) * hstar
        )  # solve for survival at time t (n x e x e)
        t_epsilon = (duration_cuts_1[-1] - duration_cuts_1[0]) / duration_cuts_1[-1]
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

        for k in range(diag_S.shape[0]):
            for e1 in range(diag_S.shape[1]):
                for e2 in range(diag_S.shape[2]):
                    self.assertEqual(
                        diag_S[k, e1, e2],
                        SatT[k, e1, e1],
                        f"diag_S[{k}, {e1}, {e2}] != SatT[{k}, {e1}, {e1}]",
                    )

        for k in range(dS1.shape[0]):
            for e1 in range(dS1.shape[1]):
                for e2 in range(dS1.shape[2]):
                    self.assertEqual(
                        dS1[k, e1, e2], diag_S[k, e1, e2] - SatT[k, e2, e1]
                    )

    def test_dS2(self):
        T = duration_cuts_1.expand(n, e, -1)
        indexSmaller = duration_cuts_1.view(1, 1, -1) <= durations.unsqueeze(2)
        t0Index = torch.sum(indexSmaller, dim=2) - 1
        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)
        t1Index = t0Index + 1  # right boundary of time interval (n x e)

        # if we run out of bounds, we match t0Index this means that dT will be
        # zero and causes NaNs in hstar, which we need to fix
        fixOOB = t1Index == len(duration_cuts_1)
        t1Index[fixOOB] = len(duration_cuts_1) - 1
        T0 = torch.gather(
            T, 2, t0Index
        )  # left boundary of time interval for all events i and time constraints j (n x e x e)
        T1 = torch.gather(
            T, 2, t1Index
        )  # right boundary of time interval for all events i and time constraints j (n x e)
        SatT0 = torch.gather(survival, 2, t0Index)  # survival at T0 (n x e x e)
        SatT1 = torch.gather(survival, 2, t1Index)  # survival at T1 (n x e x e)
        hstar = torch.gather(hazard, 2, t0Index)
        dT = T1 - T0
        positive_mask = torch.gt(dT, 0.0)
        hstar[positive_mask] = torch.div(
            torch.log(0.000001 + SatT0[positive_mask])
            - torch.log(0.000001 + SatT1[positive_mask]),
            (dT[positive_mask]),
        )  # solve for hazard given the survival at T0 and T1 (n x e x e)

        SatT = SatT0 * torch.exp(
            -(durations.unsqueeze(1).repeat(1, e, 1) - T0) * hstar
        )  # solve for survival at time t (n x e x e)
        t_epsilon = (duration_cuts_1[-1] - duration_cuts_1[0]) / duration_cuts_1[-1]
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
        )  # dS_{ij} = S_{i}(T_{j}) - S_{j}(T_{j}) (n x e x e)
        for k in range(diag_S2.shape[0]):
            for e1 in range(diag_S2.shape[1]):
                for e2 in range(diag_S2.shape[2]):
                    self.assertEqual(
                        diag_S2[k, e1, e2],
                        SatTMinus[k, e1, e1],
                        f"diag_S2[{k}, {e1}, {e2}] != SatTMinus[{k}, {e1}, {e2}]",
                    )

        for k in range(dS2.shape[0]):
            for e1 in range(dS2.shape[1]):
                for e2 in range(dS2.shape[2]):
                    self.assertEqual(
                        dS2[k, e1, e2], SatTMinus[k, e1, e2] - diag_S2[k, e2, e1]
                    )

    def test_A1(self):
        T = duration_cuts_1.expand(n, e, -1)
        indexSmaller = duration_cuts_1.view(1, 1, -1) <= durations.unsqueeze(2)
        t0Index = torch.sum(indexSmaller, dim=2) - 1
        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)
        t1Index = t0Index + 1  # right boundary of time interval (n x e)

        # if we run out of bounds, we match t0Index this means that dT will be
        # zero and causes NaNs in hstar, which we need to fix
        fixOOB = t1Index == len(duration_cuts_1)
        t1Index[fixOOB] = len(duration_cuts_1) - 1
        T0 = torch.gather(
            T, 2, t0Index
        )  # left boundary of time interval for all events i and time constraints j (n x e x e)
        T1 = torch.gather(
            T, 2, t1Index
        )  # right boundary of time interval for all events i and time constraints j (n x e)
        SatT0 = torch.gather(survival, 2, t0Index)  # survival at T0 (n x e x e)
        SatT1 = torch.gather(survival, 2, t1Index)  # survival at T1 (n x e x e)
        hstar = torch.gather(hazard, 2, t0Index)
        dT = T1 - T0
        positive_mask = torch.gt(dT, 0.0)
        hstar[positive_mask] = torch.div(
            torch.log(0.000001 + SatT0[positive_mask])
            - torch.log(0.000001 + SatT1[positive_mask]),
            (dT[positive_mask]),
        )  # solve for hazard given the survival at T0 and T1 (n x e x e)

        SatT = SatT0 * torch.exp(
            -(durations.unsqueeze(1).repeat(1, e, 1) - T0) * hstar
        )  # solve for survival at time t (n x e x e)
        t_epsilon = (duration_cuts_1[-1] - duration_cuts_1[0]) / duration_cuts_1[-1]
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
        )  # dS_{ij} = S_{i}(T_{j}) - S_{j}(T_{j}) (n x e x e)

        I = events.to(bool)
        # A_{nij}=1 if t_i < t_j and A_{ij}=0 if t_i >= t_j
        #              and A_{ij}=1 when event occured for subject i (n x e x e)
        A1 = I.unsqueeze(2).repeat(1, 1, e).float() * torch.nn.functional.relu(
            torch.sign(
                durations.unsqueeze(1).repeat(1, e, 1)
                - durations.unsqueeze(2).repeat(1, 1, e)
            )
        )

        for k in range(A1.shape[0]):
            for e1 in range(A1.shape[1]):
                for e2 in range(A1.shape[2]):
                    if A1[k, e1, e2]:
                        self.assertTrue(durations[k, e1] < durations[k, e2])
                        self.assertTrue(
                            durations.unsqueeze(2).repeat(1, 1, e)[k, e1, e2]
                            < durations.unsqueeze(1).repeat(1, e, 1)[k, e1, e2]
                        )
                        self.assertTrue(events[k, e1])
                    else:
                        self.assertEqual(A1[k, e1, e2], 0)

    def test_A2(self):
        T = duration_cuts_1.expand(n, e, -1)
        indexSmaller = duration_cuts_1.view(1, 1, -1) <= durations.unsqueeze(2)
        t0Index = torch.sum(indexSmaller, dim=2) - 1
        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)
        t1Index = t0Index + 1  # right boundary of time interval (n x e)

        # if we run out of bounds, we match t0Index this means that dT will be
        # zero and causes NaNs in hstar, which we need to fix
        fixOOB = t1Index == len(duration_cuts_1)
        t1Index[fixOOB] = len(duration_cuts_1) - 1
        T0 = torch.gather(
            T, 2, t0Index
        )  # left boundary of time interval for all events i and time constraints j (n x e x e)
        T1 = torch.gather(
            T, 2, t1Index
        )  # right boundary of time interval for all events i and time constraints j (n x e)
        SatT0 = torch.gather(survival, 2, t0Index)  # survival at T0 (n x e x e)
        SatT1 = torch.gather(survival, 2, t1Index)  # survival at T1 (n x e x e)
        hstar = torch.gather(hazard, 2, t0Index)
        dT = T1 - T0
        positive_mask = torch.gt(dT, 0.0)
        hstar[positive_mask] = torch.div(
            torch.log(0.000001 + SatT0[positive_mask])
            - torch.log(0.000001 + SatT1[positive_mask]),
            (dT[positive_mask]),
        )  # solve for hazard given the survival at T0 and T1 (n x e x e)

        SatT = SatT0 * torch.exp(
            -(durations.unsqueeze(1).repeat(1, e, 1) - T0) * hstar
        )  # solve for survival at time t (n x e x e)
        t_epsilon = (duration_cuts_1[-1] - duration_cuts_1[0]) / duration_cuts_1[-1]
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
        )  # dS_{ij} = S_{i}(T_{j}) - S_{j}(T_{j}) (n x e x e)

        I = events.to(bool)
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

        for k in range(A2.shape[0]):
            for e1 in range(A2.shape[1]):
                for e2 in range(A2.shape[2]):
                    if A2[k, e1, e2]:
                        self.assertTrue(durations[k, e1] < durations[k, e2])
                        self.assertTrue(
                            durations.unsqueeze(2).repeat(1, 1, e)[k, e1, e2]
                            < durations.unsqueeze(1).repeat(1, e, 1)[k, e1, e2]
                        )
                        self.assertTrue(events[k, e1])
                        self.assertTrue(events[k, e2])
                    else:
                        self.assertEqual(A2[k, e1, e2], 0)

    def test_A3(self):
        T = duration_cuts_1.expand(n, e, -1)
        indexSmaller = duration_cuts_1.view(1, 1, -1) <= durations.unsqueeze(2)
        t0Index = torch.sum(indexSmaller, dim=2) - 1
        t0Index = t0Index.unsqueeze(1).repeat(1, e, 1)
        t1Index = t0Index + 1  # right boundary of time interval (n x e)

        # if we run out of bounds, we match t0Index this means that dT will be
        # zero and causes NaNs in hstar, which we need to fix
        fixOOB = t1Index == len(duration_cuts_1)
        t1Index[fixOOB] = len(duration_cuts_1) - 1
        T0 = torch.gather(
            T, 2, t0Index
        )  # left boundary of time interval for all events i and time constraints j (n x e x e)
        T1 = torch.gather(
            T, 2, t1Index
        )  # right boundary of time interval for all events i and time constraints j (n x e)
        SatT0 = torch.gather(survival, 2, t0Index)  # survival at T0 (n x e x e)
        SatT1 = torch.gather(survival, 2, t1Index)  # survival at T1 (n x e x e)
        hstar = torch.gather(hazard, 2, t0Index)
        dT = T1 - T0
        positive_mask = torch.gt(dT, 0.0)
        hstar[positive_mask] = torch.div(
            torch.log(0.000001 + SatT0[positive_mask])
            - torch.log(0.000001 + SatT1[positive_mask]),
            (dT[positive_mask]),
        )  # solve for hazard given the survival at T0 and T1 (n x e x e)

        SatT = SatT0 * torch.exp(
            -(durations.unsqueeze(1).repeat(1, e, 1) - T0) * hstar
        )  # solve for survival at time t (n x e x e)
        t_epsilon = (duration_cuts_1[-1] - duration_cuts_1[0]) / duration_cuts_1[-1]
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
        )  # dS_{ij} = S_{i}(T_{j}) - S_{j}(T_{j}) (n x e x e)

        I = events.to(bool)
        I_censored = ~I  # censored indicator (n x e)
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

        for k in range(A3.shape[0]):
            for e1 in range(A3.shape[1]):
                for e2 in range(A3.shape[2]):
                    if A3[k, e1, e2]:
                        self.assertTrue(durations[k, e1] < durations[k, e2])
                        self.assertTrue(
                            durations.unsqueeze(2).repeat(1, 1, e)[k, e1, e2]
                            < durations.unsqueeze(1).repeat(1, e, 1)[k, e1, e2]
                        )
                        self.assertTrue(events[k, e1])
                        self.assertFalse(events[k, e2])
                    else:
                        self.assertEqual(A3[k, e1, e2], 0)

        self.assertTrue(torch.all(A1 == A2 + A3))


if __name__ == "__main__":
    unittest.main()
