"""label transformation functions from pycox copied here to reduce dependencies"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import warnings

import numpy as np

from sat.pycox.preprocessing.discretization import (
    DiscretizeUnknownC,
    Duration2Idx,
    IdxDiscUnknownC,
    _values_if_series,
    make_cuts,
)


class LabTransPCHazard:
    """
    Defining time intervals (`cuts`) needed for the `PCHazard` method [1].
    One can either determine the cut points in form of passing an array to this class,
    or one can obtain cut points based on the training data.

    Arguments:
        cuts {int, array} -- Defining cut points, either the number of cuts, or the actual cut points.

    Keyword Arguments:
        scheme {str} -- Scheme used for discretization. Either 'equidistant' or 'quantiles'
            (default: {'equidistant})
        min_ {float} -- Starting duration (default: {0.})
        dtype {str, dtype} -- dtype of discretization.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """

    def __init__(self, cuts, scheme="equidistant", min_=0.0, dtype=None):
        self._cuts = cuts
        self._scheme = scheme
        self._min = min_
        self._dtype_init = dtype
        self._predefined_cuts = False
        self.cuts = None
        if hasattr(cuts, "__iter__"):
            if type(cuts) is list:
                cuts = np.array(cuts)
            self.cuts = cuts
            self.idu = IdxDiscUnknownC(self.cuts)
            assert dtype is None, "Need `dtype` to be `None` for specified cuts"
            self._dtype = type(self.cuts[0])
            self._dtype_init = self._dtype
            self._predefined_cuts = True
        else:
            self._cuts += 1

    def fit(self, durations, events):
        self._dtype = self._dtype_init
        if self._dtype is None:
            if isinstance(durations[0], np.floating):
                self._dtype = durations.dtype
            else:
                self._dtype = np.dtype("float64")
        durations = durations.astype(self._dtype)
        if not self._predefined_cuts:
            self.cuts = make_cuts(
                self._cuts, self._scheme, durations, events, self._min, self._dtype
            )
        self.duc = DiscretizeUnknownC(self.cuts, right_censor=True, censor_side="right")
        self.di = Duration2Idx(self.cuts)
        return self

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        return self.transform(durations, events)

    def transform(self, durations, events):
        durations = _values_if_series(durations)
        durations = durations.astype(self._dtype)
        events = _values_if_series(events)
        dur_disc, events = self.duc.transform(durations, events)
        idx_durations = self.di.transform(dur_disc)
        cut_diff = np.diff(self.cuts)
        assert (cut_diff > 0).all(), "Cuts are not unique."
        t_frac = 1.0 - (dur_disc - durations) / cut_diff[idx_durations - 1]
        if idx_durations.min() == 0:
            warnings.warn(
                """Got event/censoring at start time. Should be removed! It is set s.t. it has no contribution to loss."""
            )
            t_frac[idx_durations == 0] = 0
            events[idx_durations == 0] = 0
        idx_durations = idx_durations - 1
        return (
            idx_durations.astype("int64"),
            events.astype("float32"),
            t_frac.astype("float32"),
        )

    @property
    def out_features(self):
        """Returns the number of output features that should be used in the torch model.

        Returns:
            [int] -- Number of output features.
        """
        if self.cuts is None:
            raise ValueError("Need to call `fit` before this is accessible.")
        return len(self.cuts) - 1
