# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# This program includes a portion of scikit-survival, authored by
# Sebastian Pölsterl and modified on October 19, 2023
import numpy as np

from sklearn.utils import check_array, check_consistent_length

from .exceptions import NoComparablePairException
from .nonparametric import CensoringDistributionEstimator
from .util import check_y_survival

__all__ = [
    "brier_score",
    "concordance_index_ipcw",
]


def _check_estimate_1d(estimate, test_time):
    estimate = check_array(estimate, ensure_2d=False, input_name="estimate")
    if estimate.ndim != 1:
        raise ValueError(
            "Expected 1D array, got {:d}D array instead:\narray={}.\n".format(
                estimate.ndim, estimate
            )
        )
    check_consistent_length(test_time, estimate)
    return estimate


def _check_inputs(event_indicator, event_time, estimate):
    check_consistent_length(event_indicator, event_time, estimate)
    event_indicator = check_array(
        event_indicator, ensure_2d=False, input_name="event_indicator"
    )
    event_time = check_array(event_time, ensure_2d=False, input_name="event_time")
    estimate = _check_estimate_1d(estimate, event_time)

    if not np.issubdtype(event_indicator.dtype, np.bool_):
        raise ValueError(
            "only boolean arrays are supported as class labels for survival analysis, got {0}".format(
                event_indicator.dtype
            )
        )

    if len(event_time) < 2:
        raise ValueError("Need a minimum of two samples")

    if not event_indicator.any():
        raise ValueError("All samples are censored")

    return event_indicator, event_time, estimate


def _check_times(test_time, times):
    times = check_array(np.atleast_1d(times), ensure_2d=False, input_name="times")
    times = np.unique(times)

    # if times.max() >= test_time.max() or times.min() < test_time.min():
    #     raise ValueError(
    #         "all times must be within follow-up time of test data: [{}; {}[".format(
    #             test_time.min(), test_time.max()
    #         )
    #     )

    return times


def _check_estimate_2d(estimate, test_time, time_points, estimator):
    estimate = check_array(
        estimate,
        ensure_2d=False,
        allow_nd=False,
        input_name="estimate",
        estimator=estimator,
    )
    time_points = _check_times(test_time, time_points)
    check_consistent_length(test_time, estimate)

    if estimate.ndim == 2 and estimate.shape[1] != time_points.shape[0]:
        raise ValueError(
            "expected estimate with {} columns, but got {}".format(
                time_points.shape[0], estimate.shape[1]
            )
        )

    return estimate, time_points


def _get_comparable(event_indicator, event_time, order):
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        start = i + 1
        end = start
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time
        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = np.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
        i = end

    return comparable, tied_time


def _estimate_concordance_index(
    event_indicator, event_time, estimate, weights, tied_tol=1e-8
):
    order = np.argsort(event_time)

    comparable, tied_time = _get_comparable(event_indicator, event_time, order)

    if len(comparable) == 0:
        # Return 0.5 (random prediction) instead of raising an exception
        # This allows training to continue even when c-index can't be computed
        return 0.5, 0, 0, 0, 0

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        w_i = weights[order[ind]]

        est = estimate[order[mask]]

        assert event_i, (
            "got censored sample at index %d, but expected uncensored" % order[ind]
        )

        ties = np.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        # an event should have a higher score
        con = est < est_i
        n_con = con[~ties].sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time


def concordance_index_ipcw(
    survival_train, survival_test, estimate, tau=None, tied_tol=1e-8
):
    """Concordance index for right-censored data based on inverse probability of censoring weights.

    This is an alternative to the estimator in :func:`concordance_index_censored`
    that does not depend on the distribution of censoring times in the test data.
    Therefore, the estimate is unbiased and consistent for a population concordance
    measure that is free of censoring.

    It is based on inverse probability of censoring weights, thus requires
    access to survival times from the training data to estimate the censoring
    distribution. Note that this requires that survival times `survival_test`
    lie within the range of survival times `survival_train`. This can be
    achieved by specifying the truncation time `tau`.
    The resulting `cindex` tells how well the given prediction model works in
    predicting events that occur in the time range from 0 to `tau`.

    The estimator uses the Kaplan-Meier estimator to estimate the
    censoring survivor function. Therefore, it is restricted to
    situations where the random censoring assumption holds and
    censoring is independent of the features.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb>`
    and [1]_ for further description.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event of test data.

    tau : float, optional
        Truncation time. The survival function for the underlying
        censoring time distribution :math:`D` needs to be positive
        at `tau`, i.e., `tau` should be chosen such that the
        probability of being censored after time `tau` is non-zero:
        :math:`P(D > \\tau) > 0`. If `None`, no truncation is performed.

    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

    Returns
    -------
    cindex : float
        Concordance index

    concordant : int
        Number of concordant pairs

    discordant : int
        Number of discordant pairs

    tied_risk : int
        Number of pairs having tied estimated risks

    tied_time : int
        Number of comparable pairs sharing the same time

    See also
    --------
    concordance_index_censored
        Simpler estimator of the concordance index.

    as_concordance_index_ipcw_scorer
        Wrapper class that uses :func:`concordance_index_ipcw`
        in its ``score`` method instead of the default
        :func:`concordance_index_censored`.

    References
    ----------
    .. [1] Uno, H., Cai, T., Pencina, M. J., D’Agostino, R. B., & Wei, L. J. (2011).
           "On the C-statistics for evaluating overall adequacy of risk prediction
           procedures with censored survival data".
           Statistics in Medicine, 30(10), 1105–1117.
    """
    test_event, test_time = check_y_survival(survival_test)

    if tau is not None:
        mask = test_time < tau
        survival_test = survival_test[mask]

    estimate = _check_estimate_1d(estimate, test_time)

    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    ipcw_test = cens.predict_ipcw(survival_test)
    if tau is None:
        ipcw = ipcw_test
    else:
        ipcw = np.empty(estimate.shape[0], dtype=ipcw_test.dtype)
        ipcw[mask] = ipcw_test
        ipcw[~mask] = 0

    w = np.square(ipcw)

    return _estimate_concordance_index(test_event, test_time, estimate, w, tied_tol)


def brier_score(survival_train, survival_test, estimate, times):
    """Estimate the time-dependent Brier score for right censored data.

    The time-dependent Brier score is the mean squared error at time point :math:`t`:

    .. math::

        \\mathrm{BS}^c(t) = \\frac{1}{n} \\sum_{i=1}^n I(y_i \\leq t \\land \\delta_i = 1)
        \\frac{(0 - \\hat{\\pi}(t | \\mathbf{x}_i))^2}{\\hat{G}(y_i)} + I(y_i > t)
        \\frac{(1 - \\hat{\\pi}(t | \\mathbf{x}_i))^2}{\\hat{G}(t)} ,

    where :math:`\\hat{\\pi}(t | \\mathbf{x})` is the predicted probability of
    remaining event-free up to time point :math:`t` for a feature vector :math:`\\mathbf{x}`,
    and :math:`1/\\hat{G}(t)` is a inverse probability of censoring weight, estimated by
    the Kaplan-Meier estimator.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Time-dependent-Brier-Score>`
    and [1]_ for details.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples, n_times)
        Estimated risk of experiencing an event for test data at `times`.
        The i-th column must contain the estimated probability of
        remaining event-free up to the i-th time point.

    times : array-like, shape = (n_times,)
        The time points for which to estimate the Brier score.
        Values must be within the range of follow-up times of
        the test data `survival_test`.

    Returns
    -------
    times : array, shape = (n_times,)
        Unique time points at which the brier scores was estimated.

    brier_scores : array , shape = (n_times,)
        Values of the brier score.

    Examples
    --------
    >>> from sksurv.datasets import load_gbsg2
    >>> from sksurv.linear_model import CoxPHSurvivalAnalysis
    >>> from sksurv.metrics import brier_score
    >>> from sksurv.preprocessing import OneHotEncoder

    Load and prepare data.

    >>> X, y = load_gbsg2()
    >>> X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)
    >>> Xt = OneHotEncoder().fit_transform(X)

    Fit a Cox model.

    >>> est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)

    Retrieve individual survival functions and get probability
    of remaining event free up to 5 years (=1825 days).

    >>> survs = est.predict_survival_function(Xt)
    >>> preds = [fn(1825) for fn in survs]

    Compute the Brier score at 5 years.

    >>> times, score = brier_score(y, y, preds, 1825)
    >>> print(score)
    [0.20881843]

    See also
    --------
    integrated_brier_score
        Computes the average Brier score over all time points.

    References
    ----------
    .. [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher,
           "Assessment and comparison of prognostic classification schemes for survival data,"
           Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545, 1999.
    """
    test_event, test_time = check_y_survival(survival_test)
    estimate, times = _check_estimate_2d(
        estimate, test_time, times, estimator="brier_score"
    )
    if estimate.ndim == 1 and times.shape[0] == 1:
        estimate = estimate.reshape(-1, 1)

    # fit IPCW estimator
    cens = CensoringDistributionEstimator().fit(survival_train)
    # calculate inverse probability of censoring weight at current time point t.
    prob_cens_t = cens.predict_proba(times)
    prob_cens_t[prob_cens_t == 0] = np.inf
    # calculate inverse probability of censoring weights at observed time point
    prob_cens_y = cens.predict_proba(np.minimum(test_time, max(times)))
    prob_cens_y[prob_cens_y == 0] = np.inf

    # Calculating the brier scores at each time point
    brier_scores = np.empty(times.shape[0], dtype=float)
    for i, t in enumerate(times):
        est = estimate[:, i]
        is_case = (test_time <= t) & test_event
        is_control = test_time > t

        brier_scores[i] = np.mean(
            np.square(est) * is_case.astype(int) / prob_cens_y
            + np.square(1.0 - est) * is_control.astype(int) / prob_cens_t[i]
        )

    return times, brier_scores


def integrated_brier_score(survival_train, survival_test, estimate, times):
    """The Integrated Brier Score (IBS) provides an overall calculation of
    the model performance at all available times :math:`t_1 \\leq t \\leq t_\\text{max}`.

    The integrated time-dependent Brier score over the interval
    :math:`[t_1; t_\\text{max}]` is defined as

    .. math::

        \\mathrm{IBS} = \\int_{t_1}^{t_\\text{max}} \\mathrm{BS}^c(t) d w(t)

    where the weighting function is :math:`w(t) = t / t_\\text{max}`.
    The integral is estimated via the trapezoidal rule.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Time-dependent-Brier-Score>`
    and [1]_ for further details.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples, n_times)
        Estimated risk of experiencing an event for test data at `times`.
        The i-th column must contain the estimated probability of
        remaining event-free up to the i-th time point.

    times : array-like, shape = (n_times,)
        The time points for which to estimate the Brier score.
        Values must be within the range of follow-up times of
        the test data `survival_test`.

    Returns
    -------
    ibs : float
        The integrated Brier score.

    Examples
    --------
    >>> import numpy as np
    >>> from sksurv.datasets import load_gbsg2
    >>> from sksurv.linear_model import CoxPHSurvivalAnalysis
    >>> from sksurv.metrics import integrated_brier_score
    >>> from sksurv.preprocessing import OneHotEncoder

    Load and prepare data.

    >>> X, y = load_gbsg2()
    >>> X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)
    >>> Xt = OneHotEncoder().fit_transform(X)

    Fit a Cox model.

    >>> est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)

    Retrieve individual survival functions and get probability
    of remaining event free from 1 year to 5 years (=1825 days).

    >>> survs = est.predict_survival_function(Xt)
    >>> times = np.arange(365, 1826)
    >>> preds = np.asarray([[fn(t) for t in times] for fn in survs])

    Compute the integrated Brier score from 1 to 5 years.

    >>> score = integrated_brier_score(y, y, preds, times)
    >>> print(score)
    0.1815853064627424

    See also
    --------
    brier_score
        Computes the Brier score at specified time points.

    as_integrated_brier_score_scorer
        Wrapper class that uses :func:`integrated_brier_score`
        in its ``score`` method instead of the default
        :func:`concordance_index_censored`.

    References
    ----------
    .. [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher,
           "Assessment and comparison of prognostic classification schemes for survival data,"
           Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545, 1999.
    """
    # Computing the brier scores
    times, brier_scores = brier_score(survival_train, survival_test, estimate, times)

    if times.shape[0] < 2:
        raise ValueError("At least two time points must be given")

    # Computing the IBS
    ibs_value = np.trapz(brier_scores, times) / (times[-1] - times[0])

    return ibs_value
