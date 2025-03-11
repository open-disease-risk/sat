"""Statistical Functions."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import multiprocessing as mp
import numpy as np
import pandas as pd
import scipy.stats as ss

from bootstrap_stat import bootstrap_stat as bp
from dataclasses import dataclass

from pathos.multiprocessing import ProcessPool as Pool

from sat.utils import logging

logger = logging.get_default_logger()


class EmpiricalDistribution:
    r"""Empirical Distribution.

    The Empirical Distribution puts probability 1/n on each of n
    observations.


    Parameters
    ----------
     data : tuple
        predictions and labels as numpy arrays

    Note
    ----
    Adapted from https://github.com/rwilson4/bootstrap-stat, which is
    licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    """

    def __init__(self, data):
        """Empirical Distribution.

        Parameters
        ----------
         data : array_like or pandas DataFrame
            The data.

        """
        self.data = data
        self.n = len(data[0])
        self.is_multi_sample = False

    def sample(self, size=None):
        """Sample from the empirical distribution.

        Parameters
        ----------
         size : int or tuple of ints, optional
            Output shape. If None (default), samples the same number
            of points as the original dataset.

        Returns
        -------
         samples : tuple of ndarray
            IID samples from the empirical distribution.
         ind : ndarray
            Indices of samples chosen.
        """

        if size is None:
            s = self.n
        else:
            s = size

        ind = np.random.choice(range(self.n), size=s, replace=True)
        preds = self.data[0][ind]
        labels = self.data[1][ind]
        samples = (preds, labels)
        return samples


def jackknife_values(x, stat, sample=None, num_threads=1):
    """Compute jackknife values.

    Parameters
    ----------
     x : array_like or pandas DataFrame or tuple of arrays/DataFrames.
        The data.
     stat : function
        The statistic.
     sample : int, optional
        When Jackknifing a multi-sample distribution, like for an A/B
        test, we generate one set of jackknife values for each
        sample. The caller should specify which sample for which
        jackknife values should be generated, calling this function
        once for each sample.
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     jv : ndarray
        The jackknife values.

    Notes
    -----
    The jackknife values consist of the statistic applied to a
    collection of datasets derived from the original by holding out
    each observation in turn. For example, let x1 be the dataset
    corresponding to x, but with the first datapoint removed. The
    first jackknife value is simply stat(x1).

    Adapted from https://github.com/rwilson4/bootstrap-stat, which is
    licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """
    if num_threads == -1:
        num_threads = mp.cpu_count()

    if sample is not None and type(x) is tuple:
        # Multi-sample jackknife. Create a new statistic that is
        # simply a wrapper around the desired `stat`. Only perform the
        # hold-out logic on the specified sample.
        x = list(x)
        x_b = x[0:sample]
        x_s = x[sample]
        x_a = x[(sample + 1) :]

        def statistic(zz):
            xx = x_b + [zz] + x_a
            return stat((*xx,))

        return jackknife_values(x_s, statistic, num_threads=num_threads)

    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        is_dataframe = True
        n = len(x.index)
    else:
        is_dataframe = False
        n = len(x[0])

    def _jackknife_sim(x, stat, is_dataframe, start, end):
        n = end - start
        theta_i = np.empty((n,))
        for i in range(start, end):
            if is_dataframe:
                zi = x.drop(x.index[i])
            else:
                xi = np.delete(x[0], i, axis=0)
                yi = np.delete(x[1], i, axis=0)
                zi = (xi, yi)
            theta_i[i - start] = stat(zi)

        return theta_i

    if num_threads == 1:
        theta_i = _jackknife_sim(x, stat, is_dataframe, 0, n)
    else:
        pool = Pool(num_threads)
        try:
            pool.restart()
        except AssertionError:
            pass

        results = []
        batch_size = n // num_threads
        extra = n % num_threads
        batch_sizes = [batch_size] * num_threads
        for i in range(extra):
            batch_sizes[i] += 1

        start = 0
        for i in range(num_threads):
            end = start + batch_sizes[i]
            r = pool.apipe(_jackknife_sim, x, stat, is_dataframe, start, end)
            results.append(r)
            start = end

        theta_i = np.hstack([res.get() for res in results])

        pool.close()
        pool.join()

    return theta_i


def bcanon_interval(
    dist,
    stat,
    x,
    alpha=0.05,
    B=1000,
    size=None,
    return_samples=False,
    theta_star=None,
    theta_hat=None,
    jv=None,
    num_threads=1,
):
    """BCa Confidence Intervals.

    Parameters
    ----------
     dist : EmpiricalDistribution
        The empirical distribution.
     stat : function
        The statistic.
     x : array_like or pandas DataFrame or tuple
        The data, used to evaluate the observed statistic and compute
        jackknife values.
     alpha : float, optional
        Number controlling the size of the interval. That is, this
        function will return a 100(1-2*`alpha`)% confidence
        interval. Defaults to 0.05.
     B : int, optional
        Number of bootstrap samples. Defaults to 1000.
     size : int or tuple of ints, optional
        Size to pass for generating samples from the distribution.
        Defaults to None, indicating the samples will be the same size
        as the original dataset.
     return_samples : boolean, optional
        If True, return the bootstrapped statistic values. Defaults to False.
     theta_star : array_like, optional
        Bootstrapped statistic values. Can be passed if they have
        already been calculated, which will speed this up
        considerably.
     theta_hat : float, optional
        Observed statistic. Can be passed if it has already been
        calculated, which will speed this up slightly.
     jv : array_like, optional
        Jackknife values. Can be passed if they have already been
        calculated, which will speed this up considerably.
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     ci_low, ci_high : float
        Lower and upper bounds on a 100(1-2*`alpha`)% confidence
        interval on theta.
     theta_star : ndarray
        Array of bootstrapped statistic values. Only returned if
        `return_samples` is True.
     jv : ndarray
        Jackknife values. Only returned if `return_samples` is True.

    Note
    ----
    Adapted from https://github.com/rwilson4/bootstrap-stat, which is
    licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """
    # The observed value of the statistic.
    if theta_hat is None:
        logger.debug("Compute statistic over the data")
        theta_hat = stat(x)

    if theta_star is None:
        logger.debug("Create the bootstrap samples")
        theta_star = bp.bootstrap_samples(
            dist, stat, B, size=size, num_threads=num_threads
        )

    zb = (theta_star < theta_hat).sum()
    z0_hat = ss.norm.ppf(zb / len(theta_star))

    if jv is None:
        if dist.is_multi_sample:
            jv = [
                jackknife_values(x, stat, sample=i, num_threads=num_threads)
                for i in range(len(x))
            ]
            jv = (*jv,)
        else:
            logger.debug("Compute the jackknife values")
            jv = jackknife_values(x, stat, num_threads=num_threads)

    a_hat = bp._bca_acceleration(jv)
    alpha1, alpha2 = bp._adjust_percentiles(alpha, a_hat, z0_hat)

    p = bp._percentile(theta_star, [alpha1, alpha2])
    if return_samples:
        return p[0], p[1], theta_star, jv
    else:
        return p[0], p[1]


def boot_interval(
    dist,
    stat,
    x,
    alpha=0.05,
    B=1000,
    size=None,
    theta_star=None,
    theta_hat=None,
    num_threads=1,
):
    """Percentile Bootstrap Confidence Intervals.

    Parameters
    ----------
     dist : EmpiricalDistribution
        The empirical distribution.
     stat : dict or function
        A dictionary of statistics functions or a single function.
     x : array_like or pandas DataFrame or tuple
        The data, used to evaluate the observed statistic and compute
        jackknife values.
     alpha : float, optional
        Number controlling the size of the interval. That is, this
        function will return a 100(1-2*`alpha`)% confidence
        interval. Defaults to 0.05.
     B : int, optional
        Number of bootstrap samples. Defaults to 1000.
     size : int or tuple of ints, optional
        Size to pass for generating samples from the distribution.
        Defaults to None, indicating the samples will be the same size
        as the original dataset.
     return_samples : boolean, optional
        If True, return the bootstrapped statistic values. Defaults to False.
     theta_star : array_like, optional
        Bootstrapped statistic values. Can be passed if they have
        already been calculated, which will speed this up
        considerably.
     theta_hat : dict, optional
        Dictionary of observed statistics. Can be passed if they have
        already been calculated, which will speed this up slightly.
     num_threads : int, optional
        Number of threads to use for multicore processing. Defaults to
        1, meaning all calculations will be done in a single
        thread. Set to -1 to use all available cores.

    Returns
    -------
     bootstrap_dict : dict
        Dictionary containing confidence intervals for all metrics.
    """
    # The observed value of the statistic.
    if theta_hat is None:
        logger.debug("Compute statistics over the data")
        theta_hat = {}
        for metric, func in stat.items():
            theta_hat[metric] = func(x)

    # Generate bootstrap samples only once and reuse for all metrics
    if theta_star is None:
        logger.debug("Create bootstrap samples (once for all metrics)")
        
        # Pre-generate all bootstrap indices for efficiency
        bootstrap_indices = []
        n_samples = len(x[0])
        for _ in range(B):
            indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
            bootstrap_indices.append(indices)
        
        # Apply each statistic function to the same bootstrap samples
        theta_star = {}
        for metric, func in stat.items():
            theta_star[metric] = np.empty(B)
            
            # Use the same bootstrap indices for all metrics
            for i, indices in enumerate(bootstrap_indices):
                bootstrap_data = (x[0][indices], x[1][indices])
                theta_star[metric][i] = func(bootstrap_data)

    bootstrap_dict = {}
    for metric, bootstraps in theta_star.items():
        deltas = bootstraps - theta_hat[metric]
        conf_interval = np.percentile(deltas, [alpha, 1.0 - alpha])
        bootstrap_dict[metric] = {
            "confidence_interval": (
                theta_hat[metric] - conf_interval[0],
                theta_hat[metric] + conf_interval[1],
            ),
            "score": theta_hat[metric],
        }

    return bootstrap_dict


@dataclass
class OnlineStats:
    n: int = 0
    old_mean: float = 0.0
    new_mean: float = 0.0
    old_s: float = 0.0
    new_s: float = 0.0

    def clear(self) -> None:
        self.n = 0

    def push(self, x: float) -> None:
        self.n += 1

        # See Knuth TAOCP vol 2, 3rd edition, page 232
        if self.n == 1:
            self.new_mean = x
            self.old_s = 0.0
        else:
            self.new_mean = self.old_mean + (x - self.old_mean) / self.n
            self.new_s = self.old_s + (x - self.old_mean) * (x - self.new_mean)

        # set up for next iteration
        self.old_mean = self.new_mean
        self.old_s = self.new_s

    def getNumValues(self) -> int:
        return self.n

    def mean(self) -> float:
        mean = 0.0

        if self.n > 1:
            mean = self.new_mean
        elif self.n == 1:
            mean = self.old_mean

        return mean

    def variance(self) -> float:
        return (self.new_s / (self.n - 1)) if self.n > 1 else 0.0

    def standardDeviation(self) -> float:
        return np.sqrt(self.variance())


def isConfidentWithPrecision(stats: OnlineStats, alpha: float, error: float):
    r"""This method calculates the confidence interval with an adjusted relative
    error given the data statistics.

    It returns a boolean value indicating
    whether the data is within a fixed confidence level of
    $100(1 - \alpha)$ percent and a relative error of
    $\gamma (0 < \gamma < 1)$. The formula is given as
    $\delta(n, \alpha) = t_{n-1,1-\alpha/2}\sqrt{S^2(n)/n}$
    evaluating to true, iff
    $\delta(n, \alpha) / \abs{\bar{X}(n)} \leq \gamma\prime$
    """

    if stats.getNumValues() < 2:
        return False
    else:
        mean = stats.mean()
        sv = stats.variance()

        df = stats.getNumValues() - 1
        nu = alpha / 2.0
        relAdjError = error / (1.0 + error)
        t = ss.t.isf(nu, df)

        ciHalfLength = t * np.sqrt(sv / stats.getNumValues())

        if (ciHalfLength / np.fabs(mean)) <= relAdjError:
            return True
        else:
            return False
