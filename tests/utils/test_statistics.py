"""Test the rand utilities."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from sat.utils import statistics


def test_online_stats_init():
    a = statistics.OnlineStats()
    assert a.getNumValues() == 0


def test_online_stats_mean_1():
    a = statistics.OnlineStats()
    a.push(1.0)
    assert a.getNumValues() == 1
    assert a.mean() == 1
    assert a.standardDeviation() == 0


def test_online_stats_mean_2():
    a = statistics.OnlineStats()
    a.push(1.0)
    a.push(3.0)
    assert a.getNumValues() == 2
    assert a.mean() == 2.0
    assert a.variance() == 2
