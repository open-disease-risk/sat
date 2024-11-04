"""Test the output utilities."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import unittest
from sat.utils.output import (
    log_metrics,
    log_metrics_from_replications,
)


class TestLogMetricsFromReplications(unittest.TestCase):
    def test_single_value(self):
        # Given a single metric value
        metrics = {"accuracy": 0.85}
        prefix = "exp1"

        # When calling the function
        result = log_metrics_from_replications(metrics, prefix)

        # Then the result should match the expected dictionary
        expected_result = {"exp1_accuracy": 0.85}
        self.assertEqual(result, expected_result)

    def test_nested_values(self):
        # Given nested metric values
        metrics = {"loss": {"train": 0.2, "validation": 0.3}, "f1_score": 0.75}
        prefix = "exp2"

        # When calling the function
        result = log_metrics_from_replications(metrics, prefix)

        # Then the result should match the expected dictionary
        expected_result = {
            "exp2_loss_train": 0.2,
            "exp2_loss_validation": 0.3,
            "exp2_f1_score": 0.75,
        }
        self.assertEqual(result, expected_result)

    def test_non_numeric_values(self):
        # Given non-numeric metric values
        metrics = {"accuracy": "high", "loss": {"train": "low", "validation": "medium"}}
        prefix = "exp3"

        # When calling the function
        result = log_metrics_from_replications(metrics, prefix)

        # Then the result should only include numeric values
        expected_result = {}
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
