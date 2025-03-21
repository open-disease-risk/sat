"""Test for the LossWeightLoggerCallback."""

import pytest
import torch
import torch.nn as nn

from sat.transformers.callbacks.loss_weight_logger import LossWeightLoggerCallback
from sat.loss.balancing import FixedWeightBalancer
from sat.loss.meta import MetaLoss
from sat.loss import Loss


# Mock model and loss classes for testing
class MockLossFn(Loss):
    def __init__(self):
        super().__init__(num_events=1)
        self.balancer = FixedWeightBalancer([0.7, 0.3])
        
    def forward(self, predictions, references):
        return torch.tensor(1.0)
        
    def get_loss_weights(self):
        return self.balancer.get_weights()


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = MockLossFn()
        self.logged_metrics = {}
        
    def log(self, metrics):
        self.logged_metrics.update(metrics)


# Mock training arguments and state
class MockTrainingArgs:
    def __init__(self):
        self.local_rank = 0


class MockTrainerState:
    def __init__(self):
        self.global_step = 10
        self.is_world_process_zero = True


class MockTrainerControl:
    def __init__(self):
        pass


def test_loss_weight_logger_init():
    """Test that the callback initializes correctly."""
    callback = LossWeightLoggerCallback(log_freq=2, prefix="test_prefix")
    assert callback.log_freq == 2
    assert callback.prefix == "test_prefix"
    assert callback.log_eval is True
    assert callback.log_train is True


def test_loss_weight_logger_evaluate():
    """Test that the callback correctly logs weights during evaluation."""
    # Setup
    callback = LossWeightLoggerCallback(prefix="test")
    model = MockModel()
    args = MockTrainingArgs()
    state = MockTrainerState()
    control = MockTrainerControl()
    
    # Call on_evaluate
    callback.on_evaluate(args, state, control, model)
    
    # Check that weights were logged
    assert "test/eval/weight_0" in model.logged_metrics
    assert "test/eval/weight_1" in model.logged_metrics
    # Using approximately equal to handle floating point precision
    assert abs(model.logged_metrics["test/eval/weight_0"] - 0.7) < 1e-5
    assert abs(model.logged_metrics["test/eval/weight_1"] - 0.3) < 1e-5


def test_loss_weight_logger_step():
    """Test that the callback correctly logs weights during training steps."""
    # Setup
    callback = LossWeightLoggerCallback(prefix="test", log_freq=1)
    model = MockModel()
    args = MockTrainingArgs()
    state = MockTrainerState()
    control = MockTrainerControl()
    
    # Call on_step_end
    callback.on_step_end(args, state, control, model)
    
    # Check that weights were logged
    assert "test/train/weight_0" in model.logged_metrics
    assert "test/train/weight_1" in model.logged_metrics
    # Using approximately equal to handle floating point precision
    assert abs(model.logged_metrics["test/train/weight_0"] - 0.7) < 1e-5
    assert abs(model.logged_metrics["test/train/weight_1"] - 0.3) < 1e-5


def test_loss_weight_logger_disabled():
    """Test that the callback respects disabled flags."""
    # Setup
    callback = LossWeightLoggerCallback(log_eval=False, log_train=False)
    model = MockModel()
    args = MockTrainingArgs()
    state = MockTrainerState()
    control = MockTrainerControl()
    
    # Call both hooks
    callback.on_evaluate(args, state, control, model)
    callback.on_step_end(args, state, control, model)
    
    # Check that no weights were logged
    assert not model.logged_metrics