"""Logging utilities"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import json
import logging
import torch

import numpy as np


# Add standard logging levels for convenient access
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


def get_default_logger(prefix="sat") -> logging.Logger:
    """
    Get the default logger instance with the given prefix.

    Args:
        prefix: Logger name prefix

    Returns:
        Logger instance
    """
    logger = logging.getLogger(prefix)

    # Add isEnabledFor method if not already present
    if not hasattr(logger, "isEnabledFor"):
        setattr(logger, "isEnabledFor", lambda level: logger.level <= level)

    return logger


def set_verbosity(level=INFO):
    """
    Set the verbosity level for the default logger.

    Args:
        level: Logging level constant (e.g., logging.DEBUG, logging.INFO)
    """
    logger = get_default_logger()
    logger.setLevel(level)

    # Also set the root logger
    logging.getLogger().setLevel(level)


def log_gpu_utilization():
    log = get_default_logger()

    if torch.cuda.is_available():
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        log.info(f"GPU memory occupied: {info.used//1024**2} MB.")
    else:
        log.info("No GPU available")


def log_summary(result):
    log = get_default_logger()

    log.info(f"Time: {result.metrics['train_runtime']:.2f}")
    log.info(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    log_gpu_utilization()


class NpEncoder(json.JSONEncoder):
    """Encode numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except ValueError:
                return str(obj)
        return super(NpEncoder, self).default(obj)
