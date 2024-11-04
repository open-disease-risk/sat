"""Logging utilities
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import json
import logging
import torch

import numpy as np


def get_default_logger(prefix="sat") -> logging.Logger:
    return logging.getLogger(prefix)


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
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
