"""Utilities for randomization."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import os
import time

from omegaconf import DictConfig
from transformers.trainer_utils import set_seed

from sat.utils import logging

logger = logging.get_default_logger()


def rand_seed():
    """
    Generate a random seed using a combination of system time and os.urandom.

    Returns:
        int: A random integer suitable for seeding random number generators.
    """
    # Get current time in nanoseconds if available, otherwise milliseconds
    try:
        current_time = time.time_ns()
    except AttributeError:
        current_time = int(time.time() * 1000)

    # Use 4 bytes from os.urandom
    random_bytes = os.urandom(4)

    # Convert bytes to integer
    random_int = int.from_bytes(random_bytes, byteorder="big")

    # Combine the two sources of randomness
    seed_value = (current_time ^ random_int) % 2**32

    logger.debug(f"Generated random seed: {seed_value}")

    return seed_value


def seed(func):
    """Decorate a function to set a random number seed before it is called."""

    def decorate(cfg: DictConfig):
        if cfg.seed is None:
            # retrieve and set the seed
            logger.debug("retrieve the random number seed")
            cfg.seed = rand_seed()
            logger.debug("set the random number seed")
            set_seed(cfg.seed)

        return func(cfg)

    return decorate


def reset_seed(func):
    """Decorate a function to reset a random number seed before it is called."""

    def decorate(cfg: DictConfig):
        # retrieve and set the seed
        logger.debug("retrieve the random number seed")
        cfg.seed = rand_seed()
        logger.debug("reset the random number seed")
        set_seed(cfg.seed)

        return func(cfg)

    return decorate
