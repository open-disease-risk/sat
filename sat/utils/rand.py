"""Utilities for randomization."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import os

from omegaconf import DictConfig
from transformers.trainer_utils import set_seed

from sat.utils import logging

logger = logging.get_default_logger()


# def rand_seed():
#
# def rand_seed() is licensed under CC-BY-SA 4.0 (since it was taken from Stack Overflow after 2018) and CC-BY-SA 4.0 
# is incompatible with GPL; accordingly, this portion needs to be rewritten or code licensed under a GPL-compatible
# license needs to be used
#
#     """
#     Random seed generation.

#     Usually the best random sample you could get in any programming language is generated through the operating system.
#     In Python, you can use the os module.

#     source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
#     """
#     RAND_SIZE = 4
#     random_data = os.urandom(
#         RAND_SIZE
#     )  # Return a string of size random bytes suitable for cryptographic use.
#     random_seed = int.from_bytes(random_data, byteorder="big")
#     logger.debug(f"Got random number seed {random_seed}")

#     return random_seed


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
