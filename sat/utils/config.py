"""Configuration Utilities
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from omegaconf import OmegaConf
from sat.utils import logging

logger = logging.get_default_logger("sat.utils.config")


class Config(object):
    """Configure the omega configuration environment."""

    _instance = None

    def __new__(cls):
        """Singleton pattern for the configuration class."""
        if cls._instance is None:
            logger.debug("Create new Config object")
            cls._instance = super(Config, cls).__new__(cls)

        return cls._instance

    def __init__(self):
        """Add new resolvers to OmegaConf."""
        logger.debug("Initialize Config object")
        OmegaConf.register_new_resolver("sum", lambda x, y: x + y)
        OmegaConf.register_new_resolver("mult", lambda x, y: x * y)
        OmegaConf.register_new_resolver("len", lambda x: len(x))
