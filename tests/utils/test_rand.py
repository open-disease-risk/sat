"""Test the rand utilities."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from dataclasses import dataclass
from typing import Optional

from omegaconf import OmegaConf

from sat.utils import rand


@dataclass
class Config:
    seed: Optional[int] = None


def config():
    return OmegaConf.structured(Config)


def test_empty_seed():
    cfg = config()

    assert cfg.seed is None

    def wrapped(cfg):
        assert cfg.seed is None

    wrapped(cfg)


def test_seed():
    cfg = config()

    assert cfg.seed is None

    @rand.seed
    def wrapped(cfg):
        assert cfg.seed is not None

    wrapped(cfg)


def test_seed_already_set():
    cfg = config()
    cfg.seed = 42

    assert cfg.seed == 42

    @rand.seed
    def wrapped(cfg):
        assert cfg.seed == 42

    wrapped(cfg)


def test_seed_in_pipeline():
    cfg = config()

    assert cfg.seed is None

    @rand.seed
    def method1(cfg):
        assert cfg.seed is not None

    @rand.seed
    def pipeline(cfg):
        assert cfg.seed is not None
        seed = cfg.seed
        method1(cfg)
        assert seed == cfg.seed

    pipeline(cfg)


def test_reset_seed_in_pipeline():
    cfg = config()

    assert cfg.seed is None

    @rand.reset_seed
    def method1(cfg):
        assert cfg.seed is not None

    @rand.seed
    def pipeline(cfg):
        assert cfg.seed is not None
        seed = cfg.seed
        method1(cfg)
        assert seed != cfg.seed

    pipeline(cfg)
