"""Test the tokenizating utilities."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import numpy as np

from sat.utils import tokenizing


def numerics():
    return {
        "numerics": [1.1, 2.1, 3.1, 4.1, 5.1],
    }


def element_no_numerics():
    return {
        "vector": [1.0, 2.0, 3.0, 4.0, 5.0],
    }


def config(
    max_seq_length: int = 5,
    truncation_direction: str = "right",
    padding_direction: str = "right",
    pooling: int = 0,
):
    """Return a configuration dictionary."""
    return {
        "max_seq_length": max_seq_length,
        "truncation_direction": truncation_direction,
        "padding_direction": padding_direction,
        "token_emb": pooling,
    }


def test_numerics_exists():
    cfg = config()
    new_element = tokenizing.numerics_padding_and_truncation(
        element_no_numerics(), **cfg
    )

    assert "numerics" not in new_element


def test_truncation_right():
    cfg = config(max_seq_length=3, truncation_direction="right")

    new_element = tokenizing.numerics_padding_and_truncation(numerics(), **cfg)

    assert len(new_element["numerics"]) == cfg["max_seq_length"]
    assert (new_element["numerics"] == np.array([1.1, 2.1, 3.1])).all()


def test_truncation_left():
    cfg = config(max_seq_length=3, truncation_direction="left")

    new_element = tokenizing.numerics_padding_and_truncation(numerics(), **cfg)

    assert len(new_element["numerics"]) == cfg["max_seq_length"]
    assert (new_element["numerics"] == np.array([3.1, 4.1, 5.1])).all()


def test_truncation_with_bert_pooling():
    cfg = config(max_seq_length=3, truncation_direction="left", pooling=5)

    new_element = tokenizing.numerics_padding_and_truncation(numerics(), **cfg)

    assert len(new_element["numerics"]) == cfg["max_seq_length"]
    assert (new_element["numerics"] == np.array([1.0, 4.1, 5.1])).all()


def test_padding_left():
    cfg = config(
        max_seq_length=8,
        padding_direction="left",
    )

    new_element = tokenizing.numerics_padding_and_truncation(numerics(), **cfg)

    assert len(new_element["numerics"]) == cfg["max_seq_length"]
    assert (
        new_element["numerics"] == np.array([1.0, 1.0, 1.0, 1.1, 2.1, 3.1, 4.1, 5.1])
    ).all()


def test_padding_left_with_bert_pooling():
    cfg = config(max_seq_length=8, padding_direction="left", pooling=5)

    new_element = tokenizing.numerics_padding_and_truncation(numerics(), **cfg)

    assert len(new_element["numerics"]) == cfg["max_seq_length"]
    assert (
        new_element["numerics"] == np.array([1.0, 1.0, 1.0, 1.1, 2.1, 3.1, 4.1, 5.1])
    ).all()


def test_padding_right():
    cfg = config(
        max_seq_length=8,
        padding_direction="right",
    )

    new_element = tokenizing.numerics_padding_and_truncation(numerics(), **cfg)

    assert len(new_element["numerics"]) == cfg["max_seq_length"]
    assert (
        new_element["numerics"] == np.array([1.1, 2.1, 3.1, 4.1, 5.1, 1.0, 1.0, 1.0])
    ).all()


def test_padding_right_with_bert_pooling():
    cfg = config(max_seq_length=8, padding_direction="right", pooling=5)

    new_element = tokenizing.numerics_padding_and_truncation(numerics(), **cfg)

    assert len(new_element["numerics"]) == cfg["max_seq_length"]
    assert (
        new_element["numerics"] == np.array([1.0, 1.1, 2.1, 3.1, 4.1, 5.1, 1.0, 1.0])
    ).all()


def test_bert_pooling():
    cfg = config(max_seq_length=5, pooling=5)

    new_element = tokenizing.numerics_padding_and_truncation(numerics(), **cfg)

    assert len(new_element["numerics"]) == cfg["max_seq_length"]
    assert (new_element["numerics"] == np.array([1.0, 2.1, 3.1, 4.1, 5.1])).all()
