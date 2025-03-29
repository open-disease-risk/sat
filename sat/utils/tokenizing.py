"""Utilities for tokenizating sequences."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from typing import Any, Dict

import numpy as np

from sat.models.heads.embeddings import TokenEmbedding
from sat.utils import logging

logger = logging.get_default_logger()


def numerics_padding_and_truncation(
    element: Dict[Any, Any],
    max_seq_length: int,
    truncation_direction: str,
    padding_direction: str,
    token_emb: int,
) -> Dict[Any, Any]:
    if "numerics" in element:
        logger.debug(f"numerics is in {element}")
        logger.debug(f"Maximum sequence length is {max_seq_length}")
        logger.debug(f"Truncation direction is {truncation_direction}")
        logger.debug(f"Padding direction is {padding_direction}")
        logger.debug(f"Pooling is {token_emb}")

        numerics = np.array(element["numerics"])
        element["old_numerics"] = numerics
        # truncate if necessary
        if numerics.size > max_seq_length:
            if truncation_direction == "right":
                logger.debug("Truncate right")
                numerics = numerics[0:max_seq_length]
            else:
                logger.debug("Truncate left")
                numerics = numerics[(numerics.size - max_seq_length) :]

        # add CLS token multiplier if we do BERT pooling
        if token_emb == TokenEmbedding.BERT.value:
            if numerics.size < max_seq_length:
                logger.debug("Prepend constant for CLS token")
                numerics = np.append([1.0], numerics)
            else:
                logger.debug("Overwrite first element with constant for CLS token")
                numerics[0] = 1.0  # overwrite the first element for the CLS token

        # pad if necessary
        if numerics.size < max_seq_length:
            if padding_direction == "right":
                logger.debug("Pad right")
                pad_width = (0, max_seq_length - numerics.size)
            else:
                logger.debug("Pad left")
                pad_width = (max_seq_length - numerics.size, 0)

            numerics = np.pad(
                numerics,
                pad_width,
                "constant",
                constant_values=1.0,
            )

        element["numerics"] = numerics

    return element
