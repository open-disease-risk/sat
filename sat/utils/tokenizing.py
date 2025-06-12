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
    """Process numeric and modality fields to ensure consistent dimensions.

    This function handles padding and truncation for variable-length fields like
    numerics and modality to ensure they have consistent dimensions, which prevents
    tensor dimension mismatches during batching.

    Args:
        element: Dict containing the data elements
        max_seq_length: Maximum sequence length to pad/truncate to
        truncation_direction: Direction for truncation ('left' or 'right')
        padding_direction: Direction for padding ('left' or 'right')
        token_emb: Token embedding type (affects CLS token handling)

    Returns:
        The processed element with consistent field dimensions
    """
    # Field configurations - add new variable-length fields here
    field_configs = {
        "numerics": {
            "pad_value": 1.0,  # Value used for padding and CLS token
            "store_original": True,  # Whether to store original values in another field
            "original_field": "old_numerics",
            "needs_cls_token": True,  # Whether this field needs CLS token handling
        },
        "modality": {
            "pad_value": 0,  # Value used for padding and CLS token
            "store_original": True,
            "original_field": "old_modality",
            "needs_cls_token": True,  # Treat modality equally to numerics for CLS token
        },
    }

    # Process each configured field that exists in the element
    for field_name, config in field_configs.items():
        if field_name in element:
            logger.debug(
                f"Processing {field_name} with max_seq_length={max_seq_length}"
            )

            # Convert to numpy array for processing (explicitly using float32 for MPS compatibility)
            field_data = np.array(
                element[field_name],
                dtype=np.float32 if field_name == "numerics" else None,
            )
            original_size = field_data.size

            # Store original values if configured
            if config["store_original"]:
                # Ensure we store the original values with the same dtype
                element[config["original_field"]] = field_data.tolist()

            # Truncate if necessary
            if field_data.size > max_seq_length:
                if truncation_direction == "right":
                    logger.debug(
                        f"Truncating {field_name} from {field_data.size} to {max_seq_length} (right)"
                    )
                    field_data = field_data[0:max_seq_length]
                else:
                    logger.debug(
                        f"Truncating {field_name} from {field_data.size} to {max_seq_length} (left)"
                    )
                    field_data = field_data[(field_data.size - max_seq_length) :]

            # Handle CLS token if needed
            if config["needs_cls_token"] and token_emb == TokenEmbedding.BERT.value:
                if field_data.size < max_seq_length:
                    logger.debug(
                        f"Prepending CLS token with pad_value ({config['pad_value']}) for {field_name}"
                    )
                    # Ensure consistent dtype when appending
                    field_data = np.append(
                        np.array([config["pad_value"]], dtype=field_data.dtype),
                        field_data,
                    )
                else:
                    logger.debug(
                        f"Overwriting first element with pad_value ({config['pad_value']}) for {field_name}"
                    )
                    field_data[0] = config["pad_value"]

            # Pad if necessary
            if field_data.size < max_seq_length:
                if padding_direction == "right":
                    logger.debug(
                        f"Padding {field_name} from {field_data.size} to {max_seq_length} (right)"
                    )
                    pad_width = (0, max_seq_length - field_data.size)
                else:
                    logger.debug(
                        f"Padding {field_name} from {field_data.size} to {max_seq_length} (left)"
                    )
                    pad_width = (max_seq_length - field_data.size, 0)

                field_data = np.pad(
                    field_data,
                    pad_width,
                    "constant",
                    constant_values=config["pad_value"],
                )

            if original_size != field_data.size:
                logger.debug(
                    f"{field_name} resized from {original_size} to {field_data.size}"
                )

            # Store the processed data back in the element
            element[field_name] = field_data.tolist()

    return element
