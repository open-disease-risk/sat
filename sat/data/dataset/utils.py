"""Utility functions for dataset processing.

This module contains common utility functions used across different dataset parsers.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

__all__ = ["tokens", "numerics"]

import numpy as np


def tokens(row, modalities, offset=0):
    """Process row data into tokens based on modality information.

    This function processes a row of data into tokens where numeric features are replaced
    with their values. It's commonly used for preparing data for transformer models.

    Args:
        row: A pandas Series or similar containing the data row
        modalities: A list where 0 indicates a token feature and 1 indicates a numeric feature
        offset: Offset to apply to the row index when accessing values (default: 0)

    Returns:
        list: Processed tokens where numeric features are replaced with their values

    Example:
        >>> row = pd.Series(['cat', 3.2, 'dog'], index=['a', 'b', 'c'])
        >>> modalities = [0, 1, 0]  # 'a' and 'c' are tokens, 'b' is numeric
        >>> tokens(row, modalities)
        ['a_cat', '3.2', 'c_dog']
    """
    (idx,) = np.where(np.array(modalities) == 0)
    toks = list(row.index[offset : len(modalities) + offset])
    for i in idx:
        toks[i] = (
            f"{toks[i]}_{row.iloc[i + offset]}" if row.iloc[i + offset] else toks[i]
        )
    return toks


def numerics(row, modalities, offset=0, default_value=1.0):
    """Process row data into numeric values based on modality information.

    This function processes a row of data into numeric values where token features are
    replaced with a default value. It's commonly used for preparing numerical features.

    Args:
        row: A pandas Series or similar containing the data row
        modalities: A list where 1 indicates a numeric feature and 0 indicates a token feature
        offset: Offset to apply to the row index when accessing values (default: 0)
        default_value: Default value to use for non-numeric features (default: 1.0)

    Returns:
        list: Processed numeric values where token features are replaced with default_value

    Example:
        >>> row = pd.Series(['cat', 3.2, 'dog'], index=['a', 'b', 'c'])
        >>> modalities = [0, 1, 0]  # 'a' and 'c' are tokens, 'b' is numeric
        >>> numerics(row, modalities)
        [1.0, 3.2, 1.0]
    """
    (idx,) = np.where(np.array(modalities) == 1)
    result = [float(default_value)] * len(modalities)
    for i in idx:
        try:
            result[i] = float(row.iloc[i + offset])
        except (ValueError, TypeError):
            result[i] = float(default_value)
    return result
