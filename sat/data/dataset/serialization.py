"""
Utility functions for dataset serialization.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from datasets import Dataset


def _datetime_json_serializer(obj: Any) -> Any:
    """
    JSON serializer for objects not serializable by default json code.
    Handles datetime objects by converting them to ISO format strings.
    
    Args:
        obj: Object to serialize
        
    Returns:
        Serializable representation of the object
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # Handle other non-serializable types
    try:
        return str(obj)
    except Exception as e:
        logging.warning(f"Could not serialize {type(obj)}: {e}")
        return None


def serialize_dataset_to_json(
    dataset: Dataset,
    output_path: Union[str, Path],
    subset_size: Optional[int] = None,
    pretty: bool = True,
) -> None:
    """
    Serialize a HuggingFace Dataset to JSON format.
    Handles special cases like datetime objects.
    
    Args:
        dataset: The HuggingFace Dataset to serialize
        output_path: Path where to save the JSON file
        subset_size: Optional limit on the number of samples to serialize (for large datasets)
        pretty: Whether to generate human-readable JSON with indentation
    """
    # Convert to dictionary format (native dataset operation)
    if subset_size is not None and subset_size < len(dataset):
        data_dict = dataset.select(range(subset_size)).to_dict()
    else:
        data_dict = dataset.to_dict()
    
    # Generate JSON with proper serialization for special types
    indent = 2 if pretty else None
    with open(output_path, 'w') as f:
        json.dump(data_dict, f, default=_datetime_json_serializer, indent=indent)
    
    logging.debug(f"Dataset serialized to JSON at {output_path}")


def serialize_dataset(
    dataset: Dataset,
    output_dir: Union[str, Path],
    name: str,
    include_json: bool = False,
    json_subset_size: Optional[int] = 100,
) -> None:
    """
    Save dataset in the standard HuggingFace format and optionally as JSON.
    
    Args:
        dataset: HuggingFace Dataset to save
        output_dir: Directory to save the dataset in
        name: Base name for the dataset files
        include_json: Whether to also save a JSON representation
        json_subset_size: Number of samples to include in JSON (None for all)
    """
    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Native Dataset serialization
    dataset_path = output_dir / name
    dataset.save_to_disk(str(dataset_path))
    logging.info(f"Dataset saved to {dataset_path}")
    
    # Optional JSON serialization
    if include_json:
        json_path = output_dir / f"{name}.json"
        serialize_dataset_to_json(
            dataset, 
            json_path, 
            subset_size=json_subset_size
        )
