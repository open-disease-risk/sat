"""
Utilities for estimating optimal MoCo buffer sizes for survival analysis.

These functions help determine the ideal buffer size for MoCo-enhanced
survival analysis based on dataset characteristics and training configuration.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import logging
import math
from typing import Dict, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_dataset_events(
    dataframe: pd.DataFrame,
    event_column: Union[str, list],
    time_column: Union[str, list],
    multi_event: bool = False,
) -> Dict[str, float]:
    """
    Analyze a dataset to extract event and censoring statistics.

    Args:
        dataframe: DataFrame containing the dataset
        event_column: Column name(s) containing event indicators
        time_column: Column name(s) containing time-to-event values
        multi_event: Whether this is a competing risks dataset with multiple event types

    Returns:
        Dictionary with dataset statistics
    """
    stats = {}
    stats["total_samples"] = len(dataframe)

    # Convert inputs to lists if they're not already
    if not isinstance(event_column, list):
        event_column = [event_column]
    if not isinstance(time_column, list):
        time_column = [time_column]

    # Determine the event structure by examining the first row
    # This handles both standard and complex event structures
    try:
        first_event_col = event_column[0]
        first_event_val = dataframe[first_event_col].iloc[0]
        is_array_event = hasattr(first_event_val, "__iter__") and not isinstance(
            first_event_val, (str, bytes)
        )
    except (IndexError, KeyError, AttributeError):
        # Default to non-array events if we can't determine
        is_array_event = False

    # Process differently based on the event structure
    if is_array_event:
        # Complex event structure (arrays/lists in cells)

        # Count samples with any event (across all event types)
        has_event = []
        for i in range(len(dataframe)):
            row_has_event = False
            for col in event_column:
                val = dataframe[col].iloc[i]
                if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                    # It's an array/list
                    if any(v > 0 for v in val if v is not None):
                        row_has_event = True
                        break
                elif val > 0:
                    # It's a scalar
                    row_has_event = True
                    break
            has_event.append(row_has_event)

        stats["samples_with_events"] = sum(has_event)

        # Count events by type
        event_counts = {}

        # Try to determine number of event types from the first non-empty array
        first_event_array = next(
            (
                val
                for val in dataframe[first_event_col]
                if hasattr(val, "__iter__")
                and not isinstance(val, (str, bytes))
                and len(val) > 0
            ),
            None,
        )

        if first_event_array is not None:
            num_event_types = len(first_event_array)

            # Count events for each type
            for i in range(num_event_types):
                count = 0
                for _, row in dataframe.iterrows():
                    for col in event_column:
                        val = row[col]
                        if (
                            hasattr(val, "__iter__")
                            and not isinstance(val, (str, bytes))
                            and i < len(val)
                            and val[i] > 0
                        ):
                            count += 1
                            break
                event_counts[f"event_type_{i}"] = count
        else:
            # Fallback - just one event type
            count = sum(1 for has_event_val in has_event if has_event_val)
            event_counts["event_type_0"] = count

        # Collect time values
        all_times = []
        for _, row in dataframe.iterrows():
            for col in time_column:
                val = row[col]
                if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                    # It's an array/list of times
                    all_times.extend([t for t in val if t is not None])
                elif val is not None:
                    # It's a scalar
                    all_times.append(val)

    elif multi_event:
        # Standard multi-event structure (each event type in its own column)

        # Count samples with any event across all columns
        has_event = []
        for i in range(len(dataframe)):
            row_has_event = False
            for col in event_column:
                if dataframe[col].iloc[i] > 0:
                    row_has_event = True
                    break
            has_event.append(row_has_event)

        stats["samples_with_events"] = sum(has_event)

        # Count events by type
        event_counts = {}
        for i, col in enumerate(event_column):
            event_counts[f"event_type_{i}"] = dataframe[col].sum()

        # Collect time values
        all_times = []
        for col in time_column:
            all_times.extend([t for t in dataframe[col].tolist() if t is not None])

    else:
        # Simple single-event dataset
        col = event_column[0]

        # Handle possible complex event structure in single-event mode
        if is_array_event:
            # If we have arrays but are in single-event mode, treat as binary
            has_event = []
            for i in range(len(dataframe)):
                val = dataframe[col].iloc[i]
                if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                    row_has_event = any(v > 0 for v in val if v is not None)
                else:
                    row_has_event = val > 0
                has_event.append(row_has_event)

            stats["samples_with_events"] = sum(has_event)
        else:
            # Standard scalar event indicator
            stats["samples_with_events"] = dataframe[col].sum()

        event_counts = {"event_type_0": stats["samples_with_events"]}

        # Collect time values
        all_times = []
        time_col = time_column[0]
        for i in range(len(dataframe)):
            val = dataframe[time_col].iloc[i]
            if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                # It's an array/list of times
                all_times.extend([t for t in val if t is not None])
            elif val is not None:
                # It's a scalar
                all_times.append(val)

    # Calculate censoring rate and set event counts
    stats["censoring_rate"] = 1 - (
        stats["samples_with_events"] / stats["total_samples"]
    )
    stats["event_counts"] = event_counts
    stats["total_events"] = sum(event_counts.values())
    stats["num_event_types"] = len(event_counts)

    # Calculate time statistics
    if all_times:
        # Filter out potential None values or empty arrays
        valid_times = [
            t
            for t in all_times
            if t is not None and not (hasattr(t, "__iter__") and len(t) == 0)
        ]
        if valid_times:
            stats["min_time"] = min(valid_times)
            stats["max_time"] = max(valid_times)
            stats["mean_time"] = float(np.mean(valid_times))
        else:
            stats["min_time"] = 0.0
            stats["max_time"] = 0.0
            stats["mean_time"] = 0.0
    else:
        stats["min_time"] = 0.0
        stats["max_time"] = 0.0
        stats["mean_time"] = 0.0

    return stats


def estimate_optimal_buffer_size(
    num_samples: int,
    censoring_rate: float,
    batch_size: int,
    min_events_per_batch: int = 10,
    max_buffer_ratio: float = 20.0,
    max_effective_batch: int = 1024,
) -> Dict[str, int]:
    """
    Estimate optimal buffer size for MoCo-enhanced survival analysis.

    Args:
        num_samples: Number of samples in the dataset
        censoring_rate: Proportion of censored samples (0-1)
        batch_size: Batch size used in training
        min_events_per_batch: Minimum desired events per effective batch
        max_buffer_ratio: Maximum buffer size as a multiple of batch size
        max_effective_batch: Maximum effective batch size (batch + buffer)

    Returns:
        Dictionary with recommended buffer sizes and rationale
    """
    # Calculate events per batch
    events_per_batch = batch_size * (1 - censoring_rate)

    # Calculate effective batch needed to reach min_events_per_batch
    if events_per_batch >= min_events_per_batch:
        # Already enough events in batch, use minimal buffer
        effective_batch_size = batch_size * 2
        buffer_size = batch_size
        reason = "sufficient_events"
    else:
        # Need buffer to reach minimum events
        required_ratio = min_events_per_batch / events_per_batch
        effective_batch_size = math.ceil(batch_size * required_ratio)
        buffer_size = effective_batch_size - batch_size
        reason = "insufficient_events"

    # Ensure we don't exceed reasonable limits
    max_by_ratio = int(batch_size * max_buffer_ratio)
    max_by_effective = max_effective_batch - batch_size
    max_by_dataset = min(num_samples // 2, 4096)

    max_reasonable = min(max_by_ratio, max_by_effective, max_by_dataset)

    if buffer_size > max_reasonable:
        buffer_size = max_reasonable
        if max_reasonable == max_by_ratio:
            limiting_factor = "buffer_ratio"
        elif max_reasonable == max_by_effective:
            limiting_factor = "effective_batch"
        else:
            limiting_factor = "dataset_size"
    else:
        limiting_factor = None

    # For very high censoring, suggest dynamic growth
    use_dynamic_growth = censoring_rate > 0.8 or buffer_size > batch_size * 5

    if use_dynamic_growth:
        # Start with smaller buffer and grow
        initial_buffer_size = min(batch_size * 2, buffer_size // 3)
    else:
        initial_buffer_size = buffer_size

    return {
        "recommended_buffer_size": buffer_size,
        "initial_buffer_size": initial_buffer_size,
        "effective_batch_size": batch_size + buffer_size,
        "events_per_batch": events_per_batch,
        "expected_events_in_effective_batch": events_per_batch
        * (1 + buffer_size / batch_size),
        "reason": reason,
        "limiting_factor": limiting_factor,
        "use_dynamic_growth": use_dynamic_growth,
    }


def generate_moco_config(
    dataset_stats: Dict[str, float], training_config: Dict[str, int]
) -> Dict[str, int]:
    """
    Generate complete MoCo configuration based on dataset statistics and training parameters.

    Args:
        dataset_stats: Dictionary with dataset statistics (from analyze_dataset_events)
        training_config: Dictionary with training configuration

    Returns:
        Dictionary with recommended MoCo configuration
    """
    # Extract key parameters
    num_samples = dataset_stats["total_samples"]
    censoring_rate = dataset_stats["censoring_rate"]
    num_event_types = dataset_stats.get("num_event_types", 1)

    batch_size = training_config.get("batch_size", 32)
    min_events = training_config.get("min_events_per_batch", 10)

    # Estimate buffer size
    buffer_config = estimate_optimal_buffer_size(
        num_samples=num_samples,
        censoring_rate=censoring_rate,
        batch_size=batch_size,
        min_events_per_batch=min_events,
    )

    # Check if adaptive buffer is recommended
    high_censoring = censoring_rate > 0.85
    large_buffer = buffer_config["recommended_buffer_size"] > batch_size * 10
    multi_event = num_event_types > 1

    use_adaptive = high_censoring or large_buffer or multi_event
    use_dynamic_weights = high_censoring or multi_event

    # Generate configuration
    config = {
        # Core MoCo parameters
        "moco_buffer_size": buffer_config["recommended_buffer_size"],
        "moco_initial_buffer_size": buffer_config["initial_buffer_size"],
        "moco_use_buffer": True,
        "moco_dynamic_buffer": buffer_config["use_dynamic_growth"],
        # Weight parameters
        "moco_batch_weight": 1.0,
        "moco_buffer_weight": 1.0,
        # Dynamic weight parameters (if using)
        "moco_initial_batch_weight": 1.0 if use_dynamic_weights else None,
        "moco_final_batch_weight": 0.5 if use_dynamic_weights else None,
        "moco_initial_buffer_weight": 0.0 if use_dynamic_weights else None,
        "moco_final_buffer_weight": 1.0 if use_dynamic_weights else None,
        "moco_warmup_steps": 1000 if use_dynamic_weights else None,
        # Adaptive parameters (if using)
        "moco_adaptive_buffer": use_adaptive,
        "moco_track_variance": use_adaptive,
        "moco_variance_window": 10 if use_adaptive else None,
        "moco_variance_threshold": 0.15 if use_adaptive else None,
    }

    # Remove None values
    return {k: v for k, v in config.items() if v is not None}


def print_buffer_recommendations(
    dataset_stats: Dict[str, float],
    training_config: Dict[str, int],
    output_format: str = "text",
) -> str:
    """
    Generate a human-readable report of buffer recommendations.

    Args:
        dataset_stats: Dictionary with dataset statistics
        training_config: Dictionary with training configuration
        output_format: Format for output ("text" or "markdown")

    Returns:
        Formatted string with recommendations
    """
    # Generate configuration
    buffer_config = estimate_optimal_buffer_size(
        num_samples=dataset_stats["total_samples"],
        censoring_rate=dataset_stats["censoring_rate"],
        batch_size=training_config["batch_size"],
        min_events_per_batch=training_config.get("min_events_per_batch", 10),
    )

    moco_config = generate_moco_config(dataset_stats, training_config)

    # Calculate events statistics
    events_per_batch = training_config["batch_size"] * (
        1 - dataset_stats["censoring_rate"]
    )
    effective_events = events_per_batch * (
        1 + buffer_config["recommended_buffer_size"] / training_config["batch_size"]
    )

    # Format output
    if output_format == "markdown":
        report = "## MoCo Buffer Size Recommendations\n\n"

        report += "### Dataset Statistics\n"
        report += f"- Total samples: {dataset_stats['total_samples']}\n"
        report += f"- Censoring rate: {dataset_stats['censoring_rate']:.1%}\n"
        report += f"- Event types: {dataset_stats.get('num_event_types', 1)}\n\n"

        report += "### Training Configuration\n"
        report += f"- Batch size: {training_config['batch_size']}\n"
        report += f"- Min events per batch: {training_config.get('min_events_per_batch', 10)}\n\n"

        report += "### Buffer Recommendations\n"
        report += f"- Recommended buffer size: **{buffer_config['recommended_buffer_size']}**\n"
        report += f"- Initial buffer size: {buffer_config['initial_buffer_size']}\n"
        report += f"- Expected events in batch: {events_per_batch:.1f}\n"
        report += f"- Expected events with buffer: {effective_events:.1f}\n"

        if buffer_config["reason"] == "insufficient_events":
            report += f"\n> **Note:** Buffer size is driven by the need to reach the minimum {training_config.get('min_events_per_batch', 10)} events per effective batch.\n"

        if buffer_config["limiting_factor"]:
            if buffer_config["limiting_factor"] == "buffer_ratio":
                report += "\n> **Note:** Buffer size is limited by the maximum buffer-to-batch ratio.\n"
            elif buffer_config["limiting_factor"] == "effective_batch":
                report += "\n> **Note:** Buffer size is limited by the maximum effective batch size.\n"
            elif buffer_config["limiting_factor"] == "dataset_size":
                report += "\n> **Note:** Buffer size is limited by the dataset size.\n"

        report += "\n### Implementation Recommendation\n"
        if moco_config["moco_adaptive_buffer"]:
            report += (
                "Use **AdaptiveMoCoLoss** with variance tracking for this dataset.\n"
            )
        elif (
            "moco_initial_batch_weight" in moco_config
            and moco_config["moco_initial_batch_weight"] is not None
        ):
            report += "Use **DynamicWeightMoCoLoss** with warmup for this dataset.\n"
        else:
            report += "Use standard **MoCoSurvivalLoss** for this dataset.\n"

        report += "\n```yaml\n"
        for k, v in moco_config.items():
            report += f"{k}: {v}\n"
        report += "```\n"

    else:  # text format
        report = "MoCo Buffer Size Recommendations\n"
        report += "===============================\n\n"

        report += "Dataset Statistics:\n"
        report += f"  Total samples: {dataset_stats['total_samples']}\n"
        report += f"  Censoring rate: {dataset_stats['censoring_rate']:.1%}\n"
        report += f"  Event types: {dataset_stats.get('num_event_types', 1)}\n\n"

        report += "Training Configuration:\n"
        report += f"  Batch size: {training_config['batch_size']}\n"
        report += f"  Min events per batch: {training_config.get('min_events_per_batch', 10)}\n\n"

        report += "Buffer Recommendations:\n"
        report += (
            f"  Recommended buffer size: {buffer_config['recommended_buffer_size']}\n"
        )
        report += f"  Initial buffer size: {buffer_config['initial_buffer_size']}\n"
        report += f"  Expected events in batch: {events_per_batch:.1f}\n"
        report += f"  Expected events with buffer: {effective_events:.1f}\n"

        if buffer_config["reason"] == "insufficient_events":
            report += f"\nNote: Buffer size is driven by the need to reach the minimum {training_config.get('min_events_per_batch', 10)} events per effective batch.\n"

        if buffer_config["limiting_factor"]:
            if buffer_config["limiting_factor"] == "buffer_ratio":
                report += "\nNote: Buffer size is limited by the maximum buffer-to-batch ratio.\n"
            elif buffer_config["limiting_factor"] == "effective_batch":
                report += "\nNote: Buffer size is limited by the maximum effective batch size.\n"
            elif buffer_config["limiting_factor"] == "dataset_size":
                report += "\nNote: Buffer size is limited by the dataset size.\n"

        report += "\nImplementation Recommendation:\n"
        if moco_config["moco_adaptive_buffer"]:
            report += "Use AdaptiveMoCoLoss with variance tracking for this dataset.\n"
        elif (
            "moco_initial_batch_weight" in moco_config
            and moco_config["moco_initial_batch_weight"] is not None
        ):
            report += "Use DynamicWeightMoCoLoss with warmup for this dataset.\n"
        else:
            report += "Use standard MoCoSurvivalLoss for this dataset.\n"

        report += "\nConfiguration:\n"
        for k, v in moco_config.items():
            report += f"  {k}: {v}\n"

    return report


def suggest_buffer_from_dataset(
    dataframe: pd.DataFrame,
    event_column: Union[str, list],
    time_column: Union[str, list],
    batch_size: int = 32,
    min_events_per_batch: int = 10,
    multi_event: bool = False,
    output_format: str = "text",
) -> str:
    """
    Complete utility function to suggest MoCo buffer configuration from a dataset.

    Args:
        dataframe: DataFrame containing the dataset
        event_column: Column name(s) containing event indicators
        time_column: Column name(s) containing time-to-event values
        batch_size: Batch size used in training
        min_events_per_batch: Minimum desired events per effective batch
        multi_event: Whether this is a competing risks dataset with multiple event types
        output_format: Format for output ("text" or "markdown")

    Returns:
        Formatted string with recommendations
    """
    # Analyze dataset
    dataset_stats = analyze_dataset_events(
        dataframe=dataframe,
        event_column=event_column,
        time_column=time_column,
        multi_event=multi_event,
    )

    # Define training config
    training_config = {
        "batch_size": batch_size,
        "min_events_per_batch": min_events_per_batch,
    }

    # Generate recommendations
    return print_buffer_recommendations(
        dataset_stats=dataset_stats,
        training_config=training_config,
        output_format=output_format,
    )


if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd

    # Create a sample dataset
    np.random.seed(42)
    n_samples = 1000
    censoring_rate = 0.8

    # Generate random events and times
    df = pd.DataFrame(
        {
            "time": np.random.exponential(scale=10, size=n_samples),
            "event": np.random.binomial(1, 1 - censoring_rate, size=n_samples),
        }
    )

    # Get buffer recommendations
    recommendations = suggest_buffer_from_dataset(
        dataframe=df,
        event_column="event",
        time_column="time",
        batch_size=32,
        min_events_per_batch=10,
        output_format="markdown",
    )

    print(recommendations)
