"""Output Utilities"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import json
import numbers
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers.utils import ModelOutput

from sat.utils import interpolate, logging

logger = logging.get_default_logger()


def write_output(
    predictions, metrics, cfg, output_dir, ids, events, durations, model, prefix=""
) -> None:
    """Write the survival analysis output to a file."""
    logger.info(f"Write prediction to {output_dir}")

    if metrics:
        logger.debug("Write the metrics to a file")
        with Path(f"{output_dir}/metrics.json").open("w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

    for i in range(cfg.data.num_events):
        if isinstance(predictions, ModelOutput):
            if "loss" in predictions:
                idx = 2
        else:
            idx = 1

        if model.is_survival:
            columns = list(
                map(
                    lambda c: "t" + str(c),
                    range(cfg.data.label_transform.cuts + 1),
                )
            )
            hazard_df = pd.DataFrame(predictions[idx][:, i], columns=columns)
            hazard_df["id"] = ids
            hazard_df.to_csv(Path(f"{output_dir}/{prefix}hazard{i}.csv"), index=False)
            risk_df = pd.DataFrame(predictions[idx + 1][:, i], columns=columns)
            risk_df["id"] = ids
            risk_df.to_csv(Path(f"{output_dir}/{prefix}risk{i}.csv"), index=False)
            survival_df = pd.DataFrame(predictions[idx + 2][:, i], columns=columns)
            survival_df["id"] = ids
            survival_df.to_csv(
                Path(f"{output_dir}/{prefix}survival{i}.csv"), index=False
            )
            idx += 3

        if model.is_regression:
            # the problem is in the following line when we reference the number of labels in the task head
            columns = list(
                map(
                    lambda c: "d" + str(c),
                    range(cfg.regression_num_labels),
                )
            )
            preds = predictions[idx][:, i]
            tte_df = pd.DataFrame(preds, columns=columns)
            tte_df["id"] = ids
            if events:
                tte_df[cfg.data.event_col] = events
            if durations:
                tte_df[cfg.data.duration_col] = durations
            tte_df.to_csv(
                Path(f"{output_dir}/{prefix}time_to_event{i}.csv"), index=False
            )
            idx += 1

    if model.is_classification:
        columns = list(
            map(
                lambda c: "e" + str(c),
                range(cfg.data.num_events),
            )
        )
        preds = np.squeeze(predictions[idx], axis=2)
        e_df = pd.DataFrame(preds, columns=columns)
        e_df["id"] = ids
        if events:
            e_df[cfg.data.event_col] = events
        if durations:
            e_df[cfg.data.duration_col] = durations
        e_df.to_csv(Path(f"{output_dir}/{prefix}event_prediction.csv"), index=False)


def write_interpolation(cfg, predictions, ids, output_dir, is_survival):
    logger.info(f"Write interpolations to {output_dir}")
    for i in range(cfg.data.num_events):
        if isinstance(predictions, ModelOutput):
            if "loss" in predictions:
                idx = 2
        else:
            idx = 1
        if is_survival:
            if cfg.interpolate:
                df = pd.read_csv(
                    cfg.data.label_transform.save_dir + "/duration_cuts.csv",
                    header=None,
                    names=["cuts"],
                )
                device = predictions[idx].device
                duration_cuts = torch.Tensor(
                    df.cuts.values, device=device
                )  # tn duration cut points
                interp = interpolate.Interpolator(duration_cuts, cfg.interpolate_points)
                t, hazard, survival = interp(
                    predictions[idx][:, i], predictions[idx + 2][:, i]
                )

                interp_df = pd.DataFrame(
                    data={
                        "id": np.broadcast_to(
                            np.array(ids).reshape((hazard.shape[0], 1)),
                            hazard.shape,
                        ).flatten(),
                        "t": t.expand(hazard.shape[0], -1).flatten().cpu().numpy(),
                        "hazard": hazard.flatten().cpu().numpy(),
                        "survival": survival.flatten().cpu().numpy(),
                    }
                )
                interp_df.to_csv(Path(f"{output_dir}/interpolations.csv"), index=False)


def log_metrics(metrics, file_path=None):
    """
    Save metrics to a JSON file.

    Args:
        metrics: Dictionary of metrics to save
        file_path: Path to save the metrics JSON file
    """
    if file_path:
        # Filter to include only numeric values
        numeric_metrics = {
            k: v for k, v in metrics.items() if isinstance(v, numbers.Number)
        }

        # Write to file
        with open(file_path, "w") as f:
            json.dump(numeric_metrics, f, indent=4)

        logger.info(f"Saved metrics to {file_path}")

    return metrics


def log_metrics_from_replications(metrics, prefix):
    """
    Convert nested metrics dictionary to a flat dictionary with prefixed keys.

    Args:
        metrics: Dictionary of metrics, potentially nested
        prefix: Prefix to add to all metric keys

    Returns:
        Flattened metrics dictionary with prefixed keys
    """
    new_dict = {}
    for k, v in metrics.items():
        if isinstance(v, numbers.Number):
            new_dict[f"{prefix}_{k}"] = v
        elif isinstance(v, dict):
            for k_n, v_n in v.items():
                if isinstance(v_n, numbers.Number):
                    new_dict[f"{prefix}_{k}_{k_n}"] = v_n
        else:
            pass  # ignore

    return new_dict
