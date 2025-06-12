"""Retrieve best hyperparameters from an Optuna study database."""

import argparse
import json
import os

import optuna
import yaml

from sat.utils import logging

logger = logging.get_default_logger()


def get_best_trial(storage_url, study_name):
    """
    Get the best trial from an Optuna study.

    Args:
        storage_url (str): The URL to the Optuna storage
        study_name (str): The name of the study

    Returns:
        The best trial object
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        logger.info(f"Loaded study '{study_name}' with {len(study.trials)} trials")
        logger.info(f"Study direction: {study.direction}")

        if not study.trials:
            logger.error(f"No trials found in study '{study_name}'")
            return None

        best_trial = study.best_trial
        logger.info(f"Best trial: #{best_trial.number} with value: {best_trial.value}")
        return best_trial

    except Exception as e:
        logger.error(f"Error loading study: {e}")
        return None


def format_best_params(best_trial, out_format="json"):
    """
    Format the best parameters into the desired output format.

    Args:
        best_trial: Optuna best trial object
        out_format: Output format ("json", "yaml", or "cli")

    Returns:
        Formatted parameters as a string
    """
    if not best_trial:
        return None

    params = best_trial.params

    if out_format == "json":
        return json.dumps(params, indent=2)
    elif out_format == "yaml":
        return yaml.dump(params, default_flow_style=False)
    elif out_format == "cli":
        return " ".join([f"{k}={v}" for k, v in params.items()])
    else:
        return str(params)


def save_best_params(best_trial, output_dir, out_format="json"):
    """
    Save the best parameters to files in various formats.

    Args:
        best_trial: Optuna best trial object
        output_dir: Directory to save the files
        out_format: Output format
    """
    if not best_trial:
        logger.error("No best trial to save")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save trial information
    trial_info = {
        "number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
        "datetime_start": (
            best_trial.datetime_start.isoformat() if best_trial.datetime_start else None
        ),
        "datetime_complete": (
            best_trial.datetime_complete.isoformat()
            if best_trial.datetime_complete
            else None
        ),
    }

    # Save in requested format
    if out_format in ["json", "all"]:
        with open(f"{output_dir}/best_params.json", "w") as f:
            json.dump(trial_info, f, indent=2)
            logger.info(f"Saved best parameters to {output_dir}/best_params.json")

    if out_format in ["yaml", "all"]:
        with open(f"{output_dir}/best_params.yaml", "w") as f:
            yaml.dump(trial_info, f, default_flow_style=False)
            logger.info(f"Saved best parameters to {output_dir}/best_params.yaml")

    if out_format in ["cli", "all"]:
        cli_args = " ".join([f"{k}={v}" for k, v in best_trial.params.items()])
        with open(f"{output_dir}/best_params_cli.txt", "w") as f:
            f.write(cli_args)
            logger.info(f"Saved CLI arguments to {output_dir}/best_params_cli.txt")

    # Always save the full trial information in JSON
    if out_format != "json" and out_format != "all":
        with open(f"{output_dir}/best_trial_full.json", "w") as f:
            json.dump(trial_info, f, indent=2)
            logger.info(
                f"Saved full trial information to {output_dir}/best_trial_full.json"
            )

    return trial_info


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve best hyperparameters from an Optuna study"
    )
    parser.add_argument(
        "--storage",
        required=True,
        help="SQLite storage URL (e.g., sqlite:///data/optuna/studies.db)",
    )
    parser.add_argument("--study", required=True, help="Name of the study")
    parser.add_argument(
        "--output-dir",
        default="data/optuna/best",
        help="Directory to save best parameters",
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml", "cli", "all"],
        default="all",
        help="Output format (default: all)",
    )
    parser.add_argument(
        "--print", action="store_true", help="Print best parameters to stdout"
    )

    args = parser.parse_args()

    # Get best trial
    best_trial = get_best_trial(args.storage, args.study)
    if not best_trial:
        logger.error("No best trial found")
        return 1

    # Save parameters
    save_best_params(best_trial, args.output_dir, args.format)

    # Print if requested
    if args.print:
        logger.info("Best parameters:")
        if args.format == "json":
            print(json.dumps(best_trial.params, indent=2))
        elif args.format == "yaml":
            print(yaml.dump(best_trial.params, default_flow_style=False))
        elif args.format == "cli":
            print(" ".join([f"{k}={v}" for k, v in best_trial.params.items()]))
        else:
            print(f"Best value: {best_trial.value}")
            for k, v in best_trial.params.items():
                print(f"{k}: {v}")

    logger.info(f"Best trial value: {best_trial.value}")

    return 0


if __name__ == "__main__":
    main()
