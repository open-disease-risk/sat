"""Exploratory Data Analysis (EDA) for Survival Analysis

This module provides EDA capabilities to analyze survival analysis datasets.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

# Apply the lifelines patch to fix scipy.integrate.trapz import issue
try:
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from fix_lifelines_import import apply_lifelines_patch

    apply_lifelines_patch()
except ImportError:
    pass

import json
import os
import re
from logging import DEBUG, ERROR
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import polars as pl
from logdecorator import log_on_end, log_on_error, log_on_start
from omegaconf import DictConfig

from sat.analysis import censoring, distribution_fitting
from sat.utils import logging, rand

logger = logging.get_default_logger()


@rand.seed
def extract_dataset_components(cfg: DictConfig) -> Tuple:
    """
    Extract the key components from a dataset for EDA using Polars for improved performance.

    Args:
        cfg: Configuration with dataset information

    Returns:
        Tuple of (durations, events, event_types, covariates)
    """
    # Load dataset using existing SAT loader
    logger.info(f"Loading dataset: {cfg.dataset}")
    dataset = hydra.utils.call(cfg.data.load)

    # Split dataset if needed
    if hasattr(cfg.data, "splits"):
        split_key = cfg.data.splits[0]  # Usually "train"
        logger.info(f"Using split: {split_key}")
        dataset = dataset[split_key]

    # Extract event and duration information
    duration_col = cfg.data.duration_col
    event_col = cfg.data.event_col
    num_events = cfg.data.num_events

    # Convert to Polars DataFrame for faster processing
    if isinstance(dataset, dict):
        # Convert dictionary of arrays to Polars DataFrame
        df_dict = {}
        for key, value in dataset.items():
            if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                df_dict[key] = value
        df = pl.DataFrame(df_dict)
    elif hasattr(dataset, "to_pandas"):
        # Handle dataset objects with to_pandas method
        df = pl.from_pandas(dataset.to_pandas())
    else:
        # Try direct conversion
        try:
            df = pl.DataFrame(dataset)
        except Exception:
            # Fallback to pandas if needed
            pdf = pd.DataFrame(dataset)
            df = pl.from_pandas(pdf)

    logger.debug(f"Dataset columns: {df.columns}")

    # Extract durations and events based on data structure
    durations = []
    events = []
    event_types = []

    # Helper to safely get numpy arrays from Polars series
    def get_numpy_array(series):
        if hasattr(series, "to_numpy"):
            return series.to_numpy()
        return np.array(series)

    # Special case for HSA synthetic dataset
    if "hsa_synthetic" in cfg.dataset.lower():
        logger.info("Handling HSA synthetic dataset format")
        # First check if we have the processed format with lists
        if "events" in df.columns and "durations" in df.columns:
            try:
                pdf = df.to_pandas()
                # Check if we have list values
                if isinstance(pdf["events"].iloc[0], list):
                    for event_idx in range(num_events):
                        # Make sure we have valid indices and non-empty data
                        if len(pdf) > 0 and len(pdf["events"].iloc[0]) > event_idx:
                            event_indicators = np.array(
                                [
                                    (
                                        float(row[event_idx])
                                        if isinstance(row, list)
                                        and len(row) > event_idx
                                        else 0.0
                                    )
                                    for row in pdf["events"]
                                ]
                            )
                            event_durations = np.array(
                                [
                                    (
                                        float(row[event_idx])
                                        if isinstance(row, list)
                                        and len(row) > event_idx
                                        else 0.0
                                    )
                                    for row in pdf["durations"]
                                ]
                            )

                            # Only add if we have some events (not all zeros)
                            if np.sum(event_indicators) > 0:
                                durations.append(event_durations)
                                events.append(event_indicators)
                                event_types.append(f"Event_{event_idx+1}")

                    logger.info(
                        f"Extracted {len(event_types)} events from HSA list format"
                    )
                else:
                    # Fallback to column-based format
                    raise ValueError("Not a list format, falling back to column search")
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(
                    f"Could not process list-based format: {str(e)}, trying column-based format"
                )
                # Continue to the column-based approach

        # Try to find duration1, duration2, event1, event2 pattern if lists didn't work
        if len(durations) == 0:
            duration_pattern = r"duration(\d+)"
            event_pattern = r"event(\d+)"

            duration_cols = sorted(
                [col for col in df.columns if re.match(duration_pattern, col)]
            )

            event_cols = sorted(
                [col for col in df.columns if re.match(event_pattern, col)]
            )

            if duration_cols and event_cols:
                for dur_col, evt_col in zip(duration_cols, event_cols):
                    evt_idx = re.match(event_pattern, evt_col).group(1)
                    dur_array = get_numpy_array(df[dur_col])
                    evt_array = get_numpy_array(df[evt_col])

                    # Only add if we have some events (not all zeros)
                    if np.sum(evt_array) > 0:
                        durations.append(dur_array)
                        events.append(evt_array)
                        event_types.append(f"Event_{evt_idx}")

                logger.info(f"Extracted {len(event_types)} events from column names")

        # Make sure we have at least one event type
        if len(durations) == 0 or len(events) == 0:
            logger.warning(
                "No events extracted from HSA synthetic dataset, trying original file fallback"
            )

            # Try to load the original CSV file directly as a fallback
            try:
                # Try different paths for finding the CSV
                csv_paths = [
                    "/Users/ddahlem/Documents/repos/open-disease-risk/sat/data/hsa-synthetic/simulated_data.csv",  # Direct path
                    os.path.join(
                        "data", "hsa-synthetic", "simulated_data.csv"
                    ),  # Relative to working dir
                ]

                csv_path = None
                for path in csv_paths:
                    if os.path.exists(path):
                        csv_path = path
                        break

                if csv_path:
                    logger.info(f"Loading original CSV file from {csv_path}")
                    raw_df = pd.read_csv(csv_path)

                    # Check for duration1, duration2, event1, event2 columns
                    if "duration1" in raw_df.columns and "event1" in raw_df.columns:
                        dur1 = np.array(raw_df["duration1"])
                        evt1 = np.array(raw_df["event1"])
                        if np.sum(evt1) > 0:  # Only add if we have some events
                            durations.append(dur1)
                            events.append(evt1)
                            event_types.append("Event_1")
                            logger.info(
                                f"Added Event_1 from original CSV with {np.sum(evt1)} events"
                            )

                    if "duration2" in raw_df.columns and "event2" in raw_df.columns:
                        dur2 = np.array(raw_df["duration2"])
                        evt2 = np.array(raw_df["event2"])
                        if np.sum(evt2) > 0:  # Only add if we have some events
                            durations.append(dur2)
                            events.append(evt2)
                            event_types.append("Event_2")
                            logger.info(
                                f"Added Event_2 from original CSV with {np.sum(evt2)} events"
                            )
            except Exception as e:
                logger.warning(f"Failed to load original CSV file: {str(e)}")

            # Final fallback if still no events
            if len(durations) == 0 or len(events) == 0:
                logger.warning("No events extracted, using final fallback approach")
                # Create a fallback with at least one event type
                if "duration1" in df.columns and "event1" in df.columns:
                    durations.append(get_numpy_array(df["duration1"]))
                    events.append(get_numpy_array(df["event1"]))
                    event_types.append("Event_1")
    # Handle multi-event cases (with arrays/lists for durations and events)
    elif num_events > 1 and duration_col in df.columns:
        # Check if we're dealing with list/array columns
        # Get first value to check
        first_dur_val = df[duration_col].item(0) if len(df) > 0 else None
        if isinstance(first_dur_val, (list, np.ndarray)):
            # Convert to pandas temporarily for list/array handling
            pdf = df.to_pandas()
            for event_idx in range(num_events):
                event_indicators = np.array([row[event_idx] for row in pdf[event_col]])
                event_durations = np.array(
                    [row[event_idx] for row in pdf[duration_col]]
                )

                durations.append(event_durations)
                events.append(event_indicators)
                event_types.append(f"Event_{event_idx}")
        else:
            # Regular multi-event structure with separate columns
            for event_idx in range(num_events):
                dur_col = (
                    f"{duration_col}_{event_idx}" if num_events > 1 else duration_col
                )
                evt_col = f"{event_col}_{event_idx}" if num_events > 1 else event_col
                evt_name = f"Event_{event_idx}" if num_events > 1 else "Event"

                if dur_col in df.columns and evt_col in df.columns:
                    durations.append(get_numpy_array(df[dur_col]))
                    events.append(get_numpy_array(df[evt_col]))
                    event_types.append(evt_name)
    else:
        # Single event case
        durations.append(get_numpy_array(df[duration_col]))
        events.append(get_numpy_array(df[event_col]))
        event_types.append("Event")

    # Extract covariates (features)
    covariate_cols = []

    # Check for explicit covariate configuration
    if hasattr(cfg.analysis, "covariates") and hasattr(
        cfg.analysis.covariates, "columns"
    ):
        covariate_cols = cfg.analysis.covariates.columns
    else:
        # Try to infer covariates: exclude standard columns like id, duration, event
        exclude_patterns = ["id", "duration", "event", "time", "status", "index"]
        covariate_cols = [
            col
            for col in df.columns
            if not any(pattern in col.lower() for pattern in exclude_patterns)
        ]

    # Create covariates DataFrame - convert to pandas for compatibility with existing analysis functions
    covariates_df = df.select(covariate_cols).to_pandas() if covariate_cols else None

    logger.info(f"Extracted {len(durations)} event types")
    if covariates_df is not None:
        logger.info(f"Extracted {len(covariates_df.columns)} covariates")

    return durations, events, event_types, covariates_df


def analyze_distribution(
    durations: List[np.ndarray],
    events: List[np.ndarray],
    event_types: List[str],
    num_events: int,
    output_dir: str,
    distributions: List[str] = None,
    prefer_metric: str = "bic",
    create_config: bool = True,
    dataset_name: str = None,
) -> Dict:
    """
    Analyze time-to-event distributions and recommend parametric models.

    Args:
        durations: List of arrays with event/censoring times
        events: List of arrays with event indicators
        event_types: List of names for each event type
        num_events: Number of event types
        output_dir: Directory to save results
        distributions: List of distributions to fit
        prefer_metric: Metric to use for recommendations ('aic' or 'bic')
        create_config: Whether to create a config file for DSM
        dataset_name: Name of the dataset

    Returns:
        Dict: Analysis results
    """
    # Create output directory
    dist_output_dir = os.path.join(output_dir, "distribution_analysis")
    os.makedirs(dist_output_dir, exist_ok=True)

    # Extract only the durations/events for actual events (not censored)
    event_durations = []
    event_indicators = []

    for i, (durs, evts) in enumerate(zip(durations, events)):
        # Keep only events that occurred
        mask = evts == 1
        event_durations.append(durs[mask])
        event_indicators.append(evts[mask])
        logger.info(f"Event type {event_types[i]}: {sum(mask)} events")

    # Default distributions to fit
    if distributions is None:
        distributions = ["weibull", "lognormal", "loglogistic"]

    # Fit distributions
    logger.info(f"Fitting distributions: {distributions}")
    fitted_models, aic_scores, bic_scores = distribution_fitting.fit_distributions(
        event_durations, event_indicators, num_events, distributions
    )

    # Plot distribution fits
    logger.info("Generating distribution fit plots")
    distribution_fitting.plot_distribution_fits(
        event_durations,
        event_indicators,
        fitted_models,
        aic_scores,
        bic_scores,
        num_events,
        dist_output_dir,
    )

    # Generate recommendations
    logger.info("Generating distribution recommendations")
    recommendations = distribution_fitting.generate_recommendations(
        aic_scores,
        bic_scores,
        num_events,
        dist_output_dir,
        prefer_metric,
        fitted_models=fitted_models,
        durations=event_durations,
    )

    # Save DSM configuration if requested
    if create_config and dataset_name:
        config = distribution_fitting.create_dsm_config(
            recommendations, dist_output_dir, dataset_name
        )

    return recommendations


def analyze_censoring(
    durations: List[np.ndarray],
    events: List[np.ndarray],
    event_types: List[str],
    covariates: Optional[pd.DataFrame] = None,
    output_dir: str = None,
) -> Dict:
    """
    Analyze censoring patterns in the dataset.

    Args:
        durations: List of arrays with event/censoring times
        events: List of arrays with event indicators
        event_types: List of names for each event type
        covariates: Optional DataFrame of covariates
        output_dir: Directory to save results

    Returns:
        Dict: Censoring analysis results
    """
    # Create output directory
    cens_output_dir = os.path.join(output_dir, "censoring_analysis")
    os.makedirs(cens_output_dir, exist_ok=True)

    results = {}

    # Analyze censoring pattern for each event type
    for i, (durs, evts, evt_type) in enumerate(zip(durations, events, event_types)):
        logger.info(f"Analyzing censoring for {evt_type}")
        event_output_dir = os.path.join(cens_output_dir, evt_type)

        # Analyze basic censoring pattern
        censoring_results = censoring.analyze_censoring_pattern(
            durs,
            evts,
            covariates=covariates,
            output_dir=event_output_dir,
            event_label=evt_type,
        )

        # If covariates are provided, analyze potential bias
        if covariates is not None and not covariates.empty:
            logger.info(f"Analyzing censoring bias for {evt_type}")
            bias_results = censoring.analyze_censoring_bias(
                durs, evts, covariates, output_dir=event_output_dir
            )

            # Merge bias results into censoring results
            censoring_results.update(bias_results)

        # Store results for this event type
        results[evt_type] = censoring_results

    # If we have multiple event types, analyze competing risks
    if len(event_types) > 1:
        logger.info("Analyzing competing risks")
        competing_risks_results = censoring.analyze_competing_risks(
            durations, events, event_types, output_dir=cens_output_dir
        )

        results["competing_risks"] = competing_risks_results

    # Save overall summary
    with open(os.path.join(cens_output_dir, "censoring_summary.json"), "w") as f:
        json.dump(results, f, indent=2, cls=logging.NpEncoder)

    return results


def analyze_covariates_effects(
    durations: List[np.ndarray],
    events: List[np.ndarray],
    event_types: List[str],
    covariates_df: pd.DataFrame,
    output_dir: str = None,
) -> Dict:
    """
    Analyze covariate relationships with survival outcomes.

    Args:
        durations: List of arrays with event/censoring times
        events: List of arrays with event indicators
        event_types: List of names for each event type
        covariates_df: DataFrame of covariates
        output_dir: Directory to save results

    Returns:
        Dict: Covariate analysis results
    """
    # Create output directory
    cov_output_dir = os.path.join(output_dir, "covariate_analysis")
    os.makedirs(cov_output_dir, exist_ok=True)

    results = {}

    # Analyze covariate distributions
    logger.info("Analyzing covariate distributions")
    distribution_output_dir = os.path.join(cov_output_dir, "distributions")

    from sat.analysis import covariates as cov_analysis

    distribution_results = cov_analysis.analyze_covariate_distributions(
        covariates_df, output_dir=distribution_output_dir
    )

    results["distributions"] = distribution_results

    # Analyze effects for each event type
    for i, (durs, evts, evt_type) in enumerate(zip(durations, events, event_types)):
        logger.info(f"Analyzing covariate effects for {evt_type}")
        event_output_dir = os.path.join(cov_output_dir, evt_type)

        # Analyze feature importance
        importance_results = cov_analysis.analyze_feature_importance(
            covariates_df,
            durs,
            evts,
            output_dir=os.path.join(event_output_dir, "importance"),
        )

        # Identify risk factors
        risk_results = cov_analysis.identify_risk_factors(
            covariates_df,
            durs,
            evts,
            output_dir=os.path.join(event_output_dir, "risk_factors"),
        )

        # Store results for this event type
        results[evt_type] = {
            "feature_importance": importance_results,
            "risk_factors": risk_results,
        }

    # Save overall summary
    with open(os.path.join(cov_output_dir, "covariates_summary.json"), "w") as f:
        json.dump(results, f, indent=2, cls=logging.NpEncoder)

    return results


@rand.seed
def _run_eda(cfg: DictConfig) -> None:
    """Run exploratory data analysis on the dataset using Polars for improved performance."""
    # Setup output directory
    output_dir = cfg.outputs.dir
    os.makedirs(output_dir, exist_ok=True)

    # Extract dataset components
    durations, events, event_types, covariates = extract_dataset_components(cfg)

    # Get CSV export setting
    export_csv = cfg.performance.get("export_csv", True)

    # Run requested analyses
    results = {}

    # Distribution analysis
    if cfg.analysis.run_distribution_analysis:
        # Check if we have any meaningful events to analyze
        if (
            len(durations) > 0
            and len(events) > 0
            and any(np.sum(e) > 0 for e in events)
        ):
            logger.info("Running time-to-event distribution analysis")
            dist_results = analyze_distribution(
                durations,
                events,
                event_types,
                cfg.data.num_events,
                output_dir,
                distributions=cfg.analysis.get("distributions", None),
                prefer_metric=cfg.analysis.get("prefer_metric", "bic"),
                create_config=cfg.analysis.get("create_config", True),
                dataset_name=cfg.dataset,
            )
            results["distribution"] = dist_results
        else:
            logger.warning("No events found for distribution analysis. Skipping.")
            results["distribution"] = {"error": "No events found for analysis"}

    # Censoring analysis
    if cfg.analysis.run_censoring_analysis:
        # Check if we have any meaningful events to analyze
        if (
            len(durations) > 0
            and len(events) > 0
            and any(np.sum(e) > 0 for e in events)
        ):
            try:
                filtered_covariates = None
                if covariates is not None and cfg.analysis.get("use_covariates", True):
                    # Clear problematic columns like arrays/lists that can't be processed
                    valid_cols = []
                    for col in covariates.columns:
                        # Check the first non-null value
                        first_val = (
                            covariates[col].iloc[0] if len(covariates) > 0 else None
                        )
                        if isinstance(first_val, (list, np.ndarray)):
                            logger.warning(
                                f"Dropping column {col} from censoring analysis which contains arrays/lists"
                            )
                        else:
                            valid_cols.append(col)

                    if valid_cols:
                        filtered_covariates = covariates[valid_cols]
                    else:
                        logger.warning(
                            "No valid covariates found for censoring analysis"
                        )

                logger.info("Running censoring analysis")
                cens_results = analyze_censoring(
                    durations,
                    events,
                    event_types,
                    covariates=filtered_covariates,
                    output_dir=output_dir,
                )
                results["censoring"] = cens_results
            except Exception as e:
                logger.error(f"Error in censoring analysis: {str(e)}")
                results["censoring"] = {"error": f"Analysis failed: {str(e)}"}
        else:
            logger.warning("No events found for censoring analysis. Skipping.")
            results["censoring"] = {"error": "No events found for analysis"}

    # Covariate analysis
    if cfg.analysis.run_covariate_analysis and covariates is not None:
        # Check if we have any meaningful events to analyze
        if (
            len(durations) > 0
            and len(events) > 0
            and any(np.sum(e) > 0 for e in events)
        ):
            try:
                # Clear problematic columns like arrays/lists that can't be processed
                valid_cols = []
                for col in covariates.columns:
                    # Check the first non-null value
                    first_val = covariates[col].iloc[0] if len(covariates) > 0 else None
                    if isinstance(first_val, (list, np.ndarray)):
                        logger.warning(
                            f"Dropping column {col} which contains arrays/lists"
                        )
                    else:
                        valid_cols.append(col)

                if valid_cols:
                    logger.info("Running covariate analysis with valid columns")
                    filtered_covariates = covariates[valid_cols]
                    cov_results = analyze_covariates_effects(
                        durations,
                        events,
                        event_types,
                        covariates_df=filtered_covariates,
                        output_dir=output_dir,
                    )
                    results["covariates"] = cov_results
                else:
                    logger.warning("No valid covariates found for analysis. Skipping.")
                    results["covariates"] = {"error": "No valid covariates found"}
            except Exception as e:
                logger.error(f"Error in covariate analysis: {str(e)}")
                results["covariates"] = {"error": f"Analysis failed: {str(e)}"}
        else:
            logger.warning("No events found for covariate analysis. Skipping.")
            results["covariates"] = {"error": "No events found for analysis"}

    # Save overall summary
    with open(os.path.join(output_dir, "eda_summary.json"), "w") as f:
        json.dump(results, f, indent=2, cls=logging.NpEncoder)

    # Also save as CSV for each major component
    if export_csv:
        logger.info("Exporting results as CSV for external visualization")
        for analysis_type, analysis_results in results.items():
            if isinstance(analysis_results, dict):
                try:
                    # Flatten the results dictionary for CSV export
                    flat_results = []
                    for key, value in analysis_results.items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, (int, float, str, bool)):
                                    flat_results.append(
                                        {
                                            "analysis": analysis_type,
                                            "component": key,
                                            "metric": subkey,
                                            "value": subvalue,
                                        }
                                    )

                    if flat_results:
                        # Save as CSV
                        summary_df = pd.DataFrame(flat_results)
                        summary_csv_path = os.path.join(
                            output_dir, f"{analysis_type}_summary.csv"
                        )
                        summary_df.to_csv(summary_csv_path, index=False)
                        logger.debug(f"Saved summary CSV to {summary_csv_path}")
                except Exception as e:
                    logger.warning(
                        f"Could not save {analysis_type} results as CSV: {str(e)}"
                    )

    logger.info(f"EDA completed. Results saved to {output_dir}")


@log_on_start(DEBUG, "Starting exploratory data analysis...", logger=logger)
@log_on_error(
    ERROR,
    "Error during exploratory data analysis: {e!r}",
    logger=logger,
    on_exceptions=Exception,
    reraise=True,
)
@log_on_end(DEBUG, "EDA complete!", logger=logger)
@hydra.main(version_base=None, config_path="../conf", config_name="eda.yaml")
def run_eda(cfg: DictConfig) -> None:
    """Entry point for exploratory data analysis."""
    logging.set_verbosity(logging.DEBUG)
    _run_eda(cfg)


if __name__ == "__main__":
    run_eda()
