"""Exploratory Data Analysis (EDA) for Survival Analysis

This module provides EDA capabilities to analyze survival analysis datasets.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

# Apply the lifelines patch to fix scipy.integrate.trapz import issue
try:
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from fix_lifelines_import import apply_lifelines_patch

    apply_lifelines_patch()
except ImportError:
    pass

import hydra
import os
import json
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
from logdecorator import log_on_start, log_on_end, log_on_error
from logging import DEBUG, ERROR
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

from sat.utils import logging, rand
from sat.data import load
from sat.analysis import distribution_fitting, censoring, covariates

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

    # Handle multi-event cases (with arrays/lists for durations and events)
    if num_events > 1 and duration_col in df.columns:
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

    # Censoring analysis
    if cfg.analysis.run_censoring_analysis:
        logger.info("Running censoring analysis")
        cens_results = analyze_censoring(
            durations,
            events,
            event_types,
            covariates=covariates if cfg.analysis.get("use_covariates", True) else None,
            output_dir=output_dir,
        )
        results["censoring"] = cens_results

    # Covariate analysis
    if cfg.analysis.run_covariate_analysis and covariates is not None:
        logger.info("Running covariate analysis")
        cov_results = analyze_covariates_effects(
            durations,
            events,
            event_types,
            covariates_df=covariates,
            output_dir=output_dir,
        )
        results["covariates"] = cov_results

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
