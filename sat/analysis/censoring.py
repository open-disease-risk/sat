"""Censoring analysis for survival data.

This module provides functions to analyze censoring patterns in survival data,
which is crucial for understanding potential biases in survival analysis models.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

# Apply the lifelines patch to fix scipy.integrate.trapz import issue
try:
    import os
    import sys

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )
    from fix_lifelines_import import apply_lifelines_patch

    apply_lifelines_patch()
except ImportError:
    pass

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import chi2_contingency

from sat.utils import logging

logger = logging.get_default_logger()


def analyze_censoring_pattern(
    durations: np.ndarray,
    events: np.ndarray,
    covariates: Optional[pd.DataFrame] = None,
    output_dir: str = None,
    event_label: str = "Event",
) -> Dict:
    """
    Analyze patterns of censoring in survival data.

    Args:
        durations: Array of event/censoring times
        events: Binary indicators (1=event, 0=censored)
        covariates: Optional DataFrame of covariates for analyzing censoring dependence
        output_dir: Directory to save plots and results
        event_label: Label for the event type

    Returns:
        Dict: Analysis results
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert to pandas Series for easier analysis
    df = pd.DataFrame({"duration": durations, "event": events})

    # Calculate basic statistics
    n_total = len(df)
    n_events = df["event"].sum()
    n_censored = n_total - n_events
    censoring_rate = n_censored / n_total * 100

    # Create result dictionary
    results = {
        "n_total": n_total,
        "n_events": int(n_events),
        "n_censored": int(n_censored),
        "censoring_rate": censoring_rate,
        "median_follow_up": np.median(durations),
        "max_follow_up": np.max(durations),
    }

    # Save basic statistics as CSV for external use
    if output_dir:
        stats_df = pd.DataFrame(
            [
                {"metric": "Total Samples", "value": n_total},
                {"metric": "Number of Events", "value": int(n_events)},
                {"metric": "Number Censored", "value": int(n_censored)},
                {"metric": "Censoring Rate (%)", "value": censoring_rate},
                {"metric": "Median Follow-up", "value": np.median(durations)},
                {"metric": "Max Follow-up", "value": np.max(durations)},
            ]
        )
        stats_df.to_csv(
            os.path.join(output_dir, "censoring_statistics.csv"), index=False
        )

        # Also save raw data for external plotting
        raw_df = pd.DataFrame(
            {
                "duration": durations,
                "event": events,
                "status": ["Event" if e == 1 else "Censored" for e in events],
            }
        )
        raw_df.to_csv(os.path.join(output_dir, "event_data.csv"), index=False)

    logger.info(f"Total samples: {n_total}")
    logger.info(f"Number of events: {n_events} ({n_events/n_total:.1%})")
    logger.info(f"Number censored: {n_censored} ({n_censored/n_total:.1%})")

    # Plot event and censoring distributions
    if output_dir:
        try:
            # Plot histogram of event times by status
            plt.figure(figsize=(10, 6))
            ax = sns.histplot(
                data=df,
                x="duration",
                hue="event",
                multiple="stack",
                bins=30,
                palette={0: "skyblue", 1: "salmon"},
                alpha=0.7,
            )
            ax.set_xlabel("Time")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Event Times and Censoring")
            # Add legend with custom labels
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, ["Censored", event_label])

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "censoring_distribution.png"))
            plt.close()

            # Plot Kaplan-Meier estimator for the censoring distribution
            # (treating events as censored and vice versa)
            plt.figure(figsize=(10, 6))
            kmf = KaplanMeierFitter()
            kmf.fit(durations, 1 - events, label="Censoring")
            kmf.plot_survival_function()

            # Export KM estimator data for external plotting (e.g., PGFPlots)
            km_df = pd.DataFrame(
                {
                    "time": kmf.timeline,
                    "survival": kmf.survival_function_.iloc[:, 0].values,
                    "confidence_lower": kmf.confidence_interval_.iloc[:, 0].values,
                    "confidence_upper": kmf.confidence_interval_.iloc[:, 1].values,
                }
            )
            km_df.to_csv(os.path.join(output_dir, "censoring_km_data.csv"), index=False)

            plt.xlabel("Time")
            plt.ylabel("Probability of Remaining Uncensored")
            plt.title("Kaplan-Meier Estimate of Censoring Distribution")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "censoring_km_curve.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Error creating censoring pattern plot: {str(e)}")
            plt.close()  # Ensure any open figure is closed

        # If covariates are provided, analyze dependency
        if covariates is not None and not covariates.empty:
            try:
                # Prepare data for Cox model with unique column names
                # First check if there are any naming conflicts with covariates
                cov_df = covariates.copy()
                # Debug column name conflicts
                logger.debug(f"Covariate columns: {list(cov_df.columns)}")

                # Use very specific column names that won't conflict - using UUID to ensure uniqueness
                import uuid

                unique_id = str(uuid.uuid4())[:8]
                time_column = f"cox_survival_duration_{unique_id}"
                event_column = f"cox_censoring_indicator_{unique_id}"

                # Add columns with guaranteed unique names
                cov_df[time_column] = durations
                cov_df[event_column] = 1 - events  # Reverse events to model censoring

                # Select only numeric columns for Cox model
                numeric_cols = cov_df.select_dtypes(include=["float", "int"]).columns
                if len(numeric_cols) > 0:
                    # Fit Cox model to check for informative censoring
                    cox = CoxPHFitter()
                    try:
                        # Create a list of column names, making sure to avoid duplicates
                        # First get unique column names from numeric_cols (excluding our newly added columns)
                        unique_numeric_cols = [
                            col
                            for col in numeric_cols
                            if col != time_column and col != event_column
                        ]

                        # Then add our duration and event columns
                        column_list = unique_numeric_cols + [time_column, event_column]
                        logger.debug(f"Cox model columns: {column_list}")
                        cox_df = cov_df[column_list]
                        cox.fit(
                            cox_df, duration_col=time_column, event_col=event_column
                        )

                        # Plot Cox model results
                        plt.figure(figsize=(12, 8))
                        cox.plot()
                        plt.title("Covariate Effects on Censoring")
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(output_dir, "censoring_covariate_effects.png")
                        )
                        plt.close()

                        # Save Cox model summary
                        cox_summary = cox.summary.copy()
                        cox_summary.to_csv(
                            os.path.join(output_dir, "censoring_cox_results.csv")
                        )

                        # Add Cox results to the output
                        results["cox_model"] = cox_summary.to_dict()
                        results["informative_censoring"] = any(cox_summary["p"] < 0.05)

                        # Flag predictors of censoring
                        if results["informative_censoring"]:
                            predictors = cox_summary[
                                cox_summary["p"] < 0.05
                            ].index.tolist()
                            results["censoring_predictors"] = predictors
                            logger.warning(
                                f"Potential informative censoring detected. Significant predictors: {predictors}"
                            )
                    except Exception as e:
                        logger.error(f"Error fitting Cox model: {str(e)}")
                else:
                    logger.warning("No numeric covariates available for Cox model")
            except Exception as e:
                logger.error(f"Error in covariate dependency analysis: {str(e)}")

    return results


def analyze_competing_risks(
    durations: List[np.ndarray],
    events: List[np.ndarray],
    labels: List[str],
    output_dir: str = None,
) -> Dict:
    """
    Analyze competing risks in survival data.

    Args:
        durations: List of arrays containing event times for each event type
        events: List of arrays containing event indicators for each event type
        labels: Names for each event type
        output_dir: Directory to save plots and results

    Returns:
        Dict: Analysis results
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    num_events = len(durations)
    results = {"num_events": num_events, "event_counts": {}, "event_rates": {}}

    # Compute event counts and rates
    for _, (_, evt, label) in enumerate(zip(durations, events, labels, strict=False)):
        count = int(np.sum(evt))
        rate = count / len(evt) * 100
        results["event_counts"][label] = count
        results["event_rates"][label] = rate
        logger.info(f"Event {label}: {count} occurrences ({rate:.1f}%)")

    # Plot cause-specific cumulative incidence functions
    if output_dir and num_events > 0:
        try:
            plt.figure(figsize=(10, 6))

            for _, (dur, evt, label) in enumerate(
                zip(durations, events, labels, strict=False)
            ):
                # Create Kaplan-Meier estimator and fit with event-specific data
                kmf = KaplanMeierFitter()
                kmf.fit(dur, evt, label=label)

                # Plot the CIF (1 - survival)
                kmf.plot_cumulative_density()

            plt.xlabel("Time")
            plt.ylabel("Cumulative Incidence")
            plt.title("Cause-Specific Cumulative Incidence Functions")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "competing_risks_cif.png"))
            plt.close()

            # Create stacked histogram of events by type
            all_times = np.concatenate(durations)
            all_events = np.concatenate(events)
            event_types = np.repeat(np.arange(num_events), [len(d) for d in durations])

            # Create DataFrame for plotting
            df = pd.DataFrame(
                {
                    "time": all_times,
                    "event": all_events,
                    "type": [
                        labels[t] if e == 1 else "Censored"
                        for t, e in zip(event_types, all_events, strict=False)
                    ],
                }
            )

            # Plot stacked histogram
            plt.figure(figsize=(12, 6))
            _ = sns.histplot(
                data=df[df["event"] == 1],  # Only include events
                x="time",
                hue="type",
                multiple="stack",
                bins=30,
                alpha=0.7,
            )
            plt.xlabel("Time")
            plt.ylabel("Count")
            plt.title("Distribution of Competing Events by Type")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "competing_risks_histogram.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Error creating competing risks plots: {str(e)}")
            plt.close()  # Ensure any open figure is closed

    return results


def analyze_censoring_bias(
    durations: np.ndarray,
    events: np.ndarray,
    covariates: pd.DataFrame,
    output_dir: str = None,
) -> Dict:
    """
    Analyze potential biases due to censoring.

    Args:
        durations: Array of event/censoring times
        events: Binary indicators (1=event, 0=censored)
        covariates: DataFrame of covariates
        output_dir: Directory to save plots and results

    Returns:
        Dict: Analysis results
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = {"n_samples": len(durations), "bias_tests": {}}

    try:
        # Create full DataFrame with duration and event info
        df = pd.DataFrame(
            {
                "time_to_event": durations,  # Use a different column name to avoid conflicts
                "event_indicator": events,
            }
        )

        # First, filter out non-numeric columns
        numeric_covariates = pd.DataFrame()
        categorical_covariates = pd.DataFrame()

        # Check each column to identify numeric vs. categorical
        for col in covariates.columns:
            try:
                # Try to convert to numeric - if it fails, it's likely categorical
                if pd.api.types.is_numeric_dtype(covariates[col]):
                    numeric_covariates[col] = covariates[col]
                else:
                    # Attempt conversion to numeric
                    converted = pd.to_numeric(covariates[col], errors="coerce")
                    if (
                        not converted.isna().all()
                    ):  # If some values converted successfully
                        numeric_covariates[col] = converted
                    else:
                        categorical_covariates[col] = covariates[col]
            except Exception as e:
                logger.warning(f"Could not process covariate {col}: {str(e)}")
                categorical_covariates[col] = covariates[col]

        logger.info(
            f"Found {len(numeric_covariates.columns)} numeric and {len(categorical_covariates.columns)} categorical covariates"
        )

        # Add covariates to main dataframe
        for col in numeric_covariates.columns:
            df[col] = numeric_covariates[col].values

        for col in categorical_covariates.columns:
            df[col] = categorical_covariates[col].values

        # Analyze censoring pattern by quartiles of each continuous covariate
        for col in numeric_covariates.columns:
            try:
                # Check for NaN values
                if df[col].isna().any():
                    logger.warning(
                        f"Column {col} contains NaN values, skipping quartile analysis"
                    )
                    continue

                # Create quartiles, handling cases with too few unique values
                num_unique = df[col].nunique()
                if num_unique < 4:
                    logger.warning(
                        f"Column {col} has only {num_unique} unique values, using value instead of quartiles"
                    )
                    df[f"{col}_quartile"] = df[col]
                else:
                    df[f"{col}_quartile"] = pd.qcut(
                        df[col], 4, labels=False, duplicates="drop"
                    )

                # Compute censoring rate by quartile
                censoring_by_quartile = df.groupby(f"{col}_quartile")[
                    "event_indicator"
                ].agg(
                    total_count="count",
                    censored_count=lambda x: (x == 0).sum(),
                    censoring_rate=lambda x: (x == 0).mean() * 100,
                )

                # Create contingency table: rows are quartiles, columns are (censored, event)
                contingency = pd.crosstab(df[f"{col}_quartile"], df["event_indicator"])

                # Ensure there are at least 2 rows and 2 columns
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    logger.warning(
                        f"Contingency table for {col} has less than 2 rows or columns, skipping chi2 test"
                    )
                    results["bias_tests"][col] = {
                        "censoring_by_quartile": censoring_by_quartile.to_dict(),
                        "chi2": None,
                        "p_value": None,
                        "significant": False,
                        "error": "Insufficient data for chi2 test",
                    }
                    continue

                chi2, p, _, _ = chi2_contingency(contingency)

                results["bias_tests"][col] = {
                    "censoring_by_quartile": censoring_by_quartile.to_dict(),
                    "chi2": chi2,
                    "p_value": p,
                    "significant": p < 0.05,
                }

                if p < 0.05:
                    logger.warning(
                        f"Potential censoring bias detected for {col} (p={p:.4f})"
                    )

                # Plot censoring rates by quartile
                if output_dir:
                    try:
                        plt.figure(figsize=(10, 6))
                        sns.barplot(
                            x=df[f"{col}_quartile"],
                            y=1 - df["event_indicator"],
                            estimator=np.mean,
                            errorbar=None,
                        )
                        plt.xlabel(f"Quartile of {col}")
                        plt.ylabel("Censoring Rate")
                        plt.title(f"Censoring Rate by Quartile of {col} (p={p:.4f})")
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(output_dir, f"censoring_bias_{col}.png")
                        )
                        plt.close()
                    except Exception as e:
                        logger.error(f"Error creating plot for {col}: {str(e)}")
                        plt.close()
            except Exception as e:
                logger.error(f"Error analyzing numeric covariate {col}: {str(e)}")
                results["bias_tests"][col] = {"error": str(e)}

        # Categorical covariates
        for col in categorical_covariates.columns:
            try:
                # Skip columns with too many unique values
                num_unique = df[col].nunique()
                if num_unique > 20:  # Skip if too many categories
                    logger.warning(
                        f"Column {col} has {num_unique} unique values, skipping categorical analysis"
                    )
                    continue

                # Compute censoring rate by category
                censoring_by_category = df.groupby(col)["event_indicator"].agg(
                    total_count="count",
                    censored_count=lambda x: (x == 0).sum(),
                    censoring_rate=lambda x: (x == 0).mean() * 100,
                )

                # Create contingency table: rows are categories, columns are (censored, event)
                contingency = pd.crosstab(df[col], df["event_indicator"])

                # Ensure there are at least 2 rows and 2 columns
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    logger.warning(
                        f"Contingency table for {col} has less than 2 rows or columns, skipping chi2 test"
                    )
                    results["bias_tests"][col] = {
                        "censoring_by_category": censoring_by_category.to_dict(),
                        "chi2": None,
                        "p_value": None,
                        "significant": False,
                        "error": "Insufficient data for chi2 test",
                    }
                    continue

                chi2, p, _, _ = chi2_contingency(contingency)

                results["bias_tests"][col] = {
                    "censoring_by_category": censoring_by_category.to_dict(),
                    "chi2": chi2,
                    "p_value": p,
                    "significant": p < 0.05,
                }

                if p < 0.05:
                    logger.warning(
                        f"Potential censoring bias detected for categorical variable {col} (p={p:.4f})"
                    )

                # Plot censoring rates by category
                if output_dir:
                    try:
                        plt.figure(figsize=(12, 6))
                        sns.barplot(
                            x=df[col],
                            y=1 - df["event_indicator"],
                            estimator=np.mean,
                            errorbar=None,
                        )
                        plt.xlabel(f"{col}")
                        plt.ylabel("Censoring Rate")
                        plt.title(f"Censoring Rate by {col} (p={p:.4f})")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(output_dir, f"censoring_bias_{col}.png")
                        )
                        plt.close()
                    except Exception as e:
                        logger.error(
                            f"Error creating plot for categorical {col}: {str(e)}"
                        )
                        plt.close()
            except Exception as e:
                logger.error(f"Error analyzing categorical covariate {col}: {str(e)}")
                results["bias_tests"][col] = {"error": str(e)}
    except Exception as e:
        logger.error(f"Error in censoring bias analysis: {str(e)}")
        results["error"] = str(e)

    return results
