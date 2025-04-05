"""Distribution fitting for time-to-event data analysis.

This module provides functionality to fit and compare various parametric distributions
to time-to-event data, which is useful for:
1. Understanding the underlying distribution of survival times
2. Selecting appropriate parametric models for survival analysis
3. Configuring Deep Survival Machines (DSM) models
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

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import (
    KaplanMeierFitter,
    LogLogisticFitter,
    LogNormalFitter,
    WeibullFitter,
)
from scipy import stats

from sat.utils import logging

logger = logging.get_default_logger()


def fit_distributions(
    durations: List[np.ndarray],
    events: List[np.ndarray],
    num_events: int,
    distributions: Optional[List[str]] = None,
) -> Tuple[Dict, Dict, Dict]:
    """
    Fit various parametric distributions to time-to-event data.

    Args:
        durations: List of arrays containing event times
        events: List of arrays containing event indicators
        num_events: Number of event types
        distributions: List of distribution names to fit

    Returns:
        tuple: (fitted_models, aic_scores, bic_scores)
    """
    fitted_models = {}
    aic_scores = {}
    bic_scores = {}

    # Default distributions to fit
    if distributions is None:
        distributions = ["weibull", "lognormal", "loglogistic"]

    for i in range(len(durations)):
        event_key = f"event_{i}" if num_events > 1 else "event"
        fitted_models[event_key] = {}
        aic_scores[event_key] = {}
        bic_scores[event_key] = {}

        times = durations[i]

        # Skip if we have no data or all zeros
        if len(times) == 0 or np.all(times == 0):
            logger.warning(
                f"No valid times for {event_key}, skipping distribution fitting"
            )
            continue

        # Make sure we have at least some non-zero values to fit
        non_zero_times = times[times > 0]
        if len(non_zero_times) < 5:  # Require at least 5 non-zero values for fitting
            logger.warning(
                f"Too few non-zero times for {event_key}, skipping distribution fitting"
            )
            continue

        # Use non-zero times for fitting
        times = non_zero_times

        # Fit Weibull distribution if requested
        if "weibull" in distributions:
            logger.debug(f"Fitting Weibull distribution for {event_key}")
            wbf = WeibullFitter()
            wbf.fit(times, event_observed=np.ones(len(times)))
            fitted_models[event_key]["weibull"] = wbf
            aic_scores[event_key]["weibull"] = wbf.AIC_
            bic_scores[event_key]["weibull"] = wbf.BIC_
            logger.debug(
                f"Weibull fit - shape: {wbf.lambda_:.3f}, scale: {wbf.rho_:.3f}, AIC: {wbf.AIC_:.3f}, BIC: {wbf.BIC_:.3f}"
            )

        # Fit LogNormal distribution if requested
        if "lognormal" in distributions:
            logger.debug(f"Fitting Log-Normal distribution for {event_key}")
            lnf = LogNormalFitter()
            lnf.fit(times, event_observed=np.ones(len(times)))
            fitted_models[event_key]["lognormal"] = lnf
            aic_scores[event_key]["lognormal"] = lnf.AIC_
            bic_scores[event_key]["lognormal"] = lnf.BIC_
            logger.debug(
                f"Log-Normal fit - mu: {lnf.mu_:.3f}, sigma: {lnf.sigma_:.3f}, AIC: {lnf.AIC_:.3f}, BIC: {lnf.BIC_:.3f}"
            )

        # Fit LogLogistic distribution if requested
        if "loglogistic" in distributions:
            logger.debug(f"Fitting Log-Logistic distribution for {event_key}")
            llf = LogLogisticFitter()
            llf.fit(times, event_observed=np.ones(len(times)))
            fitted_models[event_key]["loglogistic"] = llf
            aic_scores[event_key]["loglogistic"] = llf.AIC_
            bic_scores[event_key]["loglogistic"] = llf.BIC_
            logger.debug(
                f"Log-Logistic fit - alpha: {llf.alpha_:.3f}, beta: {llf.beta_:.3f}, AIC: {llf.AIC_:.3f}, BIC: {llf.BIC_:.3f}"
            )

    return fitted_models, aic_scores, bic_scores


def plot_distribution_fits(
    durations: List[np.ndarray],
    events: List[np.ndarray],
    fitted_models: Dict,
    aic_scores: Dict,
    bic_scores: Dict,
    num_events: int,
    output_dir: str,
) -> None:
    """
    Plot fitted distributions against empirical survival curves with
    robust error handling to prevent plotting errors from stopping the analysis.

    Args:
        durations: List of arrays containing event times
        events: List of arrays containing event indicators
        fitted_models: Dictionary of fitted models
        aic_scores: Dictionary of AIC scores
        bic_scores: Dictionary of BIC scores
        num_events: Number of event types
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(durations)):
        event_key = f"event_{i}" if num_events > 1 else "event"
        times = durations[i]

        try:
            # Skip if times array is empty or if all values are too small
            if len(times) == 0 or max(times, default=0) < 0.001:
                logger.warning(f"No valid times for {event_key}, skipping plot")
                continue

            # Create survival curve plot
            plt.figure(figsize=(12, 10))

            # Plot empirical survival curve (Kaplan-Meier)
            kmf = KaplanMeierFitter()
            kmf.fit(times, event_observed=np.ones(len(times)))
            kmf.plot(label="Kaplan-Meier", ci_show=True, color="k")

            # Plot fitted distributions - safely calculate max with a default value
            if len(times) > 0:
                max_time = max(times) * 1.1
            else:
                max_time = 100  # default to reasonable value if times is empty
            time_points = np.linspace(0, max_time, 100)

            for dist_name, model in fitted_models[event_key].items():
                survs = model.survival_function_at_times(time_points)
                plt.plot(
                    time_points,
                    survs,
                    linewidth=2,
                    label=f"{dist_name.capitalize()} (AIC={aic_scores[event_key][dist_name]:.2f}, BIC={bic_scores[event_key][dist_name]:.2f})",
                )

            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Survival Probability", fontsize=12)
            plt.title(
                f"Survival Distribution Fits for {event_key.replace('_', ' ').title()}",
                fontsize=14,
            )
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)

            # Save plot
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"distribution_fit_{event_key}.png")
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved survival curve plot to {plot_path}")
        except Exception as e:
            logger.error(
                f"Error creating survival curve plot for {event_key}: {str(e)}"
            )
            plt.close()  # Ensure any open figure is closed

        # Create QQ plots
        for dist_name, model in fitted_models[event_key].items():
            try:
                plt.figure(figsize=(10, 8))

                if dist_name == "weibull":
                    # Weibull QQ plot
                    shape = model.lambda_
                    scale = model.rho_

                    # Sort the observed data and create empirical quantiles
                    sorted_times = np.sort(times)

                    # Ensure the arrays are the same length
                    n = len(sorted_times)
                    empirical_quantiles = np.linspace(
                        0.01, 0.99, n
                    )  # Use same number of points as data

                    # Calculate theoretical quantiles based on empirical positions
                    theoretical_quantiles = scale * (
                        -np.log(1 - empirical_quantiles)
                    ) ** (1 / shape)

                    # Now both arrays have the same length
                    plt.scatter(theoretical_quantiles, sorted_times)
                    plt.xlabel("Theoretical Quantiles")
                    plt.ylabel("Observed Times")

                elif dist_name == "lognormal":
                    # Log-normal QQ plot - use our own implementation instead of stats.probplot
                    # to ensure consistency with other plots
                    sorted_times = np.sort(times)
                    n = len(sorted_times)

                    # Create empirical quantiles
                    empirical_quantiles = np.linspace(0.01, 0.99, n)

                    # Calculate theoretical normal quantiles
                    from scipy.stats import norm

                    theoretical_quantiles = norm.ppf(empirical_quantiles)

                    # For lognormal, we plot theoretical normal quantiles vs log of times
                    log_times = np.log(sorted_times)

                    plt.scatter(theoretical_quantiles, log_times)
                    plt.xlabel("Theoretical Normal Quantiles")
                    plt.ylabel("Log of Observed Times")

                elif dist_name == "loglogistic":
                    # Log-logistic QQ plot (similar to logit transform)
                    sorted_times = np.sort(times)
                    n = len(sorted_times)

                    # Create empirical quantiles, ensuring we don't get 0 or 1
                    # which would cause problems with the logit transform
                    empirical_quantiles = np.linspace(0.01, 0.99, n)

                    # Calculate logit of empirical quantiles
                    logit_empirical = np.log(
                        empirical_quantiles / (1 - empirical_quantiles)
                    )
                    log_times = np.log(sorted_times)

                    plt.scatter(logit_empirical, log_times)
                    plt.xlabel("Theoretical Quantiles (Logit)")
                    plt.ylabel("Log of Observed Times")

                plt.grid(alpha=0.3)
                plt.title(
                    f"{dist_name.capitalize()} Q-Q Plot for {event_key.replace('_', ' ').title()}"
                )
                qq_path = os.path.join(
                    output_dir, f"qq_plot_{dist_name}_{event_key}.png"
                )
                plt.savefig(qq_path)
                plt.close()
                logger.debug(f"Saved Q-Q plot for {dist_name} to {qq_path}")
            except Exception as e:
                logger.error(
                    f"Error creating Q-Q plot for {dist_name} for {event_key}: {str(e)}"
                )
                plt.close()  # Ensure any open figure is closed

        try:
            # Create histogram with density overlay
            plt.figure(figsize=(12, 8))

            # Histogram
            plt.hist(times, bins=30, density=True, alpha=0.5, label="Observed Data")

            # Overlay density functions - safely calculate max with a default value
            if len(times) > 0:
                max_time = max(times) * 1.1
            else:
                max_time = 100  # default to reasonable value if times is empty
            x = np.linspace(0, max_time, 1000)

            for dist_name, model in fitted_models[event_key].items():
                if dist_name == "weibull":
                    shape = model.lambda_
                    scale = model.rho_
                    y = stats.weibull_min.pdf(x, shape, scale=scale)
                    plt.plot(x, y, linewidth=2, label=f"{dist_name.capitalize()} PDF")

                elif dist_name == "lognormal":
                    mu = model.mu_
                    sigma = model.sigma_
                    y = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
                    plt.plot(x, y, linewidth=2, label=f"{dist_name.capitalize()} PDF")

                elif dist_name == "loglogistic":
                    alpha = model.alpha_
                    beta = model.beta_
                    # Approximate loglogistic PDF
                    y = (
                        (beta / alpha)
                        * (x / alpha) ** (beta - 1)
                        / (1 + (x / alpha) ** beta) ** 2
                    )
                    plt.plot(x, y, linewidth=2, label=f"{dist_name.capitalize()} PDF")

            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.title(
                f"Density Plot for {event_key.replace('_', ' ').title()}", fontsize=14
            )
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)

            plt.tight_layout()
            density_path = os.path.join(output_dir, f"density_{event_key}.png")
            plt.savefig(density_path)
            plt.close()
            logger.debug(f"Saved density plot to {density_path}")
        except Exception as e:
            logger.error(f"Error creating density plot for {event_key}: {str(e)}")
            plt.close()  # Ensure any open figure is closed


def generate_recommendations(
    aic_scores: Dict,
    bic_scores: Dict,
    num_events: int,
    output_dir: str,
    prefer_metric: str = "bic",
    fitted_models: Dict = None,
    durations: List[np.ndarray] = None,
) -> Dict:
    """
    Generate distribution recommendations based on analysis results.

    Args:
        aic_scores: Dictionary of AIC scores
        bic_scores: Dictionary of BIC scores
        num_events: Number of event types
        output_dir: Directory to save results
        prefer_metric: Which metric to use for primary recommendation ('aic' or 'bic')
        fitted_models: Optional dictionary of fitted models for additional exports
        durations: Optional list of duration arrays for survival function export

    Returns:
        dict: Recommendations for each event type
    """
    recommendations = {}
    summary = []

    for i in range(min(len(aic_scores), num_events)):
        event_key = f"event_{i}" if num_events > 1 else "event"

        # Find best model according to AIC
        best_aic_model = min(aic_scores[event_key].items(), key=lambda x: x[1])[0]
        best_aic_score = aic_scores[event_key][best_aic_model]

        # Find best model according to BIC
        best_bic_model = min(bic_scores[event_key].items(), key=lambda x: x[1])[0]
        best_bic_score = bic_scores[event_key][best_bic_model]

        # Determine final recommendation based on preferred metric
        if prefer_metric.lower() == "aic":
            recommendation = best_aic_model
        else:  # Default to BIC which penalizes complexity more
            recommendation = best_bic_model

        # Check if AIC and BIC disagree significantly
        aic_diff = abs(
            aic_scores[event_key][best_aic_model]
            - aic_scores[event_key][best_bic_model]
        )
        if best_aic_model != best_bic_model:
            if aic_diff > 10:
                note = f"AIC and BIC disagree significantly. AIC favors {best_aic_model}, BIC favors {best_bic_model}. Using {prefer_metric.upper()} recommendation."
            else:
                note = f"AIC favors {best_aic_model}, BIC favors {best_bic_model}. Using {prefer_metric.upper()} recommendation."
        else:
            note = f"Both AIC and BIC favor {recommendation}."

        recommendations[event_key] = {
            "recommended_distribution": recommendation,
            "aic_best": best_aic_model,
            "aic_score": best_aic_score,
            "bic_best": best_bic_model,
            "bic_score": best_bic_score,
            "note": note,
        }

        summary.append(
            {
                "event_type": event_key,
                "recommended_distribution": recommendation,
                "aic_best": best_aic_model,
                "aic_score": best_aic_score,
                "bic_best": best_bic_model,
                "bic_score": best_bic_score,
                "note": note,
            }
        )

        logger.info(
            f"{event_key}: Recommended distribution is {recommendation} ({note})"
        )

    # Create summary table
    summary_df = pd.DataFrame(summary)

    # Save recommendations to file
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "distribution_recommendations.csv")
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved recommendations to {csv_path}")

    # Generate a summary text file
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Distribution Analysis Summary\n")
        f.write("===========================\n\n")

        for i, row in summary_df.iterrows():
            f.write(f"Event Type: {row['event_type']}\n")
            f.write(f"Recommended Distribution: {row['recommended_distribution']}\n")
            f.write(f"AIC Best: {row['aic_best']} (Score: {row['aic_score']:.2f})\n")
            f.write(f"BIC Best: {row['bic_best']} (Score: {row['bic_score']:.2f})\n")
            f.write(f"Note: {row['note']}\n\n")

        # Add overall recommendation for DSM
        if len(summary_df) > 0:
            # Count distribution recommendations
            dist_counts = summary_df["recommended_distribution"].value_counts()
            most_common_dist = dist_counts.idxmax()

            f.write("Overall Recommendation for DSM\n")
            f.write("-----------------------------\n")
            f.write(f"Recommended Distribution: {most_common_dist}\n")
            f.write(
                f"(This distribution was recommended for {dist_counts[most_common_dist]} out of {len(summary_df)} event types)\n\n"
            )

            # Generate configuration snippet
            f.write("Configuration Snippet for DSM\n")
            f.write("----------------------------\n")
            f.write("Add this to your DSM experiment configuration file:\n\n")
            f.write(f"dsm_distribution: {most_common_dist}\n")
            f.write("dsm_num_mixtures: 4  # Adjust based on complexity\n")
            f.write("dsm_temp: 1000.0\n")
            f.write("dsm_discount: 1.0\n")
            f.write("dsm_elbo: true\n")

    logger.info(f"Saved detailed summary to {summary_path}")

    try:
        # Skip creating charts if no recommendations
        if not recommendations:
            logger.warning("No valid distribution recommendations to plot")
            return recommendations

        # Create a bar chart of AIC/BIC scores
        plt.figure(figsize=(12, 6 * len(recommendations)))

        for i, (event_key, rec) in enumerate(recommendations.items()):
            # Skip if we don't have valid data for this event
            if not (aic_scores.get(event_key) and bic_scores.get(event_key)):
                logger.warning(f"Missing scores for {event_key}, skipping in chart")
                continue

            plt.subplot(len(recommendations), 1, i + 1)

            # Gather data for plotting
            distributions = list(aic_scores[event_key].keys())
            aic_values = [aic_scores[event_key][d] for d in distributions]
            bic_values = [bic_scores[event_key][d] for d in distributions]

            # Save detailed metrics as CSV for each event type
            metrics_df = pd.DataFrame(
                {"distribution": distributions, "aic": aic_values, "bic": bic_values}
            )
            metrics_csv_path = os.path.join(output_dir, f"{event_key}_full_metrics.csv")
            metrics_df.to_csv(metrics_csv_path, index=False)
            logger.debug(f"Saved detailed metrics to {metrics_csv_path}")

            # Save distribution parameters if fitted_models is provided
            if fitted_models is not None and event_key in fitted_models:
                try:
                    params_data = []
                    for dist_name in distributions:
                        if dist_name in fitted_models[event_key]:
                            model = fitted_models[event_key][dist_name]
                            if dist_name == "weibull":
                                params_data.append(
                                    {
                                        "distribution": dist_name,
                                        "shape": model.lambda_,
                                        "scale": model.rho_,
                                        "aic": aic_scores[event_key][dist_name],
                                        "bic": bic_scores[event_key][dist_name],
                                    }
                                )
                            elif dist_name == "lognormal":
                                params_data.append(
                                    {
                                        "distribution": dist_name,
                                        "mu": model.mu_,
                                        "sigma": model.sigma_,
                                        "aic": aic_scores[event_key][dist_name],
                                        "bic": bic_scores[event_key][dist_name],
                                    }
                                )
                            elif dist_name == "loglogistic":
                                params_data.append(
                                    {
                                        "distribution": dist_name,
                                        "alpha": model.alpha_,
                                        "beta": model.beta_,
                                        "aic": aic_scores[event_key][dist_name],
                                        "bic": bic_scores[event_key][dist_name],
                                    }
                                )

                    if params_data:
                        params_df = pd.DataFrame(params_data)
                        params_csv_path = os.path.join(
                            output_dir, f"{event_key}_distribution_parameters.csv"
                        )
                        params_df.to_csv(params_csv_path, index=False)
                        logger.debug(
                            f"Saved distribution parameters to {params_csv_path}"
                        )
                except Exception as e:
                    logger.error(
                        f"Error saving distribution parameters for {event_key}: {str(e)}"
                    )

            x = np.arange(len(distributions))
            width = 0.35

            # Create grouped bar chart
            plt.bar(x - width / 2, aic_values, width, label="AIC")
            plt.bar(x + width / 2, bic_values, width, label="BIC")

            # Add labels and formatting
            plt.title(f"AIC and BIC Scores for {event_key.replace('_', ' ').title()}")
            plt.ylabel("Score (lower is better)")
            plt.xticks(x, [d.capitalize() for d in distributions])
            plt.legend()

            # Add value labels
            for j, v in enumerate(aic_values):
                plt.text(j - width / 2, v + 5, f"{v:.1f}", ha="center")

            for j, v in enumerate(bic_values):
                plt.text(j + width / 2, v + 5, f"{v:.1f}", ha="center")

            # Highlight best model
            if prefer_metric.lower() == "aic":
                best_idx = distributions.index(rec["aic_best"])
            else:  # BIC
                best_idx = distributions.index(rec["bic_best"])

            plt.gca().get_xticklabels()[best_idx].set_color("green")
            plt.gca().get_xticklabels()[best_idx].set_weight("bold")

        plt.tight_layout()
        bar_path = os.path.join(output_dir, "aic_bic_comparison.png")
        plt.savefig(bar_path)
        plt.close()
        logger.info(f"Saved AIC/BIC comparison chart to {bar_path}")
    except Exception as e:
        logger.error(f"Error creating comparison chart: {str(e)}")
        plt.close()  # Ensure any open figure is closed

    # Save survival function data for all distributions and event types (for external plotting)
    if fitted_models is not None and durations is not None:
        for i, (event_key, models) in enumerate(fitted_models.items()):
            try:
                if i < len(durations) and len(durations[i]) > 0:
                    # Safety check for empty arrays
                    if np.any(durations[i] > 0):
                        max_time = max(durations[i]) * 1.1
                    else:
                        max_time = 100  # Default if all values are 0 or negative

                    time_points = np.linspace(0, max_time, 200)
                    survival_data = {"time": time_points}

                    # Add empirical survival function (Kaplan-Meier)
                    kmf = KaplanMeierFitter()
                    kmf.fit(durations[i], event_observed=np.ones(len(durations[i])))
                    survival_data["kaplan_meier"] = kmf.survival_function_at_times(
                        time_points
                    )

                    # Add parametric survival functions
                    for dist_name, model in models.items():
                        survival_data[dist_name] = model.survival_function_at_times(
                            time_points
                        )

                    # Create DataFrame and save to CSV
                    survival_df = pd.DataFrame(survival_data)
                    survival_csv_path = os.path.join(
                        output_dir, f"{event_key}_survival_functions.csv"
                    )
                    survival_df.to_csv(survival_csv_path, index=False)
                    logger.debug(f"Saved survival function data to {survival_csv_path}")
            except Exception as e:
                logger.error(
                    f"Error saving survival function data for {event_key}: {str(e)}"
                )

    return recommendations


def create_dsm_config(
    recommendations: Dict, output_dir: str, dataset_name: str
) -> Dict:
    """
    Create a DSM configuration file based on recommendations.

    Args:
        recommendations: Dictionary of recommendations
        output_dir: Directory to save config
        dataset_name: Name of the dataset

    Returns:
        dict: The created configuration
    """
    # Count distribution recommendations
    rec_dists = [r["recommended_distribution"] for r in recommendations.values()]
    most_common = max(set(rec_dists), key=rec_dists.count)

    # Create DSM config
    config = {
        "dsm_distribution": most_common,
        "dsm_num_mixtures": 4,
        "dsm_temp": 1000.0,
        "dsm_discount": 1.0,
        "dsm_elbo": True,
        "dsm_bias": True,
        "dsm_batch_norm": True,
        "dsm_hidden_dropout_prob": 0.1,
        "analysis_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }

    # Save configuration
    config_path = os.path.join(output_dir, f"dsm_config_{dataset_name}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved DSM configuration to {config_path}")
    logger.info(f"Recommended distribution: {most_common}")

    return config
