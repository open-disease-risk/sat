"""Covariate analysis for survival data.

This module provides functions to analyze covariates in survival data,
examining their relationships with survival outcomes and identifying
potential predictors of survival.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

# Apply the lifelines patch to fix scipy.integrate.trapz import issue
try:
    import sys
    import os

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )
    from fix_lifelines_import import apply_lifelines_patch

    apply_lifelines_patch()
except ImportError:
    pass

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression

from sat.utils import logging

logger = logging.get_default_logger()


def analyze_covariate_distributions(
    covariates: pd.DataFrame,
    durations: Optional[np.ndarray] = None,
    events: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Analyze the distributions of covariates and their relationships with survival.

    Args:
        covariates: DataFrame of covariates
        durations: Optional array of event/censoring times
        events: Optional binary indicators (1=event, 0=censored)
        output_dir: Directory to save plots and results

    Returns:
        Dict: Analysis results
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = {
        "n_samples": len(covariates),
        "n_features": len(covariates.columns),
        "feature_stats": {},
        "correlations": {},
    }

    # Basic statistics for each covariate
    stats = covariates.describe(include="all").transpose()
    stats_dict = stats.to_dict()
    results["feature_stats"] = stats_dict

    # Analyze distributions
    if output_dir:
        # For continuous variables, create histograms
        for col in covariates.select_dtypes(include=["float", "int"]).columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(covariates[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"dist_{col}.png"))
            plt.close()

            # QQ plot to check normality
            plt.figure(figsize=(8, 8))
            from scipy import stats

            stats.probplot(covariates[col].dropna(), plot=plt)
            plt.title(f"Q-Q Plot of {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"qq_{col}.png"))
            plt.close()

        # For categorical variables, create bar plots
        for col in covariates.select_dtypes(include=["object", "category"]).columns:
            plt.figure(figsize=(10, 6))
            value_counts = covariates[col].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"dist_{col}.png"))
            plt.close()

    # If survival data is provided, analyze relationships with survival
    if durations is not None and events is not None:
        # Create full dataframe with durations and events
        df = pd.DataFrame({"duration": durations, "event": events})
        for col in covariates.columns:
            df[col] = covariates[col].values

        # Correlation with survival time
        for col in covariates.select_dtypes(include=["float", "int"]).columns:
            # Only consider events, not censored observations
            event_mask = events == 1
            if np.sum(event_mask) > 5:  # Only if we have enough events
                event_durations = durations[event_mask]
                event_values = covariates.loc[event_mask, col].values

                # Drop NaNs
                valid_mask = ~np.isnan(event_values)
                if np.sum(valid_mask) > 5:  # Only if we have enough valid values
                    corr_spearman, p_spearman = spearmanr(
                        event_durations[valid_mask], event_values[valid_mask]
                    )
                    corr_pearson, p_pearson = pearsonr(
                        event_durations[valid_mask], event_values[valid_mask]
                    )

                    results["correlations"][col] = {
                        "spearman": corr_spearman,
                        "spearman_p": p_spearman,
                        "pearson": corr_pearson,
                        "pearson_p": p_pearson,
                        "significant": p_spearman < 0.05 or p_pearson < 0.05,
                    }

                    if p_spearman < 0.05 or p_pearson < 0.05:
                        logger.info(
                            f"{col} is significantly correlated with survival time"
                        )

                    # Scatter plot for continuous variables vs. time
                    if output_dir:
                        plt.figure(figsize=(10, 6))
                        plt.scatter(
                            event_values[valid_mask],
                            event_durations[valid_mask],
                            alpha=0.5,
                        )
                        plt.xlabel(col)
                        plt.ylabel("Time to Event")
                        plt.title(
                            f"{col} vs. Survival Time (ρ={corr_spearman:.3f}, p={p_spearman:.3f})"
                        )
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"surv_scatter_{col}.png"))
                        plt.close()

        # For categorical variables, use log-rank test
        for col in covariates.select_dtypes(include=["object", "category"]).columns:
            categories = covariates[col].unique()
            if (
                len(categories) > 1 and len(categories) <= 10
            ):  # Only for reasonable number of categories
                # Run log-rank test
                results_dict = {}

                # Create Kaplan-Meier plots
                if output_dir:
                    plt.figure(figsize=(10, 6))

                # Prepare for pairwise log-rank tests
                p_values = []

                for i, cat1 in enumerate(categories):
                    mask1 = covariates[col] == cat1
                    if np.sum(mask1) > 0:  # Only if category has samples
                        durations1 = durations[mask1]
                        events1 = events[mask1]

                        # Add to KM plot
                        if output_dir:
                            kmf = KaplanMeierFitter()
                            kmf.fit(durations1, events1, label=str(cat1))
                            kmf.plot_survival_function()

                        # Pairwise log-rank tests
                        for cat2 in categories[i + 1 :]:
                            mask2 = covariates[col] == cat2
                            if np.sum(mask2) > 0:  # Only if category has samples
                                durations2 = durations[mask2]
                                events2 = events[mask2]

                                result = logrank_test(
                                    durations1, durations2, events1, events2
                                )
                                # Handle StatisticalResult object safely with comprehensive error handling
                                try:
                                    # First try to access p_value attribute
                                    if hasattr(result, "p_value"):
                                        p_value = float(result.p_value)
                                    # Then try subscripting
                                    elif (
                                        isinstance(result, (list, tuple))
                                        and len(result) > 1
                                    ):
                                        p_value = float(result[1])
                                    # Then try getting test statistic and p-value from result
                                    elif hasattr(result, "test_statistic") and hasattr(
                                        result, "p_value"
                                    ):
                                        p_value = float(result.p_value)
                                    # Final fallback
                                    else:
                                        # Use a safe default and log a warning
                                        p_value = 0.5  # Neutral value
                                        logger.warning(
                                            f"Could not extract p-value from logrank test result: {result}"
                                        )
                                except Exception as e:
                                    # Fallback if any error occurs during extraction
                                    p_value = 0.5  # Neutral value
                                    logger.warning(
                                        f"Error extracting p-value: {str(e)}"
                                    )

                                p_values.append(p_value)

                                pair_key = f"{cat1}_vs_{cat2}"
                                results_dict[pair_key] = {
                                    "p_value": p_value,
                                    "significant": p_value < 0.05,
                                }

                # Adjust for multiple comparisons
                if p_values:
                    from statsmodels.stats.multitest import multipletests

                    adjusted_p = multipletests(p_values, method="fdr_bh")[1]

                    # Update results with adjusted p-values
                    for (key, result_dict), adj_p in zip(
                        list(results_dict.items()), adjusted_p
                    ):
                        results_dict[key]["adjusted_p"] = adj_p
                        results_dict[key]["significant_adjusted"] = adj_p < 0.05

                results["correlations"][col] = {
                    "logrank_results": results_dict,
                    "significant": any(
                        v.get("significant", False) for v in results_dict.values()
                    ),
                    "significant_adjusted": any(
                        v.get("significant_adjusted", False)
                        for v in results_dict.values()
                    ),
                }

                if results["correlations"][col]["significant"]:
                    logger.info(
                        f"{col} shows significant survival differences between groups"
                    )

                # Finalize and save KM plot
                if output_dir:
                    plt.title(f"Kaplan-Meier Curves by {col}")
                    plt.xlabel("Time")
                    plt.ylabel("Survival Probability")
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"km_{col}.png"))
                    plt.close()

    # Feature correlation matrix
    numeric_covariates = covariates.select_dtypes(include=["float", "int"])
    if len(numeric_covariates.columns) > 1:
        corr_matrix = numeric_covariates.corr()
        results["feature_correlation_matrix"] = corr_matrix.to_dict()

        if output_dir:
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                mask=mask,
                vmin=-1,
                vmax=1,
            )
            plt.title("Feature Correlation Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "feature_correlation.png"))
            plt.close()

    return results


def analyze_feature_importance(
    covariates: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    output_dir: Optional[str] = None,
    export_csv: bool = True,
) -> Dict:
    """
    Analyze feature importance for survival prediction.

    Args:
        covariates: DataFrame of covariates
        durations: Array of event/censoring times
        events: Binary indicators (1=event, 0=censored)
        output_dir: Directory to save plots and results

    Returns:
        Dict: Feature importance analysis results
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = {
        "n_samples": len(covariates),
        "n_features": len(covariates.columns),
        "feature_importance": {},
    }

    # Create full dataset
    df = pd.DataFrame({"duration": durations, "event": events})
    for col in covariates.columns:
        df[col] = covariates[col].values

    # Only use numeric features for importance analysis
    numeric_cols = covariates.select_dtypes(include=["float", "int"]).columns.tolist()

    if numeric_cols:
        # 1. Cox PH model for feature importance
        try:
            cox_df = df.copy()

            # Create dummy variables for categorical features
            categorical_cols = covariates.select_dtypes(
                include=["object", "category"]
            ).columns
            if len(categorical_cols) > 0:
                dummy_df = pd.get_dummies(cox_df[categorical_cols], drop_first=True)
                cox_df = pd.concat(
                    [cox_df.drop(categorical_cols, axis=1), dummy_df], axis=1
                )

            # Drop any columns with NaNs
            cox_df = cox_df.dropna()

            if len(cox_df) > 10:  # Only if we have enough samples
                # Fit Cox model
                cph = CoxPHFitter()
                cph.fit(cox_df, duration_col="duration", event_col="event")

                # Extract feature importance
                cox_summary = cph.summary.copy()
                cox_importance = cox_summary[["coef", "exp(coef)", "p"]]
                results["feature_importance"]["cox"] = cox_importance.to_dict()

                # Save Cox model summary
                if output_dir:
                    cox_summary.to_csv(
                        os.path.join(output_dir, "cox_feature_importance.csv")
                    )

                    # Plot hazard ratios with confidence intervals
                    plt.figure(figsize=(12, max(6, len(cox_importance) * 0.3)))
                    cph.plot()
                    plt.title("Cox Proportional Hazards: Feature Effects")
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "cox_feature_effects.png"))
                    plt.close()

                # Log significant features
                significant_features = cox_summary[
                    cox_summary["p"] < 0.05
                ].index.tolist()
                results["significant_features"] = significant_features

                if significant_features:
                    logger.info(
                        f"Significant features from Cox model: {', '.join(significant_features)}"
                    )
        except Exception as e:
            logger.error(f"Error in Cox model fitting: {str(e)}")

        # 2. Mutual Information for feature importance
        try:
            # Only use samples with events for mutual information
            event_mask = events == 1
            if np.sum(event_mask) > 10:  # Only if we have enough events
                X = covariates.loc[event_mask, numeric_cols].values
                y = durations[event_mask]

                # Remove rows with NaNs
                valid_mask = ~np.isnan(X).any(axis=1)
                if np.sum(valid_mask) > 10:  # Only if we have enough valid samples
                    X_valid = X[valid_mask]
                    y_valid = y[valid_mask]

                    # Calculate mutual information
                    mi = mutual_info_regression(X_valid, y_valid)
                    mi_dict = {col: mi_val for col, mi_val in zip(numeric_cols, mi)}

                    # Sort by importance
                    sorted_mi = sorted(
                        mi_dict.items(), key=lambda x: x[1], reverse=True
                    )
                    results["feature_importance"]["mutual_info"] = dict(sorted_mi)

                    # Export mutual information to CSV
                    if output_dir and export_csv:
                        mi_df = pd.DataFrame(
                            [
                                {"feature": feature, "mutual_information": value}
                                for feature, value in sorted_mi
                            ]
                        )
                        mi_df.to_csv(
                            os.path.join(output_dir, "mutual_info_importance.csv"),
                            index=False,
                        )

                    # Plot mutual information
                    if output_dir:
                        plt.figure(figsize=(12, 8))
                        feat_names, mi_values = zip(*sorted_mi)
                        sns.barplot(x=list(mi_values), y=list(feat_names))
                        plt.title(
                            "Feature Importance: Mutual Information with Survival Time"
                        )
                        plt.xlabel("Mutual Information")
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(output_dir, "mutual_info_importance.png")
                        )
                        plt.close()
        except Exception as e:
            logger.error(f"Error in mutual information calculation: {str(e)}")

    # 3. Univariate log-rank tests for categorical features
    categorical_cols = covariates.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        categorical_importance = {}

        for col in categorical_cols:
            categories = covariates[col].unique()
            if (
                len(categories) > 1 and len(categories) <= 10
            ):  # Only for reasonable number of categories
                # Run log-rank test
                results_dict = {}
                p_values = []

                for i, cat1 in enumerate(categories):
                    mask1 = covariates[col] == cat1
                    if np.sum(mask1) > 0:
                        durations1 = durations[mask1]
                        events1 = events[mask1]

                        for cat2 in categories[i + 1 :]:
                            mask2 = covariates[col] == cat2
                            if np.sum(mask2) > 0:
                                durations2 = durations[mask2]
                                events2 = events[mask2]

                                result = logrank_test(
                                    durations1, durations2, events1, events2
                                )
                                # Handle StatisticalResult object safely
                                try:
                                    # First try to access p_value attribute
                                    if hasattr(result, "p_value"):
                                        p_value = float(result.p_value)
                                    # Then try subscripting
                                    elif (
                                        isinstance(result, (list, tuple))
                                        and len(result) > 1
                                    ):
                                        p_value = float(result[1])
                                    # Then try getting test statistic and p-value from result
                                    elif hasattr(result, "test_statistic") and hasattr(
                                        result, "p_value"
                                    ):
                                        p_value = float(result.p_value)
                                    # Final fallback
                                    else:
                                        # Use a safe default and log a warning
                                        p_value = 0.5  # Neutral value
                                        logger.warning(
                                            f"Could not extract p-value from logrank test result: {result}"
                                        )
                                except Exception as e:
                                    # Fallback if any error occurs during extraction
                                    p_value = 0.5  # Neutral value
                                    logger.warning(
                                        f"Error extracting p-value: {str(e)}"
                                    )

                                p_values.append(p_value)

                                pair_key = f"{cat1}_vs_{cat2}"
                                results_dict[pair_key] = {
                                    "p_value": p_value,
                                }

                # Calculate overall importance as -log(min(p_value))
                if p_values:
                    min_p = max(min(p_values), 1e-10)  # Avoid log(0)
                    importance = -np.log10(min_p)
                    categorical_importance[col] = importance

        # Sort by importance
        sorted_cat_imp = sorted(
            categorical_importance.items(), key=lambda x: x[1], reverse=True
        )
        results["feature_importance"]["categorical_logrank"] = dict(sorted_cat_imp)

        # Plot categorical importance
        if output_dir and categorical_importance:
            plt.figure(figsize=(12, 8))
            feat_names, imp_values = zip(*sorted_cat_imp)
            sns.barplot(x=list(imp_values), y=list(feat_names))
            plt.title(
                "Categorical Feature Importance: -log10(p-value) from Log-rank Tests"
            )
            plt.xlabel("-log10(p-value)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "categorical_importance.png"))
            plt.close()

    # Create combined feature importance ranking
    combined_importance = {}

    # Include Cox model p-values if available
    if "cox" in results["feature_importance"]:
        cox_data = results["feature_importance"]["cox"]
        if "p" in cox_data:
            for feature, p_value in cox_data["p"].items():
                if p_value < 0.05:  # Only include significant features
                    combined_importance[feature] = -np.log10(max(p_value, 1e-10))

    # Include mutual information if available
    if "mutual_info" in results["feature_importance"]:
        mi_data = results["feature_importance"]["mutual_info"]
        for feature, mi_value in mi_data.items():
            if feature in combined_importance:
                combined_importance[feature] += (
                    mi_value * 5
                )  # Scale MI to be comparable to -log(p)
            elif mi_value > 0:  # Only include features with positive MI
                combined_importance[feature] = mi_value * 5

    # Include categorical log-rank results if available
    if "categorical_logrank" in results["feature_importance"]:
        cat_data = results["feature_importance"]["categorical_logrank"]
        for feature, lr_value in cat_data.items():
            combined_importance[feature] = lr_value

    # Sort combined importance
    sorted_combined = sorted(
        combined_importance.items(), key=lambda x: x[1], reverse=True
    )
    results["feature_importance"]["combined"] = dict(sorted_combined)

    # Plot combined importance
    if output_dir and combined_importance:
        plt.figure(figsize=(12, 8))
        feat_names, imp_values = zip(*sorted_combined)
        sns.barplot(x=list(imp_values), y=list(feat_names))
        plt.title("Combined Feature Importance for Survival")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "combined_importance.png"))
        plt.close()

    return results


def identify_risk_factors(
    covariates: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Identify and quantify risk factors associated with survival outcomes.

    Args:
        covariates: DataFrame of covariates
        durations: Array of event/censoring times
        events: Binary indicators (1=event, 0=censored)
        output_dir: Directory to save plots and results

    Returns:
        Dict: Risk factor analysis results
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = {"risk_factors": {}, "protective_factors": {}, "risk_scores": {}}

    # Create full dataset
    df = pd.DataFrame({"duration": durations, "event": events})
    for col in covariates.columns:
        df[col] = covariates[col].values

    # Create dummy variables for categorical features
    categorical_cols = covariates.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        dummy_df = pd.get_dummies(df[categorical_cols], drop_first=True)
        df = pd.concat([df.drop(categorical_cols, axis=1), dummy_df], axis=1)

    # Drop any columns with NaNs and rows with NaN in duration or event
    df = df.dropna(subset=["duration", "event"])
    valid_cols = df.columns.drop(["duration", "event"]).tolist()
    df = df.dropna(subset=valid_cols)

    if len(df) > 10:  # Only proceed if we have enough samples
        try:
            # Fit Cox model
            cph = CoxPHFitter()
            cph.fit(df, duration_col="duration", event_col="event")

            # Get summary
            cox_summary = cph.summary.copy()

            # Identify risk factors (hazard ratio > 1, p < 0.05)
            risk_factors = cox_summary[
                (cox_summary["exp(coef)"] > 1) & (cox_summary["p"] < 0.05)
            ]

            # Identify protective factors (hazard ratio < 1, p < 0.05)
            protective_factors = cox_summary[
                (cox_summary["exp(coef)"] < 1) & (cox_summary["p"] < 0.05)
            ]

            # Sort by effect size
            risk_factors = risk_factors.sort_values("exp(coef)", ascending=False)
            protective_factors = protective_factors.sort_values(
                "exp(coef)", ascending=True
            )

            # Store in results
            results["risk_factors"] = risk_factors.to_dict()
            results["protective_factors"] = protective_factors.to_dict()

            # Export full Cox results to CSV for external use
            if output_dir:
                cox_summary.to_csv(os.path.join(output_dir, "cox_results.csv"))

                # Export separate CSVs for risk and protective factors
                if not risk_factors.empty:
                    risk_factors.to_csv(os.path.join(output_dir, "risk_factors.csv"))

                if not protective_factors.empty:
                    protective_factors.to_csv(
                        os.path.join(output_dir, "protective_factors.csv")
                    )

            # Log findings
            if len(risk_factors) > 0:
                logger.info(f"Identified {len(risk_factors)} significant risk factors")
                for idx, row in risk_factors.iterrows():
                    logger.info(
                        f"  {idx}: HR={row['exp(coef)']:.2f} ({row['exp(coef) lower 95%']:.2f}-{row['exp(coef) upper 95%']:.2f}), p={row['p']:.4f}"
                    )

            if len(protective_factors) > 0:
                logger.info(
                    f"Identified {len(protective_factors)} significant protective factors"
                )
                for idx, row in protective_factors.iterrows():
                    logger.info(
                        f"  {idx}: HR={row['exp(coef)']:.2f} ({row['exp(coef) lower 95%']:.2f}-{row['exp(coef) upper 95%']:.2f}), p={row['p']:.4f}"
                    )

            # Save factor plots
            if output_dir:
                # Plot hazard ratios for risk factors
                if len(risk_factors) > 0:
                    plt.figure(figsize=(12, max(6, len(risk_factors) * 0.4)))
                    # Plot function returns ax object, don't need to assign
                    cph.plot(columns=risk_factors.index.tolist())
                    plt.title("Significant Risk Factors (HR > 1)")
                    plt.grid(alpha=0.3)
                    plt.axvline(x=0, color="black", linestyle="--")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "risk_factors.png"))
                    plt.close()

                # Plot hazard ratios for protective factors
                if len(protective_factors) > 0:
                    plt.figure(figsize=(12, max(6, len(protective_factors) * 0.4)))
                    # Plot function returns ax object, don't need to assign
                    cph.plot(columns=protective_factors.index.tolist())
                    plt.title("Significant Protective Factors (HR < 1)")
                    plt.grid(alpha=0.3)
                    plt.axvline(x=0, color="black", linestyle="--")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "protective_factors.png"))
                    plt.close()

            # Calculate risk scores
            try:
                # Compute linear predictor scores
                risk_scores = cph.predict_log_partial_hazard(df)

                # Add to results
                results["risk_scores"] = {
                    "mean": risk_scores.mean(),
                    "std": risk_scores.std(),
                    "min": risk_scores.min(),
                    "max": risk_scores.max(),
                    "quartiles": [
                        risk_scores.quantile(0.25),
                        risk_scores.quantile(0.50),
                        risk_scores.quantile(0.75),
                    ],
                }

                # Plot risk score distribution
                if output_dir:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(risk_scores, kde=True)
                    plt.title("Distribution of Risk Scores")
                    plt.xlabel("Risk Score (Log Partial Hazard)")
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "risk_score_distribution.png"))
                    plt.close()

                    # Plot KM curves by risk score quartiles
                    plt.figure(figsize=(10, 6))

                    # Create quartile groups
                    df["risk_quartile"] = pd.qcut(
                        risk_scores,
                        4,
                        labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"],
                    )

                    for quartile in sorted(df["risk_quartile"].unique()):
                        mask = df["risk_quartile"] == quartile
                        if sum(mask) > 0:
                            kmf = KaplanMeierFitter()
                            kmf.fit(
                                df.loc[mask, "duration"],
                                df.loc[mask, "event"],
                                label=str(quartile),
                            )
                            kmf.plot_survival_function()

                    plt.title("Survival by Risk Score Quartiles")
                    plt.xlabel("Time")
                    plt.ylabel("Survival Probability")
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(output_dir, "survival_by_risk_quartile.png")
                    )
                    plt.close()
            except Exception as e:
                logger.error(f"Error calculating risk scores: {str(e)}")

        except Exception as e:
            logger.error(f"Error in Cox model fitting for risk factors: {str(e)}")
    else:
        logger.warning("Not enough samples with complete data to identify risk factors")

    return results
