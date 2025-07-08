"""
Example script showing how to use the SAT Exploratory Data Analysis (EDA) framework.

This script demonstrates how to perform EDA on survival analysis datasets using the SAT
framework, including CSV export for external visualization. The EDA framework uses
Polars for high-performance data processing.
"""

import json
import os

from sat.eda import run_eda


def load_and_display_results(output_dir: str) -> None:
    """
    Load and display key results from an EDA run.

    Args:
        output_dir: Directory containing EDA results
    """
    # Load summary results
    summary_path = os.path.join(output_dir, "eda_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

        print("\n=== EDA Summary ===")

        # Display distribution analysis results if available
        if "distribution" in summary:
            print("\nDistribution Analysis:")
            for event_type, recommendations in summary["distribution"].items():
                if (
                    isinstance(recommendations, dict)
                    and "recommended_distributions" in recommendations
                ):
                    print(f"  Event: {event_type}")
                    for i, rec in enumerate(
                        recommendations["recommended_distributions"]
                    ):
                        print(
                            f"    {i+1}. {rec['distribution']} (AIC: {rec['aic']:.2f}, BIC: {rec['bic']:.2f})"
                        )

        # Display censoring analysis results if available
        if "censoring" in summary:
            print("\nCensoring Analysis:")
            for event_type, results in summary["censoring"].items():
                if event_type != "competing_risks":
                    if isinstance(results, dict) and "censoring_rate" in results:
                        print(f"  Event: {event_type}")
                        print(f"    Censoring rate: {results['censoring_rate']:.2f}")
                        if "informative_censoring_p_value" in results:
                            informative = (
                                results["informative_censoring_p_value"] < 0.05
                            )
                            print(
                                f"    Informative censoring: {'Yes' if informative else 'No'}"
                            )

        # Display covariate analysis results if available
        if "covariates" in summary and isinstance(summary["covariates"], dict):
            print("\nCovariate Analysis:")
            for event_type, results in summary["covariates"].items():
                if event_type != "distributions" and isinstance(results, dict):
                    if "feature_importance" in results:
                        print(f"  Event: {event_type}")
                        if (
                            isinstance(results["feature_importance"], dict)
                            and "top_features" in results["feature_importance"]
                        ):
                            print("    Top features:")
                            for feature, score in results["feature_importance"][
                                "top_features"
                            ].items():
                                if isinstance(score, (int, float)):
                                    print(f"      {feature}: {score:.4f}")
                                else:
                                    print(f"      {feature}: {score}")

    # Display some plots if available
    dist_plot_dir = os.path.join(output_dir, "distribution_analysis")
    cens_plot_dir = os.path.join(output_dir, "censoring_analysis")
    cov_plot_dir = os.path.join(output_dir, "covariate_analysis")

    print("\n=== EDA Outputs ===")
    print(f"Full results available in: {output_dir}")

    if os.path.exists(dist_plot_dir):
        print(f"Distribution analysis plots: {dist_plot_dir}")

    if os.path.exists(cens_plot_dir):
        print(f"Censoring analysis plots: {cens_plot_dir}")

    if os.path.exists(cov_plot_dir):
        print(f"Covariate analysis plots: {cov_plot_dir}")


def main():
    """Run EDA example with various configurations"""
    print("=== SAT Exploratory Data Analysis (EDA) Example ===")

    # Run EDA with default configuration
    print("\nRunning EDA on METABRIC dataset...")
    print("(Using Polars for high-performance data processing)")
    run_eda()

    # Output directory based on default configuration
    default_output_dir = "outputs/eda/metabric"

    # Load and display results
    if os.path.exists(default_output_dir):
        load_and_display_results(default_output_dir)

        # Show CSV files generated for external visualization
        csv_files = [f for f in os.listdir(default_output_dir) if f.endswith(".csv")]
        if csv_files:
            print("\n=== CSV Files for External Visualization ===")
            for csv_file in csv_files:
                print(f"- {csv_file}")
            print(
                "\nThese CSV files can be used with LaTeX/PGFPlots for publication-quality figures."
            )
    else:
        print(
            f"No results found in {default_output_dir}. Check for errors in the EDA run."
        )


if __name__ == "__main__":
    main()
