"""Visualize Optuna study results, including Pareto fronts for multi-objective optimization."""

import argparse
import os
from pathlib import Path

import optuna
import plotly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from sat.utils import logging

logger = logging.get_default_logger()


def load_study(storage_url, study_name):
    """
    Load an Optuna study from storage.

    Args:
        storage_url (str): The URL to the Optuna storage
        study_name (str): The name of the study

    Returns:
        The loaded study or None if loading fails
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        logger.info(f"Loaded study '{study_name}' with {len(study.trials)} trials")
        logger.info(f"Study direction(s): {study.directions}")
        return study
    except Exception as e:
        logger.error(f"Error loading study: {e}")
        return None


def is_multi_objective(study):
    """Check if a study is multi-objective."""
    return len(study.directions) > 1


def create_pareto_front_plot(study, output_dir, interactive=True):
    """
    Create and save a Pareto front visualization for multi-objective studies.

    Args:
        study (optuna.study.Study): The Optuna study
        output_dir (str): Directory to save the visualization
        interactive (bool): Whether to create an interactive visualization with plotly

    Returns:
        Path to saved visualization file or None if visualization fails
    """
    if not is_multi_objective(study):
        logger.warning(
            "This is not a multi-objective study. Skipping Pareto front visualization."
        )
        return None

    if len(study.directions) > 3:
        logger.warning("More than 3 objectives. Some plots may not be possible.")

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Get objectives for completed trials
        trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if len(trials) == 0:
            logger.warning("No completed trials found.")
            return None

        if interactive:
            # Create interactive visualizations with plotly
            if len(study.directions) == 2:
                fig = optuna.visualization.plot_pareto_front(study)
                output_file = os.path.join(output_dir, "pareto_front.html")
                fig.write_html(output_file)
                logger.info(f"Saved interactive Pareto front to {output_file}")

                # Also create contour plot
                contour_fig = optuna.visualization.plot_contour(study)
                contour_file = os.path.join(output_dir, "contour_plot.html")
                contour_fig.write_html(contour_file)
                logger.info(f"Saved contour plot to {contour_file}")

                return output_file

            elif len(study.directions) == 3:
                fig = optuna.visualization.plot_pareto_front(study)
                output_file = os.path.join(output_dir, "pareto_front_3d.html")
                fig.write_html(output_file)
                logger.info(f"Saved 3D Pareto front to {output_file}")
                return output_file

            else:
                # For more than 3 objectives, create pairwise plots
                for i in range(len(study.directions)):
                    for j in range(i + 1, len(study.directions)):
                        fig = optuna.visualization.plot_pareto_front(
                            study,
                            target_names=[f"Objective {i+1}", f"Objective {j+1}"],
                            include_dominated_trials=True,
                        )
                        output_file = os.path.join(
                            output_dir, f"pareto_front_{i+1}_vs_{j+1}.html"
                        )
                        fig.write_html(output_file)
                        logger.info(f"Saved pairwise Pareto front to {output_file}")

                return os.path.join(output_dir, "pareto_front_1_vs_2.html")
        else:
            # Create static visualizations with matplotlib
            values = np.array([t.values for t in trials])

            if len(study.directions) == 2:
                plt.figure(figsize=(10, 8))

                # Mark dominated and non-dominated solutions differently
                is_pareto = study._storage.get_pareto_front_trials(study._study_id)
                pareto_trial_ids = [t._trial_id for t in is_pareto]

                for i, trial in enumerate(trials):
                    if trial._trial_id in pareto_trial_ids:
                        plt.scatter(
                            values[i, 0],
                            values[i, 1],
                            c="red",
                            s=50,
                            label="Pareto optimal" if i == 0 else None,
                        )
                    else:
                        plt.scatter(
                            values[i, 0],
                            values[i, 1],
                            c="blue",
                            s=20,
                            alpha=0.5,
                            label="Dominated" if i == 0 else None,
                        )

                # Add trial numbers
                for i, trial in enumerate(trials):
                    plt.annotate(
                        str(trial.number), (values[i, 0], values[i, 1]), fontsize=8
                    )

                # Set axis labels based on study direction
                xlabel = (
                    "Minimize"
                    if study.directions[0] == optuna.study.StudyDirection.MINIMIZE
                    else "Maximize"
                )
                ylabel = (
                    "Minimize"
                    if study.directions[1] == optuna.study.StudyDirection.MINIMIZE
                    else "Maximize"
                )

                plt.xlabel(f"Objective 1 ({xlabel})")
                plt.ylabel(f"Objective 2 ({ylabel})")
                plt.title("Pareto Front")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.7)

                output_file = os.path.join(output_dir, "pareto_front.png")
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                plt.close()
                logger.info(f"Saved static Pareto front to {output_file}")
                return output_file

            elif len(study.directions) == 3:
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection="3d")

                # Mark dominated and non-dominated solutions differently
                is_pareto = study._storage.get_pareto_front_trials(study._study_id)
                pareto_trial_ids = [t._trial_id for t in is_pareto]

                for i, trial in enumerate(trials):
                    if trial._trial_id in pareto_trial_ids:
                        ax.scatter(
                            values[i, 0],
                            values[i, 1],
                            values[i, 2],
                            c="red",
                            s=50,
                            label="Pareto optimal" if i == 0 else None,
                        )
                    else:
                        ax.scatter(
                            values[i, 0],
                            values[i, 1],
                            values[i, 2],
                            c="blue",
                            s=20,
                            alpha=0.5,
                            label="Dominated" if i == 0 else None,
                        )

                # Set axis labels based on study direction
                xlabel = (
                    "Minimize"
                    if study.directions[0] == optuna.study.StudyDirection.MINIMIZE
                    else "Maximize"
                )
                ylabel = (
                    "Minimize"
                    if study.directions[1] == optuna.study.StudyDirection.MINIMIZE
                    else "Maximize"
                )
                zlabel = (
                    "Minimize"
                    if study.directions[2] == optuna.study.StudyDirection.MINIMIZE
                    else "Maximize"
                )

                ax.set_xlabel(f"Objective 1 ({xlabel})")
                ax.set_ylabel(f"Objective 2 ({ylabel})")
                ax.set_zlabel(f"Objective 3 ({zlabel})")
                ax.set_title("3D Pareto Front")
                plt.legend()

                output_file = os.path.join(output_dir, "pareto_front_3d.png")
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                plt.close()
                logger.info(f"Saved 3D Pareto front to {output_file}")
                return output_file

            else:
                # For more than 3 objectives, create pairwise plots
                for i in range(len(study.directions)):
                    for j in range(i + 1, len(study.directions)):
                        plt.figure(figsize=(10, 8))

                        for k, trial in enumerate(trials):
                            plt.scatter(values[k, i], values[k, j])

                        # Set axis labels based on study direction
                        xlabel = (
                            "Minimize"
                            if study.directions[i]
                            == optuna.study.StudyDirection.MINIMIZE
                            else "Maximize"
                        )
                        ylabel = (
                            "Minimize"
                            if study.directions[j]
                            == optuna.study.StudyDirection.MINIMIZE
                            else "Maximize"
                        )

                        plt.xlabel(f"Objective {i+1} ({xlabel})")
                        plt.ylabel(f"Objective {j+1} ({ylabel})")
                        plt.title(f"Pairwise Objectives: {i+1} vs {j+1}")
                        plt.grid(True, linestyle="--", alpha=0.7)

                        output_file = os.path.join(
                            output_dir, f"pareto_front_{i+1}_vs_{j+1}.png"
                        )
                        plt.savefig(output_file, dpi=300, bbox_inches="tight")
                        plt.close()
                        logger.info(f"Saved pairwise Pareto front to {output_file}")

                return os.path.join(output_dir, "pareto_front_1_vs_2.png")

    except Exception as e:
        logger.error(f"Error creating Pareto front visualization: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def generate_pareto_solutions_table(study, output_dir):
    """
    Generate a table of Pareto-optimal solutions for multi-objective studies.

    Args:
        study (optuna.study.Study): The Optuna study
        output_dir (str): Directory to save the table

    Returns:
        DataFrame of Pareto-optimal solutions
    """
    if not is_multi_objective(study):
        logger.warning("This is not a multi-objective study. Skipping Pareto table.")
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Get Pareto-optimal trials
        pareto_trials = study.best_trials

        if len(pareto_trials) == 0:
            logger.warning("No Pareto-optimal trials found.")
            return None

        # Create DataFrame with trial info
        data = []
        for trial in pareto_trials:
            row = {
                "trial_number": trial.number,
                **{f"objective_{i+1}": value for i, value in enumerate(trial.values)},
                **trial.params,
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Save to CSV
        output_file = os.path.join(output_dir, "pareto_solutions.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Saved Pareto solutions table to {output_file}")

        return df

    except Exception as e:
        logger.error(f"Error generating Pareto solutions table: {e}")
        return None


def create_optimization_history_plot(study, output_dir, interactive=True):
    """
    Create and save optimization history visualization.

    Args:
        study (optuna.study.Study): The Optuna study
        output_dir (str): Directory to save the visualization
        interactive (bool): Whether to create an interactive visualization with plotly

    Returns:
        Path to saved visualization file
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        if interactive:
            # Create interactive visualization with plotly
            if is_multi_objective(study):
                # For multi-objective, create history plot for each objective
                for i in range(len(study.directions)):
                    fig = optuna.visualization.plot_optimization_history(
                        study,
                        target=lambda t: t.values[i] if t.values else None,
                        target_name=f"Objective {i+1}",
                    )
                    output_file = os.path.join(
                        output_dir, f"optimization_history_obj{i+1}.html"
                    )
                    fig.write_html(output_file)
                    logger.info(
                        f"Saved optimization history for objective {i+1} to {output_file}"
                    )
            else:
                # For single-objective, create standard history plot
                fig = optuna.visualization.plot_optimization_history(study)
                output_file = os.path.join(output_dir, "optimization_history.html")
                fig.write_html(output_file)
                logger.info(f"Saved optimization history to {output_file}")

            return output_file
        else:
            # Create static visualization with matplotlib
            trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]

            if len(trials) == 0:
                logger.warning("No completed trials found.")
                return None

            if is_multi_objective(study):
                # For multi-objective, create history plot for each objective
                for i in range(len(study.directions)):
                    plt.figure(figsize=(12, 8))

                    trial_numbers = [t.number for t in trials]
                    values = [t.values[i] if t.values else float("nan") for t in trials]

                    plt.plot(trial_numbers, values, "o-", alpha=0.6)

                    # Plot best value so far
                    best_values = []
                    direction = study.directions[i]
                    compare = (
                        min
                        if direction == optuna.study.StudyDirection.MINIMIZE
                        else max
                    )

                    best_so_far = (
                        float("inf")
                        if direction == optuna.study.StudyDirection.MINIMIZE
                        else float("-inf")
                    )
                    for val in values:
                        if not np.isnan(val):
                            best_so_far = compare(best_so_far, val)
                        best_values.append(best_so_far)

                    plt.plot(
                        trial_numbers,
                        best_values,
                        "r-",
                        linewidth=2,
                        label="Best value",
                    )

                    plt.xlabel("Trial Number")
                    plt.ylabel(f"Objective {i+1} Value")
                    plt.title(f"Optimization History for Objective {i+1}")
                    plt.legend()
                    plt.grid(True, linestyle="--", alpha=0.7)

                    output_file = os.path.join(
                        output_dir, f"optimization_history_obj{i+1}.png"
                    )
                    plt.savefig(output_file, dpi=300, bbox_inches="tight")
                    plt.close()
                    logger.info(
                        f"Saved optimization history for objective {i+1} to {output_file}"
                    )
            else:
                # For single-objective, create standard history plot
                plt.figure(figsize=(12, 8))

                trial_numbers = [t.number for t in trials]
                values = [t.value for t in trials]

                plt.plot(trial_numbers, values, "o-", alpha=0.6)

                # Plot best value so far
                best_values = []
                direction = study.direction
                compare = (
                    min if direction == optuna.study.StudyDirection.MINIMIZE else max
                )

                best_so_far = (
                    float("inf")
                    if direction == optuna.study.StudyDirection.MINIMIZE
                    else float("-inf")
                )
                for val in values:
                    if val is not None:
                        best_so_far = compare(best_so_far, val)
                    best_values.append(best_so_far)

                plt.plot(
                    trial_numbers, best_values, "r-", linewidth=2, label="Best value"
                )

                plt.xlabel("Trial Number")
                plt.ylabel("Objective Value")
                plt.title("Optimization History")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.7)

                output_file = os.path.join(output_dir, "optimization_history.png")
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                plt.close()
                logger.info(f"Saved optimization history to {output_file}")

            return output_file

    except Exception as e:
        logger.error(f"Error creating optimization history: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def create_parameter_importance_plot(study, output_dir, interactive=True):
    """
    Create and save parameter importance visualization.

    Args:
        study (optuna.study.Study): The Optuna study
        output_dir (str): Directory to save the visualization
        interactive (bool): Whether to create an interactive visualization with plotly

    Returns:
        Path to saved visualization file
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        if interactive:
            # Create interactive visualization with plotly
            if is_multi_objective(study):
                # For multi-objective, create importance plot for each objective
                for i in range(len(study.directions)):
                    try:
                        fig = optuna.visualization.plot_param_importances(
                            study,
                            target=lambda t: t.values[i] if t.values else None,
                            target_name=f"Objective {i+1}",
                        )
                        output_file = os.path.join(
                            output_dir, f"param_importance_obj{i+1}.html"
                        )
                        fig.write_html(output_file)
                        logger.info(
                            f"Saved parameter importance for objective {i+1} to {output_file}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not create parameter importance plot for objective {i+1}: {e}"
                        )
            else:
                # For single-objective, create standard importance plot
                try:
                    fig = optuna.visualization.plot_param_importances(study)
                    output_file = os.path.join(output_dir, "param_importance.html")
                    fig.write_html(output_file)
                    logger.info(f"Saved parameter importance to {output_file}")
                except Exception as e:
                    logger.warning(f"Could not create parameter importance plot: {e}")

            return output_file
        else:
            # Static visualizations not implemented for parameter importance
            # as they require complex calculation of feature importance
            logger.warning(
                "Static parameter importance plots not implemented. Use interactive=True"
            )
            return None

    except Exception as e:
        logger.error(f"Error creating parameter importance plot: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Visualize Optuna study results")
    parser.add_argument(
        "--storage",
        required=True,
        help="SQLite storage URL (e.g., sqlite:///data/optuna/studies.db)",
    )
    parser.add_argument("--study", required=True, help="Name of the study")
    parser.add_argument(
        "--output-dir",
        default="data/optuna/visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Create interactive visualizations with plotly",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Create static visualizations with matplotlib instead of interactive",
    )

    args = parser.parse_args()

    # Override interactive if static is requested
    if args.static:
        args.interactive = False

    # Load study
    study = load_study(args.storage, args.study)
    if not study:
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # If multi-objective, create Pareto front visualization
    if is_multi_objective(study):
        logger.info("Detected multi-objective study")

        # Create Pareto front visualization
        pareto_plot = create_pareto_front_plot(study, args.output_dir, args.interactive)

        # Generate table of Pareto-optimal solutions
        pareto_df = generate_pareto_solutions_table(study, args.output_dir)
        if pareto_df is not None:
            try:
                display(pareto_df)
            except:
                pass
    else:
        logger.info("Detected single-objective study")

    # Create optimization history plot
    history_plot = create_optimization_history_plot(
        study, args.output_dir, args.interactive
    )

    # Create parameter importance plot
    importance_plot = create_parameter_importance_plot(
        study, args.output_dir, args.interactive
    )

    logger.info(f"All visualizations saved to {args.output_dir}")

    return 0


if __name__ == "__main__":
    main()
