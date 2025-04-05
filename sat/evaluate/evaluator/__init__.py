"""Brier Score Metric"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"


try:
    import transformers

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from typing import Dict, List

from evaluate.evaluator.base import Evaluator

from sat.evaluate.evaluator.survival_analysis import SurvivalAnalysisEvaluator

SUPPORTED_EVALUATOR_TASKS = {
    "survival-analysis": {
        "implementation": SurvivalAnalysisEvaluator,
        "default_metric_name": "./sat/evaluate/concordance_index_ipcw",
        "num_threads": -1,
        "size": None,
    },
}


def get_supported_tasks() -> List[str]:
    """
    Returns a list of supported task strings.
    """
    return list(SUPPORTED_EVALUATOR_TASKS.keys())


def check_task(task: str) -> Dict:
    """
    Checks an incoming task string, to validate it's correct and returns the default Evaluator class and default metric
    name. It first performs a check to validata that the string is a valid `Pipeline` task, then it checks if it's a
    valid `Evaluator` task. `Evaluator` tasks are a substet of `Pipeline` tasks.
    Args:
        task (`str`):
            The task defining which evaluator will be returned. Currently accepted tasks are:
            - `"survival-analysis"`
    Returns:
        task_defaults: `dict`, contains the implementasion class of a give Evaluator and the default metric name.
    """
    if task in SUPPORTED_EVALUATOR_TASKS.keys():
        return SUPPORTED_EVALUATOR_TASKS[task]
    raise KeyError(
        f"Unknown task {task}, available tasks are: {get_supported_tasks()}."
    )


def evaluator(
    task: str = None,
    num_threads: int = -1,
    size: int = None,
) -> Evaluator:
    """
    Utility factory method to build an [`Evaluator`].
    Evaluators encapsulate a task and a default metric name. They leverage `pipeline` functionalify from `transformers`
    to simplify the evaluation of multiple combinations of models, datasets and metrics for a given task.
    Args:
        task (`str`):
            The task defining which evaluator will be returned. Currently accepted tasks are:
            - `"survival_analysis"`: will return a [`SurvivalAnalysisEvaluator`].
    Returns:
        [`Evaluator`]: An evaluator suitable for the task.
    Examples:
    ```python
    >>> from evaluate import evaluator
    >>> evaluator("survival_analysis")
    ```"""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "If you want to use the `Evaluator` you need `transformers`. Run `pip install evaluate[transformers]`."
        )
    targeted_task = check_task(task)
    evaluator_class = targeted_task["implementation"]
    default_metric_name = targeted_task["default_metric_name"]

    return evaluator_class(
        task=task,
        default_metric_name=default_metric_name,
        num_threads=num_threads,
        size=size,
    )
