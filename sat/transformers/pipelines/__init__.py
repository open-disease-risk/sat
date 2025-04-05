"""Initialize the SA pipeline."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from transformers.pipelines import PIPELINE_REGISTRY

from sat.models.tasks.heads import MTLForSurvival
from sat.transformers.pipelines.survival_analysis import SAPipeline

PIPELINE_REGISTRY.register_pipeline(
    "survival-analysis",
    pipeline_class=SAPipeline,
    pt_model=MTLForSurvival,
)
