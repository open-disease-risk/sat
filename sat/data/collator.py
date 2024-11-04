""" Data Collators for the SAT model.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from typing import List, Dict, Any

from transformers import DefaultDataCollator


class DefaultSATDataCollator(DefaultDataCollator):
    """Default data collator for the SAT model. It places the labels onto the
    device as well.
    """

    def __init__(self, device="cuda"):
        super().__init__(return_tensors="pt")
        self.device = device

    def __call__(
        self, features: List[Dict[str, Any]], return_tensors=None
    ) -> Dict[str, Any]:
        inputs = super(DefaultSATDataCollator, self).__call__(features, return_tensors)

        # Move labels to the specified device
        if "labels" in inputs:
            inputs["labels"] = inputs["labels"].to(self.device)

        return inputs
