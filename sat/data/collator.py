"""Data Collators for the SAT model."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from typing import List, Dict, Any

from transformers import DefaultDataCollator


class DefaultSATDataCollator(DefaultDataCollator):
    """Enhanced data collator for the SAT model with optimized device transfer.

    This collator places all tensors onto the device with a single operation to minimize
    CPU-GPU transfer overhead and optionally can use non-blocking transfers for improved
    performance with pinned memory.
    """

    def __init__(self, device="cuda"):
        super().__init__(return_tensors="pt")
        self.device = device

    def __call__(
        self, features: List[Dict[str, Any]], return_tensors=None
    ) -> Dict[str, Any]:
        # Process features with parent class first
        inputs = super(DefaultSATDataCollator, self).__call__(features, return_tensors)

        # Optimize device transfer: Move all tensors to device at once with non-blocking transfer
        for key, value in inputs.items():
            if hasattr(value, "to"):
                # Use non-blocking transfer when possible (with pinned memory)
                inputs[key] = value.to(self.device, non_blocking=True)

        return inputs
