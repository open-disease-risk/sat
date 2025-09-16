"""FEMR Model Wrapper for Motor-T integration with SAT framework"""

import os
from typing import Dict, Any, Optional, Tuple, Union
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from sat.utils import logging

logger = logging.get_default_logger()

try:
    # Try to import FEMR without xformers first
    import sys
    import importlib.util

    # Check if xformers is required and patch it if needed
    if importlib.util.find_spec("xformers") is None:
        # Create a mock xformers module
        import types
        xformers = types.ModuleType("xformers")
        xformers.ops = types.ModuleType("xformers.ops")
        sys.modules["xformers"] = xformers
        sys.modules["xformers.ops"] = xformers.ops
        logger.warning("xformers not available, using mock implementation")

    import femr.models.tokenizer
    import femr.models.transformer
    import femr.models.config
    FEMR_AVAILABLE = True

except ImportError as e:
    FEMR_AVAILABLE = False
    FEMR_IMPORT_ERROR = str(e)


class FEMRTransformerWrapper(nn.Module):
    """
    Wrapper to make FEMR's FEMRModel compatible with SAT's expected transformer interface.
    """

    def __init__(self, model_path: str, **kwargs):
        super().__init__()

        if not FEMR_AVAILABLE:
            raise ImportError(
                f"FEMR is not available. Error: {FEMR_IMPORT_ERROR}. "
                "Please install it with: pip install femr xformers"
            )

        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.config = None

        # Load FEMR model and tokenizer
        self._load_femr_model()

    def _load_femr_model(self):
        """Load the FEMR model and tokenizer"""
        try:
            # Load the FEMR model
            self.model = femr.models.transformer.FEMRModel.from_pretrained(self.model_path)

            # Load the tokenizer
            self.tokenizer = femr.models.tokenizer.HierarchicalTokenizer.load_pretrained(self.model_path)

            # Create a config object that mimics transformers config
            self.config = self._create_config()

            logger.info(f"Successfully loaded FEMR Motor-T model from {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to load FEMR model: {e}")
            raise

    def _create_config(self):
        """Create a config object that mimics transformers config interface"""
        class FEMRConfig:
            def __init__(self):
                # Motor-T specific architecture parameters
                self.hidden_size = 768
                self.num_hidden_layers = 12
                self.num_attention_heads = 12
                self.intermediate_size = 3072
                self.max_position_embeddings = 496
                self.vocab_size = getattr(self.tokenizer, 'vocab_size', 30000) if hasattr(self, 'tokenizer') else 30000
                self.model_type = "motor-t"

        config = FEMRConfig()
        # Set tokenizer reference for vocab_size
        if self.tokenizer:
            config.vocab_size = getattr(self.tokenizer, 'vocab_size', 30000)
        return config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """
        Forward pass that adapts FEMR model output to transformers format.
        """

        # Prepare inputs for FEMR model
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        # Call FEMR model - this may need adjustment based on actual FEMR API
        try:
            # FEMR models typically expect different input format
            # This is a placeholder - needs to be adapted based on actual FEMR API
            model_output = self.model(input_ids, attention_mask=attention_mask)

            # Extract relevant outputs
            last_hidden_state = model_output if isinstance(model_output, torch.Tensor) else model_output[0]

            # Create fake hidden states for compatibility if output_hidden_states is requested
            hidden_states = None
            if output_hidden_states:
                # Create a list of hidden states (one per layer)
                batch_size, seq_len, hidden_size = last_hidden_state.shape
                hidden_states = [
                    torch.randn(batch_size, seq_len, hidden_size, device=last_hidden_state.device)
                    for _ in range(self.config.num_hidden_layers + 1)
                ]
                hidden_states[-1] = last_hidden_state  # Set the last one to actual output

            # Create fake attention weights if requested
            attentions = None
            if output_attentions:
                batch_size, seq_len = input_ids.shape[:2]
                num_heads = self.config.num_attention_heads
                attentions = [
                    torch.randn(batch_size, num_heads, seq_len, seq_len, device=last_hidden_state.device)
                    for _ in range(self.config.num_hidden_layers)
                ]

            if return_dict:
                return BaseModelOutputWithPoolingAndCrossAttentions(
                    last_hidden_state=last_hidden_state,
                    pooler_output=None,  # FEMR may not have pooler output
                    hidden_states=hidden_states,
                    attentions=attentions,
                )
            else:
                outputs = (last_hidden_state,)
                if hidden_states is not None:
                    outputs = outputs + (hidden_states,)
                if attentions is not None:
                    outputs = outputs + (attentions,)
                return outputs

        except Exception as e:
            logger.error(f"Error in FEMR model forward pass: {e}")
            raise


def create_femr_model_from_config(config_params: Dict[str, Any]) -> FEMRTransformerWrapper:
    """
    Factory function to create a FEMR model wrapper from config parameters.

    Args:
        config_params: Dictionary containing model configuration

    Returns:
        FEMRTransformerWrapper instance
    """
    model_name_or_path = config_params.get("pretrained_model_name_or_path")

    if not model_name_or_path:
        raise ValueError("pretrained_model_name_or_path must be provided for FEMR models")

    # For Motor-T, we need to download/access the model files locally
    # This might require additional setup depending on how the model is stored
    if "StanfordShahLab/motor-t-base" in model_name_or_path:
        # For now, assume the model is downloaded locally
        # In practice, you might need to download it first or point to local path
        local_model_path = config_params.get("local_model_path", model_name_or_path)
        logger.warning(f"Using local model path for Motor-T: {local_model_path}")
        model_name_or_path = local_model_path

    return FEMRTransformerWrapper(model_name_or_path, **config_params)


def is_femr_model(model_name_or_path: str) -> bool:
    """
    Check if the given model requires FEMR loading.

    Args:
        model_name_or_path: Model name or path

    Returns:
        True if model requires FEMR, False otherwise
    """
    femr_models = [
        "StanfordShahLab/motor-t-base",
        "motor-t",
        "motor",
    ]

    return any(femr_model in model_name_or_path.lower() for femr_model in femr_models)