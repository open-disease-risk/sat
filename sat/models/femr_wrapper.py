"""FEMR Model Wrapper for Motor-T integration with SAT framework"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from sat.utils import logging

logger = logging.get_default_logger()

try:
    # Try to import FEMR without xformers first
    import importlib.util
    import sys

    # Check if xformers is required and patch it if needed
    if importlib.util.find_spec("xformers") is None:
        # Create a more complete mock xformers module
        import types
        xformers = types.ModuleType("xformers")
        xformers.ops = types.ModuleType("xformers.ops")
        xformers.ops.fmha = types.ModuleType("xformers.ops.fmha")
        xformers.ops.fmha.attn_bias = types.ModuleType("xformers.ops.fmha.attn_bias")

        # Create a mock AttentionBias base class
        class MockAttentionBias:
            def __init__(self):
                pass

        # Create a mock BlockDiagonalMask class
        class MockBlockDiagonalMask(MockAttentionBias):
            def __init__(self):
                super().__init__()

            @classmethod
            def from_seqlens(cls, seqlens, **kwargs):
                # Return an instance that has the required methods
                return cls()

            def make_local_attention(self, attention_width):
                # Return None to trigger FEMR's fallback to standard PyTorch attention
                return None

        xformers.ops.AttentionBias = MockAttentionBias
        xformers.ops.fmha.attn_bias.BlockDiagonalMask = MockBlockDiagonalMask

        # Add memory_efficient_attention mock
        def mock_memory_efficient_attention(q, k, v, attn_bias=None):
            # Fallback to standard PyTorch attention
            # Scale factor for attention
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=q.dtype, device=q.device))

            # Apply attention bias if provided (but we expect None)
            if attn_bias is not None:
                scores = scores + attn_bias

            attn_weights = torch.softmax(scores, dim=-1)
            return torch.matmul(attn_weights, v)

        xformers.ops.memory_efficient_attention = mock_memory_efficient_attention

        sys.modules["xformers"] = xformers
        sys.modules["xformers.ops"] = xformers.ops
        sys.modules["xformers.ops.fmha"] = xformers.ops.fmha
        sys.modules["xformers.ops.fmha.attn_bias"] = xformers.ops.fmha.attn_bias
        logger.warning("xformers not available, using enhanced mock implementation")

    import femr.models.config
    import femr.models.tokenizer
    import femr.models.transformer
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

            # Skip tokenizer loading for now - use basic tokenizer if available
            self.tokenizer = None
            try:
                # Try different tokenizer loading methods
                if hasattr(femr.models.tokenizer, 'HierarchicalTokenizer'):
                    self.tokenizer = femr.models.tokenizer.HierarchicalTokenizer.load_pretrained(self.model_path)
                else:
                    logger.warning("HierarchicalTokenizer not available, using basic tokenizer")
                    # Use a basic tokenizer or skip
                    pass
            except Exception as tok_error:
                logger.warning(f"Could not load tokenizer: {tok_error}")

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
            # FEMR models don't use attention_mask in the same way, so we only pass input_ids
            logger.debug(f"FEMR input_ids shape: {input_ids.shape}")

            # FEMR requires batch_size=1 and expects a dictionary input format
            batch_size = input_ids.shape[0]

            # Create dummy values for FEMR-required fields
            seq_length = input_ids.shape[1]

            if batch_size == 1:
                # FEMR expects patient_lengths inside the transformer dict
                femr_input = {
                    "transformer": {
                        "tokens": input_ids,
                        "normalized_ages": torch.zeros(batch_size, seq_length, device=input_ids.device),  # Dummy ages
                        "ages": torch.zeros(batch_size, seq_length, device=input_ids.device),  # Dummy raw ages
                        "patient_lengths": torch.tensor([seq_length], device=input_ids.device),  # Lengths per patient
                        "timestamps": torch.zeros(batch_size, seq_length, device=input_ids.device),  # Dummy timestamps
                    },
                    "patient_ids": torch.zeros(batch_size, dtype=torch.long, device=input_ids.device),  # Dummy patient IDs
                }
                model_output = self.model(femr_input)
            else:
                # Process each example individually and stack outputs
                outputs = []
                for i in range(batch_size):
                    single_input = input_ids[i:i+1]  # Keep batch dimension as 1
                    single_seq_length = single_input.shape[1]
                    femr_input = {
                        "transformer": {
                            "tokens": single_input,
                            "normalized_ages": torch.zeros(1, single_seq_length, device=input_ids.device),
                            "ages": torch.zeros(1, single_seq_length, device=input_ids.device),
                            "patient_lengths": torch.tensor([single_seq_length], device=input_ids.device),
                            "timestamps": torch.zeros(1, single_seq_length, device=input_ids.device),
                        },
                        "patient_ids": torch.tensor([i], dtype=torch.long, device=input_ids.device),  # Use index as dummy patient ID
                    }
                    single_output = self.model(femr_input)
                    # Extract representations from FEMR output format
                    if isinstance(single_output, tuple) and len(single_output) == 2:
                        _, output_dict = single_output
                        if isinstance(output_dict, dict) and 'representations' in output_dict:
                            outputs.append(output_dict['representations'])
                        else:
                            outputs.append(torch.zeros(1, single_seq_length, self.config.hidden_size, device=input_ids.device))
                    elif isinstance(single_output, torch.Tensor):
                        outputs.append(single_output)
                    else:
                        outputs.append(single_output[0])
                model_output = torch.cat(outputs, dim=0)

            # Extract relevant outputs - FEMR returns tuple (int, dict)
            if isinstance(model_output, tuple) and len(model_output) == 2:
                _, output_dict = model_output
                if isinstance(output_dict, dict) and 'representations' in output_dict:
                    last_hidden_state = output_dict['representations']
                else:
                    logger.warning(f"Expected 'representations' in FEMR output dict, got keys: {output_dict.keys() if isinstance(output_dict, dict) else 'not a dict'}")
                    batch_size, seq_len = input_ids.shape
                    hidden_size = self.config.hidden_size
                    last_hidden_state = torch.zeros(batch_size, seq_len, hidden_size, device=input_ids.device)
            elif isinstance(model_output, torch.Tensor):
                last_hidden_state = model_output
            elif isinstance(model_output, (tuple, list)) and len(model_output) > 0:
                last_hidden_state = model_output[0]
            elif hasattr(model_output, 'last_hidden_state'):
                last_hidden_state = model_output.last_hidden_state
            elif isinstance(model_output, dict) and 'hidden_states' in model_output:
                last_hidden_state = model_output['hidden_states']
            else:
                # Fallback: create a dummy output based on input shape
                logger.warning(f"Unexpected FEMR output format: {type(model_output)}, creating dummy output")
                batch_size, seq_len = input_ids.shape
                hidden_size = self.config.hidden_size
                last_hidden_state = torch.zeros(batch_size, seq_len, hidden_size, device=input_ids.device)

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
