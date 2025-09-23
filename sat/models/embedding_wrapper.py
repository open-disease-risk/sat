"""Embedding Database Wrapper for pre-computed embeddings integration with SAT framework"""

import pickle
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from sat.utils import logging

logger = logging.get_default_logger()


class EmbeddingDatabaseWrapper(nn.Module):
    """
    Wrapper to use pre-computed embeddings from a database (e.g., CLMBR) as a transformer replacement.
    """

    def __init__(self, embedding_path: str, **kwargs):
        super().__init__()

        self.embedding_path = embedding_path
        self.embeddings_data = None
        self.patient_lookup = {}  # patient_id -> list of (time, embedding_idx) tuples
        self.config = None

        # Load embeddings
        self._load_embeddings()

    def _load_embeddings(self):
        """Load the pre-computed embeddings and build lookup index"""
        try:
            logger.info(f"Loading embeddings from {self.embedding_path}")

            with open(self.embedding_path, 'rb') as f:
                self.embeddings_data = pickle.load(f)

            # Validate structure
            required_keys = ['data_matrix', 'patient_ids', 'labeling_time']
            for key in required_keys:
                if key not in self.embeddings_data:
                    raise ValueError(f"Missing required key '{key}' in embeddings file")

            data_matrix = self.embeddings_data['data_matrix']
            patient_ids = self.embeddings_data['patient_ids']
            labeling_times = self.embeddings_data['labeling_time']

            logger.info(f"Loaded {data_matrix.shape[0]} embeddings for {len(np.unique(patient_ids))} patients")
            logger.info(f"Embedding dimension: {data_matrix.shape[1]}")

            # Build efficient lookup index
            logger.info("Building patient lookup index...")
            for idx, (patient_id, time) in enumerate(zip(patient_ids, labeling_times, strict=False)):
                if patient_id not in self.patient_lookup:
                    self.patient_lookup[patient_id] = []
                self.patient_lookup[patient_id].append((time, idx))

            # Sort embeddings by time for each patient (for efficient lookup)
            for patient_id in self.patient_lookup:
                self.patient_lookup[patient_id].sort(key=lambda x: x[0])

            # Create config object that mimics transformers config interface
            self.config = self._create_config()

            logger.info(f"Successfully loaded embedding database from {self.embedding_path}")
            logger.info(f"Built lookup index for {len(self.patient_lookup)} patients")

        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise

    def _create_config(self):
        """Create a config object that mimics transformers config interface"""
        class EmbeddingConfig:
            def __init__(self, embedding_dim):
                # Set dimensions based on actual embeddings
                self.hidden_size = embedding_dim
                self.num_hidden_layers = 1  # Not applicable for embeddings
                self.num_attention_heads = 1  # Not applicable for embeddings
                self.intermediate_size = embedding_dim
                self.max_position_embeddings = 1  # Single embedding per patient-time
                self.vocab_size = 1  # Not applicable for embeddings
                self.model_type = "embedding-database"

        embedding_dim = self.embeddings_data['data_matrix'].shape[1]
        return EmbeddingConfig(embedding_dim)

    def _find_best_embedding(self, patient_id: int, anchor_time: Union[str, datetime]) -> Optional[np.ndarray]:
        """
        Find the best embedding for a patient at a given anchor time.
        Returns the most recent embedding before the anchor time.
        """
        if patient_id not in self.patient_lookup:
            logger.warning(f"Patient {patient_id} not found in embedding database")
            return None

        # Convert anchor_time to datetime if it's a string
        if isinstance(anchor_time, str):
            anchor_datetime = pd.to_datetime(anchor_time)
        else:
            anchor_datetime = anchor_time

        # Find embeddings for this patient that are before anchor time
        patient_embeddings = self.patient_lookup[patient_id]
        valid_embeddings = [(time, idx) for time, idx in patient_embeddings if time <= anchor_datetime]

        if not valid_embeddings:
            # No embeddings before anchor time, use the earliest available
            logger.warning(f"No embeddings before anchor time {anchor_datetime} for patient {patient_id}, using earliest")
            earliest_time, earliest_idx = patient_embeddings[0]
            return self.embeddings_data['data_matrix'][earliest_idx]

        # Get the most recent embedding before anchor time
        latest_time, latest_idx = max(valid_embeddings, key=lambda x: x[0])
        return self.embeddings_data['data_matrix'][latest_idx]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        patient_id: Optional[torch.Tensor] = None,
        anchor_time: Optional[list] = None,
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
        Forward pass that retrieves pre-computed embeddings based on patient_id and anchor_time.

        Args:
            input_ids: Not used for embeddings but kept for compatibility
            patient_id: Tensor of patient IDs (batch_size,)
            anchor_time: List of anchor times (batch_size,)
            **kwargs: Other arguments for compatibility
        """

        if patient_id is None:
            raise ValueError("patient_id must be provided for embedding database")
        if anchor_time is None:
            raise ValueError("anchor_time must be provided for embedding database")

        batch_size = patient_id.shape[0]
        embedding_dim = self.config.hidden_size
        device = patient_id.device if hasattr(patient_id, 'device') else torch.device('cpu')

        # Retrieve embeddings for each patient-time pair in the batch
        batch_embeddings = []
        found_embeddings = 0
        missing_embeddings = 0

        for i in range(batch_size):
            pid = patient_id[i].item() if hasattr(patient_id[i], 'item') else patient_id[i]
            atime = anchor_time[i] if isinstance(anchor_time, list) else anchor_time

            embedding = self._find_best_embedding(pid, atime)

            if embedding is not None:
                # Convert to torch tensor
                embedding_tensor = torch.from_numpy(embedding.copy()).float()

                # Validate embedding
                if torch.isnan(embedding_tensor).any():
                    logger.warning(f"NaN detected in embedding for patient {pid}, replacing with random")
                    embedding_tensor = torch.randn(embedding_dim, dtype=torch.float32) * 0.02
                elif torch.isinf(embedding_tensor).any():
                    logger.warning(f"Inf detected in embedding for patient {pid}, replacing with random")
                    embedding_tensor = torch.randn(embedding_dim, dtype=torch.float32) * 0.02

                batch_embeddings.append(embedding_tensor)
                found_embeddings += 1
            else:
                # Fallback: create small random embedding instead of zeros
                logger.warning(f"No embedding found for patient {pid}, using random embedding")
                embedding_tensor = torch.randn(embedding_dim, dtype=torch.float32) * 0.02
                batch_embeddings.append(embedding_tensor)
                missing_embeddings += 1

        # Stack into batch and move to correct device
        last_hidden_state = torch.stack(batch_embeddings).to(device)

        # Log statistics for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("EmbeddingDatabaseWrapper batch stats:")
            logger.debug(f"  Batch size: {batch_size}, Found: {found_embeddings}, Missing: {missing_embeddings}")
            logger.debug(f"  Embeddings shape: {last_hidden_state.shape}")
            logger.debug(f"  Values - Mean: {last_hidden_state.mean():.6f}, Std: {last_hidden_state.std():.6f}")
            logger.debug(f"  Values - Min: {last_hidden_state.min():.6f}, Max: {last_hidden_state.max():.6f}")
            logger.debug(f"  Zero percentage: {(last_hidden_state == 0).float().mean():.4f}")

        # Check for potential issues
        if torch.isnan(last_hidden_state).any():
            logger.error("NaN values detected in final embeddings!")
        if torch.isinf(last_hidden_state).any():
            logger.error("Inf values detected in final embeddings!")
        if last_hidden_state.abs().max() < 1e-6:
            logger.warning("All embedding values are very small (< 1e-6)")
        if last_hidden_state.std() < 1e-6:
            logger.warning("Embedding standard deviation is very small, may indicate all similar values")

        # For compatibility with transformer outputs, we need to add a sequence dimension
        # Most transformers output (batch_size, seq_len, hidden_size)
        # For embeddings, we use seq_len=1 since it's a single representation per patient
        last_hidden_state = last_hidden_state.unsqueeze(1)  # (batch_size, 1, hidden_size)

        # Create fake hidden states for compatibility if output_hidden_states is requested
        hidden_states = None
        if output_hidden_states:
            # Create a list of hidden states (one per layer)
            hidden_states = [last_hidden_state]  # Only one "layer" for embeddings

        # Create fake attention weights if requested
        attentions = None
        if output_attentions:
            # For embeddings, attention doesn't make sense, but we provide a dummy
            seq_len = 1
            num_heads = self.config.num_attention_heads
            attentions = [torch.ones(batch_size, num_heads, seq_len, seq_len, device=device)]

        if return_dict:
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=last_hidden_state,
                pooler_output=None,  # Not applicable for embeddings
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


def create_embedding_model_from_config(config_params: Dict[str, Any]) -> EmbeddingDatabaseWrapper:
    """
    Factory function to create an embedding database wrapper from config parameters.

    Args:
        config_params: Dictionary containing model configuration

    Returns:
        EmbeddingDatabaseWrapper instance
    """
    embedding_path = config_params.get("embedding_path")

    if not embedding_path:
        raise ValueError("embedding_path must be provided for embedding database models")

    # Remove embedding_path from config_params to avoid duplicate parameter
    filtered_params = {k: v for k, v in config_params.items() if k != 'embedding_path'}

    return EmbeddingDatabaseWrapper(embedding_path, **filtered_params)


def is_embedding_model(model_name_or_path: str) -> bool:
    """
    Check if the given model requires embedding database loading.

    Args:
        model_name_or_path: Model name or path

    Returns:
        True if model requires embedding database, False otherwise
    """
    embedding_models = [
        "embedding-database",
        "embedding_db",
        "clmbr-embeddings",
        "pre-computed",
    ]

    return any(emb_model in model_name_or_path.lower() for emb_model in embedding_models)


def is_embedding_database_model(model) -> bool:
    """
    Check if a model instance is an embedding database model.

    Args:
        model: Model instance to check

    Returns:
        True if model is an embedding database, False otherwise
    """
    return isinstance(model, EmbeddingDatabaseWrapper)
