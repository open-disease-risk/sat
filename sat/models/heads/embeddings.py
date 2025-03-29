"""Embedding processors for transformer outputs."""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

from enum import Enum
from typing import List, Optional, Tuple

import torch
from torch import nn

from sat.utils import logging

logger = logging.get_default_logger()


class TokenEmbedding(Enum):
    """Determine what to do with the hidden states of the encoder layers."""

    SUM = 2
    AVG = 3
    CAT = 4
    BERT = 5


class SentenceEmbedding(Enum):
    """Determine what to do to get sentence embeddings."""

    NONE = 1
    MAX = 2
    AVG = 3


class TokenEmbedder(nn.Module):
    """
    Process transformer hidden states to create token embeddings.
    Supports various token embedding strategies based on transformer representations.
    """

    def __init__(
        self,
        hidden_size: int,
        token_emb_strategy: str,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.token_emb_strategy = token_emb_strategy
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"TokenEmbedder initialized with strategy: {token_emb_strategy}"
            )

    def forward(
        self,
        hidden_states: Tuple[torch.Tensor],
        sequence_output,
        select_hidden_layers: Optional[List[int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process transformer hidden states to get token embeddings.

        Args:
            hidden_states: Tuple of tensors from transformer layers
            sequence_output: Output from transformer model
            select_hidden_layers: Optional list of specific layers to use
            attention_mask: Optional attention mask for padding

        Returns:
            Token embeddings with shape (batch_size, seq_length, embedding_dim)
        """
        # Special case for BERT pooler
        if self.token_emb_strategy == TokenEmbedding.BERT.value:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Using Bert-pooler for the token embeddings")

            # HuggingFace's BERT pooler applies a linear layer to the CLS token from the last layer
            return sequence_output[1]

        # Convert tuple of hidden states to tensor
        # layers x batches x tokens x features
        token_embeddings = torch.stack(hidden_states, dim=0)

        # Subset the layers if specified
        if select_hidden_layers:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Select hidden layers {select_hidden_layers}")
            token_embeddings = token_embeddings[select_hidden_layers, :, :, :]

        # batch x tokens x layers x features
        token_embeddings = token_embeddings.permute(1, 2, 0, 3)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Dimensions of token embeddings after layer selection {token_embeddings.shape}"
            )

        # Process token embeddings based on configuration
        if self.token_emb_strategy == TokenEmbedding.AVG.value:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Average the token embeddings across layers")
            token_embeddings = torch.mean(token_embeddings, dim=2)

        elif self.token_emb_strategy == TokenEmbedding.SUM.value:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Sum the token embeddings across layers")
            token_embeddings = torch.sum(token_embeddings, dim=2)

        elif self.token_emb_strategy == TokenEmbedding.CAT.value:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Concatenate the token embeddings across layers")
            # Split along the layer dimension
            layers = torch.tensor_split(
                token_embeddings, token_embeddings.shape[2], dim=2
            )
            token_embeddings = torch.cat(layers, dim=3).squeeze()

        # ATTENTION option removed to avoid stability issues

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Final token embeddings shape: {token_embeddings.shape}")
        return token_embeddings


class SentenceEmbedder(nn.Module):
    """
    Create sentence-level embeddings from token embeddings.
    Supports different pooling strategies and handles masked sequences.
    """

    def __init__(
        self,
        sentence_emb_strategy: str,
    ):
        super().__init__()
        self.sentence_emb_strategy = sentence_emb_strategy
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"SentenceEmbedder initialized with strategy: {sentence_emb_strategy}"
            )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create sentence embeddings from token embeddings.

        Args:
            token_embeddings: Token-level embeddings
            attention_mask: Optional mask for padding tokens

        Returns:
            Sentence embeddings with shape based on pooling strategy
        """
        # BERT pooler output is already a sentence embedding
        if len(token_embeddings.shape) == 2:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Using pre-pooled embeddings (likely from BERT pooler)")
            return token_embeddings

        # Apply pooling based on strategy
        if self.sentence_emb_strategy == SentenceEmbedding.AVG.value:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Average across tokens to produce sentence embeddings")
            if attention_mask is not None:
                # Compute masked average (only over non-padding tokens)
                expanded_mask = attention_mask.unsqueeze(-1)
                # Sum embeddings weighted by mask
                token_sum = torch.sum(token_embeddings * expanded_mask, dim=1)
                # Sum mask values (number of non-padding tokens)
                mask_sum = torch.sum(expanded_mask, dim=1)
                # Handle cases where all tokens might be masked (shouldn't happen in practice)
                # Add small epsilon to avoid division by zero
                safe_mask_sum = torch.clamp(mask_sum, min=1.0)
                # Compute mean by dividing sum by count of non-padding tokens
                sentence_embeddings = token_sum / safe_mask_sum
            else:
                sentence_embeddings = torch.mean(token_embeddings, dim=1)

        elif self.sentence_emb_strategy == SentenceEmbedding.MAX.value:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Max across tokens to produce sentence embeddings")
            if attention_mask is not None:
                # Create a mask for padding tokens (replace padding with large negative values)
                padding_mask = (1 - attention_mask).unsqueeze(-1) * -1e4
                # Apply mask to embeddings (padding tokens will have large negative values)
                masked_embeddings = token_embeddings + padding_mask
                # Take max over token dimension
                sentence_embeddings = torch.max(masked_embeddings, dim=1)[0]
            else:
                sentence_embeddings = torch.max(token_embeddings, dim=1)[0]

        else:  # SentenceEmbedding.NONE.value
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("No pooling applied to token embeddings")
            sentence_embeddings = token_embeddings

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Sentence embeddings shape: {sentence_embeddings.shape}")
        return sentence_embeddings
