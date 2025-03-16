"""Embedding utilities for transformer models.

This module provides utilities for token and sentence embedding strategies
with an emphasis on numerical stability and flexibility.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .config import SentenceEmbedding, TokenEmbedding
from sat.utils import logging

logger = logging.get_default_logger()


class BaseEmbeddingPooler(nn.Module):
    """Base class for embedding poolers."""

    def __init__(
        self,
        hidden_size: int,
        dropout_prob: float = 0.1,
        layer_norm: bool = True,
    ):
        """Initialize the base embedding pooler.

        Args:
            hidden_size: Size of the hidden state.
            dropout_prob: Dropout probability.
            layer_norm: Whether to apply layer normalization.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size) if layer_norm else None

    def apply_normalization(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply dropout and layer normalization if configured.

        Args:
            embeddings: Input embeddings.

        Returns:
            Normalized embeddings.
        """
        embeddings = self.dropout(embeddings)
        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        return embeddings


class TokenEmbeddingPooler(BaseEmbeddingPooler):
    """Pooler for token embeddings across transformer layers."""

    def __init__(
        self,
        hidden_size: int,
        token_emb_type: str,
        select_hidden_layers: Optional[List[int]] = None,
        dropout_prob: float = 0.1,
        layer_norm: bool = True,
    ):
        """Initialize the token embedding pooler.

        Args:
            hidden_size: Size of the hidden state.
            token_emb_type: Type of token embedding (AVG, SUM, CAT, BERT, NONE).
            select_hidden_layers: Optional list of hidden layers to select.
            dropout_prob: Dropout probability.
            layer_norm: Whether to apply layer normalization.
        """
        super().__init__(hidden_size, dropout_prob, layer_norm)
        self.token_emb_type = token_emb_type
        self.select_hidden_layers = select_hidden_layers

        # For attention-based pooling
        if token_emb_type == TokenEmbedding.ATTENTION.value:
            self.attention = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: Union[List[torch.Tensor], torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        pooler_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for token embedding pooling.

        Args:
            hidden_states: List of hidden states from transformer layers or stacked tensor.
            attention_mask: Optional attention mask.
            pooler_output: Optional pooler output for BERT pooling.

        Returns:
            Pooled token embeddings.
        """
        # Stack hidden states if provided as a list
        if isinstance(hidden_states, list):
            token_embeddings = torch.stack(hidden_states, dim=0)
        else:
            # Handle when hidden_states is a tuple (often the case with transformer outputs)
            if isinstance(hidden_states, tuple):
                # Take the first element which is usually the hidden states
                token_embeddings = hidden_states[0]
            else:
                token_embeddings = hidden_states

        # Subset the layers if specified
        if self.select_hidden_layers:
            logger.debug(f"Select hidden layers {self.select_hidden_layers}")
            token_embeddings = token_embeddings[self.select_hidden_layers, :, :, :]

        # Debug the shape of token_embeddings
        logger.debug(f"Token embeddings shape before permute: {token_embeddings.shape}")

        # Rearrange to batch x tokens x layers x features
        # Handle different tensor shapes based on the number of dimensions
        if len(token_embeddings.shape) == 4:  # [layers, batch, tokens, features]
            token_embeddings = token_embeddings.permute(1, 2, 0, 3).contiguous()
        elif len(token_embeddings.shape) == 3:  # [batch, tokens, features]
            # If there's only a single layer, we can reshape to add a dimension
            token_embeddings = token_embeddings.unsqueeze(2)
        else:
            raise ValueError(
                f"Unexpected token_embeddings shape: {token_embeddings.shape}"
            )

        logger.debug(
            f"Dimensions of token embeddings after permute: {token_embeddings.shape}"
        )

        # Process token embeddings based on configuration
        if self.token_emb_type == TokenEmbedding.AVG.value:
            logger.debug("Average the token embeddings")
            token_embeddings = torch.mean(token_embeddings, dim=2)

        elif self.token_emb_type == TokenEmbedding.SUM.value:
            logger.debug("Sum the token embeddings")
            token_embeddings = torch.sum(token_embeddings, dim=2)

        elif self.token_emb_type == TokenEmbedding.CAT.value:
            logger.debug("Concatenate the token embeddings")
            batch_size, num_tokens, num_layers, hidden_size = token_embeddings.shape
            token_embeddings = token_embeddings.reshape(batch_size, num_tokens, -1)

        elif self.token_emb_type == TokenEmbedding.BERT.value:
            logger.debug("Use Bert-pooler for the token embeddings")
            if pooler_output is None:
                raise ValueError("Pooler output is required for BERT embedding type")
            return pooler_output

        elif self.token_emb_type == TokenEmbedding.ATTENTION.value:
            logger.debug("Use attention-based pooling for token embeddings")
            batch_size, num_tokens, num_layers, hidden_size = token_embeddings.shape

            # Reshape for attention computation
            reshaped = token_embeddings.reshape(batch_size, num_tokens, -1)

            # Compute attention scores
            attention_scores = self.attention(reshaped).squeeze(-1)

            # Apply mask if provided
            if attention_mask is not None:
                attention_scores = attention_scores + ((1 - attention_mask) * -10000.0)

            # Apply softmax to get attention weights
            attention_weights = F.softmax(attention_scores, dim=1)

            # Apply weighted sum
            token_embeddings = torch.sum(
                reshaped * attention_weights.unsqueeze(-1), dim=1
            )

            # Return immediately as this already produces a sentence embedding
            return self.apply_normalization(token_embeddings)

        # Apply normalization
        token_embeddings = self.apply_normalization(token_embeddings)

        return token_embeddings


class SentenceEmbeddingPooler(BaseEmbeddingPooler):
    """Pooler for sentence embeddings across tokens."""

    def __init__(
        self,
        hidden_size: int,
        sentence_emb_type: str,
        dropout_prob: float = 0.1,
        layer_norm: bool = True,
    ):
        """Initialize the sentence embedding pooler.

        Args:
            hidden_size: Size of the hidden state.
            sentence_emb_type: Type of sentence embedding (AVG, MAX, NONE, ATTENTION, HIERARCHICAL).
            dropout_prob: Dropout probability.
            layer_norm: Whether to apply layer normalization.
        """
        super().__init__(hidden_size, dropout_prob, layer_norm)
        self.sentence_emb_type = sentence_emb_type

        # For attention-based pooling
        if sentence_emb_type == SentenceEmbedding.ATTENTION.value:
            self.attention = nn.Linear(hidden_size, 1)

        # For hierarchical pooling
        elif sentence_emb_type == SentenceEmbedding.HIERARCHICAL.value:
            self.projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for sentence embedding pooling.

        Args:
            token_embeddings: Token embeddings.
            attention_mask: Optional attention mask.

        Returns:
            Pooled sentence embeddings.
        """
        if self.sentence_emb_type == SentenceEmbedding.AVG.value:
            logger.debug("Average across tokens to produce sentence embeddings")

            if attention_mask is not None:
                # Apply mask to zero out padding tokens
                extended_mask = attention_mask.unsqueeze(-1).float()
                token_embeddings = token_embeddings * extended_mask

                # Sum and divide by actual sequence length
                sum_embeddings = torch.sum(token_embeddings, dim=1)
                sum_mask = torch.clamp(
                    extended_mask.sum(dim=1), min=1e-9
                )  # Prevent division by zero
                sentence_embeddings = sum_embeddings / sum_mask
            else:
                sentence_embeddings = torch.mean(token_embeddings, dim=1)

        elif self.sentence_emb_type == SentenceEmbedding.MAX.value:
            logger.debug("Max across tokens to produce sentence embeddings")

            if attention_mask is not None:
                # Create a mask for padding tokens (set to large negative value)
                mask = (1 - attention_mask).unsqueeze(-1) * -10000.0
                token_embeddings = token_embeddings + mask

            sentence_embeddings = torch.max(token_embeddings, dim=1).values

        elif self.sentence_emb_type == SentenceEmbedding.ATTENTION.value:
            logger.debug("Use attention-based pooling for sentence embeddings")

            # Compute attention scores
            attention_scores = self.attention(token_embeddings).squeeze(-1)

            # Apply mask if provided
            if attention_mask is not None:
                attention_scores = attention_scores + ((1 - attention_mask) * -10000.0)

            # Apply softmax to get attention weights
            attention_weights = F.softmax(attention_scores, dim=1)

            # Apply weighted sum
            sentence_embeddings = torch.sum(
                token_embeddings * attention_weights.unsqueeze(-1), dim=1
            )

        elif self.sentence_emb_type == SentenceEmbedding.HIERARCHICAL.value:
            logger.debug("Use hierarchical pooling for sentence embeddings")

            # Apply max and mean pooling
            if attention_mask is not None:
                # Create masks for padding tokens
                mask_for_max = (1 - attention_mask).unsqueeze(-1) * -10000.0
                max_pool = torch.max(token_embeddings + mask_for_max, dim=1)[0]

                extended_mask = attention_mask.unsqueeze(-1).float()
                token_embeddings_masked = token_embeddings * extended_mask
                sum_embeddings = torch.sum(token_embeddings_masked, dim=1)
                sum_mask = torch.clamp(extended_mask.sum(dim=1), min=1e-9)
                mean_pool = sum_embeddings / sum_mask
            else:
                max_pool = torch.max(token_embeddings, dim=1)[0]
                mean_pool = torch.mean(token_embeddings, dim=1)

            # Concatenate and project
            sentence_embeddings = torch.cat([max_pool, mean_pool], dim=1)
            sentence_embeddings = self.projection(sentence_embeddings)

        else:  # NONE
            logger.debug("No reductions to produce sentence embeddings")
            sentence_embeddings = token_embeddings

        # Apply normalization if not NONE type (which would keep the token dimension)
        if self.sentence_emb_type != SentenceEmbedding.NONE.value:
            sentence_embeddings = self.apply_normalization(sentence_embeddings)

        return sentence_embeddings


class StableEmbeddingProcessor(nn.Module):
    """Processor for stable embedding computation from transformer outputs."""

    def __init__(
        self,
        hidden_size: int,
        token_emb_type: str,
        sentence_emb_type: str,
        select_hidden_layers: Optional[List[int]] = None,
        dropout_prob: float = 0.1,
        layer_norm: bool = True,
    ):
        """Initialize the stable embedding processor.

        Args:
            hidden_size: Size of the hidden state.
            token_emb_type: Type of token embedding (AVG, SUM, CAT, BERT, NONE).
            sentence_emb_type: Type of sentence embedding (AVG, MAX, NONE, ATTENTION, HIERARCHICAL).
            select_hidden_layers: Optional list of hidden layers to select.
            dropout_prob: Dropout probability.
            layer_norm: Whether to apply layer normalization.
        """
        super().__init__()

        # If BERT token embedding is chosen, force sentence_emb to NONE
        if token_emb_type == TokenEmbedding.BERT.value:
            logger.debug(
                f"If BERT token embedding is chosen, sentence embeddings to NONE."
            )
            sentence_emb_type = SentenceEmbedding.NONE.value

        self.token_emb_type = token_emb_type
        self.sentence_emb_type = sentence_emb_type
        self.select_hidden_layers = select_hidden_layers
        self.hidden_size = hidden_size

        # Calculate output dimension
        if token_emb_type == TokenEmbedding.CAT.value:
            if select_hidden_layers:
                output_dim = hidden_size * len(select_hidden_layers)
            else:
                # Default to all layers + 1 for input embedding
                # This will be properly calculated in forward using the actual number of layers
                output_dim = hidden_size
        else:
            output_dim = hidden_size

        # Initialize poolers
        self.token_pooler = TokenEmbeddingPooler(
            hidden_size=hidden_size,
            token_emb_type=token_emb_type,
            select_hidden_layers=select_hidden_layers,
            dropout_prob=dropout_prob,
            layer_norm=layer_norm,
        )

        if (
            sentence_emb_type == SentenceEmbedding.HIERARCHICAL.value
            and token_emb_type != TokenEmbedding.BERT.value
        ):
            # Hierarchical pooling uses concatenation, so double the size
            sentence_input_size = output_dim
        else:
            sentence_input_size = output_dim

        self.sentence_pooler = SentenceEmbeddingPooler(
            hidden_size=sentence_input_size,
            sentence_emb_type=sentence_emb_type,
            dropout_prob=dropout_prob,
            layer_norm=layer_norm,
        )

    def get_output_dim(self, num_layers: int = None) -> int:
        """Calculate the output dimension based on the embedding configuration.

        Args:
            num_layers: Number of transformer layers.

        Returns:
            Output dimension.
        """
        if self.token_emb_type == TokenEmbedding.CAT.value:
            if self.select_hidden_layers:
                output_dim = self.hidden_size * len(self.select_hidden_layers)
            else:
                output_dim = self.hidden_size * (num_layers + 1)
        else:
            output_dim = self.hidden_size

        # For hierarchical pooling with concatenation
        if (
            self.sentence_emb_type == SentenceEmbedding.HIERARCHICAL.value
            and self.token_emb_type != TokenEmbedding.BERT.value
        ):
            # The projection layer in hierarchical pooling reduces back to hidden_size
            pass

        # If not doing sentence pooling, and not BERT pooling, then we'll have a token dimension
        if (self.token_emb_type != TokenEmbedding.BERT.value) and (
            self.sentence_emb_type == SentenceEmbedding.NONE.value
        ):
            # Output will be per-token, so consider this in downstream layers
            pass

        return output_dim

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        pooler_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for stable embedding processing.

        Args:
            hidden_states: List of hidden states from transformer layers.
            attention_mask: Optional attention mask.
            pooler_output: Optional pooler output.

        Returns:
            Processed embeddings.
        """
        # Apply token pooling
        token_embeddings = self.token_pooler(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            pooler_output=pooler_output,
        )

        # If BERT pooling was used, token_embeddings is already a sentence embedding
        if self.token_emb_type == TokenEmbedding.BERT.value:
            return token_embeddings

        # Apply sentence pooling
        sentence_embeddings = self.sentence_pooler(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )

        return sentence_embeddings
