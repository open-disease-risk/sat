#!/usr/bin/env python3
"""
Convert Motor-T JAX checkpoint to PyTorch format for use with FEMR.
"""

import json
import pickle
import warnings
from pathlib import Path

import msgpack
import numpy as np
import torch

warnings.filterwarnings('ignore', category=DeprecationWarning)

def load_jax_checkpoint(checkpoint_path):
    """Load the JAX checkpoint file."""
    print(f"Loading JAX checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"✓ Loaded {len(checkpoint)} parameter groups")
    return checkpoint

def convert_jax_to_pytorch_weights(jax_checkpoint):
    """Convert JAX weights to PyTorch state dict format."""
    pytorch_state_dict = {}

    # Mapping from JAX parameter names to PyTorch convention
    for jax_key, jax_weights in jax_checkpoint.items():
        # Parse the hierarchical key structure
        parts = jax_key.split('/')

        if 'embed' in jax_key:
            # Embedding layer
            if 'embeddings' in jax_weights:
                # Convert JAX embedding to PyTorch format
                embedding_weight = np.array(jax_weights['embeddings'])
                pytorch_state_dict['embeddings.weight'] = torch.from_numpy(embedding_weight)
                print(f"Converted embedding: {embedding_weight.shape}")

        elif 'loop_' in jax_key:
            # Transformer block
            layer_num = None
            for part in parts:
                if part.startswith('loop_'):
                    layer_num = part.split('_')[1]
                    break

            if layer_num is not None:
                if 'linear' in jax_key and 'linear_1' not in jax_key:
                    # First linear layer (attention projection)
                    if 'w' in jax_weights:
                        weight = np.array(jax_weights['w'])
                        pytorch_state_dict[f'transformer.layers.{layer_num}.attention.linear.weight'] = torch.from_numpy(weight.T)
                    if 'b' in jax_weights:
                        bias = np.array(jax_weights['b'])
                        pytorch_state_dict[f'transformer.layers.{layer_num}.attention.linear.bias'] = torch.from_numpy(bias)

                elif 'linear_1' in jax_key:
                    # Second linear layer (MLP)
                    if 'w' in jax_weights:
                        weight = np.array(jax_weights['w'])
                        pytorch_state_dict[f'transformer.layers.{layer_num}.mlp.linear.weight'] = torch.from_numpy(weight.T)
                    if 'b' in jax_weights:
                        bias = np.array(jax_weights['b'])
                        pytorch_state_dict[f'transformer.layers.{layer_num}.mlp.linear.bias'] = torch.from_numpy(bias)

                elif 'rms_norm' in jax_key:
                    # Layer normalization
                    for norm_key, norm_weights in jax_weights.items():
                        weight = np.array(norm_weights)
                        pytorch_state_dict[f'transformer.layers.{layer_num}.norm.{norm_key}'] = torch.from_numpy(weight)

        elif 'SurvivalCLMBRTask' in jax_key and 'linear' in jax_key:
            # Task-specific head
            if 'w' in jax_weights:
                weight = np.array(jax_weights['w'])
                pytorch_state_dict['task_head.weight'] = torch.from_numpy(weight.T)
            if 'b' in jax_weights:
                bias = np.array(jax_weights['b'])
                pytorch_state_dict['task_head.bias'] = torch.from_numpy(bias)

        elif 'rms_norm' in jax_key and 'loop_' not in jax_key:
            # Final layer norm
            for norm_key, norm_weights in jax_weights.items():
                weight = np.array(norm_weights)
                if 'rms_norm_1' in jax_key:
                    pytorch_state_dict[f'transformer.final_norm.{norm_key}'] = torch.from_numpy(weight)
                else:
                    pytorch_state_dict[f'transformer.pre_norm.{norm_key}'] = torch.from_numpy(weight)

    print(f"✓ Converted to {len(pytorch_state_dict)} PyTorch parameters")
    return pytorch_state_dict

def create_femr_config(config_msgpack_path, output_dir):
    """Create a config.json file compatible with FEMR."""

    # Load the original msgpack config
    with open(config_msgpack_path, 'rb') as f:
        original_config = msgpack.load(f)

    transformer_config = original_config['transformer']
    task_config = original_config['task']

    # Create FEMR-compatible config
    femr_config = {
        "architectures": ["FEMRModel"],
        "model_type": "femr",
        "is_encoder_decoder": False,
        "is_decoder": False,

        # Model architecture from msgpack
        "vocab_size": transformer_config['vocab_size'],
        "hidden_size": transformer_config['hidden_size'],
        "intermediate_size": transformer_config['intermediate_size'],
        "num_attention_heads": transformer_config['n_heads'],
        "num_hidden_layers": transformer_config['n_layers'],
        "max_position_embeddings": transformer_config['attention_width'],
        "hidden_dropout_prob": transformer_config.get('internal_dropout', 0.0),

        # Motor-T specific parameters
        "rotary": transformer_config.get('rotary', 'per_head'),
        "is_hierarchical": transformer_config.get('is_hierarchical', True),
        "with_age_beta": transformer_config.get('with_age_beta', True),

        # Task configuration
        "task_kwargs": {
            "type": task_config['type'],
            "num_time_bins": task_config.get('num_time_bins', 8),
            "num_codes": task_config.get('num_codes', 8192),
            "dim": task_config.get('dim', 512)
        },

        # Generation config (from FEMR tutorial)
        "max_length": 20,
        "num_beams": 1,
        "num_return_sequences": 1,
        "return_dict": True,

        # Transformers compatibility
        "transformers_version": "4.21.0",
        "torch_dtype": "float32"
    }

    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(femr_config, f, indent=2)

    print(f"✓ Created config.json at {config_path}")
    return femr_config

def create_dummy_tokenizer_files(output_dir, vocab_size=65536):
    """Create minimal tokenizer files since we don't have the original tokenizer."""

    # Create a simple dictionary.msgpack file
    # This is a placeholder - in practice, you'd need the actual tokenizer vocab
    dummy_vocab = {str(i): i for i in range(vocab_size)}

    tokenizer_path = output_dir / 'dictionary.msgpack'
    with open(tokenizer_path, 'wb') as f:
        msgpack.pack(dummy_vocab, f)

    print(f"✓ Created dummy tokenizer at {tokenizer_path}")
    print("  WARNING: This is a placeholder tokenizer. You may need the actual tokenizer vocab.")

def main():
    # Paths
    motor_t_dir = Path("/Users/ddahlem/Documents/repos/open-disease-risk/sat/data/model-hub/motor-t")
    jax_checkpoint_path = motor_t_dir / "best"
    config_msgpack_path = motor_t_dir / "config.msgpack"

    # Load and convert
    jax_checkpoint = load_jax_checkpoint(jax_checkpoint_path)
    pytorch_state_dict = convert_jax_to_pytorch_weights(jax_checkpoint)

    # Save PyTorch model
    pytorch_model_path = motor_t_dir / "pytorch_model.bin"
    torch.save(pytorch_state_dict, pytorch_model_path)
    print(f"✓ Saved PyTorch model to {pytorch_model_path}")

    # Create FEMR config
    femr_config = create_femr_config(config_msgpack_path, motor_t_dir)

    # Create dummy tokenizer
    create_dummy_tokenizer_files(motor_t_dir, vocab_size=femr_config['vocab_size'])

    print("\\n✅ Conversion complete!")
    print(f"Model files created in: {motor_t_dir}")
    print("Files:")
    print(f"  - pytorch_model.bin ({pytorch_model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print("  - config.json")
    print("  - dictionary.msgpack (dummy)")

    # Print some stats
    total_params = sum(p.numel() for p in pytorch_state_dict.values())
    print("\\nModel statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Model size: {total_params * 4 / 1024 / 1024:.1f} MB (float32)")

if __name__ == "__main__":
    main()
