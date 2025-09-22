#!/usr/bin/env python3
"""
Fix tokenizer by adding missing special tokens.
"""

import argparse
import json
from pathlib import Path

def fix_tokenizer_special_tokens(tokenizer_path):
    """Add special tokens to existing tokenizer."""

    print(f"Loading tokenizer from {tokenizer_path}...")
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)

    # Get current vocab
    vocab = tokenizer_data.get('model', {}).get('vocab', {})
    current_vocab_size = len(vocab)

    print(f"Current vocabulary size: {current_vocab_size}")

    # Special tokens to add
    special_tokens = {
        "[UNK]": current_vocab_size,
        "[PAD]": current_vocab_size + 1,
        "[CLS]": current_vocab_size + 2,
        "[SEP]": current_vocab_size + 3,
        "[MASK]": current_vocab_size + 4
    }

    # Check which tokens are missing
    missing_tokens = {}
    for token, token_id in special_tokens.items():
        if token not in vocab:
            missing_tokens[token] = token_id
            print(f"Adding missing token: {token} -> {token_id}")
        else:
            print(f"Token {token} already exists with ID {vocab[token]}")

    if not missing_tokens:
        print("All special tokens are already present!")
        return

    # Add missing tokens to vocab
    vocab.update(missing_tokens)
    tokenizer_data['model']['vocab'] = vocab

    # Update tokenizer model configuration
    if 'model' in tokenizer_data:
        # Set UNK token in model config
        tokenizer_data['model']['unk_token'] = "[UNK]"

        # Update vocab size if needed
        if 'vocab_size' in tokenizer_data['model']:
            tokenizer_data['model']['vocab_size'] = len(vocab)

    # Update special tokens configuration
    special_tokens_config = [
        {"id": vocab["[UNK]"], "content": "[UNK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": vocab["[PAD]"], "content": "[PAD]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": vocab["[CLS]"], "content": "[CLS]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": vocab["[SEP]"], "content": "[SEP]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": vocab["[MASK]"], "content": "[MASK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
    ]

    # Update added_tokens if it exists
    if 'added_tokens' not in tokenizer_data:
        tokenizer_data['added_tokens'] = []

    # Add missing special tokens to added_tokens
    existing_tokens = {token['content'] for token in tokenizer_data['added_tokens']}
    for token_config in special_tokens_config:
        if token_config['content'] not in existing_tokens:
            tokenizer_data['added_tokens'].append(token_config)

    # Keep post_processor as null to avoid compatibility issues
    # The post_processor will be set by the SAT framework if needed

    # Backup original file
    backup_path = tokenizer_path.with_suffix('.json.backup')
    if not backup_path.exists():
        print(f"Creating backup at {backup_path}")
        import shutil
        shutil.copy2(tokenizer_path, backup_path)

    # Save updated tokenizer
    print(f"Saving updated tokenizer to {tokenizer_path}...")
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer_data, f, indent=2)

    print(f"✓ Successfully updated tokenizer!")
    print(f"  - New vocabulary size: {len(vocab)}")
    print(f"  - Added special tokens: {list(missing_tokens.keys())}")

    return tokenizer_data

def main():
    parser = argparse.ArgumentParser(
        description='Fix tokenizer by adding missing special tokens'
    )
    parser.add_argument(
        'tokenizer_path',
        nargs='?',  # Optional positional argument
        default='data/model-hub/ehrshot_motor/tokenizer/tokenizer.json',
        help='Path to tokenizer.json file (default: data/model-hub/ehrshot_motor/tokenizer/tokenizer.json)'
    )

    args = parser.parse_args()
    tokenizer_path = Path(args.tokenizer_path)

    if not tokenizer_path.exists():
        print(f"Error: Tokenizer file not found at {tokenizer_path}")
        print(f"Please check the path and try again.")
        return

    fix_tokenizer_special_tokens(tokenizer_path)

    print("\n✅ Tokenizer fix complete!")
    print("You can now run the training pipeline again.")

if __name__ == "__main__":
    main()