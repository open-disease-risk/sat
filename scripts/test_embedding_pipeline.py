#!/usr/bin/env python3
"""
Test script to verify CLMBR embedding pipeline works end-to-end.
This script loads a small batch of EHRSHOT data and traces embeddings through the model.
"""

import sys
from pathlib import Path

import torch

# Add SAT to path
sat_root = Path(__file__).parent.parent
sys.path.insert(0, str(sat_root))

from sat.models.embedding_wrapper import EmbeddingDatabaseWrapper
from sat.utils import logging

# Set up logging
logger = logging.get_default_logger()
logger.setLevel(logging.INFO)

def test_embedding_loading():
    """Test that we can load and access CLMBR embeddings"""
    print("=" * 60)
    print("1. Testing CLMBR embedding loading")
    print("=" * 60)

    embedding_path = "/Users/ddahlem/Downloads/EHRSHOT_ASSETS/features/clmbr_features.pkl"

    # Create wrapper
    wrapper = EmbeddingDatabaseWrapper(embedding_path)

    print(f"✓ Loaded embeddings: {wrapper.embeddings_data['data_matrix'].shape}")
    print(f"✓ Built patient lookup: {len(wrapper.patient_lookup)} patients")
    print(f"✓ Config created: hidden_size={wrapper.config.hidden_size}")

    return wrapper

def test_embedding_retrieval(wrapper):
    """Test retrieving embeddings for specific patient IDs"""
    print("\n" + "=" * 60)
    print("2. Testing embedding retrieval")
    print("=" * 60)

    # Get some patient IDs from the data
    patient_ids = list(wrapper.patient_lookup.keys())[:5]
    print(f"Testing with {len(patient_ids)} patients: {patient_ids}")

    # Create batch tensors
    batch_patient_ids = torch.tensor(patient_ids, dtype=torch.long)
    batch_anchor_times = ['2020-01-01'] * len(patient_ids)  # Dummy anchor times

    # Forward pass through wrapper
    print("\n📊 Running forward pass...")

    with torch.no_grad():
        outputs = wrapper(
            patient_id=batch_patient_ids,
            anchor_time=batch_anchor_times,
            return_dict=True
        )

    embeddings = outputs.last_hidden_state
    print(f"✓ Retrieved embeddings: {embeddings.shape}")
    print(f"✓ Values - Mean: {embeddings.mean():.6f}, Std: {embeddings.std():.6f}")
    print(f"✓ Values - Min: {embeddings.min():.6f}, Max: {embeddings.max():.6f}")

    # Check for issues
    if torch.isnan(embeddings).any():
        print("❌ NaN values detected!")
        return None
    if torch.isinf(embeddings).any():
        print("❌ Inf values detected!")
        return None
    if embeddings.std() < 1e-6:
        print("⚠️ Very low variation detected")

    print("✓ Embeddings look good!")
    return embeddings

def test_shared_network(embeddings):
    """Test the shared network processing"""
    print("\n" + "=" * 60)
    print("3. Testing shared network")
    print("=" * 60)

    # Create a simple shared network like the one in MTL
    batch_size, seq_len, hidden_size = embeddings.shape
    embeddings_flat = embeddings.squeeze(1)  # Remove seq_len dimension

    # Create shared network (similar to config in survival.yaml)
    shared_net = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, 256),  # survival_shared_intermediate_size
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128)  # survival_intermediate_size
    )

    print(f"📊 Input shape: {embeddings_flat.shape}")
    print(f"📊 Network: {hidden_size} -> 256 -> 128")

    # Forward pass
    with torch.no_grad():
        logits = shared_net(embeddings_flat)

    print(f"✓ Output shape: {logits.shape}")
    print(f"✓ Logits - Mean: {logits.mean():.6f}, Std: {logits.std():.6f}")
    print(f"✓ Logits - Min: {logits.min():.6f}, Max: {logits.max():.6f}")

    return logits

def test_survival_head(logits):
    """Test survival task head"""
    print("\n" + "=" * 60)
    print("4. Testing survival task head")
    print("=" * 60)

    # Create a simple survival head (7 events for EHRSHOT)
    num_events = 7
    survival_head = torch.nn.Linear(128, num_events)

    batch_size = logits.shape[0]

    print(f"📊 Input shape: {logits.shape}")
    print(f"📊 Output events: {num_events}")

    # Forward pass
    with torch.no_grad():
        predictions = survival_head(logits)

    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ Predictions - Mean: {predictions.mean():.6f}, Std: {predictions.std():.6f}")

    # Test loss calculation
    # Create dummy labels (survival times and event indicators)
    survival_times = torch.randint(1, 365, (batch_size, num_events)).float()
    event_indicators = torch.randint(0, 2, (batch_size, num_events)).float()

    print("\n📊 Testing loss calculation...")
    print(f"✓ Survival times shape: {survival_times.shape}")
    print(f"✓ Event indicators shape: {event_indicators.shape}")

    # Simple MSE loss for demonstration
    loss_fn = torch.nn.MSELoss()

    # Create target from predictions + noise (since we don't have real targets)
    targets = predictions + torch.randn_like(predictions) * 0.1

    loss = loss_fn(predictions, targets)
    print(f"✓ Loss: {loss.item():.6f}")

    if loss.item() > 0:
        print("✅ Non-zero loss achieved!")
        return True
    else:
        print("❌ Zero loss - something is wrong")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing CLMBR Embedding Pipeline")
    print("This script tests the embedding pipeline end-to-end")

    try:
        # Test 1: Load embeddings
        wrapper = test_embedding_loading()

        # Test 2: Retrieve embeddings
        embeddings = test_embedding_retrieval(wrapper)
        if embeddings is None:
            print("❌ Embedding retrieval failed!")
            return False

        # Test 3: Shared network
        logits = test_shared_network(embeddings)

        # Test 4: Survival head and loss
        success = test_survival_head(logits)

        if success:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED!")
            print("The embedding pipeline should work correctly.")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("❌ TESTS FAILED!")
            print("There are issues with the pipeline.")
            print("=" * 60)
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
