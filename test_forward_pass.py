"""Test forward pass with modified loss function."""

import os
import sys
import torch
import numpy as np

from sat.models.heads.mtl import MTLForSurvival
from sat.models.heads.config import TokenEmbedding, SentenceEmbedding
from sat.loss.survival.nllpchazard import SATNLLPCHazardLoss
from sat.models.heads.survival import SAOutput

def test_forward_pass_with_mock_data():
    """Test forward pass with mock data."""
    print("=== Testing forward pass with mock data ===")
    
    # Create a simple model
    print("\nCreating test model...")
    model = MTLForSurvival(
        num_events=2,
        intermediate_size=8,
        num_hidden_layers=1,
        batch_norm=False,
        hidden_dropout_prob=0.1,
        sentence_emb=SentenceEmbedding.NONE.value,
        token_emb=TokenEmbedding.SUM.value,
        bias=True,
        num_labels=8
    )
    
    # Create mock inputs
    batch_size = 4
    num_events = 2
    seq_length = 15
    
    print("\nCreating mock inputs...")
    numerics = torch.rand(batch_size, seq_length)
    input_ids = torch.randint(0, 100, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Create mock labels
    # Format: [duration_idx1, event1, frac1, time1, duration_idx2, event2, frac2, time2]
    labels = torch.tensor([
        [0.0, 1.0, 0.5, 100.0, 1.0, 0.0, 0.3, 200.0],
        [1.0, 0.0, 0.7, 150.0, 0.0, 1.0, 0.6, 120.0],
        [2.0, 1.0, 0.8, 180.0, 2.0, 1.0, 0.9, 190.0],
        [3.0, 0.0, 0.4, 220.0, 3.0, 0.0, 0.2, 250.0]
    ])
    
    # Try different types for labels to test robustness
    label_types = [torch.float32, torch.float64, torch.int64, torch.int32]
    
    for dtype in label_types:
        print(f"\nTesting with labels of type {dtype}")
        test_labels = labels.to(dtype)
        
        try:
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerics=numerics,
                labels=test_labels
            )
            
            print(f"Forward pass successful!")
            print(f"Loss: {outputs.loss.item():.4f}")
            
            # Check shapes of outputs
            print(f"Logits shape: {outputs.logits.shape}")
            print(f"Hidden states shape: {outputs.hidden_states[-1].shape if outputs.hidden_states else 'None'}")
            
            # Check for NaN/Inf
            has_nan = torch.isnan(outputs.logits).any()
            has_inf = torch.isinf(outputs.logits).any()
            print(f"Output has NaN: {has_nan.item()}")
            print(f"Output has Inf: {has_inf.item()}")
            
        except Exception as e:
            print(f"Error during forward pass: {str(e)}")
            import traceback
            traceback.print_exc()

def test_loss_function_with_mock_data():
    """Test loss function with mock data."""
    print("\n=== Testing loss function with mock data ===")
    
    # Create a loss function
    loss_fn = SATNLLPCHazardLoss(num_events=2)
    
    # Create mock data
    batch_size = 4
    num_events = 2
    num_cuts = 5
    
    # Create logits with different shapes and types
    logits_shapes = [
        (batch_size, num_events, num_cuts),  # Regular shape
        (batch_size, num_cuts),              # Single-event shape
    ]
    
    for shape in logits_shapes:
        is_multi_event = len(shape) > 2
        event_str = "multi-event" if is_multi_event else "single-event"
        print(f"\nTesting {event_str} shape: {shape}")
        
        # Create logits
        logits = torch.randn(shape)
        
        # Create a SAOutput object with the logits
        outputs = SAOutput(
            loss=None,
            logits=logits,
            hazard=None,
            risk=None,
            survival=None,
            hidden_states=None
        )
        
        # Create targets
        if is_multi_event:
            # Multi-event: [duration_idx1, event1, frac1, time1, duration_idx2, event2, frac2, time2]
            targets = torch.tensor([
                [0.0, 1.0, 0.5, 100.0, 1.0, 0.0, 0.3, 200.0],
                [1.0, 0.0, 0.7, 150.0, 0.0, 1.0, 0.6, 120.0],
                [2.0, 1.0, 0.8, 180.0, 2.0, 1.0, 0.9, 190.0],
                [3.0, 0.0, 0.4, 220.0, 3.0, 0.0, 0.2, 250.0]
            ])
        else:
            # Single-event: [duration_idx, event, fraction, time]
            targets = torch.tensor([
                [0.0, 1.0, 0.5, 100.0],
                [1.0, 0.0, 0.7, 150.0],
                [2.0, 1.0, 0.8, 180.0],
                [3.0, 0.0, 0.4, 220.0]
            ])
        
        # Try different dtypes for targets
        target_dtypes = [torch.float32, torch.float64, torch.int64, torch.int32]
        
        for dtype in target_dtypes:
            print(f"  Testing with targets of type {dtype}")
            test_targets = targets.to(dtype)
            
            try:
                # Compute loss
                loss = loss_fn(outputs, test_targets)
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Loss is finite: {torch.isfinite(loss).item()}")
                
            except Exception as e:
                print(f"  Error computing loss: {str(e)}")

if __name__ == "__main__":
    print("=== Starting Forward Pass Test ===")
    test_loss_function_with_mock_data()
    test_forward_pass_with_mock_data()
    print("\n=== Test Complete ===")