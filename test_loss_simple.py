"""Simple script to debug the NLLPCHazardLoss with small example."""

import torch
from sat.pycox.models.loss import nll_pc_hazard_loss

def test_loss():
    """Test NLLPCHazardLoss with small example."""
    print('Testing loss with batch_size=2, num_events=2, num_cuts=3')
    
    # Create simple test inputs
    phi = torch.randn(2, 2, 3)
    idx_durations = torch.tensor([[0, 1], [2, 1]], dtype=torch.long)
    events = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
    interval_frac = torch.tensor([[0.5, 0.3], [0.7, 0.6]], dtype=torch.float32)
    
    print('Inputs:')
    print(f'phi: {phi}')
    print(f'phi shape: {phi.shape}')
    print(f'idx_durations: {idx_durations}')
    print(f'idx_durations shape: {idx_durations.shape}')
    print(f'events: {events}')
    print(f'events shape: {events.shape}')
    print(f'interval_frac: {interval_frac}')
    print(f'interval_frac shape: {interval_frac.shape}')
    
    try:
        loss = nll_pc_hazard_loss(phi, idx_durations, events, interval_frac)
        print(f'Loss: {loss.item()}')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loss()