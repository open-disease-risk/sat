"""Test script for NLLPC hazard loss"""

import torch
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("sat")
logger.setLevel(logging.DEBUG)

# Create a stream handler to output logs to stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Import the loss module
from sat.loss.survival.nllpc_hazard import SATNLLPCHazardLoss
from sat.models.tasks.heads.outputs import SAOutput

# Create test data
batch_size = 4
num_duration_bins = 5
num_events = 1

# Create test predictions
logits = torch.randn(batch_size, num_events, num_duration_bins, requires_grad=True)
hazard = torch.sigmoid(logits)
risk = 1.0 - torch.exp(-torch.cumsum(hazard, dim=2))
survival = 1.0 - risk

# Create test references 
# Format: [duration_indices, events, fraction_durations, durations]
references = torch.zeros(batch_size, 4 * num_events)
# Duration indices for each event (discrete time bins, should be Long)
references[:, :num_events] = torch.randint(0, num_duration_bins-1, (batch_size, num_events)).float()
# Event indicators (1 = event, 0 = censored)
references[:, num_events:2*num_events] = torch.randint(0, 2, (batch_size, num_events)).float()
# Fraction within the duration bin (between 0 and 1)
references[:, 2*num_events:3*num_events] = torch.rand(batch_size, num_events)
# Actual duration values (might be continuous)
references[:, 3*num_events:4*num_events] = torch.rand(batch_size, num_events) * 10.0

# Create the SAOutput object
output = SAOutput(
    loss=None,
    logits=logits,
    hazard=hazard,
    risk=risk,
    survival=survival,
    hidden_states=None,
)

# Create the loss function
loss_fn = SATNLLPCHazardLoss(num_events=num_events)

# Compute the loss
loss = loss_fn(output, references)
print(f"Loss: {loss.item()}")

# Check if we can compute gradients
loss.backward()
print(f"Gradient of logits: {logits.grad}")

print("Test successful!")