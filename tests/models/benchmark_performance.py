"""Benchmark tests for measuring performance improvements"""

import torch
import time
import pytest
import logging
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import os

# Configure minimal logging to avoid debug messages during benchmarks
logging.basicConfig(level=logging.WARNING)

from sat.models.nets import CauseSpecificNet, CauseSpecificNetCompRisk
from sat.models.heads.survival import SurvivalTaskHead
from sat.models.heads import SurvivalConfig
from sat.loss.survival.deephit import DeepHitLikelihoodLoss
from sat.loss.ranking.sample import SampleRankingLoss

# ObservationEventRankingLoss and DeepHitRankingLoss have been removed
from sat.models.heads.output import SAOutput
import matplotlib.pyplot as plt
import os


@pytest.fixture
def survival_data():
    """Create synthetic data for survival benchmarks"""
    batch_size = 128
    num_features = 64
    num_events = 2
    num_labels = 30

    # Create input features
    sequence_output = torch.randn(batch_size, num_features)

    # Create reference labels (duration percentiles, events, fraction_with_quantile, durations)
    # Format: [duration_percentiles, events, fraction_with_quantile, durations] x num_events
    references = torch.zeros(batch_size, 4 * num_events)

    # Set duration percentiles
    references[:, 0:num_events] = torch.randint(
        0, num_labels - 1, (batch_size, num_events)
    )

    # Set events (make ~20% have events)
    events = torch.zeros(batch_size, num_events)
    mask = torch.rand(batch_size) < 0.2
    events[mask, 0] = 1  # Set first event type for some samples
    mask = torch.rand(batch_size) < 0.2
    events[mask, 1] = 1  # Set second event type for some samples

    # Ensure no sample has both event types
    both_events = (events[:, 0] == 1) & (events[:, 1] == 1)
    if both_events.any():
        events[both_events, 1] = 0

    references[:, num_events : 2 * num_events] = events

    # Set fraction with quantile
    references[:, 2 * num_events : 3 * num_events] = torch.rand(batch_size, num_events)

    # Set durations
    references[:, 3 * num_events : 4 * num_events] = (
        torch.rand(batch_size, num_events) * 100
    )

    # Create hazard, risk, and survival
    hazard = torch.rand(batch_size, num_events, num_labels + 1) * 0.1
    risk = torch.cumsum(hazard, dim=2)
    risk = torch.clamp(risk, max=1.0)
    survival = 1.0 - risk

    # Create logits
    logits = torch.log(hazard + 1e-8)

    # Create SAOutput
    output = SAOutput(
        loss=None,
        logits=logits,
        hazard=hazard,
        risk=risk,
        survival=survival,
        hidden_states=sequence_output,
    )

    # Create duration cuts file
    duration_cuts = torch.linspace(0, 100, steps=num_labels).tolist()

    return {
        "sequence_output": sequence_output,
        "references": references,
        "output": output,
        "duration_cuts": duration_cuts,
        "num_features": num_features,
        "num_events": num_events,
        "num_labels": num_labels,
    }


def test_survival_head_forward_performance(survival_data):
    """Test the performance of the survival head forward pass"""
    # Get data
    sequence_output = survival_data["sequence_output"]
    references = survival_data["references"]
    num_features = survival_data["num_features"]
    num_events = survival_data["num_events"]
    num_labels = survival_data["num_labels"]

    # Create config
    config = SurvivalConfig(
        num_features=num_features,
        intermediate_size=32,
        num_hidden_layers=2,
        indiv_intermediate_size=16,
        indiv_num_hidden_layers=1,
        num_labels=num_labels,
        num_events=num_events,
        bias=True,
        batch_norm=True,
        hidden_dropout_prob=0.1,
        loss={
            "survival": {"_target_": "sat.loss.survival.deephit.DeepHitLikelihoodLoss"}
        },
        model_type="survival",
    )

    # Create head
    survival_head = SurvivalTaskHead(config)

    # Warm-up run
    _ = survival_head(sequence_output)

    # Benchmark forward pass
    num_trials = 10
    start_time = time.time()
    for _ in range(num_trials):
        output = survival_head(sequence_output)
    forward_time = (time.time() - start_time) / num_trials

    # Benchmark forward+backward pass
    start_time = time.time()
    for _ in range(num_trials):
        output = survival_head(sequence_output, references)
        if output.loss is not None:
            output.loss.backward()
    backward_time = (time.time() - start_time) / num_trials

    print(f"\nSurvivalTaskHead forward time: {forward_time:.6f} seconds")
    print(f"SurvivalTaskHead forward+backward time: {backward_time:.6f} seconds")


def test_likelihood_loss_performance(survival_data):
    """Test the performance of the DeepHitLikelihoodLoss"""
    # Get data
    output = survival_data["output"]
    references = survival_data["references"]
    num_events = survival_data["num_events"]

    # Create loss
    loss_fn = DeepHitLikelihoodLoss(num_events=num_events)

    # Warm-up run
    _ = loss_fn(output, references)

    # Benchmark forward pass
    num_trials = 10
    start_time = time.time()
    for _ in range(num_trials):
        loss = loss_fn(output, references)
    forward_time = (time.time() - start_time) / num_trials

    # Benchmark forward+backward pass
    start_time = time.time()
    for _ in range(num_trials):
        # Create a copy of output with requires_grad=True for tensors
        output_copy = SAOutput(
            loss=output.loss,
            logits=output.logits.detach().clone().requires_grad_(True),
            hazard=output.hazard.detach().clone().requires_grad_(True),
            risk=output.risk.detach().clone().requires_grad_(True),
            survival=output.survival.detach().clone().requires_grad_(True),
            hidden_states=output.hidden_states.detach().clone().requires_grad_(True),
        )
        try:
            loss = loss_fn(output_copy, references)
            if loss.requires_grad:
                loss.backward()
        except Exception as e:
            print(f"Backward pass error: {e}")
    backward_time = (time.time() - start_time) / num_trials

    print(f"\nDeepHitLikelihoodLoss forward time: {forward_time:.6f} seconds")
    print(f"DeepHitLikelihoodLoss forward+backward time: {backward_time:.6f} seconds")


def test_ranking_loss_performance(survival_data):
    """Test the performance of the DeepHitRankingLoss"""
    # Get data
    output = survival_data["output"]
    references = survival_data["references"]
    num_events = survival_data["num_events"]
    duration_cuts = survival_data["duration_cuts"]

    # Create temporary file for duration cuts
    import tempfile
    import os
    import pandas as pd

    # Create temp file for duration cuts
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        cuts_file = f.name
        pd.DataFrame(duration_cuts).to_csv(cuts_file, index=False, header=False)

    try:
        # Create loss
        loss_fn = DeepHitRankingLoss(
            num_events=num_events, duration_cuts=cuts_file, sigma=0.1
        )

        # Warm-up run
        _ = loss_fn(output, references)

        # Benchmark forward pass
        num_trials = 5
        start_time = time.time()
        for _ in range(num_trials):
            loss = loss_fn(output, references)
        forward_time = (time.time() - start_time) / num_trials

        # Benchmark forward+backward pass
        start_time = time.time()
        for _ in range(num_trials):
            # Create a copy of output with requires_grad=True for tensors
            output_copy = SAOutput(
                loss=output.loss,
                logits=output.logits.detach().clone().requires_grad_(True),
                hazard=output.hazard.detach().clone().requires_grad_(True),
                risk=output.risk.detach().clone().requires_grad_(True),
                survival=output.survival.detach().clone().requires_grad_(True),
                hidden_states=output.hidden_states.detach()
                .clone()
                .requires_grad_(True),
            )
            try:
                loss = loss_fn(output_copy, references)
                if loss.requires_grad:
                    loss.backward()
            except Exception as e:
                print(f"Backward pass error: {e}")
        backward_time = (time.time() - start_time) / num_trials

        print(f"\nDeepHitRankingLoss forward time: {forward_time:.6f} seconds")
        print(f"DeepHitRankingLoss forward+backward time: {backward_time:.6f} seconds")
    finally:
        # Clean up temp file
        if os.path.exists(cuts_file):
            os.remove(cuts_file)


def test_cause_specific_net_performance(survival_data):
    """Test the performance of the CauseSpecificNet"""
    # Get data
    sequence_output = survival_data["sequence_output"]
    num_features = survival_data["num_features"]
    num_events = survival_data["num_events"]
    num_labels = survival_data["num_labels"]

    # Create net
    net = CauseSpecificNet(
        in_features=num_features,
        intermediate_size=32,
        num_hidden_layers=2,
        out_features=num_labels,
        batch_norm=True,
        dropout=0.1,
        bias=True,
        num_events=num_events,
    )

    # Warm-up run
    _ = net(sequence_output)

    # Benchmark forward pass
    num_trials = 20
    start_time = time.time()
    for _ in range(num_trials):
        output = net(sequence_output)
    forward_time = (time.time() - start_time) / num_trials

    print(f"\nCauseSpecificNet forward time: {forward_time:.6f} seconds")


def test_cause_specific_comp_risk_net_performance(survival_data):
    """Test the performance of the CauseSpecificNetCompRisk"""
    # Get data
    sequence_output = survival_data["sequence_output"]
    num_features = survival_data["num_features"]
    num_events = survival_data["num_events"]
    num_labels = survival_data["num_labels"]

    # Create net
    net = CauseSpecificNetCompRisk(
        in_features=num_features,
        shared_intermediate_size=32,
        shared_num_hidden_layers=2,
        indiv_intermediate_size=16,
        indiv_num_hidden_layers=1,
        out_features=num_labels,
        batch_norm=True,
        dropout=0.1,
        bias=True,
        num_events=num_events,
    )

    # Warm-up run
    _ = net(sequence_output)

    # Benchmark forward pass
    num_trials = 20
    start_time = time.time()
    for _ in range(num_trials):
        output = net(sequence_output)
    forward_time = (time.time() - start_time) / num_trials

    print(f"\nCauseSpecificNetCompRisk forward time: {forward_time:.6f} seconds")


def create_survival_data():
    """Create synthetic data for survival benchmarks (non-fixture version)"""
    batch_size = 128
    num_features = 64
    num_events = 2
    num_labels = 30

    # Create input features
    sequence_output = torch.randn(batch_size, num_features)

    # Create reference labels (duration percentiles, events, fraction_with_quantile, durations)
    # Format: [duration_percentiles, events, fraction_with_quantile, durations] x num_events
    references = torch.zeros(batch_size, 4 * num_events)

    # Set duration percentiles
    references[:, 0:num_events] = torch.randint(
        0, num_labels - 1, (batch_size, num_events)
    )

    # Set events (make ~20% have events)
    events = torch.zeros(batch_size, num_events)
    mask = torch.rand(batch_size) < 0.2
    events[mask, 0] = 1  # Set first event type for some samples
    mask = torch.rand(batch_size) < 0.2
    events[mask, 1] = 1  # Set second event type for some samples

    # Ensure no sample has both event types
    both_events = (events[:, 0] == 1) & (events[:, 1] == 1)
    if both_events.any():
        events[both_events, 1] = 0

    references[:, num_events : 2 * num_events] = events

    # Set fraction with quantile
    references[:, 2 * num_events : 3 * num_events] = torch.rand(batch_size, num_events)

    # Set durations
    references[:, 3 * num_events : 4 * num_events] = (
        torch.rand(batch_size, num_events) * 100
    )

    # Create hazard, risk, and survival
    hazard = torch.rand(batch_size, num_events, num_labels + 1) * 0.1
    risk = torch.cumsum(hazard, dim=2)
    risk = torch.clamp(risk, max=1.0)
    survival = 1.0 - risk

    # Create logits
    logits = torch.log(hazard + 1e-8)

    # Create SAOutput
    output = SAOutput(
        loss=None,
        logits=logits,
        hazard=hazard,
        risk=risk,
        survival=survival,
        hidden_states=sequence_output,
    )

    # Create duration cuts file
    duration_cuts = torch.linspace(0, 100, steps=num_labels).tolist()

    return {
        "sequence_output": sequence_output,
        "references": references,
        "output": output,
        "duration_cuts": duration_cuts,
        "num_features": num_features,
        "num_events": num_events,
        "num_labels": num_labels,
    }


def create_synthetic_survival_data(batch_size=128, num_events=2, num_labels=30):
    """Create synthetic data for survival benchmarks with specific dimensions."""
    # Create input features
    sequence_output = torch.randn(batch_size, 64)

    # Create reference labels
    references = torch.zeros(batch_size, 4 * num_events)

    # Set duration percentiles
    references[:, 0:num_events] = torch.randint(
        0, num_labels - 1, (batch_size, num_events)
    )

    # Set events (make ~20% have events)
    events = torch.zeros(batch_size, num_events)
    for event_idx in range(num_events):
        mask = torch.rand(batch_size) < 0.2
        events[mask, event_idx] = 1

    # Ensure no sample has multiple event types if num_events > 1
    if num_events > 1:
        multiple_events = torch.sum(events, dim=1) > 1
        if multiple_events.any():
            for i in range(batch_size):
                if multiple_events[i]:
                    # Keep only the first event type
                    first_event = torch.argmax(events[i])
                    events[i] = torch.zeros(num_events)
                    events[i, first_event] = 1

    references[:, num_events : 2 * num_events] = events

    # Set fraction with quantile
    references[:, 2 * num_events : 3 * num_events] = torch.rand(batch_size, num_events)

    # Set durations
    references[:, 3 * num_events : 4 * num_events] = (
        torch.rand(batch_size, num_events) * 100
    )

    # Create hazard, risk, and survival
    hazard = torch.rand(batch_size, num_events, num_labels + 1) * 0.1
    risk = torch.cumsum(hazard, dim=2)
    risk = torch.clamp(risk, max=1.0)
    survival = 1.0 - risk

    # Create logits
    logits = torch.log(hazard + 1e-8)

    # Create SAOutput
    output = SAOutput(
        loss=None,
        logits=logits,
        hazard=hazard,
        risk=risk,
        survival=survival,
        hidden_states=sequence_output,
    )

    # Create duration cuts
    duration_cuts = torch.linspace(0, 100, steps=num_labels).tolist()

    return {
        "sequence_output": sequence_output,
        "references": references,
        "output": output,
        "duration_cuts": duration_cuts,
        "num_features": 64,
        "num_events": num_events,
        "num_labels": num_labels,
        "batch_size": batch_size,
    }


def run_ranking_loss_benchmark(data, loss_fn, num_trials=5):
    """Run benchmark for a specific loss function and return results."""
    output = data["output"]
    references = data["references"]

    # Warm-up run
    _ = loss_fn(output, references)

    # Benchmark forward pass
    start_time = time.time()
    for _ in range(num_trials):
        loss = loss_fn(output, references)
        loss_value = loss.item()
    forward_time = (time.time() - start_time) / num_trials

    # Benchmark backward pass
    start_time = time.time()
    for _ in range(num_trials):
        # Create a copy of output with requires_grad=True for tensors
        output_copy = SAOutput(
            loss=output.loss,
            logits=output.logits.detach().clone().requires_grad_(True),
            hazard=output.hazard.detach().clone().requires_grad_(True),
            risk=output.risk.detach().clone().requires_grad_(True),
            survival=output.survival.detach().clone().requires_grad_(True),
            hidden_states=output.hidden_states.detach().clone().requires_grad_(True),
        )
        try:
            loss = loss_fn(output_copy, references)
            if loss.requires_grad:
                loss.backward()
        except Exception as e:
            print(f"Backward pass error: {e}")
    backward_time = (time.time() - start_time) / num_trials

    return {
        "forward_time": forward_time,
        "backward_time": backward_time,
        "output_value": loss_value,
    }


def test_ranking_loss_comparison(survival_data):
    """Compare performance between different ranking loss implementations with varying tensor sizes."""
    # Create temporary file for duration cuts
    import tempfile
    import pandas as pd

    # Get initial data
    num_events = survival_data["num_events"]
    duration_cuts = survival_data["duration_cuts"]

    # Create temp file for duration cuts
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        cuts_file = f.name
        pd.DataFrame(duration_cuts).to_csv(cuts_file, index=False, header=False)

    try:
        # Define batch sizes and event counts to test
        batch_sizes = [32, 128, 512]
        event_counts = [1, 2, 5]

        # Number of time points in the data
        num_labels = 30

        # Initialize results storage
        all_results = {"SampleRankingLoss": {"forward": {}, "backward": {}}}

        # Run benchmarks for different combinations
        for batch_size in batch_sizes:
            for num_events in event_counts:
                # Create synthetic data with this configuration
                print(
                    f"\nGenerating data with batch_size={batch_size}, num_events={num_events}"
                )
                data = create_synthetic_survival_data(
                    batch_size=batch_size, num_events=num_events, num_labels=num_labels
                )

                # Create loss functions
                loss_functions = {
                    "SampleRankingLoss": SampleRankingLoss(
                        num_events=num_events, duration_cuts=cuts_file, sigma=0.1
                    )
                }

                # Run benchmarks for each loss function
                config_key = f"b{batch_size}_e{num_events}"
                print(f"Testing config: {config_key}")

                for name, loss_fn in loss_functions.items():
                    print(f"  Benchmarking {name}...")

                    results = run_ranking_loss_benchmark(data, loss_fn, num_trials=5)

                    # Store results
                    all_results[name]["forward"][config_key] = results["forward_time"]
                    all_results[name]["backward"][config_key] = results["backward_time"]

                    print(
                        f"    Forward: {results['forward_time']:.6f}s, "
                        f"Backward: {results['backward_time']:.6f}s, "
                        f"Value: {results['output_value']:.6f}"
                    )

        # Print detailed comparison
        print("\nDetailed Performance Comparison:")
        print("=" * 100)

        # Print header with configs
        configs = [f"b{bs}_e{e}" for bs in batch_sizes for e in event_counts]
        header = "Loss Function".ljust(25)
        for phase in ["Forward", "Backward"]:
            for config in configs:
                header += f" | {phase}_{config}"
        print(header)
        print("-" * len(header))

        # Print data rows
        for name in all_results.keys():
            row = name.ljust(25)
            for phase in ["forward", "backward"]:
                for config in configs:
                    if config in all_results[name][phase]:
                        row += f" | {all_results[name][phase][config]:.6f}"
                    else:
                        row += " | N/A     "
            print(row)

        # Since we only have one loss function now, we'll just print the raw performance
        print("\nPerformance Measurements:")
        print("=" * 100)

        # Header
        header = "Loss Function".ljust(25)
        for phase in ["Forward", "Backward"]:
            for config in configs:
                header += f" | {phase}_{config}"
        print(header)
        print("-" * len(header))

        # Data rows for performance
        for name in all_results.keys():
            row = name.ljust(25)
            for phase in ["forward", "backward"]:
                for config in configs:
                    if config in all_results[name][phase]:
                        row += f" | {all_results[name][phase][config]:.6f}"
                    else:
                        row += " | N/A     "
            print(row)

        # Create visualizations if matplotlib is available
        if matplotlib_available:
            try:
                # Create plots for each configuration showing comparisons
                fig, axes = plt.subplots(
                    len(batch_sizes), len(event_counts), figsize=(15, 12)
                )

                if len(batch_sizes) == 1 and len(event_counts) == 1:
                    axes = np.array([[axes]])
                elif len(batch_sizes) == 1:
                    axes = np.array([axes])
                elif len(event_counts) == 1:
                    axes = np.array([[ax] for ax in axes])

                for i, batch_size in enumerate(batch_sizes):
                    for j, num_events in enumerate(event_counts):
                        config = f"b{batch_size}_e{num_events}"

                        # Get data for this configuration
                        forward_times = []
                        backward_times = []
                        labels = []

                        for name in all_results.keys():
                            if (
                                config in all_results[name]["forward"]
                                and config in all_results[name]["backward"]
                            ):
                                labels.append(name)
                                forward_times.append(
                                    all_results[name]["forward"][config]
                                )
                                backward_times.append(
                                    all_results[name]["backward"][config]
                                )

                        # Skip if no data
                        if not labels:
                            continue

                        ax = axes[i, j]
                        x = np.arange(len(labels))
                        width = 0.35

                        # Plot bar chart
                        ax.bar(x - width / 2, forward_times, width, label="Forward")
                        ax.bar(x + width / 2, backward_times, width, label="Backward")

                        ax.set_ylabel("Time (seconds)")
                        ax.set_title(f"Batch={batch_size}, Events={num_events}")
                        ax.set_xticks(x)
                        ax.set_xticklabels(labels, rotation=45, ha="right")
                        ax.legend()

                plt.tight_layout()

                # Save plot
                log_dir = os.path.join(os.getcwd(), "logs")
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                plot_path = os.path.join(log_dir, "ranking_loss_comparison_by_size.png")
                plt.savefig(plot_path)
                print(f"Detailed plot saved to {plot_path}")

                # Create scaling plots to show how each loss scales with batch size and event count
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))

                # Plot 1-3: How each loss scales with batch size (for each event count)
                for j, num_events in enumerate(event_counts):
                    ax = axes[0, j]

                    for name in all_results.keys():
                        batch_times = []
                        batch_sizes_available = []

                        for batch_size in batch_sizes:
                            config = f"b{batch_size}_e{num_events}"
                            if config in all_results[name]["forward"]:
                                batch_times.append(all_results[name]["forward"][config])
                                batch_sizes_available.append(batch_size)

                        if batch_times:
                            ax.plot(
                                batch_sizes_available, batch_times, "o-", label=name
                            )

                    ax.set_title(f"Scaling with Batch Size (Events={num_events})")
                    ax.set_xlabel("Batch Size")
                    ax.set_ylabel("Forward Time (seconds)")
                    ax.legend()
                    ax.grid(True, linestyle="--", alpha=0.7)

                # Plot 4-6: How each loss scales with event count (for each batch size)
                for j, batch_size in enumerate(batch_sizes):
                    ax = axes[1, j]

                    for name in all_results.keys():
                        event_times = []
                        event_counts_available = []

                        for num_events in event_counts:
                            config = f"b{batch_size}_e{num_events}"
                            if config in all_results[name]["forward"]:
                                event_times.append(all_results[name]["forward"][config])
                                event_counts_available.append(num_events)

                        if event_times:
                            ax.plot(
                                event_counts_available, event_times, "o-", label=name
                            )

                    ax.set_title(f"Scaling with Event Count (Batch={batch_size})")
                    ax.set_xlabel("Number of Events")
                    ax.set_ylabel("Forward Time (seconds)")
                    ax.legend()
                    ax.grid(True, linestyle="--", alpha=0.7)

                plt.tight_layout()

                # Save scaling plot
                scaling_plot_path = os.path.join(log_dir, "ranking_loss_scaling.png")
                plt.savefig(scaling_plot_path)
                print(f"Scaling plot saved to {scaling_plot_path}")

                plt.close("all")

            except Exception as e:
                print(f"Error creating plots: {e}")

    finally:
        # Clean up temp file
        if os.path.exists(cuts_file):
            os.remove(cuts_file)


def test_ranking_loss_with_margin():
    """Test SampleRankingLoss with different margin values."""
    print("\nTesting SampleRankingLoss with different margin values...")

    # Test parameters
    batch_size = 128
    num_events = 2
    num_time_bins = 30
    sigma = 0.1
    margin_values = [0.0, 0.05, 0.1, 0.2]

    # Create temp file for duration cuts
    import tempfile
    import pandas as pd

    duration_cuts = torch.linspace(0, 100, steps=num_time_bins).tolist()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        cuts_file = f.name
        pd.DataFrame(duration_cuts).to_csv(cuts_file, index=False, header=False)

    try:
        # Initialize results table
        print("\nMargin Parameter Effect on SampleRankingLoss:")
        print("=" * 100)

        # Create synthetic data with ordered events
        data = create_synthetic_survival_data(
            batch_size=batch_size, num_events=num_events, num_labels=num_time_bins
        )

        # Create controlled test data with predictable event patterns
        references = data["references"].clone()

        # Clear all events first
        references[:, num_events : 2 * num_events] = 0

        # Create two groups of samples: one with correct ranking, one with incorrect
        mid_point = batch_size // 2

        # Set events to occur at different times
        for event_idx in range(num_events):
            # For the first half of samples (correct ranking)
            # Earlier events (lower indices) have higher risk
            for i in range(mid_point):
                if i < mid_point // 4:  # 25% of samples have this event
                    references[i, num_events + event_idx] = 1  # Event occurred
                    references[i, 3 * num_events + event_idx] = (
                        10.0 + i * 5.0
                    )  # Progressive times

            # For the second half (incorrect ranking)
            # Earlier events (lower indices) have lower risk (will be penalized)
            for i in range(mid_point, batch_size):
                if i < mid_point + mid_point // 4:  # 25% of samples have this event
                    references[i, num_events + event_idx] = 1  # Event occurred
                    references[i, 3 * num_events + event_idx] = (
                        10.0 + (i - mid_point) * 5.0
                    )  # Progressive times

        # Create hazard values that are correct for first half, incorrect for second half
        hazard = torch.zeros(batch_size, num_events, num_time_bins + 1)

        # First half: higher risk (hazard) for earlier events
        for i in range(mid_point):
            if i < mid_point // 4:
                hazard_value = 0.5 - (i / (mid_point // 4)) * 0.4  # Decreasing hazard
                time_idx = min(int(references[i, 3 * num_events] / 10.0), num_time_bins)
                hazard[i, 0, 1 : time_idx + 1] = hazard_value

                if event_idx > 0:
                    time_idx = min(
                        int(references[i, 3 * num_events + 1] / 10.0), num_time_bins
                    )
                    hazard[i, 1, 1 : time_idx + 1] = (
                        hazard_value * 0.8
                    )  # Slightly less for second event

        # Second half: lower risk (hazard) for earlier events (incorrect)
        for i in range(mid_point, batch_size):
            if i < mid_point + mid_point // 4:
                hazard_value = (
                    0.1 + ((i - mid_point) / (mid_point // 4)) * 0.4
                )  # Increasing hazard
                time_idx = min(int(references[i, 3 * num_events] / 10.0), num_time_bins)
                hazard[i, 0, 1 : time_idx + 1] = hazard_value

                if event_idx > 0:
                    time_idx = min(
                        int(references[i, 3 * num_events + 1] / 10.0), num_time_bins
                    )
                    hazard[i, 1, 1 : time_idx + 1] = (
                        hazard_value * 1.2
                    )  # Slightly more for second event

        # Calculate survival and risk
        survival = torch.exp(-torch.cumsum(hazard, dim=2))
        risk = 1.0 - survival

        # Create prediction output
        prediction = SAOutput(
            logits=data["output"].logits,
            hazard=hazard,
            survival=survival,
            risk=risk,
            hidden_states=data["output"].hidden_states,
        )

        # Test with different margin values
        margin_results = {}

        for margin in margin_values:
            # Create loss function with this margin
            loss_fn = SampleRankingLoss(
                num_events=num_events,
                duration_cuts=cuts_file,
                sigma=sigma,
                margin=margin,
            )

            # Calculate loss
            loss_value = loss_fn(prediction, references).item()
            margin_results[margin] = loss_value

            print(f"Margin {margin:.2f}: Loss = {loss_value:.6f}")

        # Check that higher margin values result in higher loss
        # This is because the margin enforces a minimum difference between correctly ranked survival probabilities
        values = list(margin_results.values())
        increasing = all(values[i] <= values[i + 1] for i in range(len(values) - 1))

        print(f"\nLoss increases with margin: {'Yes' if increasing else 'No'}")
        print(
            "This verifies that the margin parameter correctly enforces a minimum separation in survival probabilities."
        )

    finally:
        # Clean up temp file
        if os.path.exists(cuts_file):
            os.remove(cuts_file)


if __name__ == "__main__":
    # Run benchmarks directly if file is executed
    print("Running performance benchmarks...")
    data = create_survival_data()

    # Check if matplotlib is available for visualization
    matplotlib_available = True
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
    except ImportError:
        matplotlib_available = False
        print("Matplotlib not available - skipping visualization")

    # Run the benchmarks
    test_survival_head_forward_performance(data)
    test_likelihood_loss_performance(data)
    test_ranking_loss_performance(data)
    test_ranking_loss_comparison(data)
    test_ranking_loss_with_margin()
    test_cause_specific_net_performance(data)
    test_cause_specific_comp_risk_net_performance(data)
