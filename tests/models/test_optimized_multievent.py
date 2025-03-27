"""Tests for MultiEventRankingLoss implementation."""

import os
import tempfile
import pandas as pd
import torch
import time
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

from sat.models.heads import SAOutput
from sat.loss.ranking.multievent import MultiEventRankingLoss
from sat.loss.ranking.sample import SampleRankingLoss


def create_fake_data(
    batch_size: int = 16, num_events: int = 2, num_cuts: int = 10
) -> Tuple[SAOutput, torch.Tensor]:
    """Create synthetic data for testing ranking losses."""
    # Create fake hazard (ensure it's non-negative)
    hazard = torch.abs(torch.rand(batch_size, num_events, num_cuts))

    # Ensure survival is decreasing for visualization
    # First create a decreasing sequence for survival
    survival_base = (
        torch.linspace(0.9, 0.1, num_cuts)
        .view(1, 1, -1)
        .expand(batch_size, num_events, -1)
    )
    # Add some randomness but keep it decreasing
    noise = torch.rand(batch_size, num_events, num_cuts) * 0.05
    survival_base = survival_base - noise
    survival_base = torch.clamp(survival_base, min=0.01, max=0.99)

    # Add survival at time 0 (always 1.0)
    ones = torch.ones(batch_size, num_events, 1)
    survival = torch.cat([ones, survival_base], dim=2)

    # Create logits (we don't need them for the test)
    logits = torch.zeros(batch_size, num_events, num_cuts)

    # Create targets
    # Each row is: [duration_percentile, event, fraction, duration] for each event
    # For num_events=2, shape will be [batch_size, 8]
    targets = torch.zeros(batch_size, 4 * num_events)

    # Set some events to 1 and ensure all durations are > 0
    for i in range(batch_size):
        event_type = i % num_events
        targets[i, num_events + event_type] = 1  # Set event indicator
        targets[i, 3 * num_events + event_type] = (
            i % num_cuts
        ) + 1.0  # Set duration > 0
        # Set duration index (percentile) - must be valid index
        targets[i, event_type] = min(i % num_cuts, num_cuts - 1)

    # For competing risks test, make some samples have multiple events
    for i in range(0, batch_size, 5):
        if i + 1 < batch_size:
            # Set both events to 1 for this sample
            targets[i, num_events : 2 * num_events] = 1.0
            # Set different durations for each event (ensure > 0)
            targets[i, 3 * num_events] = (i % num_cuts) + 1.0  # First event duration
            targets[i, 3 * num_events + 1] = (
                (i + 3) % num_cuts
            ) + 1.0  # Second event duration

    # Create fake predictions
    predictions = SAOutput(logits=logits, hazard=hazard, survival=survival)

    return predictions, targets


def create_duration_cuts_file(num_cuts: int = 10) -> str:
    """Create a temporary file with duration cuts."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        duration_cuts = np.linspace(1, 100, num_cuts)
        for cut in duration_cuts:
            f.write(f"{cut}\n")
        return f.name


def create_importance_weights_file(num_events: int = 2) -> str:
    """Create a temporary file with importance weights."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        weights = np.ones(num_events + 1)  # Add one for censoring weight
        for weight in weights:
            f.write(f"{weight}\n")
        return f.name


def test_multievent_ranking_loss():
    """Test MultiEventRankingLoss initialization."""
    batch_size = 8
    num_events = 2
    num_cuts = 10

    # Create files needed for loss initialization
    duration_cuts_file = create_duration_cuts_file(num_cuts)
    importance_weights_file = create_importance_weights_file(num_events)

    try:
        # Create loss instance
        multi_loss = MultiEventRankingLoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=importance_weights_file,
            num_events=num_events,
            sigma=0.1,
            margin=0.05,
        )

        # Test initialization
        assert multi_loss.sigma == 0.1, "Sigma parameter not set correctly"
        assert multi_loss.margin == 0.05, "Margin parameter not set correctly"
        assert multi_loss.num_events == 2, "num_events not set correctly"
        assert (
            len(multi_loss.duration_cuts) == num_cuts
        ), "Duration cuts not set correctly"
        assert multi_loss.weights is not None, "Weights should be initialized"

        # Test with different hyperparameters
        test_params = [
            {"sigma": 0.05, "margin": 0.0},
            {"sigma": 0.1, "margin": 0.0},
            {"sigma": 0.5, "margin": 0.0},
            {"sigma": 1.0, "margin": 0.0},
            {"sigma": 0.1, "margin": 0.01},
            {"sigma": 0.1, "margin": 0.1},
            {"sigma": 0.1, "margin": 0.5},
        ]

        print("\nTesting various hyperparameters:")
        for params in test_params:
            loss = MultiEventRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                **params,
            )

            # Verify parameters are set correctly
            assert (
                loss.sigma == params["sigma"]
            ), f"Sigma not set correctly for {params}"
            assert (
                loss.margin == params["margin"]
            ), f"Margin not set correctly for {params}"
            print(f"Params {params} initialized correctly")

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts_file)
        os.unlink(importance_weights_file)


def benchmark_performance(
    batch_sizes: List[int] = [8, 16, 32, 64, 128, 256],
    num_events: int = 2,
    num_cuts: int = 10,
    num_iterations: int = 10,
):
    """Benchmark performance of MultiEventRankingLoss."""
    print("\nBenchmarking performance:")

    # Create files needed for loss initialization
    duration_cuts_file = create_duration_cuts_file(num_cuts)
    importance_weights_file = create_importance_weights_file(num_events)

    try:
        # Dictionary to store results
        results = {
            "batch_size": [],
            "forward_time": [],
            "backward_time": [],
            "memory_usage": [],
        }

        # Run benchmarks
        for batch_size in batch_sizes:
            print(f"Testing batch size {batch_size}...")

            # Create loss instance
            multi_loss = MultiEventRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                sigma=0.1,
                margin=0.05,
            )

            # Time forward pass
            forward_times = []
            backward_times = []
            for _ in range(num_iterations):
                # Create new data for each iteration
                hazard = torch.rand(
                    batch_size, num_events, num_cuts, requires_grad=True
                )
                ones = torch.ones(batch_size, num_events, 1)
                survival_base = (
                    1
                    - torch.cumsum(torch.nn.functional.softplus(hazard), dim=2)
                    / num_cuts
                )
                survival = torch.cat([ones, survival_base], dim=2)
                logits = torch.zeros_like(hazard)
                predictions = SAOutput(logits=logits, hazard=hazard, survival=survival)

                _, targets = create_fake_data(batch_size, num_events, num_cuts)

                # Clean CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Forward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss = multi_loss(predictions, targets)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                forward_times.append(end_time - start_time)

                # Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                backward_times.append(end_time - start_time)

            # Measure peak memory usage if using CUDA
            memory_usage = 0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                hazard = torch.rand(
                    batch_size, num_events, num_cuts, requires_grad=True, device="cuda"
                )
                ones = torch.ones(batch_size, num_events, 1, device="cuda")
                survival_base = (
                    1
                    - torch.cumsum(torch.nn.functional.softplus(hazard), dim=2)
                    / num_cuts
                )
                survival = torch.cat([ones, survival_base], dim=2)
                logits = torch.zeros_like(hazard)
                predictions = SAOutput(logits=logits, hazard=hazard, survival=survival)

                _, targets = create_fake_data(batch_size, num_events, num_cuts)
                targets = targets.to("cuda")

                loss = multi_loss(predictions, targets)
                loss.backward()
                memory_usage = torch.cuda.max_memory_allocated() / (
                    1024 * 1024
                )  # Convert to MB
                torch.cuda.empty_cache()

            # Calculate average times
            avg_forward_time = np.mean(forward_times) * 1000  # Convert to ms
            avg_backward_time = np.mean(backward_times) * 1000

            # Store results
            results["batch_size"].append(batch_size)
            results["forward_time"].append(avg_forward_time)
            results["backward_time"].append(avg_backward_time)
            results["memory_usage"].append(memory_usage)

            # Print results for this batch size
            print(f"  Forward time: {avg_forward_time:.4f} ms")
            print(f"  Backward time: {avg_backward_time:.4f} ms")

            if torch.cuda.is_available():
                print(f"  Memory usage: {memory_usage:.2f} MB")

            print("")

        # Plot results
        plot_performance_results(results)

        return results

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts_file)
        os.unlink(importance_weights_file)


def benchmark_num_events(
    batch_size: int = 32,
    num_events_list: List[int] = [1, 2, 4, 8, 16],
    num_cuts: int = 10,
    num_iterations: int = 10,
):
    """Benchmark performance scaling with number of events."""
    print("\nBenchmarking scaling with number of events:")

    # Dictionary to store results
    results = {
        "num_events": [],
        "forward_time": [],
        "backward_time": [],
    }

    for num_events in num_events_list:
        print(f"Testing {num_events} events...")

        # Create files needed for loss initialization
        duration_cuts_file = create_duration_cuts_file(num_cuts)
        importance_weights_file = create_importance_weights_file(num_events)

        try:
            # Create loss instance
            multi_loss = MultiEventRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                sigma=0.1,
                margin=0.05,
            )

            # Time forward and backward passes
            forward_times = []
            backward_times = []
            for _ in range(num_iterations):
                # Create new data for each iteration
                hazard = torch.rand(
                    batch_size, num_events, num_cuts, requires_grad=True
                )
                ones = torch.ones(batch_size, num_events, 1)
                survival_base = (
                    1
                    - torch.cumsum(torch.nn.functional.softplus(hazard), dim=2)
                    / num_cuts
                )
                survival = torch.cat([ones, survival_base], dim=2)
                logits = torch.zeros_like(hazard)
                predictions = SAOutput(logits=logits, hazard=hazard, survival=survival)

                _, targets = create_fake_data(batch_size, num_events, num_cuts)

                # Forward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss = multi_loss(predictions, targets)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                forward_times.append(end_time - start_time)

                # Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                backward_times.append(end_time - start_time)

            # Calculate average times
            avg_forward_time = np.mean(forward_times) * 1000  # Convert to ms
            avg_backward_time = np.mean(backward_times) * 1000

            # Store results
            results["num_events"].append(num_events)
            results["forward_time"].append(avg_forward_time)
            results["backward_time"].append(avg_backward_time)

            # Print results for this number of events
            print(f"  Forward time: {avg_forward_time:.4f} ms")
            print(f"  Backward time: {avg_backward_time:.4f} ms")
            print("")

        finally:
            # Clean up temporary files
            os.unlink(duration_cuts_file)
            os.unlink(importance_weights_file)

    # Plot results
    plot_event_scaling_results(results)

    return results


def plot_performance_results(results):
    """Plot performance results."""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot forward pass times
    ax1.plot(results["batch_size"], results["forward_time"], "b-o", label="Forward")
    ax1.set_title("Forward Pass Time")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Time (ms)")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    # Plot backward pass times
    ax2.plot(results["batch_size"], results["backward_time"], "r-^", label="Backward")
    ax2.set_title("Backward Pass Time")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Time (ms)")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig("multievent_performance.png")

    # If memory measurements are available, plot them
    if results["memory_usage"] and results["memory_usage"][0] > 0:
        plt.figure(figsize=(8, 6))

        # Plot memory usage
        plt.plot(results["batch_size"], results["memory_usage"], "g-o")
        plt.title("Memory Usage")
        plt.xlabel("Batch Size")
        plt.ylabel("Memory (MB)")
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig("multievent_memory.png")


def plot_event_scaling_results(results):
    """Plot results for event scaling benchmark."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot computation times
    ax.plot(
        results["num_events"],
        results["forward_time"],
        "b-o",
        label="Forward",
    )
    ax.plot(
        results["num_events"],
        results["backward_time"],
        "r-^",
        label="Backward",
    )
    ax.set_title("Computation Time vs Number of Events")
    ax.set_xlabel("Number of Events")
    ax.set_ylabel("Time (ms)")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.savefig("multievent_event_scaling.png")


def main():
    """Run all tests and benchmarks."""
    # Test loss functionality
    print("Testing MultiEventRankingLoss functionality...")
    test_multievent_ranking_loss()

    # Benchmark performance
    results_batch = benchmark_performance()

    # Benchmark event scaling
    results_events = benchmark_num_events()

    print("\nBenchmark summary:")
    print(f"Maximum forward time: {max(results_batch['forward_time']):.2f} ms")
    print(f"Maximum backward time: {max(results_batch['backward_time']):.2f} ms")

    if results_batch["memory_usage"] and results_batch["memory_usage"][0] > 0:
        print(f"Maximum memory usage: {max(results_batch['memory_usage']):.2f} MB")

    print("\nEvent scaling summary:")
    print(
        f"Maximum forward time with many events: {max(results_events['forward_time']):.2f} ms"
    )
    print(
        f"Maximum backward time with many events: {max(results_events['backward_time']):.2f} ms"
    )


if __name__ == "__main__":
    main()
