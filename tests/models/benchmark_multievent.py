"""Benchmark performance of MultiEventRankingLoss vs SampleRankingLoss."""

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
from sat.loss.base import RankingLoss


def create_fake_data(
    batch_size: int = 16, num_events: int = 2, num_cuts: int = 10
) -> Tuple[SAOutput, torch.Tensor]:
    """Create synthetic data for testing ranking losses."""
    # Create fake hazard and survival tensors
    hazard = torch.rand(batch_size, num_events, num_cuts)

    # Ensure survival is decreasing for visualization
    survival_base = torch.cumsum(
        torch.nn.functional.softplus(torch.randn(batch_size, num_events, num_cuts)),
        dim=2,
    )
    # Scale to 0-1 range and flip to get decreasing values
    max_vals = survival_base.max(dim=2, keepdim=True)[0]
    survival_base = 1 - (survival_base / (max_vals + 1e-6))
    # Add survival at time 0 (always 1.0)
    ones = torch.ones(batch_size, num_events, 1)
    survival = torch.cat([ones, survival_base], dim=2)

    # Create logits (we don't need them for the test)
    logits = torch.zeros(batch_size, num_events, num_cuts)

    # Create targets
    # Each row is: [duration_percentile, event, fraction, duration] for each event
    # For num_events=2, shape will be [batch_size, 8]
    targets = torch.zeros(batch_size, 4 * num_events)

    # Set some events to 1
    for i in range(batch_size):
        event_type = i % num_events
        targets[i, num_events + event_type] = 1  # Set event indicator
        targets[i, 3 * num_events + event_type] = i % num_cuts + 1  # Set duration
        # Set duration index (percentile)
        targets[i, event_type] = i % num_cuts

    # For competing risks test, make some samples have multiple events
    for i in range(0, batch_size, 5):
        if i + 1 < batch_size and num_events >= 2:
            # Set both events to 1 for this sample
            targets[i, num_events : 2 * num_events] = 1.0
            # Set different durations for each event
            targets[i, 3 * num_events] = i % num_cuts + 1  # First event duration

            # Only set second event duration if we have at least 2 events
            if num_events >= 2:
                targets[i, 3 * num_events + 1] = (
                    i + 3
                ) % num_cuts + 1  # Second event duration

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


def benchmark_losses(
    batch_sizes: List[int] = [8, 16, 32, 64, 128],
    num_events: int = 2,
    num_cuts: int = 10,
    num_iterations: int = 10,
):
    """Benchmark MultiEventRankingLoss vs SampleRankingLoss for different batch sizes."""
    # Create files needed for loss initialization
    duration_cuts_file = create_duration_cuts_file(num_cuts)
    importance_weights_file = create_importance_weights_file(num_events)

    try:
        # Dictionary to store results
        results = {
            "batch_size": [],
            "multievent_forward": [],
            "sample_forward": [],
            "multievent_backward": [],
            "sample_backward": [],
            "multievent_memory": [],
            "sample_memory": [],
        }

        # Create loss instances once to initialize any buffers
        _ = MultiEventRankingLoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=importance_weights_file,
            num_events=num_events,
            sigma=0.1,
            margin=0.05,
        )

        _ = SampleRankingLoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=importance_weights_file,
            num_events=num_events,
            sigma=0.1,
            margin=0.05,
        )

        # Run benchmarks
        for batch_size in batch_sizes:
            print(f"Benchmarking batch size {batch_size}...")

            # Create new instances for each test
            multi_loss = MultiEventRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                sigma=0.1,
                margin=0.05,
            )

            sample_loss = SampleRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                sigma=0.1,
                margin=0.05,
            )

            # Time forward pass - MultiEventRankingLoss
            multi_forward_times = []
            multi_backward_times = []
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
                multi_forward_times.append(end_time - start_time)

                # Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                multi_backward_times.append(end_time - start_time)

            # Measure peak memory usage if using CUDA
            multi_memory = 0
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
                multi_memory = torch.cuda.max_memory_allocated() / (
                    1024 * 1024
                )  # Convert to MB
                torch.cuda.empty_cache()

            # Time forward pass - SampleRankingLoss
            sample_forward_times = []
            sample_backward_times = []
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
                loss = sample_loss(predictions, targets)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                sample_forward_times.append(end_time - start_time)

                # Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                sample_backward_times.append(end_time - start_time)

            # Measure peak memory usage if using CUDA
            sample_memory = 0
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

                loss = sample_loss(predictions, targets)
                loss.backward()
                sample_memory = torch.cuda.max_memory_allocated() / (
                    1024 * 1024
                )  # Convert to MB
                torch.cuda.empty_cache()

            # Calculate average times
            avg_multi_forward = np.mean(multi_forward_times) * 1000  # Convert to ms
            avg_sample_forward = np.mean(sample_forward_times) * 1000
            avg_multi_backward = np.mean(multi_backward_times) * 1000
            avg_sample_backward = np.mean(sample_backward_times) * 1000

            # Store results
            results["batch_size"].append(batch_size)
            results["multievent_forward"].append(avg_multi_forward)
            results["sample_forward"].append(avg_sample_forward)
            results["multievent_backward"].append(avg_multi_backward)
            results["sample_backward"].append(avg_sample_backward)
            results["multievent_memory"].append(multi_memory)
            results["sample_memory"].append(sample_memory)

            # Print results for this batch size
            print(
                f"  MultiEventRankingLoss - Forward: {avg_multi_forward:.4f} ms, Backward: {avg_multi_backward:.4f} ms"
            )
            print(
                f"  SampleRankingLoss     - Forward: {avg_sample_forward:.4f} ms, Backward: {avg_sample_backward:.4f} ms"
            )
            print(
                f"  Ratio (Multi/Sample)  - Forward: {avg_multi_forward/avg_sample_forward:.2f}x, Backward: {avg_multi_backward/avg_sample_backward:.2f}x"
            )

            if torch.cuda.is_available():
                print(
                    f"  Memory Usage - Multi: {multi_memory:.2f} MB, Sample: {sample_memory:.2f} MB, Ratio: {multi_memory/sample_memory:.2f}x"
                )

            print("")

        # Return results
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
    """Benchmark MultiEventRankingLoss vs SampleRankingLoss for different numbers of events."""
    results = {
        "num_events": [],
        "multievent_forward": [],
        "sample_forward": [],
        "multievent_backward": [],
        "sample_backward": [],
    }

    for num_events in num_events_list:
        print(f"Benchmarking {num_events} events...")

        # Create files needed for loss initialization
        duration_cuts_file = create_duration_cuts_file(num_cuts)
        importance_weights_file = create_importance_weights_file(num_events)

        try:
            # Create loss instances
            multi_loss = MultiEventRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                sigma=0.1,
                margin=0.05,
            )

            sample_loss = SampleRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                sigma=0.1,
                margin=0.05,
            )

            # Time forward pass - MultiEventRankingLoss
            multi_forward_times = []
            multi_backward_times = []
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
                multi_forward_times.append(end_time - start_time)

                # Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                multi_backward_times.append(end_time - start_time)

            # Time forward pass - SampleRankingLoss
            sample_forward_times = []
            sample_backward_times = []
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
                loss = sample_loss(predictions, targets)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                sample_forward_times.append(end_time - start_time)

                # Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                sample_backward_times.append(end_time - start_time)

            # Calculate average times
            avg_multi_forward = np.mean(multi_forward_times) * 1000  # Convert to ms
            avg_sample_forward = np.mean(sample_forward_times) * 1000
            avg_multi_backward = np.mean(multi_backward_times) * 1000
            avg_sample_backward = np.mean(sample_backward_times) * 1000

            # Store results
            results["num_events"].append(num_events)
            results["multievent_forward"].append(avg_multi_forward)
            results["sample_forward"].append(avg_sample_forward)
            results["multievent_backward"].append(avg_multi_backward)
            results["sample_backward"].append(avg_sample_backward)

            # Print results for this number of events
            print(
                f"  MultiEventRankingLoss - Forward: {avg_multi_forward:.4f} ms, Backward: {avg_multi_backward:.4f} ms"
            )
            print(
                f"  SampleRankingLoss     - Forward: {avg_sample_forward:.4f} ms, Backward: {avg_sample_backward:.4f} ms"
            )
            print(
                f"  Ratio (Multi/Sample)  - Forward: {avg_multi_forward/avg_sample_forward:.2f}x, Backward: {avg_multi_backward/avg_sample_backward:.2f}x"
            )
            print("")

        finally:
            # Clean up temporary files
            os.unlink(duration_cuts_file)
            os.unlink(importance_weights_file)

    # Return results
    return results


def plot_benchmark_results(results, title, x_label, y_label, save_path=None):
    """Plot benchmark results."""
    plt.figure(figsize=(10, 6))

    # Extract data
    x = results[x_label]

    # Plot forward pass times
    plt.plot(
        x, results["multievent_forward"], "b-o", label="MultiEventRankingLoss (Forward)"
    )
    plt.plot(x, results["sample_forward"], "r-o", label="SampleRankingLoss (Forward)")

    # Plot backward pass times
    plt.plot(
        x,
        results["multievent_backward"],
        "b--^",
        label="MultiEventRankingLoss (Backward)",
    )
    plt.plot(
        x, results["sample_backward"], "r--^", label="SampleRankingLoss (Backward)"
    )

    # Add labels and legend
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_memory_usage(results, save_path=None):
    """Plot memory usage."""
    if "multievent_memory" not in results or results["multievent_memory"][0] == 0:
        # No memory measurements available
        return

    plt.figure(figsize=(10, 6))

    # Extract data
    x = results["batch_size"]

    # Plot memory usage
    plt.plot(x, results["multievent_memory"], "b-o", label="MultiEventRankingLoss")
    plt.plot(x, results["sample_memory"], "r-o", label="SampleRankingLoss")

    # Add labels and legend
    plt.title("Memory Usage Comparison")
    plt.xlabel("Batch Size")
    plt.ylabel("Memory Usage (MB)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)

    plt.show()


def main():
    """Run benchmarks and plot results."""
    print("Running batch size benchmarks...")
    batch_results = benchmark_losses(
        batch_sizes=[8, 16, 32, 64, 128], num_events=2, num_cuts=10, num_iterations=10
    )

    print("\nRunning num_events benchmarks...")
    events_results = benchmark_num_events(
        batch_size=32, num_events_list=[1, 2, 4, 8, 16], num_cuts=10, num_iterations=10
    )

    # Plot results
    plot_benchmark_results(
        batch_results,
        "Performance Comparison by Batch Size",
        "batch_size",
        "Time (ms)",
        save_path="benchmark_batch_size.png",
    )

    plot_benchmark_results(
        events_results,
        "Performance Comparison by Number of Events",
        "num_events",
        "Time (ms)",
        save_path="benchmark_num_events.png",
    )

    # Plot memory usage if available
    plot_memory_usage(batch_results, save_path="memory_usage.png")


if __name__ == "__main__":
    main()
