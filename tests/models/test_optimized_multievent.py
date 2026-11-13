"""Tests for optimized MultiEventRankingLoss implementation."""

import os
import tempfile
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from sat.loss.ranking.multievent import (
    MultiEventRankingLoss,
    OptimizedMultiEventRankingLoss,
)
from sat.models.heads import SAOutput


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
        if i + 1 < batch_size:
            # Set both events to 1 for this sample
            targets[i, num_events : 2 * num_events] = 1.0
            # Set different durations for each event
            targets[i, 3 * num_events] = i % num_cuts + 1  # First event duration
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


def test_functional_equivalence():
    """Test functional equivalence between original and optimized implementations."""
    batch_size = 32
    num_events = 2
    num_cuts = 10

    # Create test data
    predictions, targets = create_fake_data(batch_size, num_events, num_cuts)

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

        optimized_loss = OptimizedMultiEventRankingLoss(
            duration_cuts=duration_cuts_file,
            importance_sample_weights=importance_weights_file,
            num_events=num_events,
            sigma=0.1,
            margin=0.05,
        )

        # Calculate losses and compare values
        multi_loss_val = multi_loss(predictions, targets)
        optimized_loss_val = optimized_loss(predictions, targets)

        # Test numerical equivalence (allowing for small floating point differences)
        assert torch.isclose(
            multi_loss_val, optimized_loss_val, rtol=1e-4, atol=1e-5
        ), f"Loss values differ: original={multi_loss_val.item()}, optimized={optimized_loss_val.item()}"

        print(f"Original loss value: {multi_loss_val.item()}")
        print(f"Optimized loss value: {optimized_loss_val.item()}")

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
            multi_loss = MultiEventRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                **params,
            )

            optimized_loss = OptimizedMultiEventRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                **params,
            )

            multi_loss_val = multi_loss(predictions, targets)
            optimized_loss_val = optimized_loss(predictions, targets)

            is_close = torch.isclose(
                multi_loss_val, optimized_loss_val, rtol=1e-4, atol=1e-5
            )
            diff_pct = abs(
                (multi_loss_val.item() - optimized_loss_val.item())
                / multi_loss_val.item()
                * 100
            )

            print(
                f"Params {params}: Original={multi_loss_val.item():.6f}, Optimized={optimized_loss_val.item():.6f}, "
                f"Difference={diff_pct:.4f}%, {'Equivalent' if is_close else 'NOT EQUIVALENT'}"
            )

            assert is_close, f"Loss values differ for params {params}"

        # Test gradient equivalence
        hazard = torch.rand(batch_size, num_events, num_cuts, requires_grad=True)
        ones = torch.ones(batch_size, num_events, 1)
        survival_base = (
            1 - torch.cumsum(torch.nn.functional.softplus(hazard), dim=2) / num_cuts
        )
        survival = torch.cat([ones, survival_base], dim=2)
        logits = torch.zeros_like(hazard)

        predictions_for_grad = SAOutput(logits=logits, hazard=hazard, survival=survival)

        # Original loss gradient
        multi_loss_val = multi_loss(predictions_for_grad, targets)
        multi_loss_val.backward(retain_graph=True)
        multi_grad = hazard.grad.clone()
        hazard.grad.zero_()

        # Optimized loss gradient
        optimized_loss_val = optimized_loss(predictions_for_grad, targets)
        optimized_loss_val.backward()
        optimized_grad = hazard.grad.clone()

        # Check gradient similarity
        grad_correlation = torch.corrcoef(
            torch.stack([multi_grad.flatten(), optimized_grad.flatten()])
        )[0, 1].item()

        print(f"\nGradient correlation: {grad_correlation:.6f}")
        assert grad_correlation > 0.99, "Gradients are not sufficiently correlated"

        # Check sign match percentage
        sign_match = torch.sign(multi_grad) == torch.sign(optimized_grad)
        sign_match_percentage = sign_match.float().mean().item() * 100

        print(f"Gradient sign match percentage: {sign_match_percentage:.2f}%")
        assert sign_match_percentage > 95, "Gradient signs don't match sufficiently"

    finally:
        # Clean up temporary files
        os.unlink(duration_cuts_file)
        os.unlink(importance_weights_file)


def benchmark_performance(
    batch_sizes: List[int] = None,
    num_events: int = 2,
    num_cuts: int = 10,
    num_iterations: int = 10,
):
    """Benchmark performance of original vs optimized implementations."""
    if batch_sizes is None:
        batch_sizes = [8, 16, 32, 64, 128, 256]
    print("\nBenchmarking performance:")

    # Create files needed for loss initialization
    duration_cuts_file = create_duration_cuts_file(num_cuts)
    importance_weights_file = create_importance_weights_file(num_events)

    try:
        # Dictionary to store results
        results = {
            "batch_size": [],
            "original_forward": [],
            "optimized_forward": [],
            "original_backward": [],
            "optimized_backward": [],
            "original_memory": [],
            "optimized_memory": [],
            "speedup_forward": [],
            "speedup_backward": [],
            "memory_reduction": [],
        }

        # Run benchmarks
        for batch_size in batch_sizes:
            print(f"Testing batch size {batch_size}...")

            # Create loss instances
            multi_loss = MultiEventRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                sigma=0.1,
                margin=0.05,
            )

            optimized_loss = OptimizedMultiEventRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                sigma=0.1,
                margin=0.05,
            )

            # Time forward pass - Original
            original_forward_times = []
            original_backward_times = []
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
                original_forward_times.append(end_time - start_time)

                # Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                original_backward_times.append(end_time - start_time)

            # Time forward pass - Optimized
            optimized_forward_times = []
            optimized_backward_times = []
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
                loss = optimized_loss(predictions, targets)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                optimized_forward_times.append(end_time - start_time)

                # Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                optimized_backward_times.append(end_time - start_time)

            # Measure peak memory usage if using CUDA
            original_memory = 0
            optimized_memory = 0
            if torch.cuda.is_available():
                # Original memory usage
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
                original_memory = torch.cuda.max_memory_allocated() / (
                    1024 * 1024
                )  # Convert to MB
                torch.cuda.empty_cache()

                # Optimized memory usage
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

                loss = optimized_loss(predictions, targets)
                loss.backward()
                optimized_memory = torch.cuda.max_memory_allocated() / (
                    1024 * 1024
                )  # Convert to MB
                torch.cuda.empty_cache()

            # Calculate average times
            avg_original_forward = (
                np.mean(original_forward_times) * 1000
            )  # Convert to ms
            avg_optimized_forward = np.mean(optimized_forward_times) * 1000
            avg_original_backward = np.mean(original_backward_times) * 1000
            avg_optimized_backward = np.mean(optimized_backward_times) * 1000

            # Calculate speedup ratios
            speedup_forward = (
                avg_original_forward / avg_optimized_forward
                if avg_optimized_forward > 0
                else 0
            )
            speedup_backward = (
                avg_original_backward / avg_optimized_backward
                if avg_optimized_backward > 0
                else 0
            )
            memory_reduction = (
                original_memory / optimized_memory if optimized_memory > 0 else 0
            )

            # Store results
            results["batch_size"].append(batch_size)
            results["original_forward"].append(avg_original_forward)
            results["optimized_forward"].append(avg_optimized_forward)
            results["original_backward"].append(avg_original_backward)
            results["optimized_backward"].append(avg_optimized_backward)
            results["original_memory"].append(original_memory)
            results["optimized_memory"].append(optimized_memory)
            results["speedup_forward"].append(speedup_forward)
            results["speedup_backward"].append(speedup_backward)
            results["memory_reduction"].append(memory_reduction)

            # Print results for this batch size
            print(
                f"  Original  - Forward: {avg_original_forward:.4f} ms, Backward: {avg_original_backward:.4f} ms"
            )
            print(
                f"  Optimized - Forward: {avg_optimized_forward:.4f} ms, Backward: {avg_optimized_backward:.4f} ms"
            )
            print(
                f"  Speedup   - Forward: {speedup_forward:.2f}x, Backward: {speedup_backward:.2f}x"
            )

            if torch.cuda.is_available():
                print(
                    f"  Memory    - Original: {original_memory:.2f} MB, Optimized: {optimized_memory:.2f} MB"
                )
                print(f"              Reduction: {memory_reduction:.2f}x")

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
    num_events_list: List[int] = None,
    num_cuts: int = 10,
    num_iterations: int = 10,
):
    """Benchmark performance scaling with number of events."""
    if num_events_list is None:
        num_events_list = [1, 2, 4, 8, 16]
    print("\nBenchmarking scaling with number of events:")

    # Dictionary to store results
    results = {
        "num_events": [],
        "original_forward": [],
        "optimized_forward": [],
        "original_backward": [],
        "optimized_backward": [],
        "speedup_forward": [],
        "speedup_backward": [],
    }

    for num_events in num_events_list:
        print(f"Testing {num_events} events...")

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

            optimized_loss = OptimizedMultiEventRankingLoss(
                duration_cuts=duration_cuts_file,
                importance_sample_weights=importance_weights_file,
                num_events=num_events,
                sigma=0.1,
                margin=0.05,
            )

            # Time forward pass - Original
            original_forward_times = []
            original_backward_times = []
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
                original_forward_times.append(end_time - start_time)

                # Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                original_backward_times.append(end_time - start_time)

            # Time forward pass - Optimized
            optimized_forward_times = []
            optimized_backward_times = []
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
                loss = optimized_loss(predictions, targets)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                optimized_forward_times.append(end_time - start_time)

                # Backward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                loss.backward()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                optimized_backward_times.append(end_time - start_time)

            # Calculate average times
            avg_original_forward = (
                np.mean(original_forward_times) * 1000
            )  # Convert to ms
            avg_optimized_forward = np.mean(optimized_forward_times) * 1000
            avg_original_backward = np.mean(original_backward_times) * 1000
            avg_optimized_backward = np.mean(optimized_backward_times) * 1000

            # Calculate speedup
            speedup_forward = (
                avg_original_forward / avg_optimized_forward
                if avg_optimized_forward > 0
                else 0
            )
            speedup_backward = (
                avg_original_backward / avg_optimized_backward
                if avg_optimized_backward > 0
                else 0
            )

            # Store results
            results["num_events"].append(num_events)
            results["original_forward"].append(avg_original_forward)
            results["optimized_forward"].append(avg_optimized_forward)
            results["original_backward"].append(avg_original_backward)
            results["optimized_backward"].append(avg_optimized_backward)
            results["speedup_forward"].append(speedup_forward)
            results["speedup_backward"].append(speedup_backward)

            # Print results for this number of events
            print(
                f"  Original  - Forward: {avg_original_forward:.4f} ms, Backward: {avg_original_backward:.4f} ms"
            )
            print(
                f"  Optimized - Forward: {avg_optimized_forward:.4f} ms, Backward: {avg_optimized_backward:.4f} ms"
            )
            print(
                f"  Speedup   - Forward: {speedup_forward:.2f}x, Backward: {speedup_backward:.2f}x"
            )
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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot forward pass times
    ax1.plot(
        results["batch_size"], results["original_forward"], "b-o", label="Original"
    )
    ax1.plot(
        results["batch_size"], results["optimized_forward"], "r-o", label="Optimized"
    )
    ax1.set_title("Forward Pass Time")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Time (ms)")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    # Plot backward pass times
    ax2.plot(
        results["batch_size"], results["original_backward"], "b-^", label="Original"
    )
    ax2.plot(
        results["batch_size"], results["optimized_backward"], "r-^", label="Optimized"
    )
    ax2.set_title("Backward Pass Time")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Time (ms)")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    # Plot speedup
    ax3.plot(
        results["batch_size"],
        results["speedup_forward"],
        "g-o",
        label="Forward Speedup",
    )
    ax3.plot(
        results["batch_size"],
        results["speedup_backward"],
        "g-^",
        label="Backward Speedup",
    )
    ax3.axhline(y=1.0, color="gray", linestyle="--")
    ax3.set_title("Speedup (Original / Optimized)")
    ax3.set_xlabel("Batch Size")
    ax3.set_ylabel("Speedup Factor")
    ax3.grid(True, linestyle="--", alpha=0.7)
    ax3.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig("multievent_performance.png")

    # If memory measurements are available, plot them
    if results["original_memory"] and results["original_memory"][0] > 0:
        plt.figure(figsize=(12, 6))

        # Plot memory usage
        plt.subplot(1, 2, 1)
        plt.plot(
            results["batch_size"], results["original_memory"], "b-o", label="Original"
        )
        plt.plot(
            results["batch_size"], results["optimized_memory"], "r-o", label="Optimized"
        )
        plt.title("Memory Usage")
        plt.xlabel("Batch Size")
        plt.ylabel("Memory (MB)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        # Plot memory reduction
        plt.subplot(1, 2, 2)
        plt.plot(results["batch_size"], results["memory_reduction"], "g-o")
        plt.axhline(y=1.0, color="gray", linestyle="--")
        plt.title("Memory Reduction Factor")
        plt.xlabel("Batch Size")
        plt.ylabel("Reduction Factor")
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig("multievent_memory.png")


def plot_event_scaling_results(results):
    """Plot results for event scaling benchmark."""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot computation times
    ax1.plot(
        results["num_events"],
        results["original_forward"],
        "b-o",
        label="Original Forward",
    )
    ax1.plot(
        results["num_events"],
        results["optimized_forward"],
        "r-o",
        label="Optimized Forward",
    )
    ax1.plot(
        results["num_events"],
        results["original_backward"],
        "b-^",
        label="Original Backward",
    )
    ax1.plot(
        results["num_events"],
        results["optimized_backward"],
        "r-^",
        label="Optimized Backward",
    )
    ax1.set_title("Computation Time vs Number of Events")
    ax1.set_xlabel("Number of Events")
    ax1.set_ylabel("Time (ms)")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    # Plot speedup
    ax2.plot(
        results["num_events"],
        results["speedup_forward"],
        "g-o",
        label="Forward Speedup",
    )
    ax2.plot(
        results["num_events"],
        results["speedup_backward"],
        "g-^",
        label="Backward Speedup",
    )
    ax2.axhline(y=1.0, color="gray", linestyle="--")
    ax2.set_title("Speedup vs Number of Events")
    ax2.set_xlabel("Number of Events")
    ax2.set_ylabel("Speedup Factor")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("multievent_event_scaling.png")


def main():
    """Run all tests and benchmarks."""
    # Test functional equivalence
    print("Testing functional equivalence...")
    test_functional_equivalence()

    # Benchmark performance
    results_batch = benchmark_performance()

    # Benchmark event scaling
    results_events = benchmark_num_events()

    print("\nBenchmark summary:")
    print(f"Maximum forward speedup: {max(results_batch['speedup_forward']):.2f}x")
    print(f"Maximum backward speedup: {max(results_batch['speedup_backward']):.2f}x")

    if results_batch["original_memory"] and results_batch["original_memory"][0] > 0:
        print(
            f"Maximum memory reduction: {max(results_batch['memory_reduction']):.2f}x"
        )

    print("\nEvent scaling summary:")
    print(f"Maximum forward speedup: {max(results_events['speedup_forward']):.2f}x")
    print(f"Maximum backward speedup: {max(results_events['speedup_backward']):.2f}x")


if __name__ == "__main__":
    main()
